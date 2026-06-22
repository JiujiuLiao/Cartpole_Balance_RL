#!/usr/bin/env python3
"""
Cart-pole pendulum angle detector (v3 — yellow->blue dual-color vector).

Fixes vs v2:
- No more "max-area picks color": always detect BOTH yellow and blue, angle =
  atan2(blue - yellow) so the 180-deg flip is eliminated by construction.
- Angle unwrapped before differentiation; omega = 1-frame diff + LPF (no 5-frame
  lag, no +/-pi jump explosion).
- Morphology open+close + minimum-area filter for clean masks.
- Failed-detection frame: keep previous angle, mark valid=0 over ZMQ; downstream
  controller decides what to do.
- 120 fps locked via FrameDurationLimits + raw={"size":(1536,864)} sensor mode.
- Axis pivot persisted in axis_calib.json (press 'c' to recalibrate by mouse).
- HSV tuner trackbars (press 'h'), values saved with 's'.
- ZMQ legacy CSV string on port 9999 (compatible with camera_comm.py):
  "a,a,a,a,theta_rad,omega_rad_s,pole_up,ts_ns,sensor_ts_ns"; sent ONLY on valid
  frames. ts_ns = time.time_ns() at send (wall clock, legacy). sensor_ts_ns =
  SensorTimestamp (ns, CLOCK_BOOTTIME) = photon-capture time, 0 if unavailable.

Angle convention (matches the user's spec):
  theta = 0      -> pendulum tip (blue/green end) pointing UP
  theta = +/-pi  -> pendulum tip pointing DOWN (rest)
  +theta         -> clockwise rotation as seen from the camera
"""

import cv2
import numpy as np
import zmq
import time
import json
import os
import math
import argparse
from picamera2 import Picamera2


# IMX708 120fps sensor mode is 1536x864.
RAW_W, RAW_H = 1536, 864
SCALE = 0.5                      # main-stream downscale via ISP
W = int(RAW_W * SCALE)           # 768
H = int(RAW_H * SCALE)           # 432

TARGET_FPS = 120
FRAME_DURATION_US = int(round(1_000_000 / TARGET_FPS))   # 8333
EXPOSURE_US = 4000               # must be <= FRAME_DURATION_US; bump if scene dim
ANALOGUE_GAIN = 4.0
LENS_POSITION = 4.5              # tune to pendulum distance

# HSV defaults.
# Blue tightened (S>=150, V>=80) to reject low-saturation background blueish
# objects (doorknobs, shadows). Loosen via HSV tuner ('h' key) if the real
# blue rod is missed.
# Yellow S_lo bumped 90 -> 130: the paddle is a SATURATED yellow, whereas
# beige / light-brown background (cardboard, wood) shares the same hue but is
# desaturated (S ~ 40-100). Raising S_lo rejects that background. If the real
# paddle is missed under dim light, lower it via the HSV tuner; the spatial
# gate below is the second line of defense.
HSV_YELLOW_LOW = np.array([18, 130, 90], dtype=np.uint8)
HSV_YELLOW_HIGH = np.array([35, 255, 255], dtype=np.uint8)
HSV_BLUE_LOW = np.array([100, 150, 80], dtype=np.uint8)
HSV_BLUE_HIGH = np.array([130, 255, 255], dtype=np.uint8)

# Bumped from 25 to 500 to reject small background false positives.
# The real pendulum blue rod / yellow paddle should be far larger than this
# in the camera frame; if not, lower this value.
MIN_AREA_PX = 500
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Spatial gate for yellow (defense vs beige/cardboard background).
# The yellow paddle is on the SHORT arm (~3 cm) and is therefore always close
# to the pivot. Any "yellow" blob farther than this many pixels from the axis
# cannot be the paddle -> it is background -> reject it (then the angle falls
# back to the axis->blue formula instead of being hijacked by a fake yellow).
# Tune from the GUI: watch the printed yellow-to-axis distance with the real
# paddle visible, then set this ~1.5x of that. Default sized for the 768x432
# frame (~12.7 px/cm) assuming the paddle centroid sits <~9 cm from the pivot.
YELLOW_MAX_AXIS_DIST_PX = 320

OMEGA_LPF_ALPHA = 0.55           # EMA: ~3-frame time constant @120fps (~25ms)

AXIS_CALIB_FILE = "axis_calib.json"
HSV_CALIB_FILE = "hsv_calib.json"
# Port 9999 + legacy CSV format to stay compatible with camera_comm.py /
# balance_pid.py (which read parts[4]=theta, parts[5]=omega, parts[6]=pole_up).
ZMQ_ENDPOINT = "tcp://*:9999"

WINDOW = "Pendulum"
HSV_WIN = "HSV Tuner"

# Show every Nth frame to keep display from eating CPU at 120fps.
DISPLAY_EVERY_N = 2


# -------------------- Persistence --------------------

def load_axis():
    if os.path.exists(AXIS_CALIB_FILE):
        try:
            with open(AXIS_CALIB_FILE) as f:
                d = json.load(f)
            return int(d["x"]), int(d["y"])
        except Exception as e:
            print(f"[axis] load failed ({e}); using image center")
    return W // 2, H // 2


def save_axis(x, y):
    with open(AXIS_CALIB_FILE, "w") as f:
        json.dump({"x": int(x), "y": int(y)}, f)


def load_hsv():
    if os.path.exists(HSV_CALIB_FILE):
        try:
            with open(HSV_CALIB_FILE) as f:
                d = json.load(f)
            return (np.array(d["yl"], dtype=np.uint8),
                    np.array(d["yh"], dtype=np.uint8),
                    np.array(d["bl"], dtype=np.uint8),
                    np.array(d["bh"], dtype=np.uint8))
        except Exception as e:
            print(f"[hsv] load failed ({e}); using defaults")
    return (HSV_YELLOW_LOW.copy(), HSV_YELLOW_HIGH.copy(),
            HSV_BLUE_LOW.copy(), HSV_BLUE_HIGH.copy())


def save_hsv(yl, yh, bl, bh):
    with open(HSV_CALIB_FILE, "w") as f:
        json.dump({"yl": yl.tolist(), "yh": yh.tolist(),
                   "bl": bl.tolist(), "bh": bh.tolist()}, f)


# -------------------- Detection --------------------

def detect_centroid(hsv, lo, hi, min_area=None):
    """Return (cx, cy) of the largest valid blob in HSV range, or None."""
    if min_area is None:
        min_area = MIN_AREA_PX
    mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


def gate_yellow(yc, axis):
    """
    Reject a yellow blob that is too far from the pivot to be the short paddle.
    Background beige (cardboard, wood) can pass the HSV filter; this geometric
    gate removes it because the real paddle is always near the axis.

    Returns (yc_or_None, distance_px_or_None):
      - yc unchanged + distance  if accepted (or yc is None)
      - None + distance          if rejected (so caller can fall back / log it)
    """
    if yc is None:
        return None, None
    d = math.hypot(yc[0] - axis[0], yc[1] - axis[1])
    if d > YELLOW_MAX_AXIS_DIST_PX:
        return None, d
    return yc, d


def compute_angle(yellow, blue, axis):
    """
    Returns (angle_rad, mode).
    Image y points down, so y_world = -y_image inside atan2.
    """
    if yellow is not None and blue is not None:
        dx = blue[0] - yellow[0]
        dy = yellow[1] - blue[1]
        return math.atan2(dx, dy), "dual"
    if blue is not None:
        dx = blue[0] - axis[0]
        dy = axis[1] - blue[1]
        return math.atan2(dx, dy), "axis->blue"
    if yellow is not None:
        # Yellow short end is on the pivot side; flip to get tip direction.
        dx = axis[0] - yellow[0]
        dy = yellow[1] - axis[1]
        return math.atan2(dx, dy), "axis->yellow"
    return None, "none"


# -------------------- Downscaled detection (speed-up; coarser but simple) --------------------
# Run cvtColor + detect_centroid on a downscaled copy of the frame, then scale
# the centroids back to full-frame coordinates. Fewer pixels -> faster; the
# measurement (gate / compute_angle / axis) is UNCHANGED because the centroids
# come back in full-frame coords. (Replaces the old ROI cropping.)
DETECT_SCALE = 0.5     # 0.5 -> detect on 384x216. Lower = faster but coarser angle.


def detect_markers(frame, yl, yh, bl, bh):
    """Detect yellow & blue centroids in FULL-FRAME coords by detecting on a
    downscaled copy. Returns (yc_raw, bc)."""
    if DETECT_SCALE != 1.0:
        small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE,
                           interpolation=cv2.INTER_AREA)
        min_area = MIN_AREA_PX * DETECT_SCALE * DETECT_SCALE   # blob shrinks by scale^2
        inv = 1.0 / DETECT_SCALE
    else:
        small, min_area, inv = frame, MIN_AREA_PX, 1.0
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    yc = detect_centroid(hsv, yl, yh, min_area)
    bc = detect_centroid(hsv, bl, bh, min_area)
    if yc is not None:
        yc = (yc[0] * inv, yc[1] * inv)        # scaled -> full-frame coords
    if bc is not None:
        bc = (bc[0] * inv, bc[1] * inv)
    return yc, bc


def unwrap(prev, current):
    if prev is None:
        return current
    delta = current - prev
    if delta > math.pi:
        return current - 2 * math.pi
    if delta < -math.pi:
        return current + 2 * math.pi
    return current


# -------------------- UI: axis calibration --------------------

class AxisCalibrator:
    def __init__(self):
        self.active = False
        self.click = None

    def on_mouse(self, event, x, y, flags, param):
        if self.active and event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)


def open_hsv_tuner(yl, yh, bl, bh):
    cv2.namedWindow(HSV_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(HSV_WIN, 360, 320)
    bars = [
        ("Y_H_lo", int(yl[0]), 180), ("Y_H_hi", int(yh[0]), 180),
        ("Y_S_lo", int(yl[1]), 255), ("Y_V_lo", int(yl[2]), 255),
        ("B_H_lo", int(bl[0]), 180), ("B_H_hi", int(bh[0]), 180),
        ("B_S_lo", int(bl[1]), 255), ("B_V_lo", int(bl[2]), 255),
    ]
    for name, val, vmax in bars:
        cv2.createTrackbar(name, HSV_WIN, val, vmax, lambda _: None)


def read_hsv_tuner(yl, yh, bl, bh):
    yl[0] = cv2.getTrackbarPos("Y_H_lo", HSV_WIN)
    yh[0] = cv2.getTrackbarPos("Y_H_hi", HSV_WIN)
    yl[1] = cv2.getTrackbarPos("Y_S_lo", HSV_WIN)
    yl[2] = cv2.getTrackbarPos("Y_V_lo", HSV_WIN)
    bl[0] = cv2.getTrackbarPos("B_H_lo", HSV_WIN)
    bh[0] = cv2.getTrackbarPos("B_H_hi", HSV_WIN)
    bl[1] = cv2.getTrackbarPos("B_S_lo", HSV_WIN)
    bl[2] = cv2.getTrackbarPos("B_V_lo", HSV_WIN)


# -------------------- Main --------------------

def main():
    global MIN_AREA_PX, YELLOW_MAX_AXIS_DIST_PX

    parser = argparse.ArgumentParser(description="Cart-pole vision (v3)")
    parser.add_argument("--axis-x", type=int, default=None,
                        help="Override axis X pixel (can be outside frame)")
    parser.add_argument("--axis-y", type=int, default=None,
                        help="Override axis Y pixel (can be negative or > H)")
    parser.add_argument("--min-area", type=int, default=None,
                        help=f"Min contour area in px (default {MIN_AREA_PX})")
    parser.add_argument("--yellow-gate", type=int, default=None,
                        help="Max yellow-to-axis distance in px; farther blobs "
                             f"are rejected as background (default "
                             f"{YELLOW_MAX_AXIS_DIST_PX})")
    parser.add_argument("--no-gui", action="store_true",
                        help="Headless mode: no preview window, no key handler. "
                             "Cuts CPU significantly — use for swing-up/balance runs. "
                             "Quit with Ctrl+C.")
    cli = parser.parse_args()

    if cli.min_area is not None:
        MIN_AREA_PX = cli.min_area
        print(f"[cli]  MIN_AREA_PX overridden -> {MIN_AREA_PX}")

    if cli.yellow_gate is not None:
        YELLOW_MAX_AXIS_DIST_PX = cli.yellow_gate
        print(f"[cli]  YELLOW_MAX_AXIS_DIST_PX overridden -> {YELLOW_MAX_AXIS_DIST_PX}")

    GUI = not cli.no_gui
    if not GUI:
        print("[cli]  --no-gui: headless (no preview, no key bindings)")

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (W, H), "format": "RGB888"},
        raw={"size": (RAW_W, RAW_H)},
        controls={
            "FrameDurationLimits": (FRAME_DURATION_US, FRAME_DURATION_US),
            "ExposureTime": EXPOSURE_US,
            "AnalogueGain": ANALOGUE_GAIN,
            "AwbEnable": False,
            "AeEnable": False,
            "LensPosition": LENS_POSITION,
        },
        buffer_count=4,
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.4)

    axis_x, axis_y = load_axis()
    if cli.axis_x is not None:
        axis_x = cli.axis_x
        print(f"[cli]  axis_x overridden -> {axis_x}")
    if cli.axis_y is not None:
        axis_y = cli.axis_y
        print(f"[cli]  axis_y overridden -> {axis_y}")
    if not (0 <= axis_x < W and 0 <= axis_y < H):
        print(f"[warn] axis ({axis_x},{axis_y}) is OUTSIDE the {W}x{H} frame. "
              f"axis_yellow / axis_blue fallback formulas still work, but the "
              f"yellow '+' marker won't be visible on screen.")
    yl, yh, bl, bh = load_hsv()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 4)   # drop old frames rather than queue
    sock.bind(ZMQ_ENDPOINT)

    calib = AxisCalibrator()
    if GUI:
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, calib.on_mouse)
    tuner_open = False

    angle_prev = None
    omega_filt = 0.0
    t_prev = time.perf_counter()
    fps_t0 = t_prev
    fps_n = 0
    real_fps = 0.0
    frame_idx = 0

    print(f"[init] axis=({axis_x},{axis_y})  target={TARGET_FPS}fps  size={W}x{H}")
    if GUI:
        print("[keys] c=calib axis | h=HSV tuner | s=save HSV | q=quit")
    else:
        print("[keys] (disabled in --no-gui)  quit: Ctrl+C")
    print(f"[zmq]  PUB at {ZMQ_ENDPOINT}  legacy CSV: a,a,a,a,theta,omega,pole_up,ts")

    try:
        while True:
            # picamera2 "RGB888" returns memory-order BGR -> already OpenCV-native.
            # Use capture_request() (not capture_array()) so we can read the
            # hardware SensorTimestamp from frame metadata — the photon-capture
            # time (ns, CLOCK_BOOTTIME). Published as parts[8] so the controller
            # can compute true per-frame age INCLUDING the ~20ms capture latency.
            # Release the request promptly to return the buffer to the pool.
            req = picam2.capture_request()
            ts_sensor = req.get_metadata().get("SensorTimestamp")  # ns, CLOCK_BOOTTIME
            frame = req.make_array("main")
            req.release()
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            frame_idx += 1

            if tuner_open:
                read_hsv_tuner(yl, yh, bl, bh)

            # Detection on a downscaled copy (centroids returned in full coords).
            yc_raw, bc = detect_markers(frame, yl, yh, bl, bh)

            # Spatial gate: drop a yellow blob too far from the axis (background
            # beige). yc_raw is kept for display so a rejection is visible.
            yc, y_dist = gate_yellow(yc_raw, (axis_x, axis_y))
            yellow_rejected = (yc is None and yc_raw is not None)

            angle_raw, mode = compute_angle(yc, bc, (axis_x, axis_y))
            valid = angle_raw is not None

            if valid:
                angle = unwrap(angle_prev, angle_raw)
                if angle_prev is not None and dt > 1e-6:
                    omega_inst = (angle - angle_prev) / dt
                    omega_filt = (OMEGA_LPF_ALPHA * omega_inst
                                  + (1.0 - OMEGA_LPF_ALPHA) * omega_filt)
                angle_prev = angle
            else:
                angle = angle_prev if angle_prev is not None else 0.0
                # Hold omega: don't differentiate against a stale frame.

            # FPS tally
            fps_n += 1
            if t_now - fps_t0 >= 1.0:
                real_fps = fps_n / (t_now - fps_t0)
                fps_n = 0
                fps_t0 = t_now
                if not GUI:
                    ang_deg = (math.degrees(math.atan2(math.sin(angle),
                                                       math.cos(angle)))
                               if angle_prev is not None else float('nan'))
                    ydist_txt = f"{y_dist:.0f}" if y_dist is not None else "--"
                    print(f"[v3] fps={real_fps:5.1f}  ang={ang_deg:+7.1f}deg  "
                          f"w={omega_filt:+6.2f}  mode={mode}  valid={int(valid)}  "
                          f"ydist={ydist_txt}{' REJ' if yellow_rejected else ''}")

            # --- ZMQ publish (legacy CSV, ONLY on valid frames) ---
            # Format expected by camera_comm.py parse_angle():
            #   parts[0..4] = theta_rad (newest at [4]); we repeat the same value
            #   parts[5]    = omega (rad/s)
            #   parts[6]    = pole_up (1 near upright, else 0)
            #   parts[7]    = wall-clock send time, time.time_ns() (legacy)
            #   parts[8]    = SensorTimestamp (ns, CLOCK_BOOTTIME) = photon-capture
            #                 time; 0 if metadata unavailable. The controller uses
            #                 this for dynamic delay (now_BOOTTIME - parts[8]).
            # Sending ONLY on valid frames means a sustained detection loss
            # produces no messages -> downstream ZMQ_TIMEOUT stops the motor.
            if valid:
                wrapped = math.atan2(math.sin(angle), math.cos(angle))
                pole_up = 1 if abs(wrapped) < math.radians(5) else 0
                sock.send_string(
                    f"{wrapped:.6f},{wrapped:.6f},{wrapped:.6f},"
                    f"{wrapped:.6f},{wrapped:.6f},{omega_filt:.6f},"
                    f"{pole_up},{time.time_ns()},"
                    f"{ts_sensor if ts_sensor is not None else 0}"
                )

            # --- Display (throttled, GUI only) ---
            if GUI and frame_idx % DISPLAY_EVERY_N == 0:
                disp = frame.copy()
                cv2.drawMarker(disp, (axis_x, axis_y), (0, 255, 255),
                               cv2.MARKER_CROSS, 14, 2)
                # Yellow spatial gate: blobs outside this circle are rejected.
                cv2.circle(disp, (axis_x, axis_y), YELLOW_MAX_AXIS_DIST_PX,
                           (0, 255, 255), 1)
                if yc is not None:
                    cv2.circle(disp, (int(yc[0]), int(yc[1])), 6, (0, 255, 255), 2)
                if yellow_rejected:
                    # Red X on a rejected (background) yellow blob.
                    cv2.drawMarker(disp, (int(yc_raw[0]), int(yc_raw[1])),
                                   (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
                if bc is not None:
                    cv2.circle(disp, (int(bc[0]), int(bc[1])), 6, (255, 60, 0), 2)
                if yc is not None and bc is not None:
                    cv2.line(disp, (int(yc[0]), int(yc[1])),
                             (int(bc[0]), int(bc[1])), (0, 200, 255), 2)

                # Wrap unwrapped accumulator back to (-pi, pi] for display.
                wrapped_rad = math.atan2(math.sin(angle), math.cos(angle))
                ydist_txt = f"{y_dist:.0f}" if y_dist is not None else "--"
                txt = (f"ang={math.degrees(wrapped_rad):+7.2f}deg  "
                       f"w={omega_filt:+7.2f}rad/s  "
                       f"mode={mode}  fps={real_fps:5.1f}  "
                       f"valid={int(valid)}  ydist={ydist_txt}"
                       f"{'(REJ)' if yellow_rejected else ''}")
                cv2.putText(disp, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
                if calib.active:
                    cv2.putText(disp, "CLICK pivot to calibrate axis",
                                (5, H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow(WINDOW, disp)

            # Apply pending mouse click (after window exists)
            if calib.click is not None:
                axis_x, axis_y = calib.click
                save_axis(axis_x, axis_y)
                calib.click = None
                calib.active = False
                print(f"[calib] axis -> ({axis_x},{axis_y}) saved")

            if GUI:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    calib.active = True
                    print("[calib] click the pivot in the preview window")
                elif key == ord('h') and not tuner_open:
                    open_hsv_tuner(yl, yh, bl, bh)
                    tuner_open = True
                    print("[hsv]  tuner opened")
                elif key == ord('s'):
                    save_hsv(yl, yh, bl, bh)
                    print("[hsv]  saved")
    finally:
        try:
            sock.close(linger=0)
            ctx.term()
        except Exception:
            pass
        cv2.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    main()
