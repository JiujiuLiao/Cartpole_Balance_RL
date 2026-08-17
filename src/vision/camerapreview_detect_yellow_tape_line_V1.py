#!/usr/bin/env python3
"""
Cart-pole pendulum angle detector (V1 - yellow tape line fit).

This is an experimental alternative to camerapreview_detect_color_V3.py.
It uses the existing yellow HSV calibration, but treats the yellow tape as
one elongated line instead of using a yellow-centroid -> blue-centroid vector.

Detection pipeline:
  yellow HSV mask -> morphology -> elongated contour selection
  -> cv2.fitLine() -> line direction -> angle relative to calibrated pivot

The line direction alone has a 180-degree ambiguity.  The calibrated pivot
and the fitted tape centroid are used to orient the line from pivot toward the
pendulum tip.  Therefore, the tape should start near the pivot and extend
toward the tip, not be centered symmetrically around the pivot.

Angle convention (same as V3):
  theta = 0       -> yellow tape points UP
  theta = +/-pi   -> yellow tape points DOWN
  +theta          -> clockwise as seen by the camera

The ZMQ message remains compatible with the existing camera subscriber:
  "a,a,a,a,theta_rad,omega_rad_s,pole_up,ts_ns,sensor_ts_ns"

This script is for measurement and comparison first.  Do not use it for LQR
control until the static angle calibration and dynamic comparison are done.
"""

import argparse
import cv2
import json
import math
import os
import time

import numpy as np
import zmq
from picamera2 import Picamera2


# -------------------- Camera --------------------

RAW_W, RAW_H = 1536, 864
SCALE = 0.5
W = int(RAW_W * SCALE)
H = int(RAW_H * SCALE)

TARGET_FPS = 120
FRAME_DURATION_US = int(round(1_000_000 / TARGET_FPS))
EXPOSURE_US = 4000
ANALOGUE_GAIN = 4.0
LENS_POSITION = 4.5

# The defaults match camerapreview_detect_color_V3.py.
HSV_YELLOW_LOW = np.array([18, 130, 90], dtype=np.uint8)
HSV_YELLOW_HIGH = np.array([35, 255, 255], dtype=np.uint8)

# Accuracy-test default: use the full 768x432 main stream.  Set to 0.5 from
# the command line if the Raspberry Pi cannot sustain the target frame rate.
DETECT_SCALE = 1.0

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
MIN_LINE_AREA_PX = 150
MIN_LINE_LENGTH_PX = 80
MIN_LINE_ASPECT = 4.0
MAX_PIVOT_GAP_PX = 80
MIN_CENTROID_AXIS_DIST_PX = 8

OMEGA_LPF_ALPHA = 0.55
ZMQ_PORT = 9999

_VISION_DIR = os.path.dirname(os.path.abspath(__file__))
AXIS_CALIB_FILE = os.path.join(_VISION_DIR, "axis_calib.json")
HSV_CALIB_FILE = os.path.join(_VISION_DIR, "hsv_calib.json")

WINDOW = "Yellow tape line"
HSV_WIN = "Yellow HSV tuner"
DISPLAY_EVERY_N = 2


# -------------------- Calibration persistence --------------------

def load_axis():
    if os.path.exists(AXIS_CALIB_FILE):
        try:
            with open(AXIS_CALIB_FILE, encoding="utf-8") as f:
                d = json.load(f)
            return int(d["x"]), int(d["y"])
        except Exception as exc:
            print(f"[axis] load failed ({exc}); using image center")
    return W // 2, H // 2


def save_axis(x, y):
    with open(AXIS_CALIB_FILE, "w", encoding="utf-8") as f:
        json.dump({"x": int(x), "y": int(y)}, f)


def load_hsv():
    if os.path.exists(HSV_CALIB_FILE):
        try:
            with open(HSV_CALIB_FILE, encoding="utf-8") as f:
                d = json.load(f)
            return np.array(d["yl"], dtype=np.uint8), np.array(
                d["yh"], dtype=np.uint8
            )
        except Exception as exc:
            print(f"[hsv] load failed ({exc}); using defaults")
    return HSV_YELLOW_LOW.copy(), HSV_YELLOW_HIGH.copy()


def save_hsv(yl, yh):
    # Preserve the blue entries if the shared V3 calibration file already has
    # them, so opening the tuner here does not erase V3's blue thresholds.
    d = {}
    if os.path.exists(HSV_CALIB_FILE):
        try:
            with open(HSV_CALIB_FILE, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            d = {}
    d["yl"] = yl.tolist()
    d["yh"] = yh.tolist()
    with open(HSV_CALIB_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f)


# -------------------- Yellow line detection --------------------

def _line_candidate(contour, mask, axis, detect_scale):
    """Return line features for one contour, or None if it is not line-like."""
    inv = 1.0 / detect_scale
    area = cv2.contourArea(contour) * inv * inv
    if area < MIN_LINE_AREA_PX:
        return None

    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    long_side = max(width, height) * inv
    short_side = min(width, height) * inv
    if long_side < MIN_LINE_LENGTH_PX or short_side <= 0:
        return None

    aspect = long_side / short_side
    if aspect < MIN_LINE_ASPECT:
        return None

    # Use the filled component for fitting, not just the contour boundary.
    component = np.zeros_like(mask)
    cv2.drawContours(component, [contour], -1, 255, thickness=-1)
    points_small = cv2.findNonZero(component)
    if points_small is None or len(points_small) < 10:
        return None
    points = points_small.reshape(-1, 2).astype(np.float32) * inv

    contour_points = contour.reshape(-1, 2).astype(np.float32) * inv
    dx = contour_points[:, 0] - axis[0]
    dy = contour_points[:, 1] - axis[1]
    pivot_gap = float(np.min(np.hypot(dx, dy)))
    if pivot_gap > MAX_PIVOT_GAP_PX:
        return None

    centroid = points.mean(axis=0)
    centroid_dist = float(np.hypot(centroid[0] - axis[0], centroid[1] - axis[1]))
    if centroid_dist < MIN_CENTROID_AXIS_DIST_PX:
        return None

    fit_points = points.reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(
        fit_points, cv2.DIST_L2, 0, 0.01, 0.01
    )
    vx = float(vx)
    vy = float(vy)

    # Orient the otherwise ambiguous fitted line from pivot toward the tape
    # centroid.  This gives the direction needed for +/- theta.
    to_centroid = centroid - np.array(axis, dtype=np.float32)
    if vx * float(to_centroid[0]) + vy * float(to_centroid[1]) < 0:
        vx = -vx
        vy = -vy

    # Prefer a long, elongated component close to the calibrated pivot.
    score = long_side * aspect / (1.0 + pivot_gap)
    return {
        "contour": contour,
        "centroid": centroid,
        "vx": vx,
        "vy": vy,
        "length": long_side,
        "width": short_side,
        "aspect": aspect,
        "pivot_gap": pivot_gap,
        "score": score,
        "x0": float(x0) * inv,
        "y0": float(y0) * inv,
    }


def detect_yellow_line(frame, yl, yh, axis, detect_scale):
    """Detect the best elongated yellow component in full-frame coordinates."""
    if detect_scale != 1.0:
        small = cv2.resize(
            frame, None, fx=detect_scale, fy=detect_scale,
            interpolation=cv2.INTER_AREA
        )
    else:
        small = frame

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, yl, yh)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for contour in contours:
        candidate = _line_candidate(contour, mask, axis, detect_scale)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return None, mask
    return max(candidates, key=lambda item: item["score"]), mask


def compute_angle(candidate):
    """Convert an oriented image line into the V3 angle convention."""
    # Image y increases downwards; therefore -vy is the upward coordinate.
    return math.atan2(candidate["vx"], -candidate["vy"])


def unwrap(prev, current):
    if prev is None:
        return current
    delta = current - prev
    if delta > math.pi:
        return current - 2.0 * math.pi
    if delta < -math.pi:
        return current + 2.0 * math.pi
    return current


# -------------------- UI helpers --------------------

class AxisCalibrator:
    def __init__(self):
        self.active = False
        self.click = None

    def on_mouse(self, event, x, y, flags, param):
        del flags, param
        if self.active and event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)


def open_hsv_tuner(yl, yh):
    cv2.namedWindow(HSV_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(HSV_WIN, 360, 180)
    bars = [
        ("Y_H_lo", int(yl[0]), 180), ("Y_H_hi", int(yh[0]), 180),
        ("Y_S_lo", int(yl[1]), 255), ("Y_V_lo", int(yl[2]), 255),
    ]
    for name, value, maximum in bars:
        cv2.createTrackbar(name, HSV_WIN, value, maximum, lambda _: None)


def read_hsv_tuner(yl, yh):
    yl[0] = cv2.getTrackbarPos("Y_H_lo", HSV_WIN)
    yh[0] = cv2.getTrackbarPos("Y_H_hi", HSV_WIN)
    yl[1] = cv2.getTrackbarPos("Y_S_lo", HSV_WIN)
    yl[2] = cv2.getTrackbarPos("Y_V_lo", HSV_WIN)


def fmt(value, digits=6):
    return f"{value:.{digits}f}"


# -------------------- Main --------------------

def main():
    global MIN_LINE_AREA_PX, MIN_LINE_LENGTH_PX
    global MIN_LINE_ASPECT, MAX_PIVOT_GAP_PX, DETECT_SCALE

    parser = argparse.ArgumentParser(
        description="Cart-pole vision: yellow tape line angle detector"
    )
    parser.add_argument("--axis-x", type=int, default=None,
                        help="Override calibrated pivot X in the 768x432 frame")
    parser.add_argument("--axis-y", type=int, default=None,
                        help="Override calibrated pivot Y in the 768x432 frame")
    parser.add_argument("--detect-scale", type=float, default=DETECT_SCALE,
                        choices=(0.5, 1.0),
                        help="Detection scale; use 1.0 for accuracy testing")
    parser.add_argument("--min-area", type=int, default=MIN_LINE_AREA_PX,
                        help="Minimum yellow component area in full-frame px")
    parser.add_argument("--min-length", type=int, default=MIN_LINE_LENGTH_PX,
                        help="Minimum fitted line length in full-frame px")
    parser.add_argument("--min-aspect", type=float, default=MIN_LINE_ASPECT,
                        help="Minimum long/short aspect ratio")
    parser.add_argument("--pivot-gap", type=int, default=MAX_PIVOT_GAP_PX,
                        help="Maximum distance from contour to pivot in px")
    parser.add_argument("--zmq-port", type=int, default=ZMQ_PORT,
                        help="ZMQ PUB port (default 9999)")
    parser.add_argument("--record", type=str, default=None,
                        help="Optional CSV path for diagnostic recording")
    parser.add_argument("--no-gui", action="store_true",
                        help="Headless mode; quit with Ctrl+C")
    args = parser.parse_args()

    DETECT_SCALE = args.detect_scale
    MIN_LINE_AREA_PX = args.min_area
    MIN_LINE_LENGTH_PX = args.min_length
    MIN_LINE_ASPECT = args.min_aspect
    MAX_PIVOT_GAP_PX = args.pivot_gap

    gui = not args.no_gui
    axis_x, axis_y = load_axis()
    if args.axis_x is not None:
        axis_x = args.axis_x
    if args.axis_y is not None:
        axis_y = args.axis_y
    axis = (axis_x, axis_y)
    yl, yh = load_hsv()

    print("==============================================================")
    print("  Yellow tape line angle detector V1")
    print("==============================================================")
    print(f"  camera       : {W}x{H} main stream, target {TARGET_FPS} Hz")
    print(f"  detect scale : {DETECT_SCALE}")
    print(f"  pivot        : ({axis_x}, {axis_y})")
    print(f"  line rules   : length >= {MIN_LINE_LENGTH_PX}px, "
          f"aspect >= {MIN_LINE_ASPECT:.1f}, pivot gap <= {MAX_PIVOT_GAP_PX}px")
    print(f"  ZMQ PUB      : tcp://*:{args.zmq_port}")
    print("  NOTE         : measurement/comparison first; not LQR-ready yet")

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

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 4)
    sock.bind(f"tcp://*:{args.zmq_port}")

    record_file = None
    record_writer = None
    if args.record:
        import csv
        record_file = open(args.record, "w", newline="", encoding="utf-8")
        record_writer = csv.writer(record_file)
        record_writer.writerow([
            "t_s", "sensor_ts_ns", "theta_rad", "omega_rad_s", "valid",
            "line_length_px", "line_width_px", "line_aspect",
            "pivot_gap_px", "centroid_x_px", "centroid_y_px", "fps",
        ])
        print(f"  record       : {args.record}")

    calibrator = AxisCalibrator()
    tuner_open = False
    if gui:
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, calibrator.on_mouse)

    angle_prev = None
    omega_filt = 0.0
    t_prev = time.perf_counter()
    t0 = t_prev
    fps_t0 = t_prev
    fps_n = 0
    real_fps = 0.0
    frame_idx = 0

    print("[keys] c=calibrate pivot | h=HSV tuner | s=save HSV | q=quit")

    try:
        while True:
            request = picam2.capture_request()
            sensor_ts = request.get_metadata().get("SensorTimestamp")
            frame = request.make_array("main")
            request.release()

            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            frame_idx += 1

            if tuner_open:
                read_hsv_tuner(yl, yh)

            candidate, mask = detect_yellow_line(
                frame, yl, yh, axis, DETECT_SCALE
            )
            valid = candidate is not None
            if valid:
                angle_raw = compute_angle(candidate)
                angle = unwrap(angle_prev, angle_raw)
                if angle_prev is not None and dt > 1e-6:
                    omega_inst = (angle - angle_prev) / dt
                    omega_filt = (
                        OMEGA_LPF_ALPHA * omega_inst
                        + (1.0 - OMEGA_LPF_ALPHA) * omega_filt
                    )
                angle_prev = angle
            else:
                angle = angle_prev if angle_prev is not None else 0.0

            fps_n += 1
            if now - fps_t0 >= 1.0:
                real_fps = fps_n / (now - fps_t0)
                fps_n = 0
                fps_t0 = now

            if valid:
                wrapped = math.atan2(math.sin(angle), math.cos(angle))
                pole_up = int(abs(wrapped) < math.radians(5.0))
                sock.send_string(
                    f"{fmt(wrapped)},{fmt(wrapped)},{fmt(wrapped)},"
                    f"{fmt(wrapped)},{fmt(wrapped)},{fmt(omega_filt)},"
                    f"{pole_up},{time.time_ns()},"
                    f"{sensor_ts if sensor_ts is not None else 0}"
                )
            else:
                wrapped = math.atan2(math.sin(angle), math.cos(angle))

            if record_writer:
                record_writer.writerow([
                    fmt(now - t0, 6), sensor_ts or 0,
                    fmt(wrapped), fmt(omega_filt), int(valid),
                    fmt(candidate["length"], 3) if valid else "",
                    fmt(candidate["width"], 3) if valid else "",
                    fmt(candidate["aspect"], 3) if valid else "",
                    fmt(candidate["pivot_gap"], 3) if valid else "",
                    fmt(candidate["centroid"][0], 3) if valid else "",
                    fmt(candidate["centroid"][1], 3) if valid else "",
                    fmt(real_fps, 3),
                ])
                record_file.flush()

            if gui and frame_idx % DISPLAY_EVERY_N == 0:
                display = frame.copy()
                cv2.drawMarker(
                    display, axis, (0, 255, 255), cv2.MARKER_CROSS, 14, 2
                )

                if valid:
                    contour = candidate["contour"]
                    if DETECT_SCALE != 1.0:
                        contour = (contour.astype(np.float32) / DETECT_SCALE).astype(
                            np.int32
                        )
                    cv2.drawContours(display, [contour], -1, (0, 255, 255), 2)

                    cx, cy = candidate["centroid"]
                    cv2.circle(display, (int(cx), int(cy)), 6, (0, 255, 255), 2)

                    vx, vy = candidate["vx"], candidate["vy"]
                    line_half = max(80.0, candidate["length"] * 0.6)
                    p0 = (int(cx - vx * line_half), int(cy - vy * line_half))
                    p1 = (int(cx + vx * line_half), int(cy + vy * line_half))
                    cv2.line(display, p0, p1, (0, 140, 255), 2)

                display_angle = math.degrees(wrapped)
                status = "VALID" if valid else "INVALID"
                txt = (
                    f"theta={display_angle:+7.2f} deg  "
                    f"omega={omega_filt:+7.2f} rad/s  {status}  "
                    f"fps={real_fps:5.1f}"
                )
                if valid:
                    txt += (
                        f"  L={candidate['length']:.0f}px"
                        f" W={candidate['width']:.1f}px"
                        f" A={candidate['aspect']:.1f}"
                    )
                cv2.putText(
                    display, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA
                )
                cv2.imshow(WINDOW, display)

            if calibrator.click is not None:
                axis = calibrator.click
                axis_x, axis_y = axis
                save_axis(axis_x, axis_y)
                calibrator.click = None
                calibrator.active = False
                print(f"[axis] pivot -> ({axis_x}, {axis_y}) saved")

            if gui:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    calibrator.active = True
                    print("[axis] click the pivot in the preview window")
                elif key == ord("h") and not tuner_open:
                    open_hsv_tuner(yl, yh)
                    tuner_open = True
                    print("[hsv] tuner opened")
                elif key == ord("s"):
                    save_hsv(yl, yh)
                    print("[hsv] yellow thresholds saved")
    finally:
        if record_file is not None:
            record_file.close()
        try:
            sock.close(linger=0)
            ctx.term()
        except Exception:
            pass
        cv2.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    main()
