#!/usr/bin/env python3
"""
Cart-pole pendulum angle detector (V1 - yellow tape line fit).

This is the current Pi-side vision publisher as of 2026-08-18.  It supersedes
camerapreview_detect_color_V3.py for current measurement and balance work.
It uses the existing yellow HSV calibration, but treats the yellow tape as
one elongated line instead of using a yellow-centroid -> blue-centroid vector.

Detection pipeline:
  yellow HSV mask -> morphology -> connected yellow components
  -> PCA/fitLine() principal direction -> line angle

This experimental version intentionally does NOT require a particular physical
tape length, width, aspect ratio, or distance from the pivot.  Those values are
diagnostic information only.  The only hard requirement is that there are
enough yellow pixels to estimate a direction and that the pixels have a
measurable linear structure.

The angle is currently treated as a LINE ORIENTATION modulo 180 degrees:
  theta = 0       -> the tape is vertical
  +theta          -> clockwise line tilt as seen from the camera

This version deliberately does not distinguish the pendulum pointing up from
pointing down.  That requires an asymmetric marker or another sensor.
Consequently, pole_up is published as 0.  The stream is suitable for an
isolated balance experiment that is initialized upright, and for small free
oscillation tests around a known vertical equilibrium.  It is NOT suitable for
automatic swing-up or any state machine that must distinguish up from down.

The ZMQ message remains compatible with the existing camera subscriber:
  "a,a,a,a,theta_rad,omega_rad_s,pole_up,ts_ns,sensor_ts_ns,valid,frame_id"
By default only valid frames are published, preserving controller behavior.
Use --publish-invalid for raw free-swing logging so every camera frame is
published and the valid-frame ratio can be measured.

Static angle sign/zero and dynamic latency still have to be validated for the
current mechanical/camera setup before judging a controller from its output.
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

# Yellow thresholds retained from the previous dual-colour detector.
HSV_YELLOW_LOW = np.array([18, 130, 90], dtype=np.uint8)
HSV_YELLOW_HIGH = np.array([35, 255, 255], dtype=np.uint8)

# Accuracy-test default: use the full 768x432 main stream.  Set to 0.5 from
# the command line if the Raspberry Pi cannot sustain the target frame rate.
DETECT_SCALE = 1.0

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# Only a small image-space guard against single-pixel noise.  This is not a
# physical tape-size requirement.
MIN_YELLOW_PIXELS = 20

# Physical/geometric validation, not tape-size validation.  The pendulum line
# rotates about the calibrated pivot, so its fitted infinite line should pass
# close to that pivot even if the visible tape starts a little away from it.
MAX_LINE_AXIS_DISTANCE_PX = 150.0

# A background line can be yellow and linear, but it usually cannot jump to the
# pendulum angle continuously.  This is a safety gate for a single frame.
MAX_FRAME_ANGLE_STEP_DEG = 35.0

OMEGA_LPF_ALPHA = 0.55
ZMQ_PORT = 9999

_VISION_DIR = os.path.dirname(os.path.abspath(__file__))
AXIS_CALIB_FILE = os.path.join(_VISION_DIR, "axis_calib.json")
HSV_CALIB_FILE = os.path.join(_VISION_DIR, "hsv_calib.json")

WINDOW = "Yellow tape line"
HSV_WIN = "Yellow HSV tuner"
DISPLAY_EVERY_N = 2


def cv_scalar(value):
    """Convert a one-element OpenCV/NumPy result to a Python float."""
    return float(np.asarray(value).reshape(-1)[0])


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
    """Fit a line to one yellow component without fixed physical-size gates."""
    inv = 1.0 / detect_scale

    # Use the filled component for fitting, not just the contour boundary.
    component = np.zeros_like(mask)
    cv2.drawContours(component, [contour], -1, 255, thickness=-1)
    points_small = cv2.findNonZero(component)
    if points_small is None or len(points_small) < MIN_YELLOW_PIXELS:
        return None
    points = points_small.reshape(-1, 2).astype(np.float32) * inv

    # PCA measures whether the yellow pixels have a dominant line direction.
    # This is a confidence score, not a hard tape length/width requirement.
    centered = points - points.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    lambda_min = float(max(eigenvalues[0], 0.0))
    lambda_max = float(max(eigenvalues[-1], 0.0))
    linearity = (lambda_max - lambda_min) / (lambda_max + lambda_min + 1e-9)
    if lambda_max <= 1e-9 or linearity <= 0.01:
        return None

    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    long_side = max(width, height) * inv
    short_side = min(width, height) * inv
    aspect = long_side / short_side if short_side > 1e-6 else float("inf")

    fit_points = points.reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(
        fit_points, cv2.DIST_L2, 0, 0.01, 0.01
    )
    vx = cv_scalar(vx)
    vy = cv_scalar(vy)

    # Distance from the calibrated pivot to the fitted infinite line.
    # For a correct pendulum line this should be small even when the tape does
    # not physically touch the pivot.
    line_point = np.array(
        [cv_scalar(x0) * inv, cv_scalar(y0) * inv], dtype=np.float32
    )
    axis_point = np.array(axis, dtype=np.float32)
    axis_delta = axis_point - line_point
    axis_line_distance = abs(vx * float(axis_delta[1])
                             - vy * float(axis_delta[0]))
    if axis_line_distance > MAX_LINE_AXIS_DISTANCE_PX:
        return None

    # A longer and more linear component wins if there are several yellow
    # components.  No fixed length/aspect threshold is applied here.
    score = linearity * math.sqrt(float(len(points)))
    return {
        "contour": contour,
        "centroid": points.mean(axis=0),
        "vx": vx,
        "vy": vy,
        "length": long_side,
        "width": short_side,
        "aspect": aspect,
        "linearity": linearity,
        "pixels": len(points),
        "axis_line_distance": axis_line_distance,
        "score": score,
        "x0": cv_scalar(x0) * inv,
        "y0": cv_scalar(y0) * inv,
    }


def detect_yellow_line(frame, yl, yh, axis, detect_scale, previous_angle=None):
    """Detect the strongest linear yellow component in full-frame coordinates."""
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
            candidate["angle"] = compute_angle(candidate)
            if previous_angle is not None:
                candidate["angle_step"] = abs(
                    line_angle_delta(candidate["angle"], previous_angle)
                )
                if candidate["angle_step"] > math.radians(
                    MAX_FRAME_ANGLE_STEP_DEG
                ):
                    continue
            else:
                candidate["angle_step"] = 0.0
            candidates.append(candidate)

    if not candidates:
        return None, mask
    return max(candidates, key=lambda item: item["score"]), mask


def compute_angle(candidate):
    """Return line tilt from vertical, normalized to [-pi/2, +pi/2)."""
    # Image y increases downwards; therefore -vy is the upward coordinate.
    raw = math.atan2(candidate["vx"], -candidate["vy"])
    return wrap_line_angle(raw)


def wrap_line_angle(angle):
    """Wrap an unoriented line angle to one 180-degree interval."""
    half_pi = 0.5 * math.pi
    return (angle + half_pi) % math.pi - half_pi


def line_angle_delta(current, previous):
    """Smallest signed difference between two unoriented line angles."""
    return wrap_line_angle(current - previous)


def unwrap(prev, current):
    if prev is None:
        return current
    # An unoriented line repeats every pi, not every 2*pi.
    delta = (current - prev + 0.5 * math.pi) % math.pi - 0.5 * math.pi
    return prev + delta


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
    global MIN_YELLOW_PIXELS, DETECT_SCALE
    global MAX_LINE_AXIS_DISTANCE_PX, MAX_FRAME_ANGLE_STEP_DEG

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
    parser.add_argument("--min-pixels", type=int, default=MIN_YELLOW_PIXELS,
                        help="Minimum yellow pixels used to fit a line")
    parser.add_argument(
        "--axis-line-gap", type=float, default=MAX_LINE_AXIS_DISTANCE_PX,
        help="Maximum fitted-line distance from pivot in pixels"
    )
    parser.add_argument(
        "--max-angle-step-deg", type=float,
        default=MAX_FRAME_ANGLE_STEP_DEG,
        help="Reject a candidate that jumps this many degrees in one frame"
    )
    parser.add_argument("--zmq-port", type=int, default=ZMQ_PORT,
                        help="ZMQ PUB port (default 9999)")
    parser.add_argument("--record", type=str, default=None,
                        help="Optional CSV path for diagnostic recording")
    parser.add_argument("--publish-invalid", action="store_true",
                        help="Publish every frame with valid=0/1 and frame_id; "
                             "required by raw-data loggers")
    parser.add_argument("--no-gui", action="store_true",
                        help="Headless mode; quit with Ctrl+C")
    parser.add_argument(
        "--with-raw", action="store_true",
        help="Also request the explicit 1536x864 RAW stream (diagnostic only)"
    )
    args = parser.parse_args()

    DETECT_SCALE = args.detect_scale
    MIN_YELLOW_PIXELS = args.min_pixels
    MAX_LINE_AXIS_DISTANCE_PX = args.axis_line_gap
    MAX_FRAME_ANGLE_STEP_DEG = args.max_angle_step_deg

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
    print(f"  line rule    : yellow pixels >= {MIN_YELLOW_PIXELS}; "
          "linearity is used only for candidate ranking")
    print(f"  pivot rule   : fitted line gap <= {MAX_LINE_AXIS_DISTANCE_PX:.0f}px; "
          f"frame jump <= {MAX_FRAME_ANGLE_STEP_DEG:.0f}deg")
    print(f"  ZMQ PUB      : tcp://*:{args.zmq_port}")
    print(f"  RAW stream   : {'1536x864 requested' if args.with_raw else 'disabled (main only)'}")
    print("  NOTE         : line angle is modulo 180deg; up/down are indistinguishable")
    print("                 automatic swing-up is disabled for this vision source")
    if args.publish_invalid:
        print("  raw logging  : --publish-invalid enabled")

    picam2 = Picamera2()
    camera_config = {
        "main": {"size": (W, H), "format": "RGB888"},
        "controls": {
            "FrameDurationLimits": (FRAME_DURATION_US, FRAME_DURATION_US),
            "ExposureTime": EXPOSURE_US,
            "AnalogueGain": ANALOGUE_GAIN,
            "AwbEnable": False,
            "AeEnable": False,
            "LensPosition": LENS_POSITION,
        },
        "buffer_count": 4,
    }
    if args.with_raw:
        camera_config["raw"] = {"size": (RAW_W, RAW_H)}
    config = picam2.create_video_configuration(**camera_config)
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
            "linearity", "yellow_pixels", "axis_line_distance_px",
            "angle_step_rad", "centroid_x_px", "centroid_y_px",
            "fps",
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

    show_mask = False
    print("[keys] c=calibrate pivot | h=HSV tuner | s=save HSV | m=mask | q=quit")

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
                frame, yl, yh, axis, DETECT_SCALE, angle_prev
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

            wrapped = wrap_line_angle(angle)
            if valid or args.publish_invalid:
                # Up/down is intentionally not solved in this version.
                pole_up = 0
                sock.send_string(
                    f"{fmt(wrapped)},{fmt(wrapped)},{fmt(wrapped)},"
                    f"{fmt(wrapped)},{fmt(wrapped)},{fmt(omega_filt)},"
                    f"{pole_up},{time.time_ns()},"
                    f"{sensor_ts if sensor_ts is not None else 0},"
                    f"{int(valid)},{frame_idx}"
                )

            if record_writer:
                record_writer.writerow([
                    fmt(now - t0, 6), sensor_ts or 0,
                    fmt(wrapped), fmt(omega_filt), int(valid),
                    fmt(candidate["length"], 3) if valid else "",
                    fmt(candidate["width"], 3) if valid else "",
                    fmt(candidate["aspect"], 3) if valid else "",
                    fmt(candidate["linearity"], 6) if valid else "",
                    candidate["pixels"] if valid else "",
                    fmt(candidate["axis_line_distance"], 3) if valid else "",
                    fmt(candidate["angle_step"], 6) if valid else "",
                    fmt(candidate["centroid"][0], 3) if valid else "",
                    fmt(candidate["centroid"][1], 3) if valid else "",
                    fmt(real_fps, 3),
                ])
                # Do not force a filesystem flush for every frame: that can
                # perturb a 120 Hz timing test.  The file is closed/flushed in
                # the finally block, and this periodic flush limits loss if
                # the process is interrupted.
                if frame_idx % TARGET_FPS == 0:
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
                        f" lin={candidate['linearity']:.2f}"
                        f" N={candidate['pixels']}"
                        f" gap={candidate['axis_line_distance']:.0f}px"
                    )
                cv2.putText(
                    display, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA
                )
                cv2.imshow(WINDOW, display)
                if show_mask:
                    mask_display = mask
                    if DETECT_SCALE != 1.0:
                        mask_display = cv2.resize(
                            mask_display, (W, H), interpolation=cv2.INTER_NEAREST
                        )
                    cv2.imshow("Yellow mask", mask_display)

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
                elif key == ord("m"):
                    show_mask = not show_mask
                    if not show_mask:
                        cv2.destroyWindow("Yellow mask")
                    print(f"[mask] {'shown' if show_mask else 'hidden'}")
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
