#!/usr/bin/env python3
"""Yellow/white marker pendulum publisher: 0=up, left negative, right positive."""

import argparse
import cv2
import json
import math
import os
import time

import numpy as np
import zmq
from picamera2 import Picamera2


RAW_W, RAW_H = 1536, 864
SCALE = 0.5
W = int(RAW_W * SCALE)
H = int(RAW_H * SCALE)

TARGET_FPS = 120
FRAME_DURATION_US = int(round(1_000_000 / TARGET_FPS))
EXPOSURE_US = 4000
ANALOGUE_GAIN = 4.0
LENS_POSITION = 4.5

HSV_YELLOW_LOW = np.array([18, 130, 90], dtype=np.uint8)
HSV_YELLOW_HIGH = np.array([35, 255, 255], dtype=np.uint8)
HSV_WHITE_LOW = np.array([0, 0, 140], dtype=np.uint8)
HSV_WHITE_HIGH = np.array([180, 105, 255], dtype=np.uint8)

DETECT_SCALE = 1.0

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
MIN_YELLOW_PIXELS = 20
MIN_WHITE_PIXELS = 40

MAX_LINE_AXIS_DISTANCE_PX = 150.0
MAX_YELLOW_AXIS_DISTANCE_PX = 320.0

OMEGA_LPF_ALPHA = 0.55
ZMQ_PORT = 9999

_VISION_DIR = os.path.dirname(os.path.abspath(__file__))
AXIS_CALIB_FILE = os.path.join(_VISION_DIR, "axis_calib.json")
HSV_CALIB_FILE = os.path.join(_VISION_DIR, "hsv_calib.json")

WINDOW = "Yellow + white markers"
HSV_WIN = "Yellow + white HSV tuner"
DISPLAY_EVERY_N = 2


def cv_scalar(value):
    """Convert a one-element OpenCV/NumPy result to a Python float."""
    return float(np.asarray(value).reshape(-1)[0])


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
            yl = np.array(d["yl"], dtype=np.uint8)
            yh = np.array(d["yh"], dtype=np.uint8)
            wl = np.array(d.get("wl", HSV_WHITE_LOW), dtype=np.uint8)
            wh = np.array(d.get("wh", HSV_WHITE_HIGH), dtype=np.uint8)
            return yl, yh, wl, wh
        except Exception as exc:
            print(f"[hsv] load failed ({exc}); using defaults")
    return (HSV_YELLOW_LOW.copy(), HSV_YELLOW_HIGH.copy(),
            HSV_WHITE_LOW.copy(), HSV_WHITE_HIGH.copy())


def save_hsv(yl, yh, wl, wh):
    d = {}
    if os.path.exists(HSV_CALIB_FILE):
        try:
            with open(HSV_CALIB_FILE, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            d = {}
    d["yl"] = yl.tolist()
    d["yh"] = yh.tolist()
    d["wl"] = wl.tolist()
    d["wh"] = wh.tolist()
    with open(HSV_CALIB_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f)


def _line_candidate(contour, mask, axis, detect_scale, min_pixels):
    """Return a scored line candidate in full-frame coordinates."""
    inv = 1.0 / detect_scale

    component = np.zeros_like(mask)
    cv2.drawContours(component, [contour], -1, 255, thickness=-1)
    points_small = cv2.findNonZero(component)
    if points_small is None or len(points_small) < min_pixels:
        return None
    points = points_small.reshape(-1, 2).astype(np.float32) * inv

    centered = points - points.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    lambda_min = float(max(eigenvalues[0], 0.0))
    lambda_max = float(max(eigenvalues[-1], 0.0))
    linearity = (lambda_max - lambda_min) / (lambda_max + lambda_min + 1e-9)
    if lambda_max <= 1e-9 or linearity <= 0.05:
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

    line_point = np.array(
        [cv_scalar(x0) * inv, cv_scalar(y0) * inv], dtype=np.float32
    )
    axis_point = np.array(axis, dtype=np.float32)
    axis_delta = axis_point - line_point
    axis_line_distance = abs(vx * float(axis_delta[1])
                             - vy * float(axis_delta[0]))
    if axis_line_distance > MAX_LINE_AXIS_DISTANCE_PX:
        return None
    axis_weight = min(1.0, MAX_LINE_AXIS_DISTANCE_PX /
                      max(axis_line_distance, 1.0))
    score = linearity * long_side * max(axis_weight, 0.05)
    axis_point = np.asarray(axis, dtype=np.float32)
    tip = points[np.argmax(np.sum((points - axis_point) ** 2, axis=1))]
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
        "tip": tip,
    }


def _blob_candidate(contour, mask, detect_scale, min_pixels):
    inv = 1.0 / detect_scale
    component = np.zeros_like(mask)
    cv2.drawContours(component, [contour], -1, 255, thickness=-1)
    points_small = cv2.findNonZero(component)
    if points_small is None or len(points_small) < min_pixels:
        return None
    points = points_small.reshape(-1, 2).astype(np.float32) * inv
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return {
        "contour": contour,
        "centroid": np.array([
            moments["m10"] / moments["m00"] * inv,
            moments["m01"] / moments["m00"] * inv,
        ], dtype=np.float32),
        "pixels": len(points),
        "area": cv2.contourArea(contour) * inv * inv,
    }


def _mask(hsv, low, high):
    result = cv2.inRange(hsv, low, high)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, MORPH_KERNEL)
    return cv2.morphologyEx(result, cv2.MORPH_CLOSE, MORPH_KERNEL)


def detect_markers(frame, yl, yh, wl, wh, axis, detect_scale):
    """Detect the yellow near marker and white long-rod line."""
    if detect_scale != 1.0:
        small = cv2.resize(
            frame, None, fx=detect_scale, fy=detect_scale,
            interpolation=cv2.INTER_AREA
        )
    else:
        small = frame

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    yellow_mask = _mask(hsv, yl, yh)
    white_mask = _mask(hsv, wl, wh)

    yellow_contours, _ = cv2.findContours(
        yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    yellow_candidates = [
        candidate for contour in yellow_contours
        for candidate in [_blob_candidate(
            contour, yellow_mask, detect_scale, MIN_YELLOW_PIXELS
        )]
        if candidate is not None
    ]
    yellow = (
        max(yellow_candidates, key=lambda item: item["area"])
        if yellow_candidates else None
    )

    white_contours, _ = cv2.findContours(
        white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    white_candidates = []
    for contour in white_contours:
        candidate = _line_candidate(
            contour, white_mask, axis, detect_scale, MIN_WHITE_PIXELS
        )
        if candidate is not None and candidate["aspect"] >= 2.0:
            white_candidates.append(candidate)
    white = (
        max(white_candidates, key=lambda item: item["score"])
        if white_candidates else None
    )

    return yellow, white, yellow_mask, white_mask


def gate_yellow(yellow, axis):
    if yellow is None:
        return None, None
    cx, cy = yellow["centroid"]
    distance = math.hypot(float(cx) - axis[0], float(cy) - axis[1])
    if distance > MAX_YELLOW_AXIS_DISTANCE_PX:
        return None, distance
    return yellow, distance


def compute_angle(yellow, white, axis):
    """Compute 0=up, right positive, left negative from marker direction."""
    if yellow is not None and white is not None:
        near_x, near_y = yellow["centroid"]
        tip_x, tip_y = white["tip"]
    elif white is not None:
        near_x, near_y = axis
        tip_x, tip_y = white["tip"]
    elif yellow is not None:
        tip_x, tip_y = axis
        near_x, near_y = yellow["centroid"]
    else:
        return None
    dx = float(tip_x - near_x)
    dy_up = float(near_y - tip_y)
    return wrap_angle(math.atan2(dx, dy_up))


def wrap_angle(angle):
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if math.isclose(wrapped, -math.pi, abs_tol=1e-12):
        return math.pi
    return wrapped


def line_angle_delta(current, previous):
    return wrap_angle(current - previous)


def unwrap(prev, current):
    if prev is None:
        return current
    delta = wrap_angle(current - prev)
    return prev + delta


class AxisCalibrator:
    def __init__(self):
        self.active = False
        self.click = None

    def on_mouse(self, event, x, y, flags, param):
        del flags, param
        if self.active and event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)


def open_hsv_tuner(yl, yh, wl, wh):
    cv2.namedWindow(HSV_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(HSV_WIN, 420, 300)
    bars = [
        ("Y_H_lo", int(yl[0]), 180), ("Y_H_hi", int(yh[0]), 180),
        ("Y_S_lo", int(yl[1]), 255), ("Y_V_lo", int(yl[2]), 255),
        ("W_S_hi", int(wh[1]), 255), ("W_V_lo", int(wl[2]), 255),
    ]
    for name, value, maximum in bars:
        cv2.createTrackbar(name, HSV_WIN, value, maximum, lambda _: None)


def read_hsv_tuner(yl, yh, wl, wh):
    yl[0] = cv2.getTrackbarPos("Y_H_lo", HSV_WIN)
    yh[0] = cv2.getTrackbarPos("Y_H_hi", HSV_WIN)
    yl[1] = cv2.getTrackbarPos("Y_S_lo", HSV_WIN)
    yl[2] = cv2.getTrackbarPos("Y_V_lo", HSV_WIN)
    wh[1] = cv2.getTrackbarPos("W_S_hi", HSV_WIN)
    wl[2] = cv2.getTrackbarPos("W_V_lo", HSV_WIN)


def fmt(value, digits=6):
    return f"{value:.{digits}f}"


def main():
    global MIN_YELLOW_PIXELS, MIN_WHITE_PIXELS, DETECT_SCALE
    global MAX_LINE_AXIS_DISTANCE_PX

    parser = argparse.ArgumentParser(
        description="Cart-pole vision: yellow/white marker angle detector"
    )
    parser.add_argument("--axis-x", type=int, default=None,
                        help="Override calibrated pivot X in the 768x432 frame")
    parser.add_argument("--axis-y", type=int, default=None,
                        help="Override calibrated pivot Y in the 768x432 frame")
    parser.add_argument("--detect-scale", type=float, default=DETECT_SCALE,
                        choices=(0.5, 1.0),
                        help="Detection scale; use 1.0 for accuracy testing")
    parser.add_argument("--min-pixels", type=int, default=MIN_YELLOW_PIXELS,
                        help="Minimum yellow pixels in the near marker")
    parser.add_argument("--min-white-pixels", type=int,
                        default=MIN_WHITE_PIXELS,
                        help="Minimum white pixels in the rod line")
    parser.add_argument(
        "--axis-line-gap", type=float, default=MAX_LINE_AXIS_DISTANCE_PX,
        help="Maximum fitted-line distance from pivot in pixels"
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
    MIN_WHITE_PIXELS = args.min_white_pixels
    MAX_LINE_AXIS_DISTANCE_PX = args.axis_line_gap

    gui = not args.no_gui
    axis_x, axis_y = load_axis()
    if args.axis_x is not None:
        axis_x = args.axis_x
    if args.axis_y is not None:
        axis_y = args.axis_y
    axis = (axis_x, axis_y)
    yl, yh, wl, wh = load_hsv()

    print("==============================================================")
    print("  Yellow + white marker angle detector V1")
    print("==============================================================")
    print(f"  camera       : {W}x{H} main stream, target {TARGET_FPS} Hz")
    print(f"  detect scale : {DETECT_SCALE}")
    print(f"  pivot        : ({axis_x}, {axis_y})")
    print(f"  yellow rule  : marker pixels >= {MIN_YELLOW_PIXELS}")
    print(f"  white rule   : line pixels >= {MIN_WHITE_PIXELS}; "
          f"fitted line gap <= {MAX_LINE_AXIS_DISTANCE_PX:.0f}px")
    print("  angle        : 0 deg up, left negative, right positive")
    print(f"  ZMQ PUB      : tcp://*:{args.zmq_port}")
    print(f"  RAW stream   : {'1536x864 requested' if args.with_raw else 'disabled (main only)'}")
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
            "white_line_length_px", "white_line_width_px",
            "white_line_aspect", "white_linearity", "white_pixels",
            "white_axis_line_distance_px", "angle_step_rad",
            "yellow_x_px", "yellow_y_px", "white_tip_x_px", "white_tip_y_px",
            "mode", "fps",
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
                read_hsv_tuner(yl, yh, wl, wh)

            yellow_raw, white, yellow_mask, white_mask = detect_markers(
                frame, yl, yh, wl, wh, axis, DETECT_SCALE
            )
            yellow, yellow_distance = gate_yellow(yellow_raw, axis)
            valid = yellow is not None or white is not None
            if yellow is not None and white is not None:
                mode = "dual"
            elif white is not None:
                mode = "white-axis"
            elif yellow is not None:
                mode = "yellow-axis"
            else:
                mode = "none"
            angle_step = 0.0
            if valid:
                angle_raw = compute_angle(yellow, white, axis)
                angle_step = (abs(line_angle_delta(angle_raw, angle_prev))
                              if angle_prev is not None else 0.0)
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

            wrapped = wrap_angle(angle)
            if valid or args.publish_invalid:
                pole_up = int(abs(wrapped) < math.radians(5))
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
                    fmt(white["length"], 3) if white is not None else "",
                    fmt(white["width"], 3) if white is not None else "",
                    fmt(white["aspect"], 3) if white is not None else "",
                    fmt(white["linearity"], 6) if white is not None else "",
                    white["pixels"] if white is not None else "",
                    fmt(white["axis_line_distance"], 3)
                    if white is not None else "",
                    fmt(angle_step, 6) if valid else "",
                    fmt(yellow["centroid"][0], 3)
                    if yellow is not None else "",
                    fmt(yellow["centroid"][1], 3)
                    if yellow is not None else "",
                    fmt(white["tip"][0], 3) if white is not None else "",
                    fmt(white["tip"][1], 3) if white is not None else "",
                    mode,
                    fmt(real_fps, 3),
                ])
                if frame_idx % TARGET_FPS == 0:
                    record_file.flush()

            if gui and frame_idx % DISPLAY_EVERY_N == 0:
                display = frame.copy()
                cv2.drawMarker(
                    display, axis, (0, 255, 255), cv2.MARKER_CROSS, 14, 2
                )

                if yellow_raw is not None:
                    contour = yellow_raw["contour"]
                    if DETECT_SCALE != 1.0:
                        contour = (contour.astype(np.float32) / DETECT_SCALE).astype(
                            np.int32
                        )
                    yellow_color = (0, 255, 255) if yellow is not None else (0, 0, 255)
                    cv2.drawContours(display, [contour], -1, yellow_color, 2)
                    cx, cy = yellow_raw["centroid"]
                    cv2.circle(display, (int(cx), int(cy)), 6, yellow_color, 2)

                if white is not None:
                    contour = white["contour"]
                    if DETECT_SCALE != 1.0:
                        contour = (contour.astype(np.float32) / DETECT_SCALE).astype(
                            np.int32
                        )
                    cv2.drawContours(display, [contour], -1, (255, 255, 255), 2)
                    cx, cy = white["centroid"]
                    vx, vy = white["vx"], white["vy"]
                    line_half = max(80.0, white["length"] * 0.6)
                    p0 = (int(cx - vx * line_half), int(cy - vy * line_half))
                    p1 = (int(cx + vx * line_half), int(cy + vy * line_half))
                    cv2.line(display, p0, p1, (255, 180, 0), 2)
                    tip_x, tip_y = white["tip"]
                    cv2.drawMarker(
                        display, (int(tip_x), int(tip_y)), (255, 0, 255),
                        cv2.MARKER_TILTED_CROSS, 14, 2
                    )

                if yellow is not None and white is not None:
                    yx, yy = yellow["centroid"]
                    tx, ty = white["tip"]
                    cv2.line(
                        display, (int(yx), int(yy)), (int(tx), int(ty)),
                        (0, 200, 255), 2
                    )

                display_angle = math.degrees(wrapped)
                status = "VALID" if valid else "INVALID"
                txt = (
                    f"theta={display_angle:+7.2f} deg  "
                    f"omega={omega_filt:+7.2f} rad/s  {status}  "
                    f"mode={mode}  fps={real_fps:5.1f}"
                )
                if white is not None:
                    txt += (
                        f"  WL={white['length']:.0f}px"
                        f" WW={white['width']:.1f}px"
                        f" WA={white['aspect']:.1f}"
                        f" gap={white['axis_line_distance']:.0f}px"
                    )
                cv2.putText(
                    display, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA
                )
                cv2.imshow(WINDOW, display)
                if show_mask:
                    yellow_display = yellow_mask
                    white_display = white_mask
                    if DETECT_SCALE != 1.0:
                        yellow_display = cv2.resize(
                            yellow_display, (W, H), interpolation=cv2.INTER_NEAREST
                        )
                        white_display = cv2.resize(
                            white_display, (W, H), interpolation=cv2.INTER_NEAREST
                        )
                    cv2.imshow("Yellow mask", yellow_display)
                    cv2.imshow("White mask", white_display)

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
                    open_hsv_tuner(yl, yh, wl, wh)
                    tuner_open = True
                    print("[hsv] tuner opened")
                elif key == ord("m"):
                    show_mask = not show_mask
                    if not show_mask:
                        cv2.destroyWindow("Yellow mask")
                        cv2.destroyWindow("White mask")
                    print(f"[mask] {'shown' if show_mask else 'hidden'}")
                elif key == ord("s"):
                    save_hsv(yl, yh, wl, wh)
                    print("[hsv] yellow/white thresholds saved")
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
