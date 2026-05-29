"""
pi_balance.py
=============
Balance PID controller — runs entirely ON THE RASPBERRY PI.

Pipeline (single process, no ZMQ, no PC, no NTP):
    camera -> angle -> PID -> serial -> ESP32

Usage:
    python3 pi_balance.py                   # uses /dev/ttyUSB0
    python3 pi_balance.py /dev/ttyUSB1      # specify a different port

Tuneable parameters (PID gains, safety thresholds, PWM limits)
live in src/config.py — edit them there, not here.

DEPENDS ON: src/motor_comm.py, src/config.py
SUPERSEDES: src/balance_pid.py (legacy PC-loop version)
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import math
import time
import sys
import threading
from pathlib import Path
from queue import Queue

# Make src/ importable from pi/ (single source of truth lives there).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from motor_comm import MotorComm
from config import (
    Kp, Kd, Kx,
    ANGLE_SIGN, OMEGA_SIGN, CART_SIGN,
    PWM_LIMIT,
    GIVEUP_ANGLE_DEG, MAX_PULSES,
    SERIAL_BAUD,
)


# ── Return to center (local to this script) ──────────────────────────────────
RETURN_SLOW_ZONE  = 300
RETURN_PWM_MAX    = 120
RETURN_PWM_FLOOR  = 80

# ── Serial port (CLI override) ───────────────────────────────────────────────
SERIAL_PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"

# ── Camera parameters ── keep in sync with camerapreview_detect_color2.py ────
skalierung = 0.25
width  = int(640 * skalierung)   # 160
height = int(480 * skalierung)   # 120
x_ori  = int(280 * skalierung)   # 70  — pivot x in resized frame
y_ori  = int(480 * skalierung)   # 120 — pivot y in resized frame

gelb_min = np.array([ 20,  90,  90], np.uint8)
gelb_max = np.array([ 30, 255, 255], np.uint8)
blau_min = np.array([100, 150,  30], np.uint8)
blau_max = np.array([120, 255, 255], np.uint8)


# ── Vision helpers ────────────────────────────────────────────────────────────

def find_largest_contour(gelb_k, blau_k):
    if gelb_k:
        gi = int(np.argmax([cv2.contourArea(k) for k in gelb_k]))
        gf = cv2.contourArea(gelb_k[gi])
    else:
        gf, gi = 0, -1
    if blau_k:
        bi = int(np.argmax([cv2.contourArea(k) for k in blau_k]))
        bf = cv2.contourArea(blau_k[bi])
    else:
        bf, bi = 0, -1
    if gf == 0 and bf == 0:
        return None, None
    return ("gelb", gelb_k[gi]) if gf >= bf else ("blau", blau_k[bi])


def compute_angle(kind, x_m, y_m):
    """8-octant angle from marker centroid. Same as camerapreview_detect_color2.py."""
    if y_m == y_ori and x_m > x_ori:
        return -math.pi / 2 if kind == "gelb" else  math.pi / 2
    if y_m == y_ori and x_m < x_ori:
        return  math.pi / 2 if kind == "gelb" else -math.pi / 2
    if y_m > y_ori and x_m == x_ori:
        return 0.0           if kind == "gelb" else  math.pi
    if y_m < y_ori and x_m == x_ori:
        return  math.pi      if kind == "gelb" else  0.0
    if y_m < y_ori and x_m > x_ori:
        a = math.atan(abs(x_m - x_ori) / abs(y_m - y_ori))
        return -math.pi + a  if kind == "gelb" else a
    if y_m > y_ori and x_m > x_ori:
        a = math.atan(abs(x_m - x_ori) / abs(y_ori - y_m))
        return -a             if kind == "gelb" else math.pi - a
    if y_m > y_ori and x_m < x_ori:
        a = math.atan(abs(x_ori - x_m) / abs(y_ori - y_m))
        return -math.pi + a   if kind == "gelb" else a
    a = math.atan(abs(x_ori - x_m) / abs(y_m - y_ori))
    return  math.pi - a       if kind == "gelb" else -a


# ── Cart helpers ──────────────────────────────────────────────────────────────

def return_to_center(comm, timeout=10.0):
    time.sleep(0.2)
    pos = comm.get_position()
    if abs(pos) <= 20:
        print("  Cart already near center. Ready.\n")
        return
    print(f"  Cart at {pos} pulses — returning to center...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        pos = comm.get_position()
        if abs(pos) <= 10:
            break
        dist = abs(pos)
        if dist >= RETURN_SLOW_ZONE:
            pwm = RETURN_PWM_MAX
        else:
            ratio = dist / RETURN_SLOW_ZONE
            pwm   = int(RETURN_PWM_FLOOR + ratio * (RETURN_PWM_MAX - RETURN_PWM_FLOOR))
            pwm   = max(RETURN_PWM_FLOOR, pwm)
        comm.send_velocity(pwm if pos < 0 else -pwm)
        time.sleep(0.02)
    comm.stop()
    time.sleep(0.5)
    print(f"  Cart at center (pos={comm.get_position()}). Ready.\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Inverted Pendulum — Pi Balance PID")
    print("=" * 55)
    print(f"  Serial : {SERIAL_PORT}")
    print(f"  Gains  : Kp={Kp}  Kd={Kd}  Kx={Kx}")
    print(f"  Safety : ±{GIVEUP_ANGLE_DEG}° from upright, ±{MAX_PULSES} pulses")
    print()

    # ── Camera ────────────────────────────────────────────────
    picam = Picamera2()
    cam_config = picam.create_video_configuration(
        main={"format": "RGB888", "size": (640, 480)},
        controls={"FrameRate": 60.0},
    )
    picam.configure(cam_config)
    picam.start()
    time.sleep(1)
    try:
        picam.set_controls({"AfMode": 0, "LensPosition": 4.5})
    except Exception:
        pass
    print("  Camera started.")

    # ── Serial ────────────────────────────────────────────────
    try:
        comm = MotorComm(SERIAL_PORT, SERIAL_BAUD, timeout=0.05, echo_timeout=0.04)
        print("  ESP32 connected.")
        return_to_center(comm)
    except Exception as e:
        print(f"  Serial failed: {e}")
        picam.stop()
        return

    # ── Rolling angle window (5 frames, same as camerapreview_detect_color2.py) ──
    angle_q = Queue(maxsize=5)
    for _ in range(5):
        angle_q.put((math.pi, time.time()))

    # ── Diagnostic mode ───────────────────────────────────────
    print("=" * 55)
    print("  DIAGNOSTIC MODE  —  motor is OFF")
    print("=" * 55)
    print("  Hold pendulum near upright and verify:")
    print("  - theta ≈ 0° when upright, ±small when tilted")
    print("  - PWM positive = cart moves right (flip ANGLE_SIGN if wrong)")
    print()
    print("  Press ENTER to enable motor.")
    print()
    print(f"  {'theta_deg':>10}  {'omega':>8}  {'cart':>7}  {'PWM':>9}  status")
    print(f"  {'-'*60}")

    motor_enabled = False
    enable_flag   = [False]

    def wait_enter():
        input()
        enable_flag[0] = True

    threading.Thread(target=wait_enter, daemon=True).start()

    cart_pos    = 0
    pos_counter = 0
    theta_deg   = 180.0
    omega       = 0.0

    try:
        while True:
            # ── Capture frame ──────────────────────────────
            frame = picam.capture_array()
            frame = cv2.resize(frame, (width, height))
            hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            gelb_maske = cv2.inRange(hsv, gelb_min, gelb_max)
            blau_maske = cv2.inRange(hsv, blau_min, blau_max)
            gelb_k, _ = cv2.findContours(gelb_maske, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            blau_k, _ = cv2.findContours(blau_maske, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

            kind, kontur = find_largest_contour(gelb_k, blau_k)
            if kind is None:
                continue

            M = cv2.moments(kontur)
            if M["m00"] <= 0:
                continue

            x_m = int(M["m10"] / M["m00"])
            y_m = int(M["m01"] / M["m00"])
            now   = time.time()
            angle = compute_angle(kind, x_m, y_m)

            # ── Update rolling window, compute omega ───────
            if angle_q.full():
                angle_old, t_old = angle_q.get()
                angle_q.put((angle, now))
                omega = (angle - angle_old) / (now - t_old)
            else:
                angle_q.put((angle, now))
                omega = 0.0

            angle_avg = sum(a[0] for a in angle_q.queue) / angle_q.qsize()
            theta_deg = math.degrees(angle_avg)

            # ── Cart position (every 5 frames ≈ 12 Hz) ────
            pos_counter += 1
            if pos_counter >= 5:
                cart_pos    = comm.get_position()
                pos_counter = 0

            # ── PID ───────────────────────────────────────
            pwm = (ANGLE_SIGN * (-Kp * theta_deg)
                 + OMEGA_SIGN * ( Kd * omega)
                 - CART_SIGN  * ( Kx * cart_pos))
            pwm = max(-float(PWM_LIMIT), min(float(PWM_LIMIT), pwm))

            # ── Safety ────────────────────────────────────
            angle_out     = abs(theta_deg) > GIVEUP_ANGLE_DEG
            cart_boundary = abs(cart_pos)  > MAX_PULSES
            in_range      = not angle_out and not cart_boundary

            if enable_flag[0] and not motor_enabled:
                motor_enabled = True
                print("\n  *** Motor ENABLED — Ctrl+C to stop ***\n")

            if motor_enabled and in_range:
                comm.send_velocity(int(pwm))
            elif motor_enabled and cart_boundary:
                comm.stop()
                print("\n  Cart boundary reached — returning to center...")
                return_to_center(comm)
                motor_enabled  = False
                enable_flag[0] = False
                threading.Thread(target=wait_enter, daemon=True).start()
                print("  Press ENTER to re-enable motor.\n")
            else:
                comm.stop()

            # ── Status display ────────────────────────────
            if not in_range:
                oor    = "ANGLE" if angle_out else "CART"
                status = ("MOTOR ON" if motor_enabled else "MOTOR OFF") + f" | {oor} OUT OF RANGE"
            else:
                status = "MOTOR ON" if motor_enabled else "MOTOR OFF"

            print(f"\r  {theta_deg:10.1f}  {omega:8.3f}  {cart_pos:7}  "
                  f"{pwm:9.1f}  {status:<35}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
    finally:
        comm.stop()
        comm.close()
        picam.stop()
        print("  Motor off. Done.")


if __name__ == "__main__":
    main()
