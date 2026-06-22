"""
log_angle.py
============
PURPOSE : Record pendulum angle during an active swing-up experiment.
          Run this IN PARALLEL with swing_up_dc.py (two terminals on PC).
RUN ON  : PC (connects to Raspberry Pi camera remotely via ZMQ)
USE WHEN: Motor is running and you want to capture exact swing angles.

Usage (Terminal 2, while swing_up_dc.py runs in Terminal 1):
    python log_angle.py
    python log_angle.py tcp://192.168.0.109:9999

Steps:
    1. Terminal 1: python swing_up_dc.py COM6
    2. Terminal 2: python log_angle.py          ← start first, then ENTER
    3. Ctrl+C both when done

Output: angle_YYYYMMDD_HHMMSS.csv  →  analyze with analyze_swingup.py

See also: cartpole-pid/measure_pendulum.py  — records free oscillation only
          (runs on Raspberry Pi, no motor)
"""

import csv
import time
import sys
import os
import math
import statistics
import threading

from camera_comm import CameraSubscriber
from config import DEFAULT_ZMQ_ADDR

ZMQ_ADDR = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ZMQ_ADDR

# ── Connect ───────────────────────────────────────
cam = CameraSubscriber(ZMQ_ADDR, recv_timeout_ms=2000)

print("=" * 50)
print("  Angle Logger")
print("=" * 50)
print(f"  ZMQ: {ZMQ_ADDR}")
print()
print("  Press ENTER to start recording...")
input()

filename = time.strftime("angle_%Y%m%d_%H%M%S.csv")
filepath = os.path.join(os.path.dirname(__file__), filename)

# Flush buffer
cam.drain()

print(f"  Recording → {filename}")
print("  Press ENTER to mark 'motor stopped'. Ctrl+C to stop.\n")
print(f"  {'t(s)':>6}  {'θ(°)':>8}  {'ω(r/s)':>8}  pole_up")
print(f"  {'-'*40}")

rows = []
t0 = time.time()
motor_stop_t = None

def wait_for_marker():
    global motor_stop_t
    while True:
        input()
        motor_stop_t = time.time() - t0
        print(f"\n  *** Motor stop marked at t={motor_stop_t:.2f}s ***\n")

marker_thread = threading.Thread(target=wait_for_marker, daemon=True)
marker_thread.start()

try:
    while True:
        msg = cam.recv()
        if msg is None:
            print("  WARNING: no camera data")
            continue

        # Support both comma-separated and space-separated formats
        parts = msg.split(",") if "," in msg else msg.split()
        if len(parts) < 8:
            continue

        try:
            theta_rad = float(parts[4])
            omega     = float(parts[5])
            pole_up   = parts[6].strip() in ("1", "True", "true")
            ts        = float(parts[7])
        except (ValueError, IndexError):
            continue

        t = time.time() - t0
        theta_deg = theta_rad * 57.2958
        rows.append([f"{t:.4f}", f"{theta_deg:.3f}", f"{omega:.4f}",
                     1 if pole_up else 0])

        # Print every ~10 frames (~12 Hz display)
        if len(rows) % 10 == 0:
            print(f"  {t:6.2f}  {theta_deg:8.2f}  {omega:8.3f}  "
                  f"{'UP' if pole_up else ''}")

except KeyboardInterrupt:
    pass

# Save CSV
with open(filepath, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["t_s", "theta_deg", "omega_rad_s", "pole_up",
                     "motor_stop_t"])
    for i, row in enumerate(rows):
        row.append(f"{motor_stop_t:.4f}" if (i == 0 and motor_stop_t) else "")
    writer.writerows(rows)

cam.close()

print(f"\n  Saved {len(rows)} samples → {filepath}")
print(f"  Duration: {float(rows[-1][0]):.1f} s")

# ── sigma_theta for the EKF (only meaningful if the pole was STATIC) ──
# Measurement noise = how much θ scatters when the true angle is constant.
# Hold the pole still, run this, then paste the rad value into balance_lqr.py
# (EKF_SIGMA_THETA). The CSV logs degrees; the EKF wants radians, so we
# convert here (×π/180) so there is nothing left to compute by hand.
thetas_deg = [float(r[1]) for r in rows]
if len(thetas_deg) > 1:
    sd_deg = statistics.pstdev(thetas_deg)
    sd_rad = math.radians(sd_deg)
    print()
    print(f"  θ std = {sd_deg:.4f}°  =  {sd_rad:.5f} rad")
    print(f"  -> EKF_SIGMA_THETA = {sd_rad:.5f}   (ONLY valid if pole was static)")

print()
print(f"  Analyze:  python analyze_swingup.py {filename}")
