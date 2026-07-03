"""
measure_sigma_theta.py
======================
Standalone camera-angle recorder. Primary use = measure the EKF measurement
noise sigma_theta.

HOW: hold the pole STILL (no motor needed), run this, record a few hundred
frames, Ctrl+C. On exit it prints  EKF_SIGMA_THETA = <rad>  — paste that into
balance_lqr.py (EKF_SIGMA_THETA).
  ⚠️ The std is ONLY valid if the pole was static (true angle constant). If the
     pole moved, the number includes real motion, not just camera noise.

RUN ON : PC or Pi (connects to the camera ZMQ stream; no motor connection).

Usage:
    python measure_sigma_theta.py
    python measure_sigma_theta.py tcp://192.168.0.109:9999

Also writes angle_YYYYMMDD_HHMMSS.csv (t, theta_deg, omega, pole_up) for inspection.

NOTE: to log angles DURING a swing-up/balance run you do NOT need this script —
swing_up_energy_dc.py already logs the full state (theta, omega, pwm, cart_pos)
to swingup_*.csv itself. (This file used to be the swing-up companion logger;
that role is now redundant, so it was slimmed down to the sigma_theta tool.)

Wire format: reads parts[4]=theta, [5]=omega, [6]=pole_up. The V3 camera also
sends parts[7]=send-ts and parts[8]=SensorTimestamp; this script ignores both.

See also: cartpole-pid/measure_pendulum.py — records free oscillation (no motor).
"""

import csv
import time
import sys
import os
import math
import statistics
from pathlib import Path

# Expose src/ (top-level libs).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_comm import CameraSubscriber
from config import DEFAULT_ZMQ_ADDR

ZMQ_ADDR = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ZMQ_ADDR

# ── Connect ───────────────────────────────────────
cam = CameraSubscriber(ZMQ_ADDR, recv_timeout_ms=2000)

print("=" * 50)
print("  sigma_theta measurement  (hold the pole STILL)")
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
print("  Hold the pole still. Ctrl+C to stop.\n")
print(f"  {'t(s)':>6}  {'θ(°)':>8}  {'ω(r/s)':>8}  pole_up")
print(f"  {'-'*40}")

rows = []
t0 = time.time()

try:
    while True:
        msg = cam.recv()
        if msg is None:
            print("  WARNING: no camera data")
            continue

        # Support both comma- and space-separated formats
        parts = msg.split(",") if "," in msg else msg.split()
        if len(parts) < 7:
            continue

        try:
            theta_rad = float(parts[4])
            omega     = float(parts[5])
            pole_up   = parts[6].strip() in ("1", "True", "true")
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
    writer.writerow(["t_s", "theta_deg", "omega_rad_s", "pole_up"])
    writer.writerows(rows)

cam.close()

print(f"\n  Saved {len(rows)} samples → {filepath}")
if rows:
    print(f"  Duration: {float(rows[-1][0]):.1f} s")

# ── sigma_theta for the EKF (only meaningful if the pole was STATIC) ──
# Measurement noise = how much θ scatters when the true angle is constant.
# The CSV logs degrees; the EKF wants radians, so we convert here (×π/180)
# so there is nothing left to compute by hand.
thetas_deg = [float(r[1]) for r in rows]
if len(thetas_deg) > 1:
    sd_deg = statistics.pstdev(thetas_deg)
    sd_rad = math.radians(sd_deg)
    print()
    print(f"  θ std = {sd_deg:.4f}°  =  {sd_rad:.5f} rad")
    print(f"  -> EKF_SIGMA_THETA = {sd_rad:.5f}   (ONLY valid if pole was static)")
