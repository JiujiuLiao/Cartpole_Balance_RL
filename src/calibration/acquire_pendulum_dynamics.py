"""
acquire_pendulum_dynamics.py
============================
PURPOSE : Record pendulum free-oscillation angle data for parameter
          identification (natural frequency, damping coefficient).
          Run STANDALONE — no motor running.
RUN ON  : Raspberry Pi (connects to local camera via ZMQ localhost:9999)
USE WHEN: You want to identify T, omega0, damping from free oscillation.

Usage:
    python acquire_pendulum_dynamics.py              # record 45 s
    python acquire_pendulum_dynamics.py --duration 60

Steps:
    1. Start camera:
       python3 src/vision/camerapreview_detect_yellow_tape_line_V1.py --no-gui
    2. Mechanically lock the cart. Run this script and release the pendulum
       from a small displacement about the known hanging equilibrium.
    3. CSV saved automatically when recording ends

Output: pendulum_YYYYMMDD_HHMMSS.csv  ->  analyze with analyze_pendulum.py (on PC)
"""

import csv
import sys
import time
import argparse
import os
from pathlib import Path

# Allow `from camera_comm import ...` when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_comm import CameraSubscriber, parse_angle_ts
from config import CALIBRATION_DIR, CURRENT_VISION_SCRIPT

# ── Parameters ────────────────────────────────────
ZMQ_ADDRESS  = "tcp://localhost:9999"   # this script runs on the Pi, hence localhost
DURATION     = 45        # default recording duration (s)

parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=float, default=DURATION,
                    help="Recording duration in seconds (default: 45)")
parser.add_argument("--output", type=str, default=None,
                    help="Output filename (default: pendulum_YYYYMMDD_HHMMSS.csv)")
args = parser.parse_args()

# Auto-generate filename with timestamp if not specified
if args.output is None:
    args.output = "pendulum_" + time.strftime("%Y%m%d_%H%M%S") + ".csv"

# ── ZMQ setup ─────────────────────────────────────
cam = CameraSubscriber(ZMQ_ADDRESS, recv_timeout_ms=2000)

print("=" * 55)
print("  Pendulum Dynamics Data Acquisition")
print("=" * 55)
print(f"  Duration : {args.duration} s")
print(f"  Output   : {args.output}")
print(f"  Vision   : {CURRENT_VISION_SCRIPT}")
print("  Angle    : unoriented line tilt; valid locally around known hanging vertical")
print()
print("  Confirm camera is running, then press ENTER.")
print("  Push the pendulum after ENTER to start recording.")
input()

# ── Recording ─────────────────────────────────────
rows            = []
start_time      = time.time()
end_time        = start_time + args.duration
first_timestamp = None    # first SensorTimestamp (or monotonic fallback)
count           = 0

print("  Recording started... (Ctrl+C to stop early)")

try:
    while time.time() < end_time:
        msg = cam.recv()
        if msg is None:
            print("  Warning: no camera data received. Check camera script.")
            continue

        parsed = parse_angle_ts(msg)
        if parsed is None:
            continue
        angle_raw, angle_velocity, _pole_up, sensor_ts_ns = parsed

        # SensorTimestamp is generated at camera capture on CLOCK_BOOTTIME and
        # removes variable receive/processing delay from the period estimate.
        # Fall back to monotonic receive time only if the camera metadata is absent.
        timestamp = (sensor_ts_ns / 1e9
                     if sensor_ts_ns is not None else time.monotonic())
        if first_timestamp is None:
            first_timestamp = timestamp
        elapsed = timestamp - first_timestamp

        rows.append([round(elapsed, 6), angle_raw, angle_velocity,
                     sensor_ts_ns or 0])
        count += 1

        # Print progress roughly once per second
        if count % 30 == 0:
            remaining = end_time - time.time()
            print(f"  {elapsed:6.1f}s | "
                  f"angle = {angle_raw * 57.3:+7.2f} deg | "
                  f"remaining {remaining:.0f}s")

except KeyboardInterrupt:
    print("\n  Stopped early by user.")

cam.close()

# ── Save CSV ──────────────────────────────────────
os.makedirs(CALIBRATION_DIR, exist_ok=True)
output_path = (args.output if os.path.isabs(args.output)
               else os.path.join(CALIBRATION_DIR, args.output))
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_s", "angle_rad", "angle_velocity_rad_s",
                     "sensor_ts_ns"])
    writer.writerows(rows)

print()
print(f"  Acquisition complete — {count} samples recorded.")
print(f"  Saved to: {output_path}")
print()
print("  Next step: run analyze_pendulum.py on PC to identify parameters.")
