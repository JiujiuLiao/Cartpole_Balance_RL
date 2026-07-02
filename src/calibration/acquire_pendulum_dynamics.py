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
    1. Start camera:  python camerapreview_detect_color2.py --no-gui
    2. Run this script, give the pendulum a push when prompted
    3. CSV saved automatically when recording ends

Output: pendulum_YYYYMMDD_HHMMSS.csv  ->  analyze with analyze_pendulum.py (on PC)

See also: cartpole-dc/log_angle.py  - records angle during active swing-up
          (runs on PC, motor running at the same time)
"""

import csv
import sys
import time
import argparse
import os
from pathlib import Path

# Allow `from camera_comm import ...` when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_comm import CameraSubscriber

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
print()
print("  Confirm camera is running, then press ENTER.")
print("  Push the pendulum after ENTER to start recording.")
input()

# ── Recording ─────────────────────────────────────
rows            = []
start_time      = time.time()
end_time        = start_time + args.duration
first_timestamp = None    # first Pi timestamp received, used as time origin
count           = 0

print("  Recording started... (Ctrl+C to stop early)")

try:
    while time.time() < end_time:
        msg = cam.recv()
        if msg is None:
            print("  Warning: no camera data received. Check camera script.")
            continue

        parts = msg.split(",")

        # ZMQ format — current cameras (V3, color2_henry) send 8 fields:
        #   parts[0~4] : angles (queue, oldest to newest, rad, 0=up, pi=down)
        #   parts[5]   : angle_velocity_avg (rad/s)
        #   parts[6]   : pole_up (1 if near upright)
        #   parts[7]   : timestamp (V3: time.time_ns(); color2: loop start)
        # (the old 9-field format with a 2nd timestamp at parts[8] no longer exists;
        #  requiring 9 here silently dropped EVERY frame → 0 samples recorded.)
        if len(parts) < 8:
            continue

        try:
            angle_raw       = float(parts[4])   # most recent angle (rad)
            angle_velocity  = float(parts[5])   # average angular velocity (rad/s)
        except ValueError:
            continue

        # This script runs ON the Pi receiving from the LOCAL camera, so the local
        # receive time is accurate enough for period/damping ID (recv jitter ~ms vs
        # pendulum period ~1-2 s = negligible). Using it avoids the field-index /
        # unit ambiguity of the embedded camera timestamp (ns vs s, parts[7] vs [8]).
        timestamp = time.time()
        if first_timestamp is None:
            first_timestamp = timestamp
        elapsed = timestamp - first_timestamp

        rows.append([round(elapsed, 4), angle_raw, angle_velocity])
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
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_s", "angle_rad", "angle_velocity_rad_s"])
    writer.writerows(rows)

print()
print(f"  Acquisition complete — {count} samples recorded.")
print(f"  Saved to: {output_path}")
print()
print("  Next step: run analyze_pendulum.py on PC to identify parameters.")
