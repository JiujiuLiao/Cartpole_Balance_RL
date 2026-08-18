"""
record_validation_data.py
=========================
Record cart-pole state + applied action at 50 Hz for sim-vs-real validation.

The CSV produced here is replayed in the simulation notebook
(notebooks/01_simulation_environment.ipynb, Section 5):
for the SAME initial state and SAME action sequence, simulator
trajectories should overlay the real ones for the first ~1-2 seconds.

This script reads two data sources simultaneously:
  - serial:  ESP32 encoder -> cart position x (pulses -> m)
  - ZMQ:     Pi camera     -> pole angle theta (rad)

Three recording modes:
  constant_force  Hold one PWM for the first half, then PWM=0.
                  Use to validate: motor force model + friction.
  step            PWM=0 for 0.5s, PWM=value for 0.5s, then PWM=0.
                  Use to validate: cart acceleration response.
  free            PWM always 0. Push pendulum manually after start.
                  Use to validate: pole damping b_pole + coupling.

Usage:
    python record_validation_data.py --mode free --duration 8
    python record_validation_data.py --mode constant_force --pwm 100 --duration 3
    python record_validation_data.py --mode step --pwm 120 --duration 3

CSV columns:
    t_s            elapsed time (s, PC time.perf_counter)
    action_pwm     PWM sent this tick
    x_pulses       raw encoder count
    x_m            cart position (m)
    x_dot_m_s      finite-diff cart velocity (m/s)
    theta_rad      latest yellow-tape LINE orientation (rad, modulo pi);
                   upright and hanging are both 0; None if no fresh frame
    theta_dot_rad_s  camera-reported angular velocity (rad/s); None if stale
    cam_ts         Pi timestamp from camera message (s); None if stale
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Allow `from motor_comm import ...` when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor_comm import MotorComm
from camera_comm import CameraSubscriber
from config import (DEFAULT_SERIAL_PORT, SERIAL_BAUD,
                    DEFAULT_ZMQ_ADDR, M_PER_PULSE)

# Control loop matches simulation dt (notebooks/01_simulation_environment.ipynb).
CONTROL_HZ = 50
DT         = 1.0 / CONTROL_HZ

# Safety: stop and bail out if encoder reports beyond this displacement.
BOUNDARY_PULSES = 3000      # ~24 cm from center (track half-length ~42 cm)


def action_for_mode(mode, pwm, t, duration):
    """Action schedule per mode, returned as PWM in [-255, 255]."""
    if mode == "free":
        return 0
    if mode == "constant_force":
        return pwm if t < duration * 0.5 else 0
    if mode == "step":
        if t < 0.5:
            return 0
        if t < 1.0:
            return pwm
        return 0
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["constant_force", "step", "free"],
                        required=True, help="Recording mode")
    parser.add_argument("--pwm", type=int, default=100,
                        help="PWM for constant_force / step modes (ignored for free)")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Recording length (s)")
    parser.add_argument("--port", type=str, default=DEFAULT_SERIAL_PORT,
                        help="ESP32 serial port")
    parser.add_argument("--zmq", type=str, default=DEFAULT_ZMQ_ADDR,
                        help="Pi camera ZMQ address (current yellow-tape line publisher)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV filename")
    args = parser.parse_args()

    if args.output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"validation_{args.mode}_pwm{args.pwm}_{stamp}.csv"

    print("=" * 60)
    print("  Cart-Pole Validation Recorder")
    print("=" * 60)
    print(f"  Mode:        {args.mode}")
    print(f"  PWM target:  {args.pwm}")
    print(f"  Duration:    {args.duration} s   ({int(args.duration*CONTROL_HZ)} ticks @ {CONTROL_HZ} Hz)")
    print(f"  Serial:      {args.port}")
    print(f"  Camera ZMQ:  {args.zmq}")
    print(f"  Output:      {args.output}")
    print()
    print("  Setup:")
    print("    1. camerapreview_detect_yellow_tape_line_V1.py is running on the Pi.")
    print("    2. Cart near center, free to move.")
    print("       NOTE: theta is a modulo-pi line angle. Record which physical")
    print("       branch (upright/hanging) was used before simulation replay.")
    if args.mode == "free":
        print("    3. After ENTER, push the pendulum to a known angle and release.")
    elif args.mode == "constant_force":
        print(f"    3. After ENTER, motor will hold PWM={args.pwm} for "
              f"{args.duration*0.5:.1f}s, then stop.")
    else:
        print(f"    3. After ENTER, motor will pulse PWM={args.pwm} from t=0.5s to t=1.0s.")
    print()

    # Connect: short echo timeout so encoder reads don't blow our 20 ms budget.
    comm = MotorComm(args.port, SERIAL_BAUD, timeout=0.05,
                     echo_timeout=0.08, verbose=True)
    cam = CameraSubscriber(args.zmq, recv_timeout_ms=2, conflate=True)

    comm.send("h", 0)              # zero the encoder
    time.sleep(0.2)

    input("  Press ENTER to start recording...")
    comm.send("h", 0)              # zero again right before recording
    time.sleep(0.05)

    rows = []
    last_x_pulses = 0
    last_x_t      = None
    boundary_hit  = False

    t_start   = time.perf_counter()
    next_tick = t_start

    try:
        while True:
            now = time.perf_counter()
            t   = now - t_start
            if t >= args.duration:
                break

            # 1. Compute and send action
            action = action_for_mode(args.mode, args.pwm, t, args.duration)
            comm.send_velocity(action)

            # 2. Read encoder
            x_pulses = comm.get_position(echo_timeout=0.015,
                                         default=last_x_pulses)
            x_m = x_pulses * M_PER_PULSE
            if last_x_t is not None:
                dt_enc   = now - last_x_t
                x_dot_m  = ((x_pulses - last_x_pulses) * M_PER_PULSE
                            / dt_enc) if dt_enc > 0 else 0.0
            else:
                x_dot_m = 0.0

            last_x_pulses = x_pulses
            last_x_t      = now

            # 3. Pull freshest camera message (non-blocking-ish)
            theta = theta_dot = cam_ts = None
            msg = cam.latest()
            if msg is not None:
                parts = msg.split(",") if "," in msg else msg.split()
                if len(parts) >= 9:
                    try:
                        theta     = float(parts[4])
                        theta_dot = float(parts[5])
                        cam_ts    = float(parts[8])
                    except ValueError:
                        pass

            rows.append({
                "t_s":             round(t, 4),
                "action_pwm":      action,
                "x_pulses":        x_pulses,
                "x_m":             round(x_m, 5),
                "x_dot_m_s":       round(x_dot_m, 4),
                "theta_rad":       theta,
                "theta_dot_rad_s": theta_dot,
                "cam_ts":          cam_ts,
            })

            # 4. Safety: bail if cart approaches end of track
            if abs(x_pulses) > BOUNDARY_PULSES:
                print(f"\n  BOUNDARY HIT at t={t:.2f}s, x_pulses={x_pulses}. Stopping.")
                boundary_hit = True
                break

            # 5. Sleep to next tick boundary
            next_tick += DT
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
    finally:
        comm.stop()
        time.sleep(0.1)
        comm.close()
        cam.close()

    if not rows:
        print("  No samples recorded.")
        return

    # Camera coverage stats
    n_total      = len(rows)
    n_with_cam   = sum(1 for r in rows if r["theta_rad"] is not None)
    coverage_pct = 100.0 * n_with_cam / n_total

    print()
    print(f"  Recorded:        {n_total} ticks")
    print(f"  Camera coverage: {n_with_cam}/{n_total} ({coverage_pct:.1f}%)")
    if boundary_hit:
        print("  WARNING: stopped early due to boundary safety.")

    # Save CSV
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            args.output)
    fields = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved: {args.output}")
    print()
    print("  Next step:")
    print("    Open notebooks/01_simulation_environment.ipynb, Section 5,")
    print("    point REAL_DATA_PATH at this CSV, run the comparison cell.")


if __name__ == "__main__":
    main()
