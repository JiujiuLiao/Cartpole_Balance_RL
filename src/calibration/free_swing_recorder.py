"""
Record raw free-swing data for the fixed-cart pendulum experiment.

This recorder is deliberately separate from the control/validation logger:
it does not send motor commands, does not use EKF output, and does not use
ZMQ conflate mode. It expects camerapreview_detect_yellow_tape_line_V1.py to
run with --publish-invalid --down-reference, which includes invalid camera
frames and represents the known downward equilibrium near +/-pi.

Run on the Raspberry Pi, after starting the current V1 camera publisher:

    python3 src/vision/camerapreview_detect_yellow_tape_line_V1.py \
        --no-gui --publish-invalid --down-reference

    python3 free_swing_recorder.py --release-angle-deg 5 \
        --run-id 5deg_run01 --duration 25

The operator holds the pendulum at the requested angle.  After ENTER the
script records a short pre-release interval, prints RELEASE NOW, then records
the requested free-swing duration.
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

# Allow direct execution from src/calibration/.
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from camera_comm import CameraSubscriber
from config import (CALIBRATION_DIR, CURRENT_VISION_SCRIPT,
                    DEFAULT_SERIAL_PORT, DEFAULT_ZMQ_ADDR, SERIAL_BAUD)
from motor_comm import MotorComm


FIELDS = [
    "run_id",
    "sensor_ts_ns",
    "theta_raw_rad",
    "valid",
    "cart_pos_pulses",
    "release_angle_deg",
    "frame_id",
    "release_event",
]


def parse_camera_message(msg):
    """Return raw fields from the current publisher's extended message.

    The recorder intentionally rejects the old 8/9-field protocol instead of
    silently treating its already-filtered stream as 100%% valid.
    """
    if not msg:
        return None
    parts = msg.split(",") if "," in msg else msg.split()
    if len(parts) < 10:
        return None
    try:
        sensor_ts_ns = int(parts[8])
        valid = int(parts[9]) == 1
        frame_id = int(parts[10]) if len(parts) >= 11 else -1
        theta = float(parts[4]) if valid else None
        if theta is not None and not math.isfinite(theta):
            theta = None
            valid = False
        return theta, valid, sensor_ts_ns, frame_id
    except (ValueError, TypeError, IndexError):
        return None


def resolve_output(path_text, run_id):
    if path_text:
        path = Path(path_text)
        if not path.is_absolute():
            path = Path(CALIBRATION_DIR) / path
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = (Path(CALIBRATION_DIR) / "free_swing" /
                f"free_swing_{run_id}_{stamp}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description="Raw fixed-cart free-swing recorder")
    parser.add_argument("--release-angle-deg", type=float, required=True,
                        help="Signed target release angle, e.g. +5 or -8")
    parser.add_argument("--run-id", required=True,
                        help="Unique identifier, e.g. 5deg_run01")
    parser.add_argument("--duration", type=float, default=25.0,
                        help="Free-swing recording duration after release (s)")
    parser.add_argument("--pretrigger", type=float, default=2.0,
                        help="Seconds recorded before the release marker")
    parser.add_argument("--port", default=DEFAULT_SERIAL_PORT,
                        help=f"ESP32 serial port (default: {DEFAULT_SERIAL_PORT})")
    parser.add_argument("--zmq", default=DEFAULT_ZMQ_ADDR,
                        help=f"Camera ZMQ address (default: {DEFAULT_ZMQ_ADDR})")
    parser.add_argument("--output", default=None,
                        help="CSV path; default is data/calibration/free_swing/")
    args = parser.parse_args()

    if args.duration <= 0 or args.pretrigger < 0:
        parser.error("--duration must be > 0 and --pretrigger must be >= 0")

    output_path = resolve_output(args.output, args.run_id)
    comm = None
    cam = None
    rows = []
    protocol_mismatch = 0
    timeout_count = 0
    release_marked = False

    print("=" * 68)
    print("  Fixed-cart pendulum free-swing recorder")
    print("=" * 68)
    print(f"  run_id          : {args.run_id}")
    print(f"  release angle   : {args.release_angle_deg:+.2f} deg")
    print(f"  pretrigger      : {args.pretrigger:.1f} s")
    print(f"  post-release    : {args.duration:.1f} s")
    print(f"  serial          : {args.port}")
    print(f"  camera          : {args.zmq}")
    print(f"  publisher       : {CURRENT_VISION_SCRIPT} --publish-invalid")
    print(f"  output          : {output_path}")
    print()
    print("  Confirm: cart mechanically clamped at track center, motor output")
    print("  disconnected or PWM=0, and the pendulum is held at the target angle.")
    input("  Press ENTER to start the pretrigger countdown...")

    try:
        # Connecting to the ESP32 initializes the motor pins to zero in the
        # firmware.  Do not home/reset or send any nonzero motor command here.
        comm = MotorComm(args.port, SERIAL_BAUD, timeout=0.05,
                         echo_timeout=0.08, verbose=True,
                         auto_recenter_on_close=False)
        comm.start_position_stream()
        cam = CameraSubscriber(args.zmq, recv_timeout_ms=200, conflate=False)
        cam.drain()

        print("  Recording started. Keep holding the angle...")
        start_ns = time.monotonic_ns()
        release_ns = start_ns + int(args.pretrigger * 1e9)
        end_ns = release_ns + int(args.duration * 1e9)
        next_status_ns = start_ns + int(1e9)

        while time.monotonic_ns() < end_ns:
            msg = cam.recv()
            now_ns = time.monotonic_ns()

            if now_ns >= release_ns and not release_marked:
                release_marked = True
                print("\n  RELEASE NOW — do not push the pendulum.")

            if msg is None:
                timeout_count += 1
                if now_ns >= next_status_ns:
                    print("  WARNING: no camera message received recently")
                    next_status_ns += int(1e9)
                continue

            parsed = parse_camera_message(msg)
            if parsed is None:
                protocol_mismatch += 1
                continue

            theta, valid, sensor_ts_ns, frame_id = parsed
            cart_pos = comm.get_position(default=0)
            event = 1 if release_marked and not any(
                int(r["release_event"]) for r in rows) else 0
            rows.append({
                "run_id": args.run_id,
                "sensor_ts_ns": sensor_ts_ns,
                "theta_raw_rad": "" if theta is None else f"{theta:.9f}",
                "valid": int(valid),
                "cart_pos_pulses": cart_pos,
                "release_angle_deg": f"{args.release_angle_deg:.6f}",
                "frame_id": frame_id,
                "release_event": event,
            })

            if now_ns >= next_status_ns:
                elapsed = (now_ns - start_ns) / 1e9
                print(f"  t={elapsed:6.1f}s  rows={len(rows):5d}  "
                      f"valid={sum(int(r['valid']) for r in rows):5d}")
                next_status_ns += int(1e9)

    except KeyboardInterrupt:
        print("\n  Stopped early by user.")
    finally:
        if cam is not None:
            cam.close()
        if comm is not None:
            # Explicitly disable auto-recenter above; close only sends PWM=0.
            comm.close()

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_rows = len(rows)
    n_valid = sum(int(r["valid"]) for r in rows)
    frame_ids = [int(r["frame_id"]) for r in rows if int(r["frame_id"]) >= 0]
    if len(frame_ids) >= 2:
        expected = max(frame_ids) - min(frame_ids) + 1
    else:
        expected = n_rows
    valid_ratio = 100.0 * n_valid / expected if expected else 0.0

    print()
    print(f"  Saved rows       : {n_rows}")
    print(f"  Valid frames     : {n_valid}/{expected} ({valid_ratio:.2f}%)")
    print(f"  Camera timeouts  : {timeout_count}")
    print(f"  Old/bad protocol : {protocol_mismatch}")
    print(f"  Output           : {output_path}")
    if not release_marked:
        print("  WARNING: release marker was not reached before stopping.")
    if protocol_mismatch:
        print("  ERROR: start the current V1 vision publisher with "
              "--publish-invalid before accepting this run.")


if __name__ == "__main__":
    main()
