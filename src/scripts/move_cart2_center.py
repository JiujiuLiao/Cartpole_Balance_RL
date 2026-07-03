"""
set_center.py
=============
Move cart by a set distance, then reset encoder to 0 there.
Run this once to calibrate the physical center of the track.

  OFFSET_M > 0  →  move right
  OFFSET_M < 0  →  move left

Usage:
    python set_center.py COM6
"""

import time
import sys
from pathlib import Path

# Expose src/ (top-level libs).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor_comm import MotorComm
from config import DEFAULT_SERIAL_PORT, SERIAL_BAUD, pulses_for_distance

PORT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERIAL_PORT

# ── Adjust this value to move the cart ────────────
# Positive = right,  Negative = left
OFFSET_M      = -0.39          # m  (e.g. -0.0625 = 6.25 cm left)
OFFSET_PULSES = pulses_for_distance(OFFSET_M)

DRIVE_PWM  = 155   # PWM for moving (must be above deadband ~26)
TOLERANCE  = 10    # pulses — stop within this distance of target


# ── Main ──────────────────────────────────────────
def main():
    direction_label = "right" if OFFSET_M >= 0 else "left"

    print("=" * 50)
    print("  Track Center Calibration")
    print("=" * 50)
    print(f"  Move   : {abs(OFFSET_M)*100:.2f} cm {direction_label} "
          f"({OFFSET_PULSES:+d} pulses)")
    print(f"  Port   : {PORT}")
    print()

    comm = MotorComm(PORT, SERIAL_BAUD, verbose=True,
                     auto_recenter_on_close=False)   # this script IS the recenter

    pos = comm.get_position(default=None)
    if pos is None:
        print("  ERROR: could not read position.")
        comm.close()
        return

    target = pos + OFFSET_PULSES
    print(f"  Current position : {pos:+d} pulses")
    print(f"  Target position  : {target:+d} pulses")
    print()
    input(f"  Make sure cart has room to move {direction_label}. Press ENTER to start.")
    print()

    try:
        # Drive toward target
        while True:
            pos = comm.get_position(default=None)
            if pos is None:
                continue
            remaining = target - pos
            if abs(remaining) <= TOLERANCE:
                break
            drive = DRIVE_PWM if remaining > 0 else -DRIVE_PWM   # right or left
            comm.send("v", drive)
            time.sleep(0.02)
            print(f"\r  pos={pos:+6d}  target={target:+d}  remaining={remaining:+5d} pulses",
                  end="", flush=True)

        # Stop and reset encoder
        comm.stop()
        time.sleep(0.3)
        comm.send("h", 0)
        time.sleep(0.1)

        pos = comm.get_position(default=None)
        print(f"\n  Encoder reset to 0.  Verify: pos={pos} (should be near 0)")
        print("  Done. This position is now encoder 0 = track center.")

    except KeyboardInterrupt:
        print("\n  Interrupted — stopping motor.")
    finally:
        comm.close()


if __name__ == "__main__":
    main()
