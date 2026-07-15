"""
hold_pwm.py — hold a constant PWM for N seconds (for multimeter probing).

Purpose (2026-07-15): measured full-PWM force is only 3.6% of the motor
nameplate. To find where the power is lost, probe DC voltages along the
chain (PSU output / IBT-2 B+B- / motor terminals) while the motor is
STALLED at full command. This script fires the PWM after ENTER so both
hands are free for the probes.

Usage (on the Pi):
    python3 hold_pwm.py            # +255 for 5 s
    python3 hold_pwm.py -255       # reverse direction
    python3 hold_pwm.py 255 8      # custom duration (s)

Stall setup: press the cart against the track end IN the drive direction
(+ = right end, - = left end) BEFORE firing, so nothing moves.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from motor_comm import MotorComm
from config import DEFAULT_SERIAL_PORT, SERIAL_BAUD

pwm = int(sys.argv[1]) if len(sys.argv) > 1 else 255
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

comm = MotorComm(DEFAULT_SERIAL_PORT, SERIAL_BAUD, verbose=True,
                 auto_recenter_on_close=False)
try:
    end = "RIGHT" if pwm > 0 else "LEFT"
    input(f"  Cart pressed against the {end} end, probes in place.\n"
          f"  ENTER to hold PWM={pwm:+d} for {dur:.0f}s (Ctrl-C to abort)...")
    comm.send_velocity(pwm)
    t0 = time.time()
    while time.time() - t0 < dur:
        print(f"  holding... {dur - (time.time()-t0):.1f}s left", end="\r")
        time.sleep(0.2)
finally:
    comm.stop()
    print("\n  stopped.")
    comm.close()
