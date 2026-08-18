"""
collect_b_data.py
=================
Record a CLEAN dataset for calibrating the EKF input gain b (EKF_INPUT_GAIN).

WHAT IT DOES: with the pole HANGING at rest, it shuttles the cart back and forth
with known PWM steps (alternating sign, a few magnitudes). The hanging pole swings
in response. It logs t / theta / omega / pwm every camera frame for ~RUN_S seconds,
keeping the cart inside a safe position band. Then run fit_ekf_b.py on the CSV.

WHY hanging + PWM steps (not swing-up, not balance):
  - Hanging is a STABLE equilibrium → the pole swings smoothly, the pendulum model
    holds, the EKF gating stays quiet → clean data for regression.
  - PWM steps excite the b*u*cos(theta) term we want to fit.
  (Swing-up is too violent; balance is too short and logs filtered theta.)

SETUP (read before running):
  1. Pole HANGING straight down, at rest. Cart near track center.
  2. The cart WILL move (PWM drives it). Keep hands clear. The script bounds the
     cart to +/-POS_LIMIT pulses and recenters on exit, but watch it.
  3. The current yellow-tape line camera must be publishing. Because this
     experiment fixes the operating branch as HANGING, the recorded line angle
     is lifted by +pi before fitting the oriented pendulum model.

Usage:
    python collect_b_data.py                       # /dev/ttyUSB0, default ZMQ
    python collect_b_data.py /dev/ttyUSB0 tcp://127.0.0.1:9999
    Ctrl+C to stop early (motor stops, CSV saved, cart recentered).

Output: bdata_YYYYMMDD_HHMMSS.csv  ->  python fit_ekf_b.py <that file>
"""

import csv
import math
import sys
import time
import os
from pathlib import Path

# motor_comm / camera_comm / config live in the parent src/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor_comm import MotorComm
from camera_comm import CameraSubscriber, parse_angle
from config import (DEFAULT_SERIAL_PORT, DEFAULT_ZMQ_ADDR, MAX_PULSES,
                    PWM_DEADBAND, CURRENT_VISION_SCRIPT)

PORT     = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERIAL_PORT
ZMQ_ADDR = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ZMQ_ADDR

# ── experiment knobs ────────────────────────────────────────────────
RUN_S        = 25.0                  # total recording time
STEP_S       = 0.40                  # hold each PWM step this long (transient = signal)
PWM_MAGS     = [120, 160, 200]       # cycle through these magnitudes (> deadband)
POS_LIMIT    = int(MAX_PULSES * 0.30)  # ~1780 pulses: stay well inside the track
assert PWM_MAGS[0] > PWM_DEADBAND, "PWM steps must exceed the dead-band"


def main():
    motor = MotorComm(PORT, verbose=True)
    cam   = CameraSubscriber(ZMQ_ADDR, recv_timeout_ms=300)

    # Home the encoder so POS_LIMIT is measured from the true center.
    motor.send("h", 0)
    time.sleep(0.2)

    print("=" * 56)
    print("  b-calibration data collection")
    print("=" * 56)
    print(f"  Port {PORT}   ZMQ {ZMQ_ADDR}")
    print(f"  Vision {CURRENT_VISION_SCRIPT} (hanging branch: theta=line_angle+pi)")
    print(f"  steps={PWM_MAGS} every {STEP_S}s   pos limit=+/-{POS_LIMIT}")
    print()
    print("  Pole HANGING at rest, cart centered. Press ENTER to start...")
    input()
    cam.drain()

    filename = time.strftime("bdata_%Y%m%d_%H%M%S.csv")
    filepath = os.path.join(os.path.dirname(__file__), filename)
    rows = []

    t0          = time.time()
    next_step   = t0
    step_idx    = 0
    sign        = 1
    pwm         = 0
    last_print  = 0.0

    print(f"  Recording -> {filename}   (Ctrl+C to stop)\n")
    try:
        while time.time() - t0 < RUN_S:
            now = time.time()

            # ---- new PWM step? ------------------------------------
            if now >= next_step:
                pos = motor.get_position(default=0)
                mag = PWM_MAGS[step_idx % len(PWM_MAGS)]
                # Safety override: if near an edge, push back toward center,
                # otherwise just alternate sign.
                if pos > POS_LIMIT:
                    sign = -1
                elif pos < -POS_LIMIT:
                    sign = +1
                else:
                    sign = -sign
                pwm = sign * mag
                motor.send_velocity(pwm)
                next_step = now + STEP_S
                step_idx += 1

            # ---- log one camera frame -----------------------------
            msg = cam.recv()
            if msg is not None:
                parsed = parse_angle(msg)          # (theta_rad, omega, pole_up)
                if parsed is not None:
                    theta_line, omega, _ = parsed
                    # The line detector is modulo pi. This experiment guarantees
                    # the pole is near the hanging branch, so theta=line+pi is
                    # the corresponding oriented model angle. Trig/derivatives
                    # remain continuous for the small oscillations used here.
                    theta_rad = theta_line + math.pi
                    t = now - t0
                    rows.append([f"{t:.4f}", f"{theta_rad*57.2958:.3f}",
                                 f"{omega:.4f}", pwm, motor.get_position(default=0)])
                    if t - last_print > 0.5:
                        print(f"  t={t:5.1f}s  theta={theta_rad*57.2958:+7.1f}  "
                              f"pwm={pwm:+5d}  n={len(rows)}")
                        last_print = t
    except KeyboardInterrupt:
        print("\n  stopped early")
    finally:
        motor.stop()

    # save
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "theta_deg", "omega_rad_s", "pwm", "cart_pos"])
        w.writerows(rows)
    print(f"\n  Saved {len(rows)} samples -> {filepath}")
    print(f"  Next:  python fit_ekf_b.py {filename}")

    motor.close()   # auto-recenters the cart
    cam.close()


if __name__ == "__main__":
    main()
