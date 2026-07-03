"""
balance_pd.py - PD balance controller, called inline from swing_up_energy_dc.py.
================================================================================
Drop-in replacement for `balance_lqr_loop`:
    from balance_pd import balance_pd_loop
    bal_result = balance_pd_loop(comm, cam, t0, log_writer)

Same return contract:  'fall' | 'boundary' | 'interrupt'.

Control law (PD + cart-position drift correction):
    pwm = ANGLE_SIGN · (-Kp · err)
        + OMEGA_SIGN · ( Kd · omega)
        -  CART_SIGN · ( Kx · pos)
where err = theta_deg (camera convention: upright = 0°).

Gains (Kp/Kd/Kx) and signs (ANGLE_SIGN/OMEGA_SIGN/CART_SIGN) live in
config.py — historical exception noted in ../CLAUDE.md.
"""

import math
import sys
import time
from pathlib import Path

# Expose src/ (top-level libs: config, motor_comm, camera_comm, ekf).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    Kp, Kd, Kx, ANGLE_SIGN, OMEGA_SIGN, CART_SIGN,
    PWM_DEADBAND, PWM_LIMIT, GIVEUP_ANGLE_DEG, MAX_PULSES,
)

# Control loop
CTRL_HZ           = 100
DT                = 1.0 / CTRL_HZ
CART_POLL_EVERY_N = 5      # 20 Hz cart polling at 100 Hz loop (less serial load)


def balance_pd_loop(comm, cam, t0, log_writer):
    """
    Inline PD + cart-pos balance. Drop-in replacement for balance_lqr_loop.

    Args:
      comm        : MotorComm  (send_velocity, get_position)
      cam         : CameraSubscriber  (latest_angle -> (theta_rad, omega, pole_up))
      t0          : run start time (for log timestamps, matches swing-up clock)
      log_writer  : csv.writer — same columns as swing-up's log_writer

    Returns 'fall' | 'boundary' | 'interrupt'.
    """
    pos       = comm.get_position()
    pos_count = 0
    t_next    = time.time() + DT

    print(f"\n  --- BALANCE PD (Kp={Kp} Kd={Kd} Kx={Kx}) ---")
    print(f"  Will give up if |θ|>{GIVEUP_ANGLE_DEG}° or |cart|>{MAX_PULSES} pulses.")

    last_print = 0.0
    try:
        while True:
            # ---- pace loop ------------------------------------------
            now = time.time()
            sleep_t = t_next - now
            if sleep_t > 0:
                time.sleep(sleep_t)
            t_next += DT

            # ---- pull latest vision --------------------------------
            result = cam.latest_angle()
            if result is None:
                continue
            theta_rad, omega, _pole_up = result
            theta_deg = math.degrees(theta_rad)

            # ---- safety: pendulum fell -----------------------------
            if abs(theta_deg) > GIVEUP_ANGLE_DEG:
                comm.send_velocity(0)
                print(f"\n  [PD] FELL  θ={theta_deg:+.1f}°  ω={omega:+.2f} rad/s")
                return 'fall'

            # ---- cart position (subsampled) ------------------------
            pos_count += 1
            if pos_count >= CART_POLL_EVERY_N:
                pos = comm.get_position()
                pos_count = 0
                if abs(pos) > MAX_PULSES:
                    comm.send_velocity(0)
                    print(f"\n  [PD] CART BOUNDARY pos={pos}")
                    return 'boundary'

            # ---- PD + cart-pos drift correction --------------------
            err = theta_deg     # camera convention: upright = 0°
            pwm = (ANGLE_SIGN * (-Kp * err)
                 + OMEGA_SIGN * ( Kd * omega)
                 -  CART_SIGN * ( Kx * pos))

            # Deadband compensation
            if pwm != 0 and abs(pwm) < PWM_DEADBAND:
                pwm = PWM_DEADBAND if pwm > 0 else -PWM_DEADBAND
            pwm = int(max(-PWM_LIMIT, min(PWM_LIMIT, pwm)))

            comm.send_velocity(pwm)

            # ---- log every cycle, print throttled ------------------
            t = time.time() - t0
            log_writer.writerow([f"{t:.4f}", f"{theta_deg:.2f}", f"{omega:.4f}",
                                  "0.0000", pwm, pos])
            if t - last_print > 0.2:
                print(f"  {t:5.2f}  [PD] θ={theta_deg:+6.1f}  ω={omega:+6.2f}  "
                      f"pwm={pwm:+5}  pos={pos:+5}")
                last_print = t

    except KeyboardInterrupt:
        comm.send_velocity(0)
        print("\n  [PD] interrupted")
        return 'interrupt'
