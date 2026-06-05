"""
balance_lqr.py - LQR balance controller, called inline from swing_up_energy_dc.py.
==================================================================================
Drop-in replacement for `balance_loop` in swing_up_energy_dc.py:
    from balance_lqr import balance_lqr_loop
    bal_result = balance_lqr_loop(comm, cam, t0, log_writer)

Same return contract:  'fall' | 'boundary' | 'interrupt'.

State vector:
    x = [theta_rad, theta_dot_rad_s, x_cart_m, x_cart_dot_m_s]

Control law (K precomputed by compute_lqr_K.py):
    pwm = -(K_theta · theta  +  K_theta_dot · theta_dot
          +  K_x · x_cart    +  K_x_dot · x_cart_dot)

K is in PWM units because compute_lqr_K.py uses FORCE_PER_PWM in B.
"""

import math
import time

from config import (
    M_PER_PULSE, MAX_PULSES, PWM_DEADBAND, PWM_LIMIT, GIVEUP_ANGLE_DEG,
)

# ----------------------------------------------------------------------
# LQR GAINS  ←  paste from compute_lqr_K.py output
# ----------------------------------------------------------------------
# These are PLACEHOLDERS.  Run `python compute_lqr_K.py` on PC, then
# copy the printed K values here.
K_THETA      =  -800.0    # PLACEHOLDER
K_THETA_DOT  =  -120.0    # PLACEHOLDER
K_X          =    -5.0    # PLACEHOLDER
K_X_DOT      =   -45.0    # PLACEHOLDER

# Hardware sign tweak: the model assumes θ>0 = tilt right, +PWM = cart right.
# This matches the PD controller (which uses ANGLE_SIGN=-1 in its formula),
# so leave PWM_SIGN=+1 here. Flip to -1 only if the cart pushes the wrong
# way on the very first balance attempt.
PWM_SIGN = +1

# Control loop
CTRL_HZ            = 100
DT                 = 1.0 / CTRL_HZ

# Cart velocity estimate
CART_POLL_EVERY_N  = 2        # 50 Hz cart polling at 100 Hz loop
CART_VEL_LPF_ALPHA = 0.30     # EMA on raw (Δpos/Δt) — high = more responsive,
                              # low = smoother (and more lag)


def _deadband(pwm):
    if pwm == 0:
        return 0
    if abs(pwm) < PWM_DEADBAND:
        return PWM_DEADBAND if pwm > 0 else -PWM_DEADBAND
    return pwm


def balance_lqr_loop(comm, cam, t0, log_writer):
    """
    Inline LQR balance. Drop-in replacement for swing_up_energy_dc.balance_loop.

    Args:
      comm        : MotorComm  (send_velocity, get_position)
      cam         : CameraSubscriber  (latest_angle -> (theta_rad, omega, pole_up))
      t0          : run start time (for log timestamps, matches swing-up clock)
      log_writer  : csv.writer — same columns as swing-up's log_writer

    Returns 'fall' | 'boundary' | 'interrupt'.
    """
    print(f"\n  --- BALANCE LQR (K=[{K_THETA:+.1f}, {K_THETA_DOT:+.1f}, "
          f"{K_X:+.2f}, {K_X_DOT:+.1f}]) ---")
    print(f"  Will give up if |θ|>{GIVEUP_ANGLE_DEG}° or |cart|>{MAX_PULSES} pulses.")

    # State estimates
    theta_rad        = 0.0
    omega            = 0.0
    pos              = comm.get_position()
    cart_pos_m       = pos * M_PER_PULSE
    cart_pos_m_prev  = cart_pos_m
    cart_vel_m_s     = 0.0
    last_cart_t      = time.time()
    cart_poll_count  = 0

    t_next     = time.time() + DT
    last_print = 0.0

    try:
        while True:
            # ---- pace loop ------------------------------------------
            now      = time.time()
            sleep_t  = t_next - now
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
                print(f"\n  [LQR] FELL  θ={theta_deg:+.1f}°  ω={omega:+.2f} rad/s")
                return 'fall'

            # ---- cart pos + velocity (50 Hz) -----------------------
            cart_poll_count += 1
            if cart_poll_count >= CART_POLL_EVERY_N:
                cart_poll_count = 0
                pos             = comm.get_position()
                cart_pos_m      = pos * M_PER_PULSE

                t_cart = time.time()
                dt_c   = max(t_cart - last_cart_t, 1e-6)
                v_inst = (cart_pos_m - cart_pos_m_prev) / dt_c
                cart_vel_m_s    = (CART_VEL_LPF_ALPHA * v_inst
                                   + (1.0 - CART_VEL_LPF_ALPHA) * cart_vel_m_s)
                cart_pos_m_prev = cart_pos_m
                last_cart_t     = t_cart

                # ---- safety: cart boundary -------------------------
                if abs(pos) > MAX_PULSES:
                    comm.send_velocity(0)
                    print(f"\n  [LQR] CART BOUNDARY pos={pos}")
                    return 'boundary'

            # ---- LQR -----------------------------------------------
            u_pwm = -(K_THETA     * theta_rad
                    + K_THETA_DOT * omega
                    + K_X         * cart_pos_m
                    + K_X_DOT     * cart_vel_m_s)
            pwm = int(max(-PWM_LIMIT, min(PWM_LIMIT, _deadband(PWM_SIGN * u_pwm))))

            comm.send_velocity(pwm)

            # ---- log every cycle, print throttled -----------------
            t = time.time() - t0
            # CSV columns match swing-up's log writer (t, θ, ω, _, pwm, pos).
            # We dump cart_vel_m_s into the E_diff column to keep the schema.
            log_writer.writerow([f"{t:.4f}", f"{theta_deg:.2f}", f"{omega:.4f}",
                                  f"{cart_vel_m_s:.4f}", pwm, pos])
            if t - last_print > 0.2:
                print(f"  {t:5.2f}  [LQR] θ={theta_deg:+6.1f}  ω={omega:+6.2f}  "
                      f"ẋ={cart_vel_m_s:+5.2f}  pwm={pwm:+5}  pos={pos:+5}")
                last_print = t

    except KeyboardInterrupt:
        comm.send_velocity(0)
        print("\n  [LQR] interrupted")
        return 'interrupt'
