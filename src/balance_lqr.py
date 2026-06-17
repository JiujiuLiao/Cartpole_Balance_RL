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
K_THETA      = -1752.4887    # theta gain
K_THETA_DOT  = -306.8017    # theta_dot gain
K_X          = -3.1623    # cart-position gain
K_X_DOT      = -25.3450    # cart-velocity gain

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

# ── Delay compensation (Tier-2 angle predictor) ──────────────────────
# Vision theta/omega arrive ~T_DELAY seconds late (glass->motion latency).
# Extrapolate them forward to "now" before feeding the LQR, using the
# pendulum's free near-upright dynamics: theta_ddot ≈ (g/l_eff)*theta.
#   theta_pred = theta + omega*T + 0.5*(g/l_eff)*theta*T^2
#   omega_pred = omega +          (g/l_eff)*theta*T
# Note: this ignores the cart's own motion during the delay (2nd-order); a
# Level-2 observer would include it. Safety/giveup still uses MEASURED theta.
PREDICT_ENABLE = True
T_DELAY        = 0.043     # s, sensor->actuation latency (tune: glass->motion 43-57ms)
GRAV_OVER_LEFF = 31.5       # = ω₀² 实测 (单摆自由摆动, 2026-06-17, FFT 31.2 + 过零点 31.9)
                            # 注意: 这不是 g/几何长(g/0.34=28.85, 那是点质量近似);
                            # 你的摆有配重 → 真身 = m·g·d/I_pivot = ω₀² ≈ 31.5,
                            # 等效 g/l_eff, l_eff≈0.31m. 旧值 28.85 偏低~9%, 已弃.


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

    # --- loop-rate diagnostics (remove once tuned) -------------------
    # Tells us WHO gates the rate: the 100 Hz loop, the camera feed, or
    # the serial get_position() call.
    diag_t0     = time.time()
    diag_ticks  = 0      # every while iteration (incl. None-frame skips)
    diag_frames = 0      # iterations that got a FRESH vision frame
    diag_none   = 0      # iterations where vision was None -> continue
    diag_gp_sum = 0.0    # total seconds spent inside get_position()
    diag_gp_n   = 0      # number of get_position() calls

    try:
        while True:
            # ---- pace loop ------------------------------------------
            now      = time.time()
            sleep_t  = t_next - now
            if sleep_t > 0:
                time.sleep(sleep_t)
            t_next += DT

            # ---- loop-rate diagnostics (remove once tuned) ----------
            diag_ticks += 1
            now_d = time.time()
            if now_d - diag_t0 >= 1.0:
                el    = now_d - diag_t0
                gp_ms = (diag_gp_sum / diag_gp_n * 1000.0) if diag_gp_n else 0.0
                print(f"  [diag] loop={diag_ticks/el:4.0f}Hz  "
                      f"vision={diag_frames/el:4.0f}Hz  "
                      f"none={diag_none/el:4.0f}Hz  "
                      f"get_pos={gp_ms:4.1f}ms")
                diag_t0     = now_d
                diag_ticks  = 0
                diag_frames = 0
                diag_none   = 0
                diag_gp_sum = 0.0
                diag_gp_n   = 0

            # ---- pull latest vision --------------------------------
            result = cam.latest_angle()
            if result is None:
                diag_none += 1
                continue
            diag_frames += 1
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
                _t_gp           = time.time()
                pos             = comm.get_position()
                diag_gp_sum    += time.time() - _t_gp
                diag_gp_n      += 1
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

            # ---- delay compensation: predict theta/omega to "now" --
            if PREDICT_ENABLE:
                theta_ctrl = (theta_rad + omega * T_DELAY
                              + 0.5 * GRAV_OVER_LEFF * theta_rad * T_DELAY**2)
                omega_ctrl = omega + GRAV_OVER_LEFF * theta_rad * T_DELAY
            else:
                theta_ctrl = theta_rad
                omega_ctrl = omega

            # ---- LQR (fed the delay-compensated state) -------------
            u_pwm = -(K_THETA     * theta_ctrl
                    + K_THETA_DOT * omega_ctrl
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
