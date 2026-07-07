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

import csv
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

# Expose src/ (top-level libs: config, motor_comm, camera_comm, ekf).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    M_PER_PULSE, MAX_PULSES, PWM_DEADBAND, PWM_LIMIT, GIVEUP_ANGLE_DEG,
    BALANCE_DIAG_DIR,
)
from ekf import PendulumEKF

# SensorTimestamp (parts[8]) is on CLOCK_BOOTTIME; "now" must be read on the SAME
# clock for the age to be valid. CLOCK_BOOTTIME is Linux-only (the Pi); guard so
# the module still imports on a dev box (Windows/macOS).
try:
    _CLOCK_BOOTTIME = time.CLOCK_BOOTTIME
except AttributeError:
    _CLOCK_BOOTTIME = None

# ----------------------------------------------------------------------
# LQR GAINS  ←  paste from compute_lqr_K.py output
# ----------------------------------------------------------------------
# These are PLACEHOLDERS.  Run `python compute_lqr_K.py` on PC, then
# copy the printed K values here.
# LQR state-feedback gains  →  state = [theta, theta_dot, x_cart, x_cart_dot]
# Control law: pwm = -(K_THETA*theta + K_THETA_DOT*theta_dot
#                      + K_X*x + K_X_DOT*x_dot)   (see ~line 294)
# Computed by compute_lqr_K.py (Riccati solution), NOT hand-tuned.
# Larger magnitude = that state is weighted more heavily / corrected harder.
#K_THETA      = -1752.4887   # pole angle theta [rad] — largest; primary balancer
#K_THETA_DOT  = -306.8017    # pole rate theta_dot [rad/s] — damping, kills overshoot
#K_X          = -3.1623      # cart position x [m] — smallest; gently recenters cart
#K_X_DOT      = -25.3450     # cart velocity x_dot [m/s] — damps cart motion
# 2026-07-07 (evening): DELAY-AWARE gain set + outer-loop low-pass.
# History of this day's iterations (all hand-test verified, see activity log):
#   hot  [-7912,-1372,-1700,-2971] and soft [-5281,-925,-800,-2103] both hit a
#   GROWING ~2.6-3.1 Hz oscillation. Frequency-domain analysis of the full
#   chain (plant + 45ms camera + EKF + 62ms forward + 27ms actuation + 40ms
#   v-window), calibrated against 4 rounds of hardware data, showed why:
#   theta-loop natural freq sqrt(|B2|*K_THETA - w0^2) must stay inside the
#   delay-tolerable window (~8 rad/s, like the OLD -1780 gains, which never
#   oscillated at x1.0); the new hot/soft sets put it at 16-21 rad/s where the
#   ~72ms total delay flips damping negative. The outer loop (K_X_DOT) also
#   pumps the 2.5 Hz mode unless band-limited — hence OUTER_LP_W below.
# Model worst closed-loop root: -0.11 (reference: old K, which hardware shows
# as non-oscillating, maps to +1.44 in the same model -> real margin larger).
K_THETA      = -1780    # theta gain        (old, delay-compatible scale)
K_THETA_DOT  = -284     # theta_dot gain    (old, delay-compatible scale)
K_X          = -200     # cart-position gain  (anti-drift, band-limited)
K_X_DOT      = -2103    # cart-velocity gain  (anti-drift, band-limited)

# First-order low-pass on the OUTER-loop term (K_X*x + K_X_DOT*v). The outer
# loop only needs to act on the seconds-scale drift; at 2.5 Hz its lagged
# feedback destabilizes the pole mode (measured rounds 3-4). 10 rad/s corner
# keeps drift control intact and removes its gain where it hurts.
OUTER_LP_W   = 10.0     # rad/s (~1.6 Hz corner)
# Hardware sign tweak: the model assumes θ>0 = tilt right, +PWM = cart right.
# This matches the PD controller (which uses ANGLE_SIGN=-1 in its formula),
# so leave PWM_SIGN=+1 here. Flip to -1 only if the cart pushes the wrong
# way on the very first balance attempt.
PWM_SIGN = +1

# Control loop
CTRL_HZ            = 100
DT                 = 1.0 / CTRL_HZ

# Cart velocity estimate
# 2026-07-07: velocity = Δpos over a FIXED ~40ms window, NOT between consecutive
# distinct cache updates. The old Δpos/Δt used "when the loop noticed the value"
# as the clock; serial/thread/loop jitter (±3-5ms on a 10ms baseline) produced
# physically impossible spikes (measured +0.94 m/s in one tick, motor limit
# ~0.29) which K_X_DOT amplified into ±2500 PWM garbage (testbal_ 194211-194245).
# A 40ms baseline dilutes the same jitter to <~12% error; quantization noise is
# 1 pulse/40ms ≈ 0.003 m/s; the estimate lags ~20ms — small vs τ=178ms.
CART_POLL_EVERY_N  = 1        # read cached position every 100 Hz loop
VEL_WINDOW_S       = 0.040    # velocity differencing baseline (30-50ms all fine)

# ── Delay compensation (Tier-2 angle predictor) ──────────────────────
# Vision arrives late (glass->motion). We extrapolate the state forward by the
# delay before feeding the LQR. The delay is NOT a constant (measured 38-58ms),
# so it is computed PER FRAME from the camera's hardware capture timestamp:
#
#   T_delay_dyn = frame_age + T_ACTUATION
#     frame_age   = now_BOOTTIME - SensorTimestamp   (MEASURED each loop via
#                   parts[8] on CLOCK_BOOTTIME). This is glass->now: it INCLUDES
#                   the ~20ms photon-capture latency + processing + ZMQ transport
#                   + control-loop staleness, and all their jitter.
#     T_ACTUATION = FIXED part STILL in the future when we predict: controller
#                   compute + serial + motor mechanical response AFTER the command
#                   (~17ms). Not in frame_age because it hasn't happened yet.
#
# CALIBRATE T_ACTUATION: run balance, watch the printed "age=..ms Tdly=..ms".
# `age` should sit near your measured glass->now (~25-35ms, incl. the 20ms
# capture); set T_ACTUATION (~motor response) so Tdly ~ glass->motion median (~40ms).
#
# GRACEFUL FALLBACK: if the camera sends no SensorTimestamp (parts[8]=0 / old
# publisher) or the OS lacks CLOCK_BOOTTIME, frame_age stays 0 and the horizon is
# just T_ACTUATION (a one-time warning prints) — bump T_ACTUATION in that case.
#
# The predictor uses the pendulum's free near-upright dynamics: θ̈ ≈ (g/l_eff)·θ.
# Safety/giveup still uses MEASURED theta, never the prediction.
PREDICT_ENABLE = True
T_ACTUATION    = 0.017     # s, post-command serial+motor response (CALIBRATE on hw)
T_DELAY_MIN    = 0.015     # s, clamp floor — never predict less than this
T_DELAY_MAX    = 0.090     # s, clamp ceiling — reject a stale/garbage timestamp
GRAV_OVER_LEFF = 31.5       # = ω₀² 实测 (单摆自由摆动, 2026-06-17, FFT 31.2 + 过零点 31.9)
                            # 注意: 这不是 g/几何长(g/0.34=28.85, 那是点质量近似);
                            # 你的摆有配重 → 真身 = m·g·d/I_pivot = ω₀² ≈ 31.5,
                            # 等效 g/l_eff, l_eff≈0.31m. 旧值 28.85 偏低~9%, 已弃.

# ── EKF state estimator (clean ω; replaces camera diff+EMA) ────────────
# Why: camera ω (1-frame diff + EMA) noise ~0.6 rad/s × K_θ̇(307) ≈ ±184 PWM of
# PURE noise → PWM saturates from noise alone (proven in swingup_..._145909.csv).
# The EKF infers ω from the physics, killing most of that noise term.
USE_EKF         = True     # master switch; set False to revert to raw camera ω
EKF_SIGMA_A     = 8.0      # process noise: unknown angular-accel std (rad/s²)
                           #   ↑ trust camera more (faster, noisier)
                           #   ↓ trust model more (smoother, more lag)
EKF_SIGMA_THETA = 0.02     # camera angle-noise std (rad ≈1.1°).
                           #   TODO: measure from a static recording, set exact.
EKF_U_TAU_S     = 0.060    # EMA time constant for the PWM fed to the EKF.
                           # NEVER feed raw last_pwm with b≠0: the delay-
                           # compensation forward(T≈62ms, u) makes this cycle's
                           # command depend on last cycle's with loop gain
                           # |b|·K_THETA_DOT·T = 0.015·1372·0.062 ≈ 1.28 > 1 →
                           # 50 Hz ±255 chatter (measured testbal_ 201405:
                           # corr(pwm, ω_ctrl−ω_ekf)=+0.93, 95 sign flips/100
                           # ticks). A 60ms EMA attenuates the alternating
                           # component ~20x (loop gain →0.07) while sustained
                           # pushes still get compensated.
EKF_INPUT_GAIN  = -0.015   # b = PWM→angular-accel (rad/s² per PWM unit) in the
                           # EKF model ω̇ = ω₀²·sinθ + b·u·cosθ.
                           # Calibrated 2026-07-07 by OFFLINE REPLAY of the 7
                           # testbal_ segments: sweeping b, one-step prediction
                           # RMS error minimized at b≈-0.015 (0.93°→0.66°).
                           # Sign matches physics (+PWM → restoring θ̈<0).
                           # |b| < ideal 0.054 = ω₀²/g · c/m because PWM ≠ force
                           # (back-EMF drag + 63ms motor lag dilute it).


def _deadband(pwm):
    if pwm == 0:
        return 0
    if abs(pwm) < PWM_DEADBAND:
        return PWM_DEADBAND if pwm > 0 else -PWM_DEADBAND
    return pwm


def balance_lqr_loop(comm, cam, t0, log_writer, diag_prefix="balance_diag"):
    """
    Inline LQR balance. Drop-in replacement for swing_up_energy_dc.balance_loop.

    Args:
      comm        : MotorComm  (send_velocity, get_position)
      cam         : CameraSubscriber  (latest_angle -> (theta_rad, omega, pole_up))
      t0          : run start time (for log timestamps, matches swing-up clock)
      log_writer  : csv.writer — same columns as swing-up's log_writer
      diag_prefix : filename prefix of the per-loop diagnostic CSV. Default is
                    the swing-up path ("balance_diag"); test_balance_lqr passes
                    "testbal_diag" so hand-test segments are distinguishable.

    Returns 'fall' | 'boundary' | 'interrupt'.
    """
    print(f"\n  --- BALANCE LQR (K=[{K_THETA:+.1f}, {K_THETA_DOT:+.1f}, "
          f"{K_X:+.2f}, {K_X_DOT:+.1f}]) ---")
    print(f"  state estimator: {'EKF (clean ω, grav_coef=%.1f)' % GRAV_OVER_LEFF if USE_EKF else 'RAW camera ω (EKF off)'}")
    print(f"  Will give up if |θ|>{GIVEUP_ANGLE_DEG}° or |cart|>{MAX_PULSES} pulses.")

    # State estimates
    theta_rad        = 0.0
    omega            = 0.0
    pos              = comm.get_position()
    cart_pos_m       = pos * M_PER_PULSE
    cart_vel_m_s     = 0.0
    cart_poll_count  = 0
    # (t, pos_m) history for the fixed-window velocity estimate. Seeded with
    # the current sample; velocity stays 0 until ~half a window of history
    # exists (true at engage: the cart starts at rest).
    pos_hist         = deque([(time.time(), cart_pos_m)])

    # --- EKF angle/rate estimator (replaces raw camera ω) ------------
    ekf        = PendulumEKF(grav_coef=GRAV_OVER_LEFF,
                             sigma_a=EKF_SIGMA_A,
                             sigma_theta=EKF_SIGMA_THETA,
                             b=EKF_INPUT_GAIN)
    ekf_ready  = False     # lazily reset() to the first valid frame
    last_pwm   = 0         # last PWM sent
    u_ema      = 0.0       # low-passed PWM — what the EKF sees as input u
                           # (see EKF_U_TAU_S: raw last_pwm self-oscillates)
    outer_f    = 0.0       # low-passed outer-loop term (see OUTER_LP_W)
    last_ekf_t = time.time()
    last_sensor_ts_ns = None   # newest camera SensorTimestamp (ns, CLOCK_BOOTTIME)
    frame_age   = 0.0          # now_BOOTTIME - last_sensor_ts_ns; grows on dropouts
    t_delay_dyn = T_ACTUATION  # per-loop horizon; updated once a frame arrives
    warned_no_ts = False       # one-time warning if SensorTimestamp is absent

    # --- diagnostic CSV (balance-only; separate from swing-up's log_writer) ---
    # ONE ROW PER LOOP with the full control chain so we can tell delay from
    # parameter from hardware offline:
    #   raw camera meas  ->  EKF state  ->  predicted (LQR-input) state  ->
    #   pre-clamp force  ->  post-clamp PWM,  plus timing + gating + covariance.
    # Analyze with analysis/analyze_balance_diag.py.
    os.makedirs(BALANCE_DIAG_DIR, exist_ok=True)   # shared dir (config); create if new
    _diag_name = time.strftime(f"{diag_prefix}_%Y%m%d_%H%M%S.csv")
    _diag_path = os.path.join(BALANCE_DIAG_DIR, _diag_name)
    _diag_f    = open(_diag_path, "w", newline="")
    _diag_w    = csv.writer(_diag_f)
    _diag_w.writerow([
        "loop_t_mono", "t", "sensor_ts_ns", "frame_age", "t_delay_dyn", "dt_ekf",
        "theta_meas_deg", "omega_meas", "theta_ekf_deg", "omega_ekf",
        "gated", "n_rejected", "theta_ctrl_deg", "omega_ctrl",
        "P_theta", "P_omega", "u_pwm", "pwm", "saturated", "in_deadband",
        "cart_pos_m", "cart_vel_m_s",
    ])
    _diag_rows = 0
    def _fmt(v, nd=4):        # format an optional float; blank for None/dropout
        return "" if v is None else f"{v:.{nd}f}"
    print(f"  [LQR] diag log -> {_diag_path}")

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
            if now_d - diag_t0 >= 0.008:
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
            result     = cam.latest_angle_ts()
            now_loop   = time.time()
            dt_ekf     = now_loop - last_ekf_t
            last_ekf_t = now_loop

            if result is not None:
                diag_frames += 1
                theta_meas, omega_meas, _pole_up, sensor_ts_ns = result
                theta_meas = -theta_meas
                omega_meas = -omega_meas
                if sensor_ts_ns is not None:
                    last_sensor_ts_ns = sensor_ts_ns      # freshest capture time (BOOTTIME)
                elif not warned_no_ts:
                    print("  [LQR] WARN: camera sent no SensorTimestamp — dynamic "
                          "delay degraded to T_ACTUATION only. Update V3 camera or "
                          "raise T_ACTUATION.")
                    warned_no_ts = True
            else:
                diag_none += 1
                theta_meas, omega_meas = None, None
                if not USE_EKF:
                    continue            # legacy: skip the whole cycle on a dropout

            # ---- state estimate: EKF (clean ω) or raw camera --------
            gated = False        # True if this frame was rejected by the EKF gate
            if USE_EKF:
                if not ekf_ready:
                    if theta_meas is None:
                        continue        # wait for the first real frame to seed
                    ekf.reset(theta_meas, omega_meas)
                    ekf_ready = True
                else:
                    u_ema += (last_pwm - u_ema) * min(1.0, dt_ekf / EKF_U_TAU_S)
                    ekf.predict(dt_ekf, u=u_ema)         # roll model fwd by real dt
                    if theta_meas is not None:
                        gated = not ekf.update(theta_meas)  # fuse; True→frame rejected by gate
                theta_rad, omega = ekf.state()
            else:
                theta_rad, omega = theta_meas, omega_meas
            theta_deg = math.degrees(theta_rad)

            # ---- safety: pendulum fell (prefer MEASURED θ; never trust
            #      the estimate to hide a real fall) -------------------
            theta_safety = (math.degrees(theta_meas)
                            if theta_meas is not None else theta_deg)
            if abs(theta_safety) > GIVEUP_ANGLE_DEG:
                comm.send_velocity(0)
                print(f"\n  [LQR] FELL  θ={theta_safety:+.1f}°  ω={omega:+.2f} rad/s")
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

                # Fixed-window velocity: Δpos over ~VEL_WINDOW_S, appended every
                # loop tick (repeated cache values are fine — the baseline spans
                # ~4 firmware pushes, so push/loop jitter dilutes to <~12%
                # instead of the 3-4x spikes the old consecutive-update Δpos/Δt
                # produced; see header comment at VEL_WINDOW_S).
                t_now = time.time()
                pos_hist.append((t_now, cart_pos_m))
                # keep pos_hist[0] as the newest sample that is >= window old
                while len(pos_hist) >= 2 and t_now - pos_hist[1][0] >= VEL_WINDOW_S:
                    pos_hist.popleft()
                t_old, pos_m_old = pos_hist[0]
                if t_now - t_old >= VEL_WINDOW_S * 0.5:
                    cart_vel_m_s = (cart_pos_m - pos_m_old) / (t_now - t_old)
                # else: not enough history yet (first ~20ms) → keep previous

                # ---- safety: cart boundary -------------------------
                if abs(pos) > MAX_PULSES:
                    comm.send_velocity(0)
                    print(f"\n  [LQR] CART BOUNDARY pos={pos}")
                    return 'boundary'

            # ---- dynamic delay: how far ahead to predict THIS cycle ----
            # frame_age = glass->now from the camera's hardware capture timestamp,
            # read on the SAME clock (CLOCK_BOOTTIME). It naturally GROWS on
            # dropouts (last_sensor_ts_ns stays put), so we predict further when
            # vision goes stale. Falls back to 0 if no timestamp / no BOOTTIME.
            if last_sensor_ts_ns is not None and _CLOCK_BOOTTIME is not None:
                now_bt    = time.clock_gettime_ns(_CLOCK_BOOTTIME)
                frame_age = (now_bt - last_sensor_ts_ns) / 1e9
            t_delay_dyn = frame_age + T_ACTUATION
            t_delay_dyn = max(T_DELAY_MIN, min(T_DELAY_MAX, t_delay_dyn))

            # ---- delay compensation: predict state to "command effect" ----
            if USE_EKF and PREDICT_ENABLE:
                # Roll the EKF state forward by the (dynamic) delay using the same
                # nonlinear model. Input = LOW-PASSED pwm, not last_pwm — raw
                # last_pwm here closes an unstable algebraic loop (EKF_U_TAU_S).
                theta_ctrl, omega_ctrl = ekf.forward(t_delay_dyn, u=u_ema)
            elif PREDICT_ENABLE:
                theta_ctrl = (theta_rad + omega * t_delay_dyn
                              + 0.5 * GRAV_OVER_LEFF * theta_rad * t_delay_dyn**2)
                omega_ctrl = omega + GRAV_OVER_LEFF * theta_rad * t_delay_dyn
            else:
                theta_ctrl = theta_rad
                omega_ctrl = omega

            # ---- LQR (fed the delay-compensated state) -------------
            # Outer term goes through a first-order low-pass: drift control
            # needs no bandwidth at 2.5 Hz, where its lag destabilizes the
            # pole mode (OUTER_LP_W header comment).
            outer_raw = -(K_X * cart_pos_m + K_X_DOT * cart_vel_m_s)
            outer_f  += (outer_raw - outer_f) * min(1.0, dt_ekf * OUTER_LP_W)
            u_pwm = -(K_THETA     * theta_ctrl
                    + K_THETA_DOT * omega_ctrl) + outer_f
            pwm = int(max(-PWM_LIMIT, min(PWM_LIMIT, _deadband(PWM_SIGN * u_pwm))))

            comm.send_velocity(pwm)
            last_pwm = pwm        # feed back as EKF input u next cycle

            # ---- log every cycle, print throttled -----------------
            t = time.time() - t0

            # ---- diagnostic row: full control chain (balance-only) --------
            _diag_w.writerow([
                f"{now_loop:.6f}",
                f"{t:.4f}",
                "" if last_sensor_ts_ns is None else str(last_sensor_ts_ns),
                f"{frame_age:.6f}",
                f"{t_delay_dyn:.6f}",
                f"{dt_ekf:.6f}",
                _fmt(None if theta_meas is None else math.degrees(theta_meas), 3),
                _fmt(omega_meas, 4),
                f"{theta_deg:.3f}",
                f"{omega:.4f}",
                1 if gated else 0,
                ekf.n_rejected,
                f"{math.degrees(theta_ctrl):.3f}",
                f"{omega_ctrl:.4f}",
                f"{ekf.P[0, 0]:.6e}",
                f"{ekf.P[1, 1]:.6e}",
                f"{u_pwm:.2f}",
                pwm,
                1 if abs(pwm) >= PWM_LIMIT else 0,
                1 if abs(u_pwm) < PWM_DEADBAND else 0,
                f"{cart_pos_m:.4f}",
                f"{cart_vel_m_s:.4f}",
            ])
            _diag_rows += 1

            # CSV columns match swing-up's log writer (t, θ, ω, _, pwm, pos).
            # We dump cart_vel_m_s into the E_diff column to keep the schema.
            log_writer.writerow([f"{t:.4f}", f"{theta_deg:.2f}", f"{omega:.4f}",
                                  f"{cart_vel_m_s:.4f}", pwm, pos])
            if t - last_print > 0.2:
                print(f"  {t:5.2f}  [LQR] θ={theta_deg:+6.1f}  ω={omega:+6.2f}  "
                      f"ẋ={cart_vel_m_s:+5.2f}  pwm={pwm:+5}  pos={pos:+5}  "
                      f"age={frame_age*1000:4.0f}ms Tdly={t_delay_dyn*1000:4.0f}ms")
                last_print = t

    except KeyboardInterrupt:
        comm.send_velocity(0)
        print("\n  [LQR] interrupted")
        return 'interrupt'
    finally:
        # runs on every exit (fall / boundary / interrupt / exception) so the
        # diagnostic CSV is always flushed and closed.
        try:
            _diag_f.flush()
            _diag_f.close()
            print(f"  [LQR] diag saved {_diag_rows} rows -> {_diag_path}")
        except Exception:
            pass
