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
# LQR GAINS — from compute_lqr_K.py (measured model: c=0.0195, b=16.0).
# "R2" golden config, rollback 2026-07-08. Do NOT soften K or re-"fix" the
# 50 Hz dither without A/B data — full audit in activity_log.md 2026-07-08.
# ----------------------------------------------------------------------
# 2026-07-22 v2: R2 golden K transplanted to 80T by the c-ratio
# (0.0195/0.04221 = 0.462) -> linear loop identical to R2, but with 2x
# saturation/v_max/deadband margins. The re-solved -6753 set ran the loop
# 1.8x hotter than R2 -> 2.3Hz growing oscillation, 14/14 died <1.5s
# (testbal_20260722_1517xx). User-approved change.
K_THETA      = -3656    # = R2 -7912 x 0.462
K_THETA_DOT  = -634     # = R2 -1372 x 0.462
K_X          = -785     # = R2 -1700 x 0.462
K_X_DOT      = -1373    # = R2 -2971 x 0.462
PWM_SIGN     = +1       # verified: θ>0 = tilt right, +PWM = cart right

# Zero-bias trim v2 (2026-07-09): adapts BETWEEN runs, frozen within a run.
# Purpose: a proportional law parks the pole at whatever angle cancels the
# total bias (camera + friction asymmetry + dither mean) and the cart drifts
# into the 0.29 m/s ceiling. The trim moves the angle target to the true
# no-drift point.
# v1 (in-run integral) FAILED on hardware: with 60-90% PWM saturation the
# outer loop loses the damping an extra in-loop integrator state needs
# (s^2 Routh term ~ 0) -> slow unstable spiral; drift flipped direction and
# railed left 12/13 (testbal_20260709_16xx). Between-run stepping has no
# in-loop dynamics: contraction map, converges in ~4 runs.
TRIM_ADAPT    = 0.03     # rad per (m/s) of a run's mean drift velocity
TRIM_STEP_MAX = 0.004    # rad (~0.23 deg) max trim change per run
TRIM_LIMIT    = 0.035    # rad ~ 2 deg total clamp (authority <= 277 PWM)
_trim_persist = 0.0      # session state; resets on process restart

# Control loop
CTRL_HZ            = 100
DT                 = 1.0 / CTRL_HZ

# Cart velocity = Δpos over a FIXED 40ms window. Do not go back to
# consecutive-update Δpos/Δt: read-time jitter on a 10ms baseline produced
# impossible spikes (0.94 m/s → ±2500 PWM garbage via K_X_DOT).
CART_POLL_EVERY_N  = 1        # read cached position every 100 Hz loop
VEL_WINDOW_S       = 0.040    # differencing baseline; adds ~20ms lag (ok vs τ=178ms)

# ── Delay compensation ────────────────────────────────────────────────
# Vision is ~40-60ms stale; the state fed to the LQR is extrapolated by
#   T_delay_dyn = frame_age (camera SensorTimestamp vs CLOCK_BOOTTIME, per
#                 frame) + T_ACTUATION (serial + motor response, fixed).
# No SensorTimestamp / no BOOTTIME → frame_age=0, warns once; raise T_ACTUATION.
# Safety/giveup always uses MEASURED theta, never the prediction.
PREDICT_ENABLE = True
T_ACTUATION    = 0.017     # s, calibrated: Tdly print ≈ glass->motion median 40ms
T_DELAY_MIN    = 0.015     # s, horizon clamp floor
T_DELAY_MAX    = 0.090     # s, horizon clamp ceiling (rejects garbage timestamps)
GRAV_OVER_LEFF = 31.5      # ω₀² 实测 2026-06-17 = m·g·d/I_pivot（非 g/L 点质量近似）

# ── EKF state estimator (clean ω instead of camera diff+EMA) ──────────
USE_EKF         = True     # False = raw camera ω (noisy, ~0.6 rad/s)
EKF_SIGMA_A     = 8.0      # process noise (rad/s²): ↑跟相机紧, ↓更平滑但滞后
EKF_SIGMA_THETA = 0.02     # camera angle-noise std (rad ≈ 1.1°)
EKF_INPUT_GAIN  = -0.032   # b in ω̇ = ω₀²·sinθ + b·u·cosθ.
                           # 2026-07-22: rescaled for 80T by the c ratio
                           # (0.04221/0.0195 = 2.16x on the 40T replay value
                           # -0.015). TODO: refine by offline replay once the
                           # first 80T testbal_ data exists.
                           # Raw last_pwm as u causes 50Hz dither via forward()
                           # — KNOWN & KEPT (R2 golden config, activity_log
                           # 2026-07-08). Don't low-pass it without A/B.


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
      diag_prefix : per-loop diag CSV prefix ("balance_diag" = swing-up runs;
                    test_balance_lqr passes "testbal_diag")

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
    pos_hist         = deque([(time.time(), cart_pos_m)])  # (t, pos_m) for VEL_WINDOW_S

    # --- EKF angle/rate estimator (replaces raw camera ω) ------------
    ekf        = PendulumEKF(grav_coef=GRAV_OVER_LEFF,
                             sigma_a=EKF_SIGMA_A,
                             sigma_theta=EKF_SIGMA_THETA,
                             b=EKF_INPUT_GAIN)
    ekf_ready  = False     # lazily reset() to the first valid frame
    last_pwm   = 0         # last PWM sent — fed RAW to the EKF (see NOTE above)
    global _trim_persist
    theta_trim = _trim_persist   # frozen for this run; stepped at run end
    drift_v_sum, drift_n = 0.0, 0
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
        "cart_pos_m", "cart_vel_m_s", "theta_trim_deg",
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
                    ekf.predict(dt_ekf, u=last_pwm)      # roll model fwd by real dt
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

                # Fixed-window velocity (see VEL_WINDOW_S header comment).
                t_now = time.time()
                pos_hist.append((t_now, cart_pos_m))
                # keep pos_hist[0] as the newest sample that is >= window old
                while len(pos_hist) >= 2 and t_now - pos_hist[1][0] >= VEL_WINDOW_S:
                    pos_hist.popleft()
                t_old, pos_m_old = pos_hist[0]
                if t_now - t_old >= VEL_WINDOW_S * 0.5:
                    cart_vel_m_s = (cart_pos_m - pos_m_old) / (t_now - t_old)
                # else: <20ms of history yet → keep previous value

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
                # Roll EKF state forward by the delay (raw last_pwm as input —
                # see EKF_INPUT_GAIN note).
                theta_ctrl, omega_ctrl = ekf.forward(t_delay_dyn, u=last_pwm)
            elif PREDICT_ENABLE:
                theta_ctrl = (theta_rad + omega * t_delay_dyn
                              + 0.5 * GRAV_OVER_LEFF * theta_rad * t_delay_dyn**2)
                omega_ctrl = omega + GRAV_OVER_LEFF * theta_rad * t_delay_dyn
            else:
                theta_ctrl = theta_rad
                omega_ctrl = omega

            # ---- collect drift evidence for the between-run trim step ----
            if abs(theta_deg) < 4.0:
                drift_v_sum += cart_vel_m_s
                drift_n     += 1

            # ---- LQR (fed the delay-compensated state) -------------
            u_pwm = -(K_THETA     * (theta_ctrl - theta_trim)
                    + K_THETA_DOT * omega_ctrl
                    + K_X         * cart_pos_m
                    + K_X_DOT     * cart_vel_m_s)
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
                f"{math.degrees(theta_trim):.3f}",
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
        # Between-run trim step: nudge the angle target against this run's
        # mean drift (>=0.5s of near-upright evidence required).
        if drift_n >= 50:
            step = -TRIM_ADAPT * (drift_v_sum / drift_n)
            step = max(-TRIM_STEP_MAX, min(TRIM_STEP_MAX, step))
            _trim_persist = max(-TRIM_LIMIT, min(TRIM_LIMIT, theta_trim + step))
            print(f"  [LQR] trim {math.degrees(theta_trim):+.2f} -> "
                  f"{math.degrees(_trim_persist):+.2f} deg "
                  f"(mean drift v={drift_v_sum/drift_n:+.3f} m/s)")
        # runs on every exit (fall / boundary / interrupt / exception) so the
        # diagnostic CSV is always flushed and closed.
        try:
            _diag_f.flush()
            _diag_f.close()
            print(f"  [LQR] diag saved {_diag_rows} rows -> {_diag_path}")
        except Exception:
            pass
