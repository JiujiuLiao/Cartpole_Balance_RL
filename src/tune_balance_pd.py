"""
tune_balance_pd.py — Standalone PD tuner for DC inverted pendulum balance.
========================================================================
Use this to find good Kp/Kd/Kx by hand, then copy the winners into
`config.py` so `balance_pd.py` (called by swing-up) picks them up.

Pairs with `balance_pd.py` (library) — same control law, different entry.


TWO-PROCESS architecture, both ON the Raspberry Pi:
    Terminal 1:  python3 vision/camerapreview_detect_color_V3.py
                 -> vision (verified correct), PUBs theta/omega/pole_up on :9999
    Terminal 2:  python3 tune_balance_pd.py        (this file)
                 -> SUBs the angle stream -> PID -> serial -> ESP32

This file no longer does any image processing. The angle comes from V3 over
ZMQ (localhost). Start V3 FIRST, confirm its preview angle is correct, then run
this controller.

Control law (matches Varghese 2017 "2-PID with Ki=0" structure):
    pwm = ANGLE_SIGN * (-Kp * theta_deg)     # angle proportional
        + OMEGA_SIGN * ( Kd * omega )        # angle derivative (damping)
        -  CART_SIGN * ( Kx * cart_pos )     # cart-position regulation
Then: deadband compensation -> PWM saturation -> serial.

========================================================================
*** LAB CHECKLIST — verify before first motor-on ***
========================================================================
[x] PULSES_PER_REV    : 700  (measured 2026-05-27); MAX_PULSES depends on this
[x] PWM_DEADBAND      : 22   (measured 2026-05-27)
[x] Vision angle      : verified correct in V3 preview (0=up, +-180=hanging)
[ ] ANGLE_SIGN        : diagnostic mode must show
                        "pwm_raw POSITIVE when pendulum tilts right"
                        if not, flip ANGLE_SIGN to -1

Tuning order (after diagnostic passes):
    1. Kd=0, Kx=0. Increase Kp until cart actively chases pendulum.
    2. Increase Kd to damp oscillation.
    3. Add small Kx (0.01-0.05) to prevent slow drift.
"""

import time
import sys
import csv
import os
import threading
import math
from datetime import datetime

from motor_comm import MotorComm
from camera_comm import CameraSubscriber
from config import (
    DEFAULT_SERIAL_PORT, SERIAL_BAUD, DEFAULT_ZMQ_ADDR,
    PENDULUM_L_M, MAX_PULSES, BALANCE_LOGS_DIR,
    PWM_LIMIT, PWM_DEADBAND, GIVEUP_ANGLE_DEG,
)


# ------------------------------------------------------------------------
# Connections
# ------------------------------------------------------------------------
SERIAL_PORT    = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERIAL_PORT
ZMQ_ADDR       = DEFAULT_ZMQ_ADDR     # V3 publisher on the same Pi
ZMQ_TIMEOUT_MS = 200


# ------------------------------------------------------------------------
# PID gains (algorithm-specific — tune in lab, NOT in config.py)
# ------------------------------------------------------------------------
Kp = 80.0    # angle proportional (input: degrees)
Kd =  5.0    # angle derivative   (input: rad/s)   — add after Kp works
Kx =  0.0    # cart position      (input: pulses)  — add last

ANGLE_SIGN = +1
OMEGA_SIGN = +1
CART_SIGN  = +1


# ------------------------------------------------------------------------
# This-controller-specific safety (global GIVEUP/MAX_PULSES come from config)
# ------------------------------------------------------------------------
CATCH_ANGLE_DEG = 15.0     # motor enables only if |theta| <= this for CATCH_HOLD_S
                           # (tighter than GIVEUP_ANGLE_DEG; PID-specific handoff)
CATCH_HOLD_S    = 0.5
CAM_TIMEOUT_S   = 0.5


# ------------------------------------------------------------------------
# Return to center (algorithm-specific recovery)
# ------------------------------------------------------------------------
RETURN_PWM_MAX   = 120
RETURN_PWM_FLOOR = max(80, PWM_DEADBAND + 5)
RETURN_SLOW_ZONE = int(MAX_PULSES * 0.30)   # last 30% of half-track: slow down


# ------------------------------------------------------------------------
# Control loop
# ------------------------------------------------------------------------
CTRL_HZ         = 100
DT              = 1.0 / CTRL_HZ


# ========================================================================
# Deadband compensation
# ========================================================================

def deadband_compensate(pwm, deadband=PWM_DEADBAND):
    """
    Push small commanded PWM out of the static-friction dead zone.
      pwm == 0                -> 0
      0 < |pwm| < deadband    -> +-deadband (sign-preserving)
      |pwm| >= deadband       -> unchanged
    """
    if pwm == 0:
        return 0
    if abs(pwm) < deadband:
        return deadband if pwm > 0 else -deadband
    return pwm


# ========================================================================
# Return to center
# ========================================================================

def return_to_center(comm, timeout=10.0):
    time.sleep(0.2)
    pos = comm.get_position()
    if abs(pos) <= 20:
        print("  Cart already near center. Ready.\n")
        return
    print(f"  Cart at {pos} pulses - returning to center...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        pos = comm.get_position()
        if abs(pos) <= 10:
            break
        dist = abs(pos)
        if dist >= RETURN_SLOW_ZONE:
            pwm = RETURN_PWM_MAX
        else:
            ratio = dist / RETURN_SLOW_ZONE
            pwm   = int(RETURN_PWM_FLOOR + ratio * (RETURN_PWM_MAX - RETURN_PWM_FLOOR))
            pwm   = max(RETURN_PWM_FLOOR, pwm)
        comm.send_velocity(pwm if pos < 0 else -pwm)
        time.sleep(0.02)
    comm.stop()
    time.sleep(0.5)
    print(f"  Cart at center (pos={comm.get_position()}). Ready.\n")


# ========================================================================
# Main
# ========================================================================

def main():
    print("=" * 64)
    print("  Inverted Pendulum - Balance PD (sim-to-real, 2026-05-15)")
    print("=" * 64)
    print(f"  Serial         : {SERIAL_PORT}")
    print(f"  Gains          : Kp={Kp}  Kd={Kd}  Kx={Kx}")
    print(f"  Boundary       : ±{MAX_PULSES} pulses (config.USABLE_HALF_M)")
    print(f"  Catch / giveup : ±{CATCH_ANGLE_DEG}° / ±{GIVEUP_ANGLE_DEG}°")
    print(f"  Deadband       : {PWM_DEADBAND}")
    print(f"  Vision (ZMQ)   : {ZMQ_ADDR}")
    print()

    # ---- Vision: subscribe to V3 ----------------------------------------
    cam = CameraSubscriber(ZMQ_ADDR, recv_timeout_ms=ZMQ_TIMEOUT_MS, conflate=True)
    print(f"  Subscribed to V3 vision at {ZMQ_ADDR}")
    print("  >>> Make sure camerapreview_detect_color_V3.py is running first! <<<")

    # ---- Serial ----------------------------------------------------------
    try:
        comm = MotorComm(SERIAL_PORT, SERIAL_BAUD, timeout=0.05, echo_timeout=0.04)
        print("  ESP32 connected.")
        return_to_center(comm)
    except Exception as e:
        print(f"  Serial failed: {e}")
        cam.close()
        return

    # ---- Diagnostic banner -----------------------------------------------
    print("=" * 64)
    print("  DIAGNOSTIC MODE - motor OFF")
    print("=" * 64)
    print("  Hold pendulum near upright; theta should track it (V3 must be running).")
    print(f"  Motor will only run when |theta| <= {CATCH_ANGLE_DEG} deg "
          f"for {CATCH_HOLD_S}s.")
    print("  Verify in diagnostic:")
    print("    - theta around 0 when upright, ~+-180 when hanging")
    print("    - omega > 0 when falling right")
    print("    - pwm_raw POSITIVE when tilted right  (else flip ANGLE_SIGN)")
    print()
    print("  Press ENTER to ARM motor (it still won't run until catch met).")
    print()
    print(f"  {'theta':>8} {'err':>7} {'omega':>8} {'cart':>7} "
          f"{'pwm_raw':>8} {'pwm':>6} {'pu':>4}  status")
    print(f"  {'-' * 78}")

    # ---- CSV log (one file per run) -------------------------------------
    os.makedirs(BALANCE_LOGS_DIR, exist_ok=True)
    log_name = f"balance_log_{datetime.now():%Y%m%d_%H%M%S}_Kp{Kp:g}_Kd{Kd:g}_Kx{Kx:g}.csv"
    log_path = os.path.join(BALANCE_LOGS_DIR, log_name)
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["t", "theta_deg", "omega", "cart_pos",
                         "pwm_raw", "pwm", "pole_up", "status"])
    t_start = time.time()
    print(f"  Logging to {log_path}\n")

    motor_armed   = False
    motor_running = False
    enable_flag   = [False]

    def wait_enter():
        input()
        enable_flag[0] = True
    threading.Thread(target=wait_enter, daemon=True).start()

    theta_deg    = 180.0
    omega        = 0.0
    pole_up      = False
    cart_pos     = 0
    pos_counter  = 0
    last_cam_t   = time.time()
    catch_start  = None
    pwm_raw      = 0.0
    pwm          = 0
    valid        = False

    try:
        while True:
            t0 = time.time()

            # ---- Vision (from V3 over ZMQ) -----------------------------
            # latest_angle() drains the buffer -> freshest (theta, omega, pole_up).
            # theta is already wrapped to (-pi, pi]; omega already LPF'd by V3.
            result = cam.latest_angle()
            if result is not None:
                theta_rad, omega, pole_up = result
                theta_deg  = math.degrees(theta_rad)
                last_cam_t = time.time()
                valid      = True
            else:
                valid = False
            # else: hold previous theta_deg / omega; cam_ok timeout handles dropout

            # ---- Cart position (20Hz) --------------------------------
            pos_counter += 1
            if pos_counter >= 5:
                cart_pos    = comm.get_position()
                pos_counter = 0

            # ---- PD + cart-pos -----------------------------------------
            err = theta_deg
            pwm_raw = (ANGLE_SIGN * (-Kp * err)
                     + OMEGA_SIGN * ( Kd * omega)
                     -  CART_SIGN * ( Kx * cart_pos))
            pwm_db  = deadband_compensate(pwm_raw)
            pwm     = int(max(-PWM_LIMIT, min(PWM_LIMIT, pwm_db)))

            # ---- Safety state -----------------------------------------
            cam_ok       = (time.time() - last_cam_t) <= CAM_TIMEOUT_S
            angle_in_run = abs(err) <= GIVEUP_ANGLE_DEG
            cart_in_run  = abs(cart_pos) <= MAX_PULSES
            in_catch     = cam_ok and abs(err) <= CATCH_ANGLE_DEG

            if in_catch:
                if catch_start is None:
                    catch_start = time.time()
                catch_hold_ok = (time.time() - catch_start) >= CATCH_HOLD_S
            else:
                catch_start   = None
                catch_hold_ok = False

            # ---- Arm motor on ENTER ----------------------------------
            if enable_flag[0] and not motor_armed:
                motor_armed = True
                print("\n  Motor ARMED. Will run when pendulum is held near upright.\n")

            # ---- State machine ---------------------------------------
            if not cam_ok:
                comm.stop()
                motor_running = False
            elif motor_armed and not motor_running and catch_hold_ok:
                motor_running = True
                print("\n  *** Motor RUNNING (catch condition met) ***\n")
            elif motor_running and not (angle_in_run and cart_in_run):
                comm.stop()
                motor_running = False
                if not cart_in_run:
                    print(f"\n  Cart boundary ({cart_pos} pulses). Returning to center.")
                    return_to_center(comm)
                    motor_armed    = False
                    enable_flag[0] = False
                    threading.Thread(target=wait_enter, daemon=True).start()
                    print("  Press ENTER to re-ARM.\n")
                else:
                    print(f"\n  Angle out of range ({err:+.1f} deg). Motor stopped.\n")
            elif motor_running:
                comm.send_velocity(pwm)
            else:
                comm.stop()

            # ---- Display ----------------------------------------------
            if not cam_ok:
                status = "CAM LOST"
            elif motor_running:
                status = "RUNNING"
            elif motor_armed:
                status = f"ARMED catch={int(in_catch)} hold={int(catch_hold_ok)}"
            else:
                status = "IDLE"
            print(f"\r  {theta_deg:8.1f} {err:7.1f} {omega:8.3f} {cart_pos:7} "
                  f"{pwm_raw:8.1f} {pwm:6d} {int(pole_up):>4}  {status:<32}",
                  end="", flush=True)

            # ---- CSV row ----------------------------------------------
            log_writer.writerow([f"{t0 - t_start:.4f}", f"{theta_deg:.2f}",
                                 f"{omega:.4f}", cart_pos, f"{pwm_raw:.1f}",
                                 pwm, int(pole_up), status])

            # ---- Loop timing -----------------------------------------
            elapsed = time.time() - t0
            if elapsed < DT:
                time.sleep(DT - elapsed)

    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
    finally:
        comm.stop()
        comm.close()
        cam.close()
        log_file.close()
        print(f"  Motor off. Log saved: {log_path}")


if __name__ == "__main__":
    main()
