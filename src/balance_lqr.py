"""
balance_lqr.py - DC inverted pendulum, LQR full-state-feedback controller.
========================================================================
Runs ON the Raspberry Pi (single process):
    picamera2 -> inline v3 vision (yellow->blue dual-color vector)
    serial    -> ESP32 (dc_motor_controller.ino)

Control law (LQR, gains K precomputed by compute_lqr_K.py):
    state = [theta_rad, theta_dot_rad_s, x_cart_m, x_cart_dot_m_s]
    u     = -K @ state
    pwm   = u * LQR_OUTPUT_SCALE   (saturated to +-255)

========================================================================
*** LAB CHECKLIST 2026-05-15 ***
========================================================================
[ ] Same hardware params as balance_pid.py (now all from config.py):
    PULSES_PER_REV, PWM_DEADBAND, AXIS_X/Y, HSV thresholds, ANGLE_SIGN
[ ] Cart mass M_c (kg)                       — for compute_lqr_K.py
[ ] Pendulum mass m_p (kg)                   — for compute_lqr_K.py
[ ] Pivot-to-CoM distance l_p (m)            — for compute_lqr_K.py (NOT 0.34!)
[ ] Run compute_lqr_K.py on PC -> paste K_*  values below
[ ] LQR_OUTPUT_SCALE                         — start 1.0, tune empirically
                                               (final pwm magnitude should
                                                land in [-200, 200] in steady
                                                small-perturbation regime)
"""

import time
import sys
import threading
import math
import cv2
import numpy as np
from picamera2 import Picamera2

from motor_comm import MotorComm
from config import (
    DEFAULT_SERIAL_PORT, SERIAL_BAUD,
    PENDULUM_L_M, MAX_PULSES, M_PER_PULSE,
    PWM_LIMIT, PWM_DEADBAND, GIVEUP_ANGLE_DEG,
)


# ------------------------------------------------------------------------
# LQR GAINS - !!! PASTE FROM compute_lqr_K.py OUTPUT !!!
# ------------------------------------------------------------------------
K_THETA      = -50.0     # !!! PLACEHOLDER — replace with computed value
K_THETA_DOT  =  -8.0     # !!! PLACEHOLDER
K_X          =  -5.0     # !!! PLACEHOLDER
K_X_DOT      =  -3.0     # !!! PLACEHOLDER

LQR_OUTPUT_SCALE = 1.0   # !!! PLACEHOLDER — tune so pwm stays inside +-200


# ------------------------------------------------------------------------
# Connections
# ------------------------------------------------------------------------
SERIAL_PORT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERIAL_PORT


# ------------------------------------------------------------------------
# Camera (same as balance_pid.py)
# ------------------------------------------------------------------------
CAM_RAW_W, CAM_RAW_H = 640, 480
CAM_SCALE  = 0.25
CAM_W, CAM_H = int(CAM_RAW_W * CAM_SCALE), int(CAM_RAW_H * CAM_SCALE)
TARGET_FPS = 60
FRAME_US   = int(round(1_000_000 / TARGET_FPS))

AXIS_X, AXIS_Y = 70, 120     # !!! PLACEHOLDER

GELB_MIN = np.array([18,  80,  80], np.uint8)
GELB_MAX = np.array([35, 255, 255], np.uint8)
BLAU_MIN = np.array([95, 110,  60], np.uint8)
BLAU_MAX = np.array([130, 255, 255], np.uint8)
MIN_AREA_PX = 25
MORPH_K     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


# ------------------------------------------------------------------------
# Sign flips (verify in diagnostic mode)
# ------------------------------------------------------------------------
ANGLE_SIGN = +1     # flip if cart moves opposite to expected direction
                    # (LQR K is signed correctly by construction, but a wrong
                    #  hardware/camera direction still needs flipping here)


# ------------------------------------------------------------------------
# This-controller-specific safety (PWM_LIMIT / PWM_DEADBAND / GIVEUP from config)
# ------------------------------------------------------------------------
CATCH_ANGLE_DEG  = 5.0       # LQR catch is much tighter than PID (linearization
                             # is only valid near upright)
CATCH_HOLD_S     = 0.5
CAM_TIMEOUT_S    = 0.5


# ------------------------------------------------------------------------
# Return to center (same as balance_pid)
# ------------------------------------------------------------------------
RETURN_PWM_MAX   = 120
RETURN_PWM_FLOOR = max(80, PWM_DEADBAND + 5)
RETURN_SLOW_ZONE = int(MAX_PULSES * 0.30)


# ------------------------------------------------------------------------
# Control loop
# ------------------------------------------------------------------------
CTRL_HZ            = 100
DT                 = 1.0 / CTRL_HZ
OMEGA_LPF_ALPHA    = 0.35
CART_VEL_LPF_ALPHA = 0.30     # EMA on x_dot (LQR needs this)
CART_POLL_EVERY_N  = 2        # 50 Hz cart position polling (vs 10 Hz in PD)


# ========================================================================
# v3 VISION (identical to balance_pid.py — kept inline for parallelism)
# ========================================================================

def _detect_centroid(hsv, lo, hi):
    mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  MORPH_K)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_K)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA_PX:
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


def _compute_angle(yellow, blue):
    if yellow is not None and blue is not None:
        dx = blue[0] - yellow[0]
        dy = yellow[1] - blue[1]
        return math.atan2(dx, dy), "dual"
    if blue is not None:
        return math.atan2(blue[0] - AXIS_X, AXIS_Y - blue[1]), "axis_blue"
    if yellow is not None:
        return math.atan2(AXIS_X - yellow[0], yellow[1] - AXIS_Y), "axis_yellow"
    return None, "none"


def _unwrap(prev, current):
    if prev is None:
        return current
    d = current - prev
    if d >  math.pi:
        return current - 2 * math.pi
    if d < -math.pi:
        return current + 2 * math.pi
    return current


# ========================================================================
# Deadband
# ========================================================================

def deadband_compensate(pwm, deadband=PWM_DEADBAND):
    if pwm == 0:
        return 0
    if abs(pwm) < deadband:
        return deadband if pwm > 0 else -deadband
    return pwm


# ========================================================================
# Return-to-center (same as balance_pid)
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
    print("  Inverted Pendulum - Balance LQR (sim-to-real, 2026-05-15)")
    print("=" * 64)
    print(f"  Serial         : {SERIAL_PORT}")
    print(f"  K              : [{K_THETA:+.3f}, {K_THETA_DOT:+.3f}, "
          f"{K_X:+.3f}, {K_X_DOT:+.3f}]")
    print(f"  Output scale   : {LQR_OUTPUT_SCALE}")
    print(f"  Boundary       : ±{MAX_PULSES} pulses (config.USABLE_HALF_M)")
    print(f"  Catch / giveup : +-{CATCH_ANGLE_DEG} deg / +-{GIVEUP_ANGLE_DEG} deg")
    print(f"  Deadband       : {PWM_DEADBAND}")
    print()

    # ---- Camera ---------------------------------------------------------
    picam = Picamera2()
    cfg = picam.create_video_configuration(
        main={"format": "RGB888", "size": (CAM_RAW_W, CAM_RAW_H)},
        controls={
            "FrameDurationLimits": (FRAME_US, FRAME_US),
            "ExposureTime": 5000,
            "AnalogueGain": 4.0,
            "AwbEnable": False,
            "AeEnable": False,
        },
        buffer_count=4,
    )
    picam.configure(cfg)
    picam.start()
    time.sleep(0.6)
    try:
        picam.set_controls({"AfMode": 0, "LensPosition": 4.5})
    except Exception:
        pass
    print("  Camera started.")

    # ---- Serial ---------------------------------------------------------
    try:
        comm = MotorComm(SERIAL_PORT, SERIAL_BAUD, timeout=0.05, echo_timeout=0.04)
        print("  ESP32 connected.")
        return_to_center(comm)
    except Exception as e:
        print(f"  Serial failed: {e}")
        picam.stop()
        return

    # ---- Diagnostic banner ---------------------------------------------
    print("=" * 64)
    print("  DIAGNOSTIC MODE - motor OFF")
    print("=" * 64)
    print("  Hold pendulum near upright (mode = 'dual').")
    print(f"  Motor will only run when |theta| <= {CATCH_ANGLE_DEG} deg "
          f"for {CATCH_HOLD_S} s.")
    print("  Verify:")
    print("    - theta_rad ~ 0 when upright, ~ +-3.14 when hanging")
    print("    - omega > 0 when falling right")
    print("    - x_cart_m matches physical cart position (try push left/right)")
    print("    - x_dot follows cart motion direction")
    print("    - pwm_raw POSITIVE when tilted right  (else flip ANGLE_SIGN)")
    print()
    print("  Press ENTER to ARM motor.")
    print()
    print(f"  {'theta':>8} {'th_dot':>8} {'x_m':>8} {'x_dot':>8} "
          f"{'pwm':>6} {'mode':>10}  status")
    print(f"  {'-' * 80}")

    motor_armed   = False
    motor_running = False
    enable_flag   = [False]

    def wait_enter():
        input()
        enable_flag[0] = True
    threading.Thread(target=wait_enter, daemon=True).start()

    angle_prev   = None
    theta_rad    = math.pi
    omega        = 0.0
    cart_pos     = 0
    cart_pos_m   = 0.0
    cart_pos_m_prev = None
    cart_vel_m_s = 0.0
    cart_poll_counter = 0
    last_cam_t   = time.time()
    last_cart_t  = time.time()
    catch_start  = None
    mode         = "init"
    pwm          = 0
    valid        = False

    try:
        while True:
            t0 = time.time()

            # ---- Vision ---------------------------------------------
            frame = picam.capture_array()
            frame = cv2.resize(frame, (CAM_W, CAM_H))
            hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            yc = _detect_centroid(hsv, GELB_MIN, GELB_MAX)
            bc = _detect_centroid(hsv, BLAU_MIN, BLAU_MAX)
            angle_raw, mode = _compute_angle(yc, bc)
            valid = angle_raw is not None

            if valid:
                angle_unwrap = _unwrap(angle_prev, angle_raw)
                now = time.time()
                if angle_prev is not None:
                    dt_v = max(now - last_cam_t, 1e-6)
                    omega_inst = (angle_unwrap - angle_prev) / dt_v
                    omega = (OMEGA_LPF_ALPHA * omega_inst
                             + (1.0 - OMEGA_LPF_ALPHA) * omega)
                angle_prev = angle_unwrap
                last_cam_t = now
                # Wrap to (-pi, pi]
                theta_rad = math.atan2(math.sin(angle_unwrap),
                                       math.cos(angle_unwrap))

            # ---- Cart position + velocity (50 Hz) -------------------
            cart_poll_counter += 1
            if cart_poll_counter >= CART_POLL_EVERY_N:
                cart_poll_counter = 0
                cart_pos    = comm.get_position()
                cart_pos_m  = cart_pos * M_PER_PULSE
                t_cart      = time.time()
                if cart_pos_m_prev is not None:
                    dt_c       = max(t_cart - last_cart_t, 1e-6)
                    v_inst     = (cart_pos_m - cart_pos_m_prev) / dt_c
                    cart_vel_m_s = (CART_VEL_LPF_ALPHA * v_inst
                                    + (1.0 - CART_VEL_LPF_ALPHA) * cart_vel_m_s)
                cart_pos_m_prev = cart_pos_m
                last_cart_t     = t_cart

            # ---- LQR --------------------------------------------------
            state = np.array([theta_rad, omega, cart_pos_m, cart_vel_m_s])
            K     = np.array([K_THETA, K_THETA_DOT, K_X, K_X_DOT])
            u     = -float(K @ state)            # natural units (~Newton)
            pwm_lqr = ANGLE_SIGN * u * LQR_OUTPUT_SCALE
            pwm_db  = deadband_compensate(pwm_lqr)
            pwm     = int(max(-PWM_LIMIT, min(PWM_LIMIT, pwm_db)))

            # ---- Safety state -----------------------------------------
            theta_deg    = math.degrees(theta_rad)
            cam_ok       = (time.time() - last_cam_t) <= CAM_TIMEOUT_S
            angle_in_run = abs(theta_deg) <= GIVEUP_ANGLE_DEG
            cart_in_run  = abs(cart_pos)  <= MAX_PULSES
            in_catch     = (abs(theta_deg) <= CATCH_ANGLE_DEG
                            and valid and mode == "dual")

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
                print("\n  Motor ARMED. Will run when catch condition met.\n")

            # ---- State machine ---------------------------------------
            if not cam_ok:
                comm.stop()
                motor_running = False
            elif motor_armed and not motor_running and catch_hold_ok:
                motor_running = True
                print("\n  *** Motor RUNNING (LQR active) ***\n")
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
                    print(f"\n  Angle out of range ({theta_deg:+.1f} deg). Motor stopped.\n")
            elif motor_running:
                comm.send_velocity(pwm)
            else:
                comm.stop()

            # ---- Display ---------------------------------------------
            if not cam_ok:
                status = "CAM LOST"
            elif motor_running:
                status = "RUNNING"
            elif motor_armed:
                status = f"ARMED catch={int(in_catch)} hold={int(catch_hold_ok)}"
            else:
                status = "IDLE"
            print(f"\r  {theta_rad:8.3f} {omega:8.3f} {cart_pos_m:8.3f} "
                  f"{cart_vel_m_s:8.3f} {pwm:6d} {mode:>10}  {status:<28}",
                  end="", flush=True)

            # ---- Loop timing ----------------------------------------
            elapsed = time.time() - t0
            if elapsed < DT:
                time.sleep(DT - elapsed)

    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
    finally:
        comm.stop()
        comm.close()
        picam.stop()
        print("  Motor off. Done.")


if __name__ == "__main__":
    main()
