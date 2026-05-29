"""
swing_up_energy_dc.py
======================
Energy-based pendulum swing-up for DC motor + IBT-2 + ESP32.
Uses camera angle data (ZMQ) for real-time closed-loop control.

Control law (Åström 2000):
    E     = ω² - (2g/L)·cos(θ)        normalized energy
    E_ref = -2g/L                       energy at upright (θ=0, ω=0)
    u     = sign[(E_ref - E)·ω·cos(θ)] push direction to add energy

Angle convention (from camera): θ=0 at top, θ=π at hanging down.

Usage:
    python swing_up_energy_dc.py COM6
    python swing_up_energy_dc.py COM6 tcp://192.168.0.109:9999
"""

import time
import sys
import math
import csv
import os

from motor_comm import MotorComm
from camera_comm import CameraSubscriber
from config import (
    DEFAULT_SERIAL_PORT, SERIAL_BAUD, DEFAULT_ZMQ_ADDR,
    GRAVITY, PENDULUM_L_M, M_PER_PULSE, MAX_PULSES,
    PWM_DEADBAND,
)

# ── Connections ───────────────────────────────────────────────────────
PORT      = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERIAL_PORT
ZMQ_ADDR  = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ZMQ_ADDR

# ── Algorithm tuning (swing-up specific) ──────────────────────────────
SWING_PWM = 255           # PWM applied when pushing (0-255)

CATCH_ANGLE_DEG = 25.0    # |θ| < this → caught upright, hand off to balance.
                          # Lenient for self-test. Tighten to 15° for handoff.

CTRL_HZ   = 50            # control loop frequency (Hz)
POS_EVERY = 10            # check cart position every N cycles (saves time)


# ── Energy control ────────────────────────────────────────────────────
def compute_pwm(theta, omega, cart_pos):
    """
    Returns PWM to apply based on energy control.

    Control law (Åström 2000):
      u = (E - E_ref) * omega * cos(theta)
      This drives the cart so the pendulum gains energy toward the top.

    Startup kick when omega ≈ 0 (motor would otherwise do nothing).
    """
    E     = omega**2 - (2 * GRAVITY / PENDULUM_L_M) * math.cos(theta)
    E_ref = -2 * GRAVITY / PENDULUM_L_M   # upright with zero velocity (most negative)

    # Startup kick: pendulum is nearly stationary at the bottom
    # Give a fixed push to initiate oscillation
    if abs(omega) < 0.15:
        # Push toward center when nearly stopped
        return SWING_PWM if cart_pos <= 0 else -SWING_PWM

    # Energy control: u = (E - E_ref) * omega * cos(theta)
    # (E - E_ref) > 0 always during swing-up (pendulum below top)
    # sign(u) = sign(omega * cos(theta)) → push in direction that adds energy
    u = (E - E_ref) * omega * math.cos(theta)

    if abs(u) < 0.5:       # near inflection point — skip push
        return 0

    base_pwm = SWING_PWM if u > 0 else -SWING_PWM

    # Soft centering: keep cart from drifting to boundary
    center_correction = -0.12 * SWING_PWM * (cart_pos / MAX_PULSES)
    total_pwm = max(-255, min(255, int(base_pwm + center_correction)))
    return total_pwm


# ── Main ──────────────────────────────────────────────────────────────
def main():
    CATCH_RAD = math.radians(CATCH_ANGLE_DEG)
    DT        = 1.0 / CTRL_HZ

    print("=" * 62)
    print("  Energy-Based Swing-Up  (DC Motor + Camera)")
    print("=" * 62)
    print(f"  Port:    {PORT}")
    print(f"  ZMQ:     {ZMQ_ADDR}")
    print(f"  L={PENDULUM_L_M} m   g={GRAVITY}   E_ref={-2*GRAVITY/PENDULUM_L_M:.2f}")
    print(f"  PWM={SWING_PWM}   deadband={PWM_DEADBAND}   catch={CATCH_ANGLE_DEG}°")
    print(f"  Boundary: ±{MAX_PULSES} pulses (~{MAX_PULSES*M_PER_PULSE*100:.1f} cm)")
    print()

    # Connect serial
    try:
        comm = MotorComm(PORT, SERIAL_BAUD, timeout=0.05, echo_timeout=0.08)
        print("  Serial: connected")
    except Exception as e:
        print(f"  Serial error: {e}")
        return

    # Connect ZMQ
    try:
        cam = CameraSubscriber(ZMQ_ADDR)
        print("  ZMQ: connected — waiting for camera data...")
    except Exception as e:
        print(f"  ZMQ error: {e}")
        comm.close()
        return

    # Wait until camera stream is live
    time.sleep(0.5)
    n_drained = cam.drain()
    if n_drained == 0:
        print("  WARNING: no camera data received. Is the camera script running?")
        print("  On Raspberry Pi: python3 hardware_measurement/camerapreview_detect_color_V3.py")
        comm.close()
        cam.close()
        return
    print(f"  Camera: live ({n_drained} messages drained)")

    # Reset encoder
    comm.send("h", 0, echo_timeout=0.3)
    time.sleep(0.1)

    print()
    input("  Pendulum hanging at center. Press ENTER to start swing-up...")
    cam.drain()   # clear buffer right before loop
    print()
    print(f"  {'t':>5}  {'θ(°)':>7}  {'ω':>7}  {'E-Eref':>8}  {'PWM':>5}  pos")
    print(f"  {'-'*55}")

    # Open log file
    log_filename = time.strftime("swingup_%Y%m%d_%H%M%S.csv")
    log_path = os.path.join(os.path.dirname(__file__), log_filename)
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["t_s", "theta_deg", "omega_rad_s", "E_minus_Eref", "pwm", "cart_pos"])
    print(f"  Logging to: {log_filename}")

    t0         = time.time()
    t_next     = t0
    pos        = 0
    pos_count  = 0

    OMEGA_MAX = math.sqrt(4 * GRAVITY / PENDULUM_L_M)  # physical max ω ≈ 10.7 rad/s

    try:
        while True:
            # Pace loop
            now = time.time()
            sleep_t = t_next - now
            if sleep_t > 0:
                time.sleep(sleep_t)
            t_next += DT

            # Get latest angle
            result = cam.latest_angle()
            if result is None:
                continue
            theta, omega, pole_up = result

            # Skip wrap-around glitch frames (|ω| physically impossible)
            if abs(omega) > OMEGA_MAX * 1.2:
                continue

            # Check upright catch condition
            if abs(theta) < CATCH_RAD or pole_up:
                comm.send_velocity(0)
                print(f"\n  *** UPRIGHT! θ={math.degrees(theta):.1f}°  ω={omega:.2f} rad/s ***")
                print("  Switch to balance controller now.")
                break

            # Check cart boundary (every POS_EVERY cycles)
            pos_count += 1
            if pos_count >= POS_EVERY:
                pos = comm.get_position()
                pos_count = 0
                if abs(pos) > MAX_PULSES:
                    comm.send_velocity(0)
                    print(f"\n  BOUNDARY HIT at pos={pos} — stopped.")
                    break

            # Compute and apply PWM
            pwm = compute_pwm(theta, omega, pos)
            comm.send_velocity(pwm)

            # Log to screen and file
            E_err = (omega**2
                     - (2*GRAVITY/PENDULUM_L_M)*math.cos(theta)
                     - (-2*GRAVITY/PENDULUM_L_M))
            t = time.time() - t0
            theta_deg = math.degrees(theta)
            print(f"  {t:5.2f}  {theta_deg:7.1f}  {omega:7.3f}  "
                  f"{E_err:8.3f}  {pwm:5}  {pos}")
            log_writer.writerow([f"{t:.4f}", f"{theta_deg:.2f}", f"{omega:.4f}",
                                  f"{E_err:.4f}", pwm, pos])

    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        comm.send_velocity(0)
        time.sleep(0.1)
        comm.close()
        cam.close()
        log_file.close()
        print(f"  Motor stopped. Log saved to: {log_path}")


if __name__ == "__main__":
    main()
