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
from balance_lqr import balance_lqr_loop
from balance_pd import balance_pd_loop
from config import (
    DEFAULT_SERIAL_PORT, SERIAL_BAUD, DEFAULT_ZMQ_ADDR,
    GRAVITY, PENDULUM_L_M, M_PER_PULSE, MAX_PULSES,
    PWM_DEADBAND, GIVEUP_ANGLE_DEG,
)

# ── Connections ───────────────────────────────────────────────────────
PORT      = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERIAL_PORT
ZMQ_ADDR  = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ZMQ_ADDR

# ── Algorithm tuning (swing-up specific) ──────────────────────────────
SWING_PWM = 200           # PWM applied when pushing (0-255)
                          # 2026-06-02: 255 → 200, reduce per-cycle energy
                          # injection. Goal: smaller ω at catch → balance
                          # starts closer to stable manifold. Trade-off:
                          # swing-up takes a couple more cycles to reach top.

CATCH_ANGLE_DEG = 10.0    # |θ| < this → caught upright, hand off to balance.
                          # Was 8°; relaxed to 10° on 2026-05-29 (4th round)
                          # because empirically the pendulum's natural energy
                          # equilibrium has it peaking at ±9° with ω≈0
                          # consistently — 8° was unreachable, 10° lets the
                          # peaks trigger catch. Right at the cart's recovery
                          # ceiling (arctan(1.5/9.81)≈8.7°) but with ω≈0
                          # there's no fall acceleration to fight at t=0.
CATCH_OMEGA_MAX = 0.5     # rad/s — also require |ω| below this for a clean catch.
                          # Tightened 1.0 → 0.5: at ω=0.5 the pendulum travels
                          # ~6° in 200ms, leaving balance ~2° of working room
                          # before hitting the 8° recovery ceiling.

CTRL_HZ   = 50            # control loop frequency (Hz)
POS_EVERY = 10            # check cart position every N cycles (saves time)

# 2026-06-05: which balance controller to invoke after a clean catch.
# True  → balance_lqr.balance_lqr_loop  (LQR full-state feedback, gains in balance_lqr.py)
# False → balance_pd.balance_pd_loop    (PD + cart-pos,           gains in config.py)
USE_LQR_BALANCE = True


# ── Energy control ────────────────────────────────────────────────────
def compute_pwm(theta, omega, cart_pos):
    """
    Returns PWM to apply based on energy control.

    Control law (Åström 2000):
      u = (E - E_ref) * omega * cos(theta)
      This drives the cart so the pendulum gains energy toward the top.

    Startup kick when omega ≈ 0 (motor would otherwise do nothing).
    """
    # Mechanical energy excess relative to upright equilibrium (θ=0, ω=0).
    # Below ref (swing-up regime): E_diff < 0  → controller PUMPS energy
    # At ref (upright, ω=0):       E_diff = 0  → no action
    # Above ref (overshoot):       E_diff > 0  → controller DRAINS energy
    # Because u = E_diff·ω·cos(θ) carries E_diff's sign, the controller
    # auto-switches between pumping and draining without a special case.
    # This fixes the old (E - E_ref) formula which was ALWAYS ≥ 0 and so
    # kept pumping even after the pendulum had passed upright with excess ω.
    E_diff = 0.5 * omega**2 + (GRAVITY / PENDULUM_L_M) * (math.cos(theta) - 1)

    # Startup kick: pendulum is nearly stationary at the bottom
    # Give a fixed push to initiate oscillation
    if abs(omega) < 0.15:
        # Push toward center when nearly stopped
        return SWING_PWM if cart_pos <= 0 else -SWING_PWM

    # Negative sign because of this hardware's PWM/encoder direction convention
    # (empirically verified 2026-05-29: with this sign the pendulum swings up
    # and auto-drains at overshoot; the unsigned version drains during swing-up
    # and never gets up).
    u = -E_diff * omega * math.cos(theta)

    if abs(u) < 0.5:       # near inflection point — skip push
        return 0

    base_pwm = SWING_PWM if u > 0 else -SWING_PWM

    # Centering: keep cart from drifting to boundary.
    # 2026-05-29: coefficient bumped 0.12 → 0.40 because the previous run
    # drifted cart from 0 to +3678 in 21s. With 0.40, cart at ±2000 gives
    # ±63 PWM correction (significant), cart at ±500 gives only ±16 (gentle
    # in the middle so it doesn't interfere with the energy pump).
    center_correction = -0.40 * SWING_PWM * (cart_pos / MAX_PULSES)
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
    print(f"  L={PENDULUM_L_M} m   g={GRAVITY}   "
          f"E_diff target=0 (mechanical energy excess vs upright)")
    print(f"  PWM={SWING_PWM}   deadband={PWM_DEADBAND}   "
          f"catch: |θ|<{CATCH_ANGLE_DEG}° AND |ω|<{CATCH_OMEGA_MAX} rad/s")
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
    last_print = 0.0   # 2026-06-02: throttle screen print to ~200ms (same as
                       # balance_loop) — printing every 20ms at 50Hz added
                       # ~5-10ms jitter to the control cycle.

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

            # Check upright catch condition — BOTH small angle AND small ω.
            # The old "or pole_up" branch fired even when pendulum was slamming
            # through upright at high speed, then PWM=0 let it spin past, energy
            # piled up cycle after cycle. Require slow ω so the catch is real.
            if abs(theta) < CATCH_RAD and abs(omega) < CATCH_OMEGA_MAX:
                print(f"\n  *** CAUGHT! θ={math.degrees(theta):.1f}°  ω={omega:.2f} rad/s ***")
                print("  Handing off to inline balance (no process switch).")
                # Run balance in this same process so we don't lose control
                # during a process-start gap (pendulum falls in <100ms).
                balance_fn = balance_lqr_loop if USE_LQR_BALANCE else balance_pd_loop
                bal_result = balance_fn(comm, cam, t0, log_writer)
                if bal_result == 'fall':
                    # Pendulum fell — let outer loop resume swing-up from current state.
                    cam.drain()                # discard stale frames
                    t_next = time.time() + DT  # reset pacing
                    pos_count = POS_EVERY      # force a fresh pos read next cycle
                    print("  Resuming swing-up...\n")
                    continue
                else:
                    # 'boundary' or 'interrupt' — give up entirely
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

            # Log E_diff: mechanical energy excess vs upright equilibrium.
            # Negative during swing-up; crosses 0 at perfect catch; positive
            # if pendulum has overshot with excess speed.
            E_err = (0.5*omega**2
                     + (GRAVITY/PENDULUM_L_M)*(math.cos(theta) - 1))
            t = time.time() - t0
            theta_deg = math.degrees(theta)
            # CSV log every cycle (full data integrity), screen print throttled
            log_writer.writerow([f"{t:.4f}", f"{theta_deg:.2f}", f"{omega:.4f}",
                                  f"{E_err:.4f}", pwm, pos])
            if t - last_print > 0.2:
                print(f"  {t:5.2f}  {theta_deg:7.1f}  {omega:7.3f}  "
                      f"{E_err:8.3f}  {pwm:5}  {pos}")
                last_print = t

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
 