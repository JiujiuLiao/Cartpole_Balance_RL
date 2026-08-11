"""
swing_up_dc.py
==============
Resonance-based pendulum swing-up for DC motor + IBT-2 + ESP32.

Method: Drive the cart back and forth at the pendulum's natural frequency.
        Each push is timed to add energy to the pendulum (resonance pumping).
        Period stretches gradually as the pendulum swings higher.

Measured parameters (2026-04-09):
    HALF_PERIOD   = 0.550 s   (measured via camera)
    omega0        = 5.71 rad/s
    DAMPING       = 0.0354
    Cart max speed = 0.231 m/s  @ PWM 200  (measured 2026-05-27, new motor + GT2 80T)
                    ≈0.30  m/s  @ PWM 255  (extrapolated, untested for safety)
    Cart max accel = TBD                   (Phase 3 force-curve fit pending)

Usage:
    python swing_up_dc.py
"""

import time
import sys
from pathlib import Path

# Expose src/ (top-level libs).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor_comm import MotorComm
from config import (DEFAULT_SERIAL_PORT, SERIAL_BAUD,
                    PULSES_PER_REV, WHEEL_DIAMETER_M, M_PER_PULSE)


# ── Hardware ──────────────────────────────────────
PORT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERIAL_PORT

# ── Pendulum (measured 2026-04-09) ────────────────
HALF_PERIOD    = 0.550        # s  - natural half-period at small angles
PERIOD_STRETCH = 0.80         # total stretch over all swings (matches swing_up.py)
NUM_SWINGS     = 33           # number of half-swings to attempt

# ── Motor ─────────────────────────────────────────
SWING_PWM      = 255          # PWM during swing-up  (0-255)
RETURN_PWM     = 100          # PWM used to return cart to center

# ── Safety ───────────────────────────────────────
MAX_PULSES     = 1500         # resonance swing-up keeps cart in a tight band
                              # (~34 cm with M_PER_PULSE=0.2286mm). Intentionally
                              # smaller than config.MAX_PULSES (algorithm-specific).


# ── Cart control ─────────────────────────────────
def return_to_center(comm):
    """Drive cart back to encoder position 0."""
    print("  Returning to center...", end="", flush=True)
    timeout = time.time() + 8      # give up after 8 seconds to avoid infinite loop
    while time.time() < timeout:
        pos = comm.get_position()
        if abs(pos) <= 7:          # within 7 pulses (~1 mm) → close enough, stop
            break
        # if cart is left of center (pos < 0) → drive right (+PWM)
        # if cart is right of center (pos > 0) → drive left  (-PWM)
        pwm = RETURN_PWM if pos < 0 else -RETURN_PWM
        comm.send("v", pwm)
        time.sleep(0.02)           # wait 20 ms then re-check position
    comm.send("v", 0)              # stop motor
    time.sleep(0.3)                # let cart settle before reading final position
    final = comm.get_position()
    print(f" done (pos={final})")


def check_boundary(comm):
    """Return True if cart is too close to the track end."""
    pos = comm.get_position()
    if abs(pos) > MAX_PULSES:      # exceeded ±1200 pulses (~16 cm from center)
        print(f"\n  BOUNDARY HIT at pos={pos} — stopping!")
        comm.send("v", 0)          # emergency stop
        return True                # signal to caller: abort swing-up
    return False                   # still safe, continue


# ── Swing-up ─────────────────────────────────────
def swing_up(comm, num_swings, half_period, period_stretch, pwm):
    """
    Resonance swing-up with period stretching.

    Drives cart alternately forward/reverse.
    Half-period increases gradually to track the pendulum's
    longer period at large angles.
    """
    print(f"\n  Starting swing-up: {num_swings} swings, "
          f"T/2={half_period:.3f}s, stretch={period_stretch*100:.0f}%, PWM={pwm}")
    print(f"  {'Swing':<8} {'Direction':<12} {'T/2 (s)':<10} {'Cart pos'}")
    print(f"  {'-'*45}")

    direction = 1   # +1 = forward, -1 = reverse
    top_swing = 0   # which swing reached the top (0 = didn't)

    for i in range(num_swings):
        if check_boundary(comm):
            break

        # Stretch half-period as pendulum swings higher.
        # First WARMUP_SWINGS use fixed period; stretch only applies after that.
        WARMUP_SWINGS = num_swings // 3
        if i < WARMUP_SWINGS:
            current_half_period = half_period
        else:
            progress = (i - WARMUP_SWINGS + 1) / (num_swings - WARMUP_SWINGS)
            current_half_period = half_period * (1.0 + period_stretch * progress)
            #current_half_period = half_period * (1.0 + period_stretch * progress**2)

        # Drive cart
        motor_pwm = pwm * direction
        comm.send("v", motor_pwm)

        dir_str = "forward →" if direction > 0 else "← reverse"
        pos = comm.get_position()
        print(f"  {i+1:<8} {dir_str:<12} {current_half_period:.3f}      {pos}")

        # Wait for this half-period, checking for keyboard interrupt
        t_end = time.time() + current_half_period
        while time.time() < t_end:
            if check_boundary(comm):
                comm.send("v", 0)
                return top_swing
            time.sleep(0.01)       # check boundary every 10 ms during each half-swing

        # Mark top if user presses SPACE (non-blocking not easy in Python,
        # so we just note each swing — user watches and reports)

        direction *= -1   # flip direction

    comm.send("v", 0)
    time.sleep(0.3)                # wait for cart to physically stop before returning
    return top_swing


# ── Main ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("  DC Motor Pendulum Swing-Up")
    print("=" * 60)
    print(f"  Port:         {PORT}")
    print(f"  Half-period:  {HALF_PERIOD} s")
    print(f"  Swings:       {NUM_SWINGS}")
    print(f"  PWM:          {SWING_PWM}")
    print(f"  Stretch:      {PERIOD_STRETCH*100:.0f}%")
    print(f"  Boundary:     ±{MAX_PULSES} pulses "
          f"(~{MAX_PULSES * M_PER_PULSE * 100:.1f} cm)")
    print()

    try:
        comm = MotorComm(PORT, SERIAL_BAUD)
        print(f"  Connected to {PORT}")
    except Exception as e:
        print(f"  Connection failed: {e}")
        print(f"  Try: python swing_up_dc.py /dev/ttyUSB0")
        return

    try:
        comm.send("h", 0)
        time.sleep(0.2)            # wait for encoder reset to complete
        print("  Initialized.\n")

        # ── Phase 1: Quick test (6 swings, no stretch) ──
        print("=" * 60)
        print("  PHASE 1: Quick test (6 swings, no stretch)")
        print("=" * 60)
        print("  Watch the pendulum — it should start swinging.")
        print()
        input("  Place cart at center, pendulum hanging. Press ENTER...")
        print()
        comm.send("h", 0)
        time.sleep(0.2)            # wait for encoder reset to complete

        swing_up(comm, num_swings=6, half_period=HALF_PERIOD,
                 period_stretch=0, pwm=SWING_PWM)

        return_to_center(comm)

        resp = input("\n  Did the pendulum swing noticeably? (y/n): ").strip().lower()
        if not resp.startswith("y"):
            print("  Pendulum not responding. Check:")
            print("    - HALF_PERIOD matches your pendulum")
            print("    - Motor moves the cart correctly")
            print("    - Encoder is counting correctly")
            comm.close()
            return

        # ── Phase 2: Full swing-up ──────────────────────
        print()
        print("=" * 60)
        print(f"  PHASE 2: Full swing-up ({NUM_SWINGS} swings)")
        print("=" * 60)
        print()

        tune_period  = HALF_PERIOD
        tune_stretch = PERIOD_STRETCH
        tune_swings  = NUM_SWINGS
        tune_pwm     = SWING_PWM

        while True:
            return_to_center(comm)
            print()
            print(f"  Settings: T/2={tune_period:.3f}s  stretch={tune_stretch*100:.0f}%"
                  f"  swings={tune_swings}  PWM={tune_pwm}")
            print("  [r] Run   [t] Period  [s] Stretch  [n] Swings  [p] PWM  [q] Quit")
            cmd = input("  > ").strip().lower()

            if cmd == "q":
                break
            elif cmd == "r":
                comm.send("h", 0)
                time.sleep(0.2)    # wait for encoder reset to complete
                input("  Pendulum hanging at center. Press ENTER to start...")
                print()
                swing_up(comm, tune_swings, tune_period, tune_stretch, tune_pwm)
            elif cmd == "t":
                try:
                    tune_period = float(input(f"  Half-period s ({tune_period:.3f}): "))
                except ValueError:
                    pass
            elif cmd == "s":
                try:
                    tune_stretch = float(input(f"  Stretch 0-1 ({tune_stretch:.2f}): "))
                except ValueError:
                    pass
            elif cmd == "n":
                try:
                    tune_swings = int(input(f"  Num swings ({tune_swings}): "))
                except ValueError:
                    pass
            elif cmd == "p":
                try:
                    tune_pwm = int(input(f"  PWM 0-255 ({tune_pwm}): "))
                    tune_pwm = max(0, min(255, tune_pwm))
                except ValueError:
                    pass

    except KeyboardInterrupt:
        print("\n  Interrupted")
    finally:
        comm.send("v", 0)
        comm.close()
        print("  Motor stopped. Done.")


if __name__ == "__main__":
    main()
