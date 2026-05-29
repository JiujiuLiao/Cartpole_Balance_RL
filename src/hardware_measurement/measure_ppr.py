"""
measure_ppr.py
==============
Measure the true PULSES_PER_REV for your encoder + gearbox.

Two methods:
  A) Motor drive across a long known distance (recommended if gearbox cannot be backdriven)
  B) Turn the output pulley by hand exactly one revolution (requires backdriving)

Usage:
    python measure_ppr.py COM6          # Windows
    python measure_ppr.py /dev/ttyUSB0  # Linux
"""

import time
import sys
import math
import threading
from pathlib import Path

# Allow `from motor_comm import ...` / `from config import ...` when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor_comm import MotorComm
from config import DEFAULT_SERIAL_PORT, SERIAL_BAUD, WHEEL_DIAMETER_M, PULSES_PER_REV

PORT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERIAL_PORT


def print_result(pulses, dist_m):
    ppr = pulses * (math.pi * WHEEL_DIAMETER_M) / dist_m
    m_per_pulse = dist_m / pulses

    print()
    print("=" * 55)
    print("  RESULT")
    print("=" * 55)
    print(f"  Pulses measured:   {pulses}")
    print(f"  Distance measured: {dist_m*100:.1f} cm")
    print(f"  M_PER_PULSE:       {m_per_pulse*1000:.4f} mm/pulse")
    print()
    print(f"  >>> PULSES_PER_REV (measured) = {ppr:.1f}")
    print(f"  >>> Current value in code     = {PULSES_PER_REV}")
    print()
    error_pct = (ppr - PULSES_PER_REV) / PULSES_PER_REV * 100
    if abs(error_pct) < 5:
        print(f"  OK: only {error_pct:+.1f}% off — {PULSES_PER_REV} is correct.")
    else:
        print(f"  WARNING: {error_pct:+.1f}% off from {PULSES_PER_REV}!")
        print(f"  --> Update PULSES_PER_REV in src/config.py to {round(ppr)}.")


def measure_by_motor(comm):
    """
    Method A: Drive cart across a long known distance.
    User marks start, runs motor at low PWM, presses ENTER to stop,
    then measures distance with a ruler.
    Using a long distance reduces ruler measurement error.
    """
    print("\n" + "=" * 55)
    print("  Method A: Motor drive + ruler")
    print("=" * 55)
    print()
    print("  Steps:")
    print("  1. Push cart to one end of the track by hand")
    print("  2. Mark the cart's starting position with tape")
    print("  3. Press ENTER — motor starts at PWM 150")
    print("  4. Press ENTER again when cart is near the other end")
    print("  5. Measure the distance with a ruler (more distance = more accurate)")
    print()
    input("  Ready? Press ENTER to start motor...")

    comm.send("h", 0)       # reset encoder to 0
    time.sleep(0.2)

    print("  Motor running... press ENTER to stop.")
    comm.send("v", 150)     # drive at moderate PWM

    input()                 # wait for user to press ENTER to stop
    comm.send("v", 0)
    time.sleep(0.3)

    pulses = comm.get_position()
    print(f"\n  Encoder count: {pulses} pulses")

    if pulses <= 0:
        print("  ERROR: encoder read 0 — check wiring or firmware.")
        return

    print()
    try:
        dist_cm = float(input("  Measure cart displacement with ruler (cm): "))
    except ValueError:
        print("  Invalid input.")
        return

    if dist_cm <= 0:
        print("  Distance must be > 0.")
        return

    print_result(pulses, dist_cm / 100.0)


def measure_by_hand(comm):
    """
    Method B: Turn the output pulley exactly one full revolution by hand.
    Requires the gearbox to be backdriven (not possible with worm gears).
    Shows live encoder count; user presses ENTER when done.
    """
    print("\n" + "=" * 55)
    print("  Method B: Hand turn — one full revolution")
    print("=" * 55)
    print()
    print("  NOTE: This requires the gearbox to be backdriven.")
    print("  If the output shaft is hard to turn by hand → use Method A.")
    print()
    print("  Steps:")
    print("  1. Make a mark on the output pulley (tape or pen)")
    print("  2. Press ENTER to reset encoder to 0")
    print("  3. Slowly turn the output pulley EXACTLY one full revolution")
    print("     (stop when the mark returns to the same position)")
    print("  4. Press ENTER to capture the count")
    print()
    input("  Press ENTER to reset encoder...")

    comm.send("h", 0)
    time.sleep(0.2)
    print("  Encoder reset to 0. Turn the pulley now.")
    print("  Press ENTER when you complete exactly one full revolution.\n")

    # Use a background thread to keep showing live encoder count
    # while the main thread waits for ENTER
    stop_flag = threading.Event()

    def live_display():
        while not stop_flag.is_set():
            pos = comm.get_position()
            print(f"\r  Live encoder: {pos:>8} pulses", end="", flush=True)
            time.sleep(0.1)

    t = threading.Thread(target=live_display, daemon=True)
    t.start()

    input()                  # blocks until ENTER
    stop_flag.set()
    t.join(timeout=0.5)

    pulses = abs(comm.get_position())   # absolute value — direction doesn't matter
    print(f"\n\n  Captured: {pulses} pulses for one full revolution")

    if pulses == 0:
        print("  ERROR: encoder read 0 — gearbox may not be backdriven, try Method A.")
        return

    # For method B: distance = one full revolution of output shaft = pi * D
    dist_m = math.pi * WHEEL_DIAMETER_M
    print_result(pulses, dist_m)


def main():
    print("=" * 55)
    print("  Measure PULSES_PER_REV")
    print("=" * 55)
    print(f"  Port:           {PORT}")
    print(f"  Wheel diameter: {WHEEL_DIAMETER_M*1000:.0f} mm")
    print(f"  Current code:   PULSES_PER_REV = {PULSES_PER_REV}")
    print()

    try:
        comm = MotorComm(PORT, SERIAL_BAUD)
        print("  Connected to ESP32.")
    except Exception as e:
        print(f"  Connection failed: {e}")
        return

    try:
        ok = comm.send("h", 0)
        if not ok:
            print("  ERROR: ESP32 not responding. Check dc_motor_controller.ino is flashed.")
            return
        print("  ESP32 responding OK.")

        print()
        print("  Choose method:")
        print("  [A] Motor drive + ruler  (recommended — no backdriving needed)")
        print("  [B] Hand turn one revolution  (only if gearbox can be backdriven)")
        print()
        choice = input("  > ").strip().lower()

        if choice == "a":
            measure_by_motor(comm)
        elif choice == "b":
            measure_by_hand(comm)
        else:
            print("  Unknown option.")

    except KeyboardInterrupt:
        print("\n  Interrupted")
    finally:
        comm.send("v", 0)
        comm.close()
        print("\n  Motor stopped. Done.")


if __name__ == "__main__":
    main()
