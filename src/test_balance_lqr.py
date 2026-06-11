"""
test_balance_lqr.py - Isolated balance test (NO swing-up).
===========================================================
Lets you test/tune the LQR balance controller from a clean, repeatable start,
without the variability of swing-up catches.

How it works:
  1. Connect, then you put the CART at track center and press ENTER -> encoder
     is zeroed there (so 0 = center, boundary check is symmetric).
  2. The script enters an ARMED state: motor OFF, live theta/omega readout.
  3. You hold the pendulum up and steady it near 0 deg. The controller engages
     ONLY when |theta| < ENGAGE_ANGLE_DEG AND |omega| < ENGAGE_OMEGA, held
     steady for ENGAGE_HOLD_S — i.e. only once it's genuinely upright & still.
     It will NOT start while you're still moving it toward 0.
  4. On engage it prints "BALANCING — let go", runs balance_lqr_loop, and logs
     to balance_<timestamp>.csv (same columns as swing-up, so plot_swingup.py
     works on it).
  5. After it falls / hits boundary, you can re-center and arm again.
  Ctrl-C quits.

Usage:
    python test_balance_lqr.py [PORT] [ZMQ_ADDR]
"""

import sys
import time
import csv
import math
import os

from motor_comm import MotorComm
from camera_comm import CameraSubscriber
from balance_lqr import balance_lqr_loop
from config import DEFAULT_SERIAL_PORT, SERIAL_BAUD, DEFAULT_ZMQ_ADDR

# ── Engage gate (the "is it genuinely upright & steady?" condition) ───
ENGAGE_ANGLE_DEG = 3.0     # must be within this of upright
ENGAGE_OMEGA     = 0.5     # rad/s — must be slower than this
ENGAGE_HOLD_S    = 0.05    # must satisfy both for this long (anti false-trigger)

PORT     = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERIAL_PORT
ZMQ_ADDR = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ZMQ_ADDR

LOG_COLS = ["t_s", "theta_deg", "omega_rad_s", "E_minus_Eref", "pwm", "cart_pos"]


def wait_until_upright(cam):
    """Block (motor off) until the pendulum is steady near upright. Returns True,
    or False if Ctrl-C / no camera."""
    print(f"\n  ARMED — hold the pendulum UP and steady near 0°.")
    print(f"  Engages when |θ|<{ENGAGE_ANGLE_DEG}° AND |ω|<{ENGAGE_OMEGA} rad/s "
          f"(steady {ENGAGE_HOLD_S*1000:.0f}ms). Motor is OFF until then.")
    gate_since = None
    last_disp  = 0.0
    while True:
        res = cam.latest_angle()
        if res is None:
            time.sleep(0.005)
            continue
        theta_rad, omega, _ = res
        theta_deg = math.degrees(theta_rad)

        if time.time() - last_disp > 0.1:
            ready = abs(theta_deg) < ENGAGE_ANGLE_DEG and abs(omega) < ENGAGE_OMEGA
            sys.stdout.write(f"\r  θ={theta_deg:+7.1f}°  ω={omega:+5.2f} rad/s   "
                             f"{'[READY -> engaging]' if ready else '[hold near 0°]  '}   ")
            sys.stdout.flush()
            last_disp = time.time()

        if abs(theta_deg) < ENGAGE_ANGLE_DEG and abs(omega) < ENGAGE_OMEGA:
            if gate_since is None:
                gate_since = time.time()
            elif time.time() - gate_since >= ENGAGE_HOLD_S:
                return True
        else:
            gate_since = None
        time.sleep(0.005)


def main():
    print("=" * 62)
    print("  Isolated Balance Test (LQR, no swing-up)")
    print("=" * 62)
    print(f"  Port: {PORT}   ZMQ: {ZMQ_ADDR}")

    try:
        comm = MotorComm(PORT, SERIAL_BAUD, timeout=0.05, echo_timeout=0.08)
        print("  Serial: connected")
    except Exception as e:
        print(f"  Serial error: {e}")
        return

    try:
        cam = CameraSubscriber(ZMQ_ADDR)
    except Exception as e:
        print(f"  ZMQ error: {e}")
        comm.close()
        return

    time.sleep(0.5)
    if cam.drain() == 0:
        print("  WARNING: no camera data. Is camerapreview_detect_color_V3.py running on the Pi?")
        comm.close()
        cam.close()
        return
    print("  Camera: live")

    trial = 0
    try:
        while True:
            input("\n  Put the CART at track CENTER, then press ENTER to zero & arm "
                  "(Ctrl-C to quit)...")
            comm.send("h", 0, echo_timeout=0.3)   # zero encoder at center
            comm.send_velocity(0)                 # ensure motor off
            cam.drain()

            if not wait_until_upright(cam):
                continue

            trial += 1
            print(f"\n\n  ▶ BALANCING (trial {trial}) — LET GO!\n")

            log_name = time.strftime("balance_%Y%m%d_%H%M%S.csv")
            log_path = os.path.join(os.path.dirname(__file__), log_name)
            with open(log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(LOG_COLS)
                t0 = time.time()
                result = balance_lqr_loop(comm, cam, t0, w)
                held = time.time() - t0

            comm.send_velocity(0)
            print(f"\n  Trial {trial}: ended '{result}' after {held:.2f}s  "
                  f"-> {log_name}")
            print(f"  Plot it:  python plot_swingup.py {log_name}")

    except KeyboardInterrupt:
        print("\n  Quit.")
    finally:
        comm.send_velocity(0)
        comm.close()
        cam.close()


if __name__ == "__main__":
    main()
