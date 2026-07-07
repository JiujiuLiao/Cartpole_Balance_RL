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
from collections import deque
from pathlib import Path

# Expose src/ (top-level libs). balance_lqr sits next to this file so it
# resolves via the script's own directory automatically.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor_comm import MotorComm
from camera_comm import CameraSubscriber
from balance_lqr import balance_lqr_loop
from config import (DEFAULT_SERIAL_PORT, SERIAL_BAUD, DEFAULT_ZMQ_ADDR,
                    BALANCE_DIAG_DIR)

# ── Engage gate (the "is it genuinely upright & steady?" condition) ───
# Raw camera omega is noisy (~0.6 rad/s std — the reason balance uses an EKF),
# so the gate averages over a window instead of requiring every sample to pass:
# engage when ALL samples in the window have |θ| < ENGAGE_ANGLE_DEG, and the
# WINDOW-MEAN |ω| < ENGAGE_OMEGA. Per-sample ω checks never trigger by design.
ENGAGE_ANGLE_DEG = 3.0     # every sample in the window within this of upright
ENGAGE_OMEGA     = 0.3     # rad/s — window-mean angular rate below this
ENGAGE_WINDOW_S  = 0.3     # judge steadiness over this much history

PORT     = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERIAL_PORT
ZMQ_ADDR = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ZMQ_ADDR

LOG_COLS = ["t_s", "theta_deg", "omega_rad_s", "E_minus_Eref", "pwm", "cart_pos"]


def wait_until_upright(cam):
    """Block (motor off) until the pendulum is steady near upright. Returns True,
    or False if Ctrl-C / no camera."""
    print(f"\n  ARMED — hold the pendulum UP and steady near 0°.")
    print(f"  Engages when |θ|<{ENGAGE_ANGLE_DEG}° for {ENGAGE_WINDOW_S}s "
          f"AND mean |ω|<{ENGAGE_OMEGA} rad/s. Motor is OFF until then.")
    window     = deque()     # (t, theta_deg, omega) samples inside ENGAGE_WINDOW_S
    last_disp  = 0.0
    while True:
        res = cam.latest_angle()
        if res is None:
            time.sleep(0.005)
            continue
        theta_rad, omega, _ = res
        theta_deg = math.degrees(theta_rad)

        now = time.time()
        window.append((now, theta_deg, omega))
        while window and now - window[0][0] > ENGAGE_WINDOW_S:
            window.popleft()

        full     = window[-1][0] - window[0][0] >= ENGAGE_WINDOW_S * 0.9
        all_up   = all(abs(s[1]) < ENGAGE_ANGLE_DEG for s in window)
        mean_om  = sum(s[2] for s in window) / len(window)
        ready    = full and all_up and abs(mean_om) < ENGAGE_OMEGA

        if now - last_disp > 0.1:
            sys.stdout.write(f"\r  θ={theta_deg:+7.1f}°  ω̄={mean_om:+5.2f} rad/s   "
                             f"{'[READY -> engaging]' if ready else '[hold near 0°]  '}   ")
            sys.stdout.flush()
            last_disp = now

        if ready:
            return True
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
            # Blocking send() must not race the position-stream reader thread:
            # stop the stream, home, then restart it. balance_lqr_loop polls
            # get_position() at 100 Hz and NEEDS the cached stream — without it
            # every call is a blocking serial round-trip and the loop rate dies
            # (swing_up_energy_dc.py does the same home-then-stream sequence).
            comm.stop_position_stream()
            comm.send("h", 0, echo_timeout=0.3)   # zero encoder at center
            comm.send_velocity(0)                 # ensure motor off
            comm.start_position_stream()
            cam.drain()

            if not wait_until_upright(cam):
                continue

            trial += 1
            print(f"\n\n  ▶ BALANCING (trial {trial}) — LET GO!\n")

            # Both CSVs of a hand-test trial go to the shared diag folder with a
            # "testbal_" prefix, so they never mix with swing-up-phase segments:
            #   testbal_log_*.csv   swing-up-schema log (this file, plot_swingup)
            #   testbal_diag_*.csv  22-col per-loop diag written by balance_lqr_loop
            os.makedirs(BALANCE_DIAG_DIR, exist_ok=True)
            log_name = time.strftime("testbal_log_%Y%m%d_%H%M%S.csv")
            log_path = os.path.join(BALANCE_DIAG_DIR, log_name)
            with open(log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(LOG_COLS)
                t0 = time.time()
                result = balance_lqr_loop(comm, cam, t0, w,
                                          diag_prefix="testbal_diag")
                held = time.time() - t0

            comm.send_velocity(0)
            plot_py = Path(__file__).resolve().parent.parent / "analysis" / "plot_swingup.py"
            print(f"\n  Trial {trial}: ended '{result}' after {held:.2f}s  "
                  f"-> {log_path}")
            print(f"  Plot it:  python {plot_py} {log_path}")

    except KeyboardInterrupt:
        print("\n  Quit.")
    finally:
        comm.send_velocity(0)
        comm.close()
        cam.close()


if __name__ == "__main__":
    main()
