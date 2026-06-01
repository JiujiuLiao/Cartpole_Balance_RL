"""
test_accel.py
=============
Measure cart acceleration for a set of PWM values.
Each PWM is tested N_REPEAT times; results are averaged to reduce noise.

Improvements over v1:
  - Repeated trials per PWM + median (reduces friction/startup jitter)
  - Linear least-squares fit for initial acceleration (more robust than two-point diff)
  - Active standstill check before each trial (ensures consistent start)

IMPORTANT: Run test_pwm_range.py first to find the motor deadband.
           Set TEST_PWM so all values are above the deadband.

Usage:
    python test_accel.py 

Output: accel_YYYYMMDD_HHMMSS.csv  (full time-series, all trials)
"""

import time
import sys
import os
import csv
from pathlib import Path

import numpy as np

# Allow `from motor_comm import ...` / `from config import ...` when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor_comm import MotorComm
from config import (DEFAULT_SERIAL_PORT, SERIAL_BAUD,
                    PULSES_PER_REV, WHEEL_DIAMETER_M, M_PER_PULSE)

# ── Parameters ────────────────────────────────────
PORT          = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERIAL_PORT

TEST_PWM      = [100, 150, 200, 255,
                 -100, -150, -200, -255]  # bidirectional: forward then reverse
                                          # set after running test_pwm_range.py
                                          # all |values| must be above the measured deadband

N_REPEAT      = 5        # trials per PWM — median taken across trials to reduce noise
RUN_TIME_S    = 1.0      # s         — motor run duration per trial
                         # Was 1.5; reduced 2026-06-01 because at PWM=255 with
                         # steady speed ~0.23 m/s, 1.5s gave 0.32 m travel —
                         # too close to the 0.343 m boundary. 1.0s gives ~0.20 m
                         # which is safe. Linear fit uses only first 0.5s anyway.
SAMPLE_INT_S  = 0.005    # s         — interval between encoder reads (5 ms → ~200 samples/s)
FIT_WINDOW_S  = 0.5      # s         — only first this many seconds used for accel fit (before speed saturates)
STANDSTILL_S  = 0.4      # s         — must stay below STANDSTILL_THR for this long before trial starts
STANDSTILL_THR = 3       # pulses/s  — below this the cart is considered stopped

RETURN_PWM    = 120      # PWM 0–255 — return speed; must be above motor deadband (update after test_pwm_range)
BOUNDARY      = 3000     # pulses    ≈ 24.5 cm from center (track half-length ~42 cm = 5147 pulses)

# ── Cart control ──────────────────────────────────
def return_to_origin(comm, timeout=8.0):
    """Drive back to encoder 0."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        pos = comm.get_position()
        if abs(pos) <= 5:
            break
        comm.send("v", RETURN_PWM if pos < 0 else -RETURN_PWM)
        time.sleep(0.02)
    comm.stop()
    time.sleep(0.3)


def wait_for_standstill(comm, timeout=3.0):
    """
    Wait until cart velocity is near zero for STANDSTILL_S seconds.
    Returns True if standstill confirmed, False if timeout.
    """
    still_since = None
    deadline = time.time() + timeout
    prev_pos  = comm.get_position()
    prev_t    = time.time()

    while time.time() < deadline:
        time.sleep(0.05)
        now = time.time()
        pos = comm.get_position()
        dt  = now - prev_t
        vel = abs(pos - prev_pos) / dt if dt > 0 else 0   # pulses/s — instantaneous speed estimate

        if vel < STANDSTILL_THR:
            if still_since is None:
                still_since = now
            elif now - still_since >= STANDSTILL_S:
                return True        # confirmed still long enough
        else:
            still_since = None     # reset if cart moved

        prev_pos = pos
        prev_t   = now

    return False   # timed out


# ── Single trial ──────────────────────────────────
def run_trial(comm, pwm):
    """
    One trial at a fixed PWM from standstill.
    Returns list of sample dicts.
    """
    comm.send("h", 0)
    comm.stop()
    wait_for_standstill(comm)   # ensure truly stopped before starting

    samples   = []
    prev_pos  = 0
    prev_time = 0.0

    comm.send("v", pwm)
    t_start = time.time()

    while True:
        now     = time.time()
        elapsed = now - t_start
        if elapsed >= RUN_TIME_S:
            break

        pos = comm.get_position()
        dt  = elapsed - prev_time

        vel = (pos - prev_pos) / dt if (dt > 0 and prev_time > 0) else 0.0  # pulses/s

        samples.append({
            "t_s":       round(elapsed, 4),               # s
            "pos_pulses": pos,                            # pulses — encoder position from origin
            "vel_pps":   round(vel, 1),                   # pulses/s — instantaneous speed
            "vel":       round(vel * M_PER_PULSE, 5),     # m/s
        })

        prev_pos  = pos
        prev_time = elapsed
        time.sleep(SAMPLE_INT_S)

    comm.stop()
    time.sleep(0.3)
    return samples


# ── Acceleration from one trial ───────────────────
def compute_accel_from_trial(samples):
    """
    Returns (accel_fit, accel_avg) both in m/s²:
      accel_fit : linear fit slope over first FIT_WINDOW_S  (robust, use this)
      accel_avg : (v_end - v_start) / total_time            (reference only)
    """
    # Use points where motor is actually moving
    moving = [s for s in samples if abs(s["vel_pps"]) > 2]
    if len(moving) < 3:
        return 0.0, 0.0

    # Linear fit over first FIT_WINDOW_S — captures initial acceleration before speed saturates
    window = [s for s in moving if s["t_s"] <= FIT_WINDOW_S]
    if len(window) >= 3:
        ts   = np.array([s["t_s"]  for s in window])   # s
        vs   = np.array([s["vel"]  for s in window])   # m/s
        coeffs = np.polyfit(ts, vs, 1)   # fit v = a·t + b
        accel_fit = coeffs[0]            # m/s² — slope of v-t line = acceleration (use this)
    else:
        accel_fit = 0.0

    # Average over full run (reference only — motor may have reached steady-state, so this underestimates)
    v_start = moving[0]["vel"]    # m/s
    v_end   = moving[-1]["vel"]   # m/s
    dt_full = moving[-1]["t_s"] - moving[0]["t_s"]         # s
    accel_avg = (v_end - v_start) / dt_full if dt_full > 0 else 0.0  # m/s²

    return accel_fit, accel_avg


# ── Main ──────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Cart Acceleration Test  (repeat + linear fit)")
    print("=" * 60)
    print(f"  Port:      {PORT}")
    print(f"  PWM:       {TEST_PWM}  (must be above deadband)")
    print(f"  Trials:    {N_REPEAT} per PWM, median taken")
    print(f"  Fit window: first {FIT_WINDOW_S}s")
    print(f"  Encoder:   {PULSES_PER_REV} pulses/rev, "
          f"{WHEEL_DIAMETER_M*1000:.0f}mm pulley")
    print()

    try:
        comm = MotorComm(PORT, SERIAL_BAUD, verbose=True)
    except Exception as e:
        print(f"\n  Failed to connect: {e}")
        return

    input("  Press ENTER when cart is near center and free to move...")

    all_rows = []
    summary  = []

    try:
        for pwm in TEST_PWM:
            print()
            print(f"  ══ PWM = {pwm} ({'─'*40})")

            # Safety check before starting this PWM group
            pos = comm.get_position()
            if abs(pos) > BOUNDARY:
                print(f"  Boundary exceeded ({pos} pulses), skipping PWM={pwm}")
                continue

            fit_accels = []
            avg_accels = []

            for trial in range(1, N_REPEAT + 1):
                print(f"  Trial {trial}/{N_REPEAT}...", end="", flush=True)

                return_to_origin(comm)

                samples = run_trial(comm, pwm)
                fit_a, avg_a = compute_accel_from_trial(samples)

                fit_accels.append(fit_a)
                avg_accels.append(avg_a)

                print(f"  fit={fit_a:.4f} m/s²  avg={avg_a:.4f} m/s²")

                # Save raw samples
                for s in samples:
                    all_rows.append({"pwm": pwm, "trial": trial, **s})

            # Median across trials
            fit_median = float(np.median(fit_accels))
            avg_median = float(np.median(avg_accels))

            summary.append({
                "pwm":       pwm,
                "accel_fit": round(fit_median, 4),   # m/s²
                "accel_avg": round(avg_median, 4),   # m/s²
            })

            print(f"  ── Median fit accel : {fit_median:.4f} m/s²  "
                  f"← use for control analysis")
            print(f"  ── Median avg accel : {avg_median:.4f} m/s²")

    except KeyboardInterrupt:
        print("\n  Interrupted")
    finally:
        comm.stop()

    # Summary table
    print()
    print("=" * 60)
    print("  Summary (median across trials)")
    print(f"  {'PWM':>5}  {'Fit accel (m/s²)':>18}  {'Avg accel (m/s²)':>18}")
    print(f"  {'─'*50}")
    for s in summary:
        print(f"  {s['pwm']:>5}  {s['accel_fit']:>18.4f}  "
              f"{s['accel_avg']:>18.4f}")
    print("=" * 60)

    # Save CSV
    if all_rows:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        folder    = os.path.dirname(os.path.abspath(__file__))
        filename  = f"accel_{timestamp}.csv"
        filepath  = os.path.join(folder, filename)

        fields = ["pwm", "trial", "t_s", "pos_pulses", "vel_pps", "vel"]
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"  Saved: {filename}")

    comm.close()


if __name__ == "__main__":
    main()
