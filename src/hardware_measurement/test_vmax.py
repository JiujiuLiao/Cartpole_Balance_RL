"""
test_vmax.py
============
Distinguish TORQUE-limited vs SPEED-limited cart actuator regime.

Hypothesis (2026-06-03):
    Theoretical max linear speed = 251 RPM * 80mm / 60s = 0.335 m/s.
    If the cart hits this ceiling in <100 ms, the actuator is SPEED-limited
    (back-EMF kills available torque once cart matches no-load speed).
    If pos(t) is parabolic over the whole run (no plateau), it is
    TORQUE-limited.

Procedure (bidirectional, fully automatic):
    1. Auto-home cart to encoder 0.
    2. Drive cart to pos = -RUNWAY_PULSES (~28 cm to the left).
    3. Wait for standstill (no manual handling).
    4. Send <v, +255>, sample pos every 5 ms for up to 1.5 s.
    5. Stop when pos >= +RUNWAY_PULSES (~28 cm to the right) or t >= 1.5 s.
    6. Repeat in the reverse direction (drive to +RUNWAY_PULSES, then PWM=-255).
    7. Analyze both runs:
         - first 50 ms average accel  -> high (>10 m/s^2) means torque rich
         - 200-500 ms mean velocity   -> compare to v_max_theoretical
         - time to reach 90 % v_max   -> short means speed-limited
       Print verdict.

Output: vmax_YYYYMMDD_HHMMSS.csv  (raw t/pos/dir samples for both runs)

Usage:
    python3 test_vmax.py [serial_port]
"""

import time
import sys
import os
import csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor_comm import MotorComm
from config import (
    DEFAULT_SERIAL_PORT, SERIAL_BAUD,
    M_PER_PULSE, MAX_PULSES,
    PWM_DEADBAND,
)

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
PORT             = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERIAL_PORT

TEST_PWM         = 255                # absolute PWM used for both directions
RUN_TIME_S       = 1.5                # max sample time per run
SAMPLE_INT_S     = 0.005              # 200 Hz sampling
RUNWAY_PULSES    = 2500               # start each run this far from center
                                      # (~28 cm), leaves ~6 cm safety margin
                                      # before the +-MAX_PULSES boundary
SAFE_STOP_PULSES = MAX_PULSES - 100   # emergency stop if we get this close to wall

REPOSITION_PWM   = 120                # PWM used during automatic repositioning
                                      # (slow, predictable, well above deadband)
STANDSTILL_S     = 0.3
STANDSTILL_THR   = 3                  # pulses/s

# Theoretical no-load cart speed = motor_RPM * pulley_circumference / 60s
#   251 RPM * (40 teeth * 2 mm) / 60 = 334.7 mm/s
V_MAX_THEORETICAL = 0.3347            # m/s, reference target for analysis


# ----------------------------------------------------------------------
# Cart positioning helpers
# ----------------------------------------------------------------------
def drive_to(comm, target_pulses, timeout=10.0, tol=15):
    """
    Drive cart to `target_pulses` using bang-bang positioning.
    Blocks until |pos - target| <= tol or timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        pos = comm.get_position()
        err = target_pulses - pos
        if abs(err) <= tol:
            break
        pwm = REPOSITION_PWM if err > 0 else -REPOSITION_PWM
        comm.send_velocity(pwm)
        time.sleep(0.02)
    comm.stop()
    time.sleep(0.3)


def wait_for_standstill(comm, timeout=2.0):
    """Block until cart has been still for STANDSTILL_S seconds."""
    still_since = None
    deadline    = time.time() + timeout
    prev_pos    = comm.get_position()
    prev_t      = time.time()

    while time.time() < deadline:
        time.sleep(0.05)
        now = time.time()
        pos = comm.get_position()
        dt  = now - prev_t
        vel = abs(pos - prev_pos) / dt if dt > 0 else 0

        if vel < STANDSTILL_THR:
            if still_since is None:
                still_since = now
            elif now - still_since >= STANDSTILL_S:
                return True
        else:
            still_since = None

        prev_pos, prev_t = pos, now

    return False


# ----------------------------------------------------------------------
# Single run
# ----------------------------------------------------------------------
def run_one_direction(comm, pwm, start_pulses, stop_pulses):
    """
    Position cart at start_pulses, then fire pwm and record samples
    until pos crosses stop_pulses (in the direction of pwm) or time
    runs out. Returns list of (t_s, pos_pulses) tuples.
    """
    label = "FORWARD (+)" if pwm > 0 else "REVERSE (-)"
    print(f"\n  -- {label}: PWM = {pwm:+d} --")

    print(f"  positioning to pos={start_pulses}...", end="", flush=True)
    drive_to(comm, start_pulses)
    print(f" done (pos={comm.get_position()}).")

    print(f"  waiting for standstill...", end="", flush=True)
    wait_for_standstill(comm)
    print(" still.")

    samples = []
    comm.send_velocity(pwm)
    t_start = time.perf_counter()

    while True:
        t = time.perf_counter() - t_start
        if t >= RUN_TIME_S:
            break

        pos = comm.get_position()
        samples.append((t, pos))

        # Stop when we cross the target end (direction-aware)
        if pwm > 0 and pos >= stop_pulses:
            break
        if pwm < 0 and pos <= stop_pulses:
            break

        # Emergency wall check
        if abs(pos) >= SAFE_STOP_PULSES:
            print(f"  ! safety stop at pos={pos}")
            break

        time.sleep(SAMPLE_INT_S)

    comm.stop()
    time.sleep(0.3)
    print(f"  collected {len(samples)} samples in {t*1000:.0f} ms.")
    return samples


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
def analyze(samples, pwm):
    """
    Returns a dict of derived numbers and prints a per-run verdict.
    """
    if len(samples) < 10:
        print("  (not enough samples to analyze)")
        return None

    ts  = np.array([s[0] for s in samples])
    pos = np.array([s[1] for s in samples])

    # Make direction-agnostic: signed displacement in meters
    disp_m = (pos - pos[0]) * M_PER_PULSE
    if pwm < 0:
        disp_m = -disp_m  # so positive displacement always means "in PWM direction"

    # Velocity by central differences (m/s)
    vel = np.gradient(disp_m, ts)

    # Smooth a touch (3-sample moving average) for the printed numbers only
    vel_smooth = np.convolve(vel, np.ones(3) / 3, mode="same")

    # First 50 ms: instantaneous acceleration estimate
    mask_50 = ts <= 0.050
    if mask_50.sum() >= 3:
        coeffs    = np.polyfit(ts[mask_50], vel_smooth[mask_50], 1)
        accel_50  = float(coeffs[0])  # slope of v(t) in the first 50 ms
        v_at_50ms = float(np.interp(0.050, ts, vel_smooth))
    else:
        accel_50, v_at_50ms = 0.0, 0.0

    # Steady-state estimate: mean velocity in 200..500 ms (if we got there)
    mask_ss = (ts >= 0.200) & (ts <= 0.500)
    if mask_ss.sum() >= 3:
        v_steady = float(np.mean(vel_smooth[mask_ss]))
    else:
        v_steady = float(np.max(vel_smooth))  # if run was short, use peak

    # Peak observed velocity
    v_peak = float(np.max(vel_smooth))

    # Time to reach 90 % of v_peak
    target = 0.9 * v_peak
    above  = np.where(vel_smooth >= target)[0]
    t_90   = float(ts[above[0]]) if len(above) else float("nan")

    print(f"  -- analysis (PWM={pwm:+d}) --")
    print(f"    v_peak              = {v_peak:.3f} m/s")
    print(f"    v(50 ms)            = {v_at_50ms:.3f} m/s")
    print(f"    v_steady (200-500ms)= {v_steady:.3f} m/s")
    print(f"    accel (first 50 ms) = {accel_50:.2f} m/s^2")
    print(f"    t to reach 90% peak = {t_90*1000:.1f} ms")
    print(f"    v_max theoretical   = {V_MAX_THEORETICAL:.3f} m/s "
          f"(ratio: {v_peak / V_MAX_THEORETICAL:.2f})")

    return {
        "v_peak": v_peak,
        "v_at_50ms": v_at_50ms,
        "v_steady": v_steady,
        "accel_50": accel_50,
        "t_90_ms": t_90 * 1000.0,
    }


def verdict(stats_fwd, stats_rev):
    """Combined verdict across both directions."""
    print()
    print("=" * 60)
    print("  VERDICT")
    print("=" * 60)

    if not stats_fwd or not stats_rev:
        print("  insufficient data")
        return

    # Use the better-conditioned (larger v_peak) direction for headline numbers
    s = max([stats_fwd, stats_rev], key=lambda d: d["v_peak"])
    accel = max(stats_fwd["accel_50"], stats_rev["accel_50"])
    ratio = s["v_peak"] / V_MAX_THEORETICAL

    # Diagnostic rules (loose -- the user reads the numbers, this is only a hint):
    speed_limited  = (s["t_90_ms"] < 150) and (0.80 < ratio < 1.15)
    torque_limited = (accel < 5.0) and (s["t_90_ms"] > 500)

    if speed_limited and not torque_limited:
        msg = ("SPEED-LIMITED.  Cart reaches no-load speed quickly and plateaus.\n"
               "  -> Torque is plentiful. To raise control bandwidth, use a\n"
               "     larger drive pulley (40T -> 60T or 80T) or a higher-RPM\n"
               "     motor. LQR alone cannot break this ceiling.")
    elif torque_limited and not speed_limited:
        msg = ("TORQUE-LIMITED.  Cart never plateaus; pos(t) stays parabolic.\n"
               "  -> Available motor torque is the bottleneck. Bigger pulley\n"
               "     would make this worse. Investigate belt slip, current\n"
               "     limit, or PWM frequency before changing the drive train.")
    else:
        msg = ("MIXED / INCONCLUSIVE.  Compare v_peak to v_max_theoretical\n"
               "  and the shape of pos(t) in the CSV. Re-run with cart fully\n"
               "  centered and the belt re-tensioned if uncertain.")

    print(f"  v_peak / v_max_theoretical = {ratio:.2f}")
    print(f"  first-50ms accel           = {accel:.2f} m/s^2")
    print(f"  fastest 90% rise time      = {min(stats_fwd['t_90_ms'], stats_rev['t_90_ms']):.0f} ms")
    print()
    print("  " + msg.replace("\n", "\n  "))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  test_vmax.py  --  speed-vs-torque diagnostic")
    print("=" * 60)
    print(f"  port              : {PORT}")
    print(f"  test PWM          : +-{TEST_PWM}")
    print(f"  runway            : +-{RUNWAY_PULSES} pulses "
          f"({RUNWAY_PULSES * M_PER_PULSE * 1000:.0f} mm)")
    print(f"  sample period     : {SAMPLE_INT_S*1000:.0f} ms")
    print(f"  deadband (ref)    : {PWM_DEADBAND}")
    print(f"  v_max theoretical : {V_MAX_THEORETICAL:.3f} m/s")
    print()

    try:
        comm = MotorComm(PORT, SERIAL_BAUD, verbose=True)
    except Exception as e:
        print(f"\n  failed to open serial: {e}")
        return

    input("  press ENTER when track is clear and cart is free to move...")

    try:
        # ---- home encoder to its current cart position ------------
        comm.send("h", 0)
        time.sleep(0.2)

        # ---- forward run --------------------------------------------
        samples_fwd = run_one_direction(
            comm,
            pwm          =  TEST_PWM,
            start_pulses = -RUNWAY_PULSES,
            stop_pulses  = +RUNWAY_PULSES,
        )
        stats_fwd = analyze(samples_fwd, +TEST_PWM)

        # ---- reverse run --------------------------------------------
        samples_rev = run_one_direction(
            comm,
            pwm          = -TEST_PWM,
            start_pulses = +RUNWAY_PULSES,
            stop_pulses  = -RUNWAY_PULSES,
        )
        stats_rev = analyze(samples_rev, -TEST_PWM)

        # ---- verdict -----------------------------------------------
        verdict(stats_fwd, stats_rev)

        # ---- return cart to center --------------------------------
        print("\n  returning cart to center...")
        drive_to(comm, 0)

    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        comm.stop()

    # ---- save CSV -----------------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    folder    = os.path.dirname(os.path.abspath(__file__))
    filepath  = os.path.join(folder, f"vmax_{timestamp}.csv")

    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["direction", "t_s", "pos_pulses", "disp_m"])
        for t, p in samples_fwd:
            w.writerow(["fwd", f"{t:.4f}", p,
                        f"{(p - samples_fwd[0][1]) * M_PER_PULSE:.5f}"])
        for t, p in samples_rev:
            w.writerow(["rev", f"{t:.4f}", p,
                        f"{-(p - samples_rev[0][1]) * M_PER_PULSE:.5f}"])

    print(f"\n  saved: {os.path.basename(filepath)}")
    comm.close()


if __name__ == "__main__":
    main()
