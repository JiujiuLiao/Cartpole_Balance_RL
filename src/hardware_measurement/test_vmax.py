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
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor_comm import MotorComm
from config import (
    DEFAULT_SERIAL_PORT, SERIAL_BAUD,
    M_PER_PULSE, MAX_PULSES,
    PWM_DEADBAND,
    CART_MASS_KG, PENDULUM_MASS_KG,
)

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
PORT             = (sys.argv[1]
                    if len(sys.argv) > 1 and not sys.argv[1].endswith(".csv")
                    else DEFAULT_SERIAL_PORT)

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

# Total moving mass at the cart (cart body + the pendulum it carries).
# Needed to convert measured v(t) dynamics into a force estimate via F = m*V/tau.
MASS_TOTAL_KG     = CART_MASS_KG + PENDULUM_MASS_KG    # 1.141 kg

# Nameplate force = stall_torque / pulley_radius from the motor spec sheet
# (1.766 N*m / 0.01273 m = 138.8 N). Used only as a *comparison* point;
# the measured force is the truth.
NAMEPLATE_FORCE_N = 139.0


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
# Analysis  (curve fit + empirical sanity check)
# ----------------------------------------------------------------------
def _first_order(t, V, tau):
    """v(t) = V * (1 - exp(-t/tau)) — DC motor + load step response."""
    return V * (1.0 - np.exp(-t / tau))


def analyze(samples, pwm):
    """
    Fit v(t) = V*(1 - exp(-t/tau)) to the run, derive force/accel from
    the fit, and additionally compute empirical metrics as a sanity check.

    Returns a dict with both sets of numbers. Prints the result.
    """
    if len(samples) < 10:
        print("  (not enough samples to analyze)")
        return None

    ts  = np.array([s[0] for s in samples], dtype=float)
    pos = np.array([s[1] for s in samples], dtype=float)

    # signed displacement in meters (positive = in PWM direction)
    disp_m = (pos - pos[0]) * M_PER_PULSE
    if pwm < 0:
        disp_m = -disp_m

    # velocity by central differences (np.gradient handles uneven dt)
    vel = np.gradient(disp_m, ts)

    # 3-point moving average with replication padding (no edge zero-bias)
    vp         = np.concatenate([[vel[0]], vel, [vel[-1]]])
    vel_smooth = (vp[:-2] + vp[1:-1] + vp[2:]) / 3.0

    # ---- curve fit -----------------------------------------------------
    fit_ok = False
    V_fit = tau_fit = a_initial = t_dies = r_squared = F_eff = float("nan")
    try:
        V0   = float(max(np.max(vel_smooth) * 1.02, 0.05))
        tau0 = 0.07
        popt, _ = curve_fit(
            _first_order, ts, vel,
            p0     = [V0, tau0],
            bounds = ([0.01, 0.001], [1.0, 2.0]),
            maxfev = 5000,
        )
        V_fit, tau_fit = popt
        v_model        = _first_order(ts, *popt)
        ss_res         = float(np.sum((vel - v_model) ** 2))
        ss_tot         = float(np.sum((vel - np.mean(vel)) ** 2))
        r_squared      = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        a_initial      = V_fit / tau_fit
        t_dies         = 3.0 * tau_fit       # accel @ 3*tau ≈ 5% of peak
        F_eff          = MASS_TOTAL_KG * V_fit / tau_fit
        fit_ok         = True
    except Exception as e:
        print(f"  (curve_fit failed: {e})")

    # ---- empirical (sanity check, unchanged from old logic) ----------
    mask_50 = ts <= 0.050
    if mask_50.sum() >= 3:
        coeffs    = np.polyfit(ts[mask_50], vel_smooth[mask_50], 1)
        accel_50  = float(coeffs[0])
        v_at_50ms = float(np.interp(0.050, ts, vel_smooth))
    else:
        accel_50, v_at_50ms = 0.0, 0.0

    mask_ss = (ts >= 0.200) & (ts <= 0.500)
    if mask_ss.sum() >= 3:
        v_steady = float(np.mean(vel_smooth[mask_ss]))
    else:
        v_steady = float(np.max(vel_smooth))

    v_peak = float(np.max(vel_smooth))
    above  = np.where(vel_smooth >= 0.9 * v_peak)[0]
    t_90   = float(ts[above[0]]) if len(above) else float("nan")

    # ---- print --------------------------------------------------------
    print(f"  -- analysis (PWM={pwm:+d}) --")
    if fit_ok:
        warn = []
        if r_squared < 0.95:
            warn.append("R^2 < 0.95 — model may not be first-order")
        if not (0.005 <= tau_fit <= 1.0):
            warn.append(f"tau = {tau_fit*1000:.0f} ms is outside [5, 1000]")
        if V_fit > 1.2 * V_MAX_THEORETICAL:
            warn.append("V_fit > 1.2 * v_max_theoretical — suspicious")

        print(f"    CURVE FIT  v(t) = V * (1 - exp(-t/tau))")
        print(f"      V (asymptotic v_max)  = {V_fit:.3f} m/s   "
              f"({100*V_fit/V_MAX_THEORETICAL:.0f}% of theoretical "
              f"{V_MAX_THEORETICAL:.3f})")
        print(f"      tau (time constant)   = {tau_fit*1000:.1f} ms")
        print(f"      a(0) (peak accel)     = {a_initial:.2f} m/s^2")
        print(f"      3*tau (accel @ ~5%)   = {t_dies*1000:.0f} ms")
        print(f"      R^2 (fit quality)     = {r_squared:.3f}")
        print(f"      F_eff = m*V/tau       = {F_eff:.2f} N   "
              f"(nameplate {NAMEPLATE_FORCE_N:.0f} N -> "
              f"{100*F_eff/NAMEPLATE_FORCE_N:.1f}%)")
        for w in warn:
            print(f"      ! {w}")
    else:
        print(f"    CURVE FIT: failed — see empirical only")

    print(f"    EMPIRICAL (sanity check)")
    print(f"      v_peak (np.max)       = {v_peak:.3f} m/s")
    print(f"      v(50 ms)              = {v_at_50ms:.3f} m/s")
    print(f"      v_steady (200-500ms)  = {v_steady:.3f} m/s")
    print(f"      avg accel (0-50ms)    = {accel_50:.2f} m/s^2")
    print(f"      t to 90% v_peak       = {t_90*1000:.0f} ms")

    return {
        "fit_ok":    fit_ok,
        "V_fit":     V_fit,
        "tau_fit":   tau_fit,
        "a_initial": a_initial,
        "t_dies":    t_dies,
        "r_squared": r_squared,
        "F_eff":     F_eff,
        "v_peak":    v_peak,
        "v_at_50ms": v_at_50ms,
        "v_steady":  v_steady,
        "accel_50":  accel_50,
        "t_90_ms":   t_90 * 1000.0,
    }


def verdict(stats_fwd, stats_rev):
    """Combined verdict across both directions, driven by the curve fit."""
    print()
    print("=" * 60)
    print("  VERDICT")
    print("=" * 60)

    runs = [s for s in (stats_fwd, stats_rev) if s]
    if not runs:
        print("  insufficient data")
        return

    valid = [s for s in runs if s["fit_ok"]]
    if not valid:
        print("  no valid curve fit — read the empirical numbers above and")
        print("  re-run with cleaner data (cart centered, belt tensioned).")
        return

    V_avg   = float(np.mean([s["V_fit"]     for s in valid]))
    tau_avg = float(np.mean([s["tau_fit"]   for s in valid]))
    a0_avg  = float(np.mean([s["a_initial"] for s in valid]))
    F_avg   = float(np.mean([s["F_eff"]     for s in valid]))
    r2_min  = float(min(s["r_squared"] for s in valid))

    print(f"  V (asymptotic v_max) = {V_avg:.3f} m/s   "
          f"({100*V_avg/V_MAX_THEORETICAL:.0f}% of theoretical "
          f"{V_MAX_THEORETICAL:.3f})")
    print(f"  tau (time constant)  = {tau_avg*1000:.1f} ms")
    print(f"  a(0) initial accel   = {a0_avg:.2f} m/s^2")
    print(f"  F_eff at cart        = {F_avg:.2f} N   "
          f"({100*F_avg/NAMEPLATE_FORCE_N:.1f}% of nameplate "
          f"{NAMEPLATE_FORCE_N:.0f} N)")
    print(f"  min R^2              = {r2_min:.3f}")
    print()

    if r2_min < 0.80:
        msg = ("R^2 < 0.80 in at least one direction. The system is NOT a clean\n"
               "first-order step response. Derived numbers (V, tau, a(0), F_eff)\n"
               "are unreliable — inspect the raw v(t) curve in the CSV before\n"
               "drawing physical conclusions.")
    elif r2_min < 0.95:
        msg = ("R^2 in [0.80, 0.95]. System is approximately first-order but with\n"
               "measurable deviation (likely static friction or a 2nd-order term).\n"
               "Numbers are usable but uncertain to roughly 10-20 %.")
    else:
        msg = (f"Clean first-order response (R^2 >= 0.95). The cart actuator is\n"
               f"well modelled as v_dot = (F/m)*(1 - v/V). The force available at\n"
               f"the cart is {F_avg:.1f} N — {100*F_avg/NAMEPLATE_FORCE_N:.1f}% of the "
               f"{NAMEPLATE_FORCE_N:.0f} N nameplate calculation.\n"
               f"This is the hardware ceiling; control-side tweaks cannot exceed it.")

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


def analyze_csv(path):
    """
    Re-analyze an existing vmax_*.csv with the current analyze() / verdict().
    No hardware needed. Usage:  python test_vmax.py path/to/vmax_*.csv
    """
    import csv as csv_mod

    samples_fwd, samples_rev = [], []
    with open(path, newline="") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            t   = float(row["t_s"])
            pos = int(row["pos_pulses"])
            (samples_fwd if row["direction"] == "fwd"
             else samples_rev).append((t, pos))

    print("=" * 60)
    print(f"  RE-ANALYZE: {os.path.basename(path)}")
    print("=" * 60)
    print(f"  fwd samples: {len(samples_fwd)}   rev samples: {len(samples_rev)}")
    print()

    stats_fwd = analyze(samples_fwd, +TEST_PWM) if samples_fwd else None
    stats_rev = analyze(samples_rev, -TEST_PWM) if samples_rev else None
    verdict(stats_fwd, stats_rev)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].endswith(".csv"):
        analyze_csv(sys.argv[1])
    else:
        main()
