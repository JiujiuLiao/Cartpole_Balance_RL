"""
pwm_force_calibration.py
========================
Calibrate the effective relationship between PWM command and cart force.

Primary purpose:
    Identify how much effective cart force is produced by each PWM level.

The script also estimates the additional actuator/dynamics terms needed to
interpret that relationship:
    c      PWM-to-force gain                  [N/PWM]
    b      speed-dependent drag/back-EMF term [N*s/m]
    F_c    Coulomb friction                   [N]
    tau    first-order velocity time constant [s]

The fitted model is:

    m * v_dot = c * u - b * v - F_c * sgn(v)

        c    motor force constant        (N per PWM unit)
        b    speed-proportional drag     (N*s/m, back-EMF + viscous)
        F_c  Coulomb friction            (N, fitted per direction)

METHOD: for each PWM level (both directions) the cart starts from one end
of a fixed runway, gets a constant-PWM step, and position is sampled at
200 Hz until it crosses the far end. Steady state gives one point of
    v_ss(u) = (c*u - F_c*sgn(u)) / b        -> slope c/b, intercept F_c/b
and the exponential transient gives tau = m/b  -> pins b, then c and F_c.

SETUP:
  1. Pole REMOVED if possible (cleanest), else HANGING at rest
     (its swing adds ~0.9 Hz ripple; steady-state averaging spans it).
     Run with --no-pole when the pole is off so the right mass is used.
  2. Cart free to move over the whole track, hands clear.
  3. Camera not needed.

Usage (on the Pi, like the other calibration scripts):
    python3 pwm_force_calibration.py                     # /dev/ttyUSB0, pole hanging
    python3 pwm_force_calibration.py --no-pole           # pole removed
    python3 pwm_force_calibration.py /dev/ttyUSB0
    python3 pwm_force_calibration.py motor_ident_*.csv   # re-analyze offline (no hardware)

Output: data/calibration/motor_ident_YYYYMMDD_HHMMSS.csv
        (columns: pwm, t_s, pos_pulses) + fitted c / b / F_c summary.
"""

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DEFAULT_SERIAL_PORT, SERIAL_BAUD,
    M_PER_PULSE, MAX_PULSES, pulses_for_distance,
    PWM_DEADBAND, FORCE_PER_PWM,
    CART_MASS_KG, PENDULUM_MASS_KG,
    CALIBRATION_DIR,
)

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
PWM_LEVELS       = [40, 60, 90, 120, 160, 200, 255]   # tested in BOTH directions
# Distances below are measured from the track center to the start/stop point.
# The actual pulse limits are derived from config.M_PER_PULSE so they stay
# correct when the pulley or encoder conversion changes.
RUNWAY_HALF_DISTANCE_CM = 25.0  # one-sided start/stop distance from center
SAFE_STOP_DISTANCE_CM   = 27.7  # one-sided emergency limit from center

RUNWAY_PULSES    = pulses_for_distance(RUNWAY_HALF_DISTANCE_CM / 100.0)
SAFE_STOP_PULSES = pulses_for_distance(SAFE_STOP_DISTANCE_CM / 100.0)

if RUNWAY_PULSES >= SAFE_STOP_PULSES:
    raise ValueError("RUNWAY_HALF_DISTANCE_CM must be smaller than SAFE_STOP_DISTANCE_CM")
if SAFE_STOP_PULSES >= MAX_PULSES:
    raise ValueError("SAFE_STOP_DISTANCE_CM must stay inside config.MAX_PULSES")

SAMPLE_INT_S     = 0.005              # 200 Hz

REPOSITION_PWM   = 120
STANDSTILL_S     = 0.3
STANDSTILL_THR   = 3                  # pulses/s

TAU_FIT_MIN_PWM  = 90                 # below this, stick-slip corrupts the tau fit
STEADY_WINDOW_S  = 0.8                # average v over the last part of each run
STEADY_SKIP_S    = 0.6                # ...but never earlier than this after the step


def run_time_for(pwm_abs):
    """Slow runs need more time to reach and hold steady state."""
    return 2.5 if pwm_abs <= 90 else 1.5


def total_mass(no_pole):
    return CART_MASS_KG if no_pole else CART_MASS_KG + PENDULUM_MASS_KG


# ----------------------------------------------------------------------
# Cart positioning helpers (same logic as test_vmax.py)
# ----------------------------------------------------------------------
def drive_to(comm, target_pulses, timeout=10.0, tol=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        pos = comm.get_position()
        err = target_pulses - pos
        if abs(err) <= tol:
            break
        comm.send_velocity(REPOSITION_PWM if err > 0 else -REPOSITION_PWM)
        time.sleep(0.02)
    comm.stop()
    time.sleep(0.3)


def wait_for_standstill(comm, timeout=3.0):
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
# One step run
# ----------------------------------------------------------------------
def run_step(comm, pwm):
    """Position the cart, fire `pwm`, sample (t, pos) until the far end."""
    start = -RUNWAY_PULSES if pwm > 0 else +RUNWAY_PULSES
    stop  = -start
    print(f"  PWM {pwm:+4d}: positioning...", end="", flush=True)
    drive_to(comm, start)
    wait_for_standstill(comm)
    print(" running...", end="", flush=True)

    samples = []
    max_t   = run_time_for(abs(pwm))
    comm.send_velocity(pwm)
    t0 = time.perf_counter()
    while True:
        t = time.perf_counter() - t0
        if t >= max_t:
            break
        pos = comm.get_position()
        samples.append((t, pos))
        if (pwm > 0 and pos >= stop) or (pwm < 0 and pos <= stop):
            break
        if abs(pos) >= SAFE_STOP_PULSES:
            print(" ! safety stop", end="")
            break
        time.sleep(SAMPLE_INT_S)
    comm.stop()
    time.sleep(0.4)
    print(f" {len(samples)} samples, {t:.2f} s")
    return samples


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
def _first_order(t, V, tau):
    return V * (1.0 - np.exp(-t / tau))


def velocity_of(samples, pwm):
    """t, v arrays with v positive in the PWM direction."""
    ts  = np.array([s[0] for s in samples], float)
    pos = np.array([s[1] for s in samples], float)
    disp = (pos - pos[0]) * M_PER_PULSE
    if pwm < 0:
        disp = -disp
    v = np.gradient(disp, ts)
    vp = np.concatenate([[v[0]], v, [v[-1]]])
    return ts, (vp[:-2] + vp[1:-1] + vp[2:]) / 3.0


def analyze_run(samples, pwm):
    """Return dict(pwm, v_ss, tau or nan). Steady state = mean of the tail."""
    if len(samples) < 30:
        return None
    ts, v = velocity_of(samples, pwm)
    t_end = ts[-1]
    lo = max(STEADY_SKIP_S, t_end - STEADY_WINDOW_S)
    m = ts >= lo
    if m.sum() < 15:            # run ended too early for a steady tail
        m = ts >= t_end * 0.5
    v_ss = float(np.mean(v[m]))

    tau = float("nan")
    if abs(pwm) >= TAU_FIT_MIN_PWM:
        try:
            popt, _ = curve_fit(_first_order, ts, v,
                                p0=[max(v_ss, 0.05), 0.12],
                                bounds=([0.01, 0.005], [1.0, 1.0]), maxfev=5000)
            v_model = _first_order(ts, *popt)
            ss_tot  = np.sum((v - v.mean()) ** 2)
            r2 = 1 - np.sum((v - v_model) ** 2) / ss_tot if ss_tot > 0 else 0
            if r2 >= 0.80:
                tau = float(popt[1])
        except Exception:
            pass
    return {"pwm": pwm, "v_ss": v_ss, "tau": tau}


def fit_model(results, mass):
    """
    Per-direction line fit v_ss = (c/b)*|u| - F_c/b, plus b = m/tau.
    Prints the summary and returns (c, b, F_c_pos, F_c_neg).
    """
    print("\n" + "=" * 62)
    print("  MODEL FIT   m*v_dot = c*u - b*v - F_c*sgn(v)")
    print("=" * 62)
    print(f"  {'pwm':>5}  {'v_ss m/s':>9}  {'tau ms':>7}")
    for r in sorted(results, key=lambda r: r["pwm"]):
        tau_s = f"{r['tau']*1000:7.1f}" if np.isfinite(r["tau"]) else "      -"
        print(f"  {r['pwm']:+5d}  {r['v_ss']:9.3f}  {tau_s}")

    taus = [r["tau"] for r in results if np.isfinite(r["tau"])]
    if not taus:
        print("  ! no valid tau fit — cannot separate c from b. Check raw CSV.")
        return None
    tau_med = float(np.median(taus))
    b = mass / tau_med

    out = {}
    for sgn, name in [(+1, "forward"), (-1, "reverse")]:
        pts = [(abs(r["pwm"]), r["v_ss"]) for r in results
               if np.sign(r["pwm"]) == sgn and r["v_ss"] > 0.005]
        if len(pts) < 3:
            print(f"  ! {name}: <3 usable points, skipped")
            continue
        u  = np.array([p[0] for p in pts], float)
        v  = np.array([p[1] for p in pts], float)
        slope, icpt = np.polyfit(u, v, 1)
        pred = slope * u + icpt
        r2 = 1 - np.sum((v - pred) ** 2) / np.sum((v - v.mean()) ** 2)
        c   = slope * b
        F_c = -icpt * b
        u_dead = F_c / c if c > 0 else float("nan")
        out[name] = (c, F_c, u_dead, r2)

    print(f"\n  tau (median of {len(taus)} fits) = {tau_med*1000:.1f} ms")
    print(f"  b   = m/tau = {mass:.3f}/{tau_med:.3f} = {b:.2f} N*s/m")
    for name, (c, F_c, u_dead, r2) in out.items():
        print(f"  {name:8s}: c = {c:.5f} N/PWM   F_c = {F_c:.3f} N   "
              f"eff. deadband = {u_dead:.1f} PWM   (line R^2 = {r2:.3f})")
    if out:
        c_avg = float(np.mean([v[0] for v in out.values()]))
        print(f"\n  c averaged = {c_avg:.5f} N/PWM   "
              f"(config FORCE_PER_PWM = {FORCE_PER_PWM})")
        print(f"  -> for compute_lqr_K:  b = {b:.2f},  c = {c_avg:.5f}")
    return out


# ----------------------------------------------------------------------
# CSV I/O
# ----------------------------------------------------------------------
def save_csv(all_runs):
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    path = os.path.join(CALIBRATION_DIR,
                        f"motor_ident_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pwm", "t_s", "pos_pulses"])
        for pwm, samples in all_runs:
            for t, p in samples:
                w.writerow([pwm, f"{t:.4f}", p])
    print(f"\n  saved: {path}")
    return path


def load_csv(path):
    runs = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pwm = int(row["pwm"])
            runs.setdefault(pwm, []).append((float(row["t_s"]),
                                             int(row["pos_pulses"])))
    return [(pwm, s) for pwm, s in runs.items()]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    no_pole = "--no-pole" in sys.argv
    args    = [a for a in sys.argv[1:] if not a.startswith("--")]
    port    = args[0] if args else DEFAULT_SERIAL_PORT
    mass    = total_mass(no_pole)

    from motor_comm import MotorComm   # deferred: offline analysis needs no serial

    print("=" * 62)
    print("  pwm_force_calibration.py — PWM-to-force calibration")
    print("=" * 62)
    print(f"  port    : {port}")
    print(f"  levels  : +-{PWM_LEVELS}")
    print(f"  mass    : {mass:.3f} kg ({'pole removed' if no_pole else 'pole hanging'})")
    print(f"  runway  : ±{RUNWAY_HALF_DISTANCE_CM:.1f} cm "
          f"(derived ±{RUNWAY_PULSES} pulses)")
    print(f"  safe stop: ±{SAFE_STOP_DISTANCE_CM:.1f} cm "
          f"(derived ±{SAFE_STOP_PULSES} pulses)")
    print()

    comm = MotorComm(port, SERIAL_BAUD, verbose=True)
    input("  press ENTER when track is clear and cart is free to move...")

    all_runs = []
    try:
        # Homing sets 0 wherever the cart currently is: START THE SCRIPT
        # WITH THE CART CENTERED (same convention as test_vmax.py).
        comm.send("h", 0)
        time.sleep(0.2)

        for level in PWM_LEVELS:
            for pwm in (+level, -level):   # out and back: minimal repositioning
                all_runs.append((pwm, run_step(comm, pwm)))

        print("\n  returning cart to center...")
        drive_to(comm, 0)
    except KeyboardInterrupt:
        print("\n  interrupted — analyzing what was collected")
    finally:
        comm.stop()

    path = save_csv(all_runs)
    results = [analyze_run(s, pwm) for pwm, s in all_runs]
    fit_model([r for r in results if r], mass)
    comm.close()


def main_offline(path):
    no_pole = "--no-pole" in sys.argv
    mass    = total_mass(no_pole)
    print(f"  RE-ANALYZE: {os.path.basename(path)}   mass = {mass:.3f} kg")
    runs = load_csv(path)
    results = [analyze_run(s, pwm) for pwm, s in runs]
    fit_model([r for r in results if r], mass)


if __name__ == "__main__":
    csv_args = [a for a in sys.argv[1:] if a.endswith(".csv")]
    if csv_args:
        main_offline(csv_args[0])
    else:
        main()
