"""
fit_force_curve.py
==================
Fit an asymmetric Coulomb-friction motor model from test_accel.py output:

    F_net(u) = sgn(u) * max(0, K * (|u| - U_DEAD))

Two parameter pairs are fitted INDEPENDENTLY:
    (K_POS, U_DEAD_POS)   for u > 0
    (K_NEG, U_DEAD_NEG)   for u < 0
A quadratic saturation term per direction can be added with --saturation
(use only if linear residuals show a systematic trend at high PWM).

Why this model and not a quadratic F = a*u**2 + b*u + c:
  - Each parameter has physical meaning:
      U_DEAD = static-friction PWM threshold
      K      = motor force constant (N per PWM unit, includes gearbox eff.)
  - Linear in (|u| - U_DEAD) matches DC-motor physics at low velocity
    (back-EMF ~ 0, current ~ V/R, torque proportional to current).
  - PPO + Domain Randomization can perturb each parameter physically:
      K       *= U(0.9, 1.1)       (~ motor strength variation)
      U_DEAD  += U(-5, +5)         (~ friction variation)
  - Independent positive/negative directions catch real H-bridge and
    mechanical asymmetry (IBT-2 channel imbalance, belt pre-tension, etc.)

Workflow:
    1. test_pwm_range.py  -> find deadband, pick PWM grid
    2. test_accel.py      -> record cart acceleration at each PWM
       (edit TEST_PWM to cover BOTH directions, e.g.
        [-255,-200,-150,-100, 100,150,200,255])
    3. fit_force_curve.py -> this script

The fitted function pwm_to_force(u) becomes the action -> force mapping
inside the simulation env (notebooks/01_simulation_environment.ipynb).
Belt + transmission friction is absorbed into K and U_DEAD, so the sim
must NOT subtract f_cart again (set F_CART = 0 in the notebook).

Usage:
    python fit_force_curve.py                          # newest accel_*.csv
    python fit_force_curve.py --file accel_20260513_120530.csv
    python fit_force_curve.py --m_total 1.141          # override total mass
    python fit_force_curve.py --saturation             # add quadratic sat term
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Allow `from config import ...` when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import M_PER_PULSE  # noqa: E402

# Mass values; override with --m_total. Confirmed 2026-05-11.
M_CART_DEFAULT = 1.014
M_POLE_DEFAULT = 0.127

# Must match the window used inside test_accel.py so the two scripts agree.
FIT_WINDOW_S = 0.5

# Mechanical context for the sanity check
GEARBOX_TORQUE_NM = 1.766     # nominal stall torque
PULLEY_RADIUS_M   = 0.008     # 16 mm diameter / 2


# ── I/O ──────────────────────────────────────────────────────────────
def find_latest_csv(folder):
    pattern = os.path.join(folder, "accel_*.csv")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


def load_trials(csv_path):
    """Returns {pwm: {trial: [(t_s, vel_m_s), ...]}}."""
    trials = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pwm   = int(row["pwm"])
            trial = int(row["trial"])
            t_s   = float(row["t_s"])
            vel   = float(row["vel"])
            trials.setdefault(pwm, {}).setdefault(trial, []).append((t_s, vel))
    return trials


def trial_acceleration(samples):
    """Linear fit of velocity vs time over FIT_WINDOW_S; same logic as test_accel.py."""
    arr = np.array([(t, v) for t, v in samples if abs(v) > 0.001])
    if len(arr) < 3:
        return None
    window = arr[arr[:, 0] <= FIT_WINDOW_S]
    if len(window) < 3:
        return None
    slope, _ = np.polyfit(window[:, 0], window[:, 1], 1)
    return float(slope)


def aggregate(trials):
    """Returns (pwms, accel_median, accel_std) sorted by pwm."""
    pwms, medians, stds = [], [], []
    for pwm in sorted(trials.keys()):
        accels = []
        for trial_samples in trials[pwm].values():
            a = trial_acceleration(trial_samples)
            if a is not None:
                accels.append(a)
        if not accels:
            continue
        pwms.append(pwm)
        medians.append(np.median(accels))
        stds.append(np.std(accels))
    return np.array(pwms), np.array(medians), np.array(stds)


# ── Models (per direction; K is always >= 0) ────────────────────────
def linear_deadband_pos(u, k, u_dead):
    """For u > 0:  F = max(0, k * (u - u_dead))."""
    return np.maximum(0.0, k * (u - u_dead))


def linear_deadband_neg(u, k, u_dead):
    """For u < 0:  F = min(0, k * (u + u_dead))."""
    return np.minimum(0.0, k * (u + u_dead))


def saturated_pos(u, k, u_dead, sat):
    """Linear with quadratic saturation; clamps to 0 if model dips negative."""
    delta = u - u_dead
    return np.where(delta > 0,
                    np.maximum(0.0, k * delta - sat * delta**2),
                    0.0)


def saturated_neg(u, k, u_dead, sat):
    delta = u + u_dead
    return np.where(delta < 0,
                    np.minimum(0.0, k * delta + sat * delta**2),
                    0.0)


def r_squared(y_actual, y_pred):
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ── Fitting ──────────────────────────────────────────────────────────
def fit_direction(pwms, forces, direction, use_saturation=False):
    """
    direction = +1 (positive PWM) or -1 (negative PWM).
    Returns dict with K, U_DEAD, (SAT,) r2, n_points, u, f, f_pred, residuals.
    Returns None if fewer than 2 (or 3 with saturation) points in this direction.
    """
    if direction > 0:
        mask  = pwms > 0
        model = saturated_pos if use_saturation else linear_deadband_pos
    else:
        mask  = pwms < 0
        model = saturated_neg if use_saturation else linear_deadband_neg

    min_points = 3 if use_saturation else 2
    if mask.sum() < min_points:
        return None

    u = pwms[mask].astype(float)
    f = forces[mask].astype(float)

    # Initial guesses: K from peak slope, U_DEAD ~30
    abs_u = np.abs(u)
    abs_f = np.abs(f)
    k_guess = float(abs_f.max() / max(abs_u.max() - 30.0, 1.0))
    p0 = [k_guess, 30.0]
    bounds_lo, bounds_hi = [0.0, 0.0], [10.0, 200.0]
    if use_saturation:
        p0.append(0.0)
        bounds_lo.append(0.0)
        bounds_hi.append(1.0)

    try:
        popt, _ = curve_fit(model, u, f, p0=p0,
                            bounds=(bounds_lo, bounds_hi), maxfev=5000)
    except RuntimeError:
        return None

    f_pred = model(u, *popt)
    out = {
        "K":         float(popt[0]),
        "U_DEAD":    float(popt[1]),
        "r2":        r_squared(f, f_pred),
        "n_points":  int(mask.sum()),
        "u":         u,
        "f":         f,
        "f_pred":    f_pred,
        "residuals": f - f_pred,
    }
    if use_saturation:
        out["SAT"] = float(popt[2])
    return out


# ── Reporting ────────────────────────────────────────────────────────
def print_fit_block(label, params, use_saturation):
    if params is None:
        print(f"  {label}: NO DATA")
        print(f"    Tip: edit TEST_PWM in test_accel.py to include this sign")
        print(f"         (e.g. [-255,-200,-150,-100, 100,150,200,255]) and re-run.")
        return
    print(f"  {label}  ({params['n_points']} points,  R^2 = {params['r2']:.4f})")
    print(f"    K      = {params['K']:.6f} N/PWM")
    print(f"    U_DEAD = {params['U_DEAD']:.2f}")
    if use_saturation:
        print(f"    SAT    = {params['SAT']:.6e}")


def check_symmetry(p_pos, p_neg):
    if p_pos is None or p_neg is None:
        return
    k_pos, k_neg = p_pos["K"], p_neg["K"]
    k_rel = abs(k_pos - k_neg) / max(k_pos, k_neg)
    u_abs = abs(p_pos["U_DEAD"] - p_neg["U_DEAD"])
    print()
    print("  ── Symmetry check ───────────────────────────────")
    k_verdict = "symmetric (within 10%)"      if k_rel < 0.10 else \
                "mild asymmetry (10-20%)"     if k_rel < 0.20 else \
                "ASYMMETRIC (> 20%)"
    u_verdict = "symmetric (within 10 PWM)"   if u_abs < 10  else \
                "ASYMMETRIC"
    print(f"  K_POS = {k_pos:.4f}  vs  K_NEG = {k_neg:.4f}")
    print(f"    relative diff = {k_rel*100:.1f}%   [{k_verdict}]")
    print(f"  U_DEAD_POS = {p_pos['U_DEAD']:.1f}  vs  "
          f"U_DEAD_NEG = {p_neg['U_DEAD']:.1f}")
    print(f"    absolute diff = {u_abs:.1f} PWM   [{u_verdict}]")


def print_sanity_check(p_pos, p_neg):
    f_stall = GEARBOX_TORQUE_NM / PULLEY_RADIUS_M
    print()
    print("  ── Sanity check ─────────────────────────────────")
    print(f"  Theoretical stall force at the cart:")
    print(f"    {GEARBOX_TORQUE_NM} N.m / {PULLEY_RADIUS_M*1000:.0f} mm "
          f"= {f_stall:.0f} N  (gearbox limit, no losses)")
    print(f"    realistic ~60% efficiency           ~ {f_stall*0.6:.0f} N")
    print(f"    expected F_max(u=255) in the 50-250 N range")
    for label, p in [("Positive", p_pos), ("Negative", p_neg)]:
        if p is None:
            continue
        f_max = p["K"] * (255 - p["U_DEAD"])
        verdict = "OK" if 50 < f_max < 250 else \
                  "TOO LOW (belt slip? undervolted? PWM not reaching motor?)" \
                  if f_max < 50 else \
                  "TOO HIGH (wrong mass? encoder calibration?)"
        print(f"  {label} F_max(u=255) = {f_max:6.1f} N   [{verdict}]")


def emit_sim_snippet(p_pos, p_neg, use_saturation):
    print()
    print("  ══ Paste into notebooks/01_simulation_environment.ipynb ══")
    print()
    if p_pos:
        print(f"    K_POS      = {p_pos['K']:.6f}")
        print(f"    U_DEAD_POS = {p_pos['U_DEAD']:.2f}")
        if use_saturation:
            print(f"    SAT_POS    = {p_pos['SAT']:.6e}")
    if p_neg:
        print(f"    K_NEG      = {p_neg['K']:.6f}")
        print(f"    U_DEAD_NEG = {p_neg['U_DEAD']:.2f}")
        if use_saturation:
            print(f"    SAT_NEG    = {p_neg['SAT']:.6e}")
    print()
    print("    def pwm_to_force(u):")
    print("        \"\"\"Map PWM (-255..255) to net cart force [N].\"\"\"")
    print("        if u > U_DEAD_POS:")
    if use_saturation:
        print("            d = u - U_DEAD_POS")
        print("            return max(0.0, K_POS * d - SAT_POS * d * d)")
    else:
        print("            return K_POS * (u - U_DEAD_POS)")
    print("        elif u < -U_DEAD_NEG:")
    if use_saturation:
        print("            d = u + U_DEAD_NEG")
        print("            return min(0.0, K_NEG * d + SAT_NEG * d * d)")
    else:
        print("            return K_NEG * (u + U_DEAD_NEG)")
    print("        else:")
    print("            return 0.0")
    print()
    print("  Reminder: set F_CART = 0 in cell-2; friction is absorbed into K/U_DEAD.")


# ── Plotting ─────────────────────────────────────────────────────────
def make_plot(pwms, forces, force_stds, p_pos, p_neg,
              use_saturation, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: data + fits
    ax = axes[0]
    ax.errorbar(pwms, forces, yerr=force_stds, fmt='o', capsize=4,
                color='black', label='Measured (median +/- std)')
    if p_pos:
        u_fit = np.linspace(0, max(255, p_pos["u"].max()), 200)
        if use_saturation:
            f_fit = saturated_pos(u_fit, p_pos["K"], p_pos["U_DEAD"], p_pos["SAT"])
        else:
            f_fit = linear_deadband_pos(u_fit, p_pos["K"], p_pos["U_DEAD"])
        ax.plot(u_fit, f_fit, '-', color='red', linewidth=2,
                label=f'Fit u>0  (R^2 = {p_pos["r2"]:.3f})')
    if p_neg:
        u_fit = np.linspace(min(-255, p_neg["u"].min()), 0, 200)
        if use_saturation:
            f_fit = saturated_neg(u_fit, p_neg["K"], p_neg["U_DEAD"], p_neg["SAT"])
        else:
            f_fit = linear_deadband_neg(u_fit, p_neg["K"], p_neg["U_DEAD"])
        ax.plot(u_fit, f_fit, '-', color='blue', linewidth=2,
                label=f'Fit u<0  (R^2 = {p_neg["r2"]:.3f})')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('PWM duty cycle')
    ax.set_ylabel('Net force F_net [N]')
    ax.set_title('Force vs PWM  (linear with deadband)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Right: residuals
    ax = axes[1]
    if p_pos:
        ax.scatter(p_pos["u"], p_pos["residuals"], color='red',
                   label=f'u > 0  (n={p_pos["n_points"]})')
    if p_neg:
        ax.scatter(p_neg["u"], p_neg["residuals"], color='blue',
                   label=f'u < 0  (n={p_neg["n_points"]})')
    ax.axhline(0, color='gray', linewidth=1)
    ax.set_xlabel('PWM duty cycle')
    ax.set_ylabel('Residual = measured - fitted [N]')
    ax.set_title('Residuals  (systematic trend = bad model, try --saturation)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"\n  Plot saved: {os.path.basename(out_png)}")
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None,
                        help="accel_*.csv filename (default: newest in this folder)")
    parser.add_argument("--m_total", type=float,
                        default=M_CART_DEFAULT + M_POLE_DEFAULT,
                        help="m_cart + m_pole (kg)")
    parser.add_argument("--saturation", action="store_true",
                        help="Add quadratic saturation term per direction "
                             "(use only if linear residuals show systematic "
                             "trend at high PWM)")
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    folder   = os.path.dirname(os.path.abspath(__file__))
    csv_path = (os.path.join(folder, args.file) if args.file
                else find_latest_csv(folder))
    if not csv_path or not os.path.exists(csv_path):
        print(f"  ERROR: no accel CSV found in {folder}")
        return

    print("=" * 66)
    print("  Force Curve Fit  (asymmetric Coulomb-friction motor model)")
    print("=" * 66)
    print(f"  Source CSV: {os.path.basename(csv_path)}")
    print(f"  m_total:    {args.m_total} kg")
    print(f"  Model:      F = sgn(u) * max(0, K * (|u| - U_DEAD))"
          + ("  + quadratic saturation" if args.saturation else ""))
    print(f"  Fit window: first {FIT_WINDOW_S}s of each trial")
    print()

    trials = load_trials(csv_path)
    pwms, accels, stds = aggregate(trials)
    if len(pwms) == 0:
        print("  ERROR: no usable trials found.")
        return

    forces     = args.m_total * accels
    force_stds = args.m_total * stds

    print(f"  {'PWM':>5}  {'a [m/s^2]':>11}  {'F_net [N]':>10}  "
          f"{'std_F [N]':>10}  trials")
    print(f"  {'-'*56}")
    for pwm, a, f, s in zip(pwms, accels, forces, force_stds):
        n = len(trials[pwm])
        print(f"  {pwm:>5}  {a:>11.4f}  {f:>10.4f}  {s:>10.4f}  {n}")

    p_pos = fit_direction(pwms, forces, +1, args.saturation)
    p_neg = fit_direction(pwms, forces, -1, args.saturation)
    if p_pos is None and p_neg is None:
        print("\n  ERROR: insufficient data for either direction.")
        return

    print()
    print("  ── Fitted parameters ────────────────────────────")
    print_fit_block("Positive direction (u > 0)", p_pos, args.saturation)
    print_fit_block("Negative direction (u < 0)", p_neg, args.saturation)

    check_symmetry(p_pos, p_neg)
    print_sanity_check(p_pos, p_neg)
    emit_sim_snippet(p_pos, p_neg, args.saturation)

    if not args.no_plot:
        out_png = csv_path.replace('.csv', '_force_fit.png')
        make_plot(pwms, forces, force_stds, p_pos, p_neg,
                  args.saturation, out_png)


if __name__ == "__main__":
    main()
