"""
fit_ekf_b.py
============
Estimate the EKF input gain  b  (EKF_INPUT_GAIN in balance_lqr.py) from a logged
run, by least-squares regression.

Model (ekf.py):   omega_dot = w0sq*sin(theta) + b*u*cos(theta)
Rearrange:        omega_dot - w0sq*sin(theta)  =  b * (u*cos(theta))
                  └──────── Y (residual) ─────┘     └──── X ────┘
Regress Y on X through the origin → slope = b.

b means: angular acceleration produced per unit PWM (rad/s^2 per PWM), via
"PWM -> cart acceleration -> reaction torque on the pole". The cos(theta) factor
is built into X (max coupling upright, zero when the pole is horizontal).

Method notes:
  - omega and omega_dot are the 1st/2nd derivative of theta. Differentiating raw
    theta twice is hopeless, so we use a NON-CAUSAL Savitzky-Golay derivative
    (zero phase lag, smooths noise) — same trick as compare_omega_filters.py.
  - Use a SWING-UP log: motor pushing + pole swinging gives rich (u, theta)
    variation, which the regression needs. A pure balance log has little range.
  - ⚠️ CONVENTION: the EKF in balance_lqr runs on the NEGATED camera theta
    (theta_meas = -theta_meas). Swing-up CSVs log RAW camera theta. So by default
    we negate theta here (--no-neg to disable) → the b we print is already in the
    EKF's convention and can be pasted straight into EKF_INPUT_GAIN (sign incl.).
  - Below the PWM dead-band the cart does not actually move, so those samples
    have u != 0 but ~zero real force; we drop |pwm| < --min-pwm to avoid biasing
    b toward zero.

Sanity: the physics-derived magnitude is |b| ~ 0.039 (= m*l/p * FORCE_PER_PWM,
from compute_lqr_K.py). The regression should land near that; the SIGN comes out
of the data.

Usage:
    python fit_ekf_b.py ../swingup_20260617_145909.csv
    python fit_ekf_b.py <csv> --t0 2 --t1 20 --min-pwm 30 --savgol-win 15
    python fit_ekf_b.py <csv> --no-show
"""

import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# config lives in the parent src/ dir — make it importable for defaults.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from config import PWM_DEADBAND
except Exception:
    PWM_DEADBAND = 30


def load(fn):
    """Load t_s, theta_deg, pwm columns from a swing-up / balance CSV."""
    t, th, u = [], [], []
    with open(fn) as f:
        for row in csv.DictReader(f):
            try:
                t.append(float(row["t_s"]))
                th.append(float(row["theta_deg"]))
                u.append(float(row["pwm"]))
            except (ValueError, KeyError):
                continue
    t = np.array(t)
    th = np.unwrap(np.radians(np.array(th)))   # rad, unwrapped for derivatives
    u = np.array(u)
    return t, th, u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--grav-coef", type=float, default=31.5,
                    help="EKF gravity coef w0sq = omega0^2 (match balance_lqr)")
    ap.add_argument("--no-neg", action="store_true",
                    help="do NOT negate theta (use if the CSV is already in EKF "
                         "convention; default negates camera->EKF like balance_lqr)")
    ap.add_argument("--min-pwm", type=float, default=float(PWM_DEADBAND),
                    help="drop |pwm| below this (dead-band: no real force)")
    ap.add_argument("--savgol-win", type=int, default=15,
                    help="odd; Savitzky-Golay window for the theta derivatives")
    ap.add_argument("--t0", type=float, default=None, help="window start (s)")
    ap.add_argument("--t1", type=float, default=None, help="window end (s)")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    t, th_cam, u = load(args.csv)
    if len(t) < 50:
        sys.exit("too few rows")

    # uniform resample (median dt) so Savitzky-Golay is consistent
    dt = float(np.median(np.diff(t)))
    tu = np.arange(t[0], t[-1], dt)
    th_cam = np.interp(tu, t, th_cam)
    u = np.interp(tu, t, u)

    # EKF convention: balance_lqr feeds the EKF the NEGATED camera theta.
    sign = 1.0 if args.no_neg else -1.0
    theta = sign * th_cam

    win = args.savgol_win | 1                       # force odd
    omega     = savgol_filter(theta, win, 3, deriv=1, delta=dt)   # rad/s
    omega_dot = savgol_filter(theta, win, 3, deriv=2, delta=dt)   # rad/s^2

    # regression variables
    Y = omega_dot - args.grav_coef * np.sin(theta)   # gravity-removed accel
    X = u * np.cos(theta)                            # the b regressor

    # mask: comparison window, drop dead-band PWM, trim SavGol edge transients
    m = np.ones_like(tu, bool)
    if args.t0 is not None:
        m &= tu >= args.t0
    if args.t1 is not None:
        m &= tu <= args.t1
    m &= np.abs(u) >= args.min_pwm
    m[:win] = False
    m[-win:] = False
    Xm, Ym = X[m], Y[m]
    if len(Xm) < 30:
        sys.exit(f"too few usable samples ({len(Xm)}) after masking — lower "
                 f"--min-pwm or widen --t0/--t1")

    # through-origin slope = b  (the model has no constant term)
    b0 = float(np.dot(Xm, Ym) / np.dot(Xm, Xm))
    # with-intercept fit — diagnostic: a big intercept = unmodeled bias (friction?)
    A = np.vstack([Xm, np.ones_like(Xm)]).T
    (b1, c1), *_ = np.linalg.lstsq(A, Ym, rcond=None)
    # R^2 of the through-origin fit
    resid = Ym - b0 * Xm
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((Ym - Ym.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print("=" * 64)
    print(f"  fit_ekf_b   {Path(args.csv).name}")
    print(f"  dt={dt*1000:.1f}ms  samples used={len(Xm)}  "
          f"theta {'NEGATED (EKF conv)' if not args.no_neg else 'raw'}  "
          f"grav_coef={args.grav_coef}  min_pwm={args.min_pwm:.0f}")
    print("=" * 64)
    print(f"  b (through origin)   = {b0:+.5f}   <- paste into EKF_INPUT_GAIN")
    print(f"  b (with intercept)   = {b1:+.5f}   intercept = {c1:+.4f} rad/s^2")
    print(f"  R^2 (through origin) = {r2:.3f}   (closer to 1 = cleaner fit)")
    print(f"  physics-derived |b|  ~ 0.039  (sanity: same order of magnitude?)")
    if abs(c1) > 2.0:
        print("  ⚠ large intercept → unmodeled bias (friction / grav_coef off / "
              "wrong theta sign). Treat b with caution.")
    if r2 < 0.2:
        print("  ⚠ low R^2 → weak fit. Try a cleaner swing-up segment (--t0/--t1), "
              "another --savgol-win, or record a dedicated PWM-step run.")
    print("=" * 64)

    # plot: the regression scatter + fitted line
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(Xm, Ym, s=6, alpha=0.3, label="samples (|pwm|≥deadband)")
    xs = np.array([Xm.min(), Xm.max()])
    ax.plot(xs, b0 * xs, "r", lw=2, label=f"through-origin  b={b0:+.4f}")
    ax.plot(xs, b1 * xs + c1, "g--", lw=1.2,
            label=f"with intercept  b={b1:+.4f}")
    ax.set_xlabel("X = pwm · cos(theta)")
    ax.set_ylabel("Y = omega_dot − grav_coef·sin(theta)   [rad/s²]")
    ax.set_title(f"EKF input gain b fit — {Path(args.csv).name}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = str(Path(args.csv).with_suffix("")) + "_fit_b.png"
    fig.savefig(out, dpi=120)
    print(f"  [saved] {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
