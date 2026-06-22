"""
compare_omega_filters.py
========================
Judge how accurate the cart-pole's angular-velocity (omega) estimate is, and
which real-time filter gives the best lag/noise trade-off.

Idea: there is no direct omega sensor, so we build a near-truth REFERENCE by
differentiating the recorded angle theta with a NON-CAUSAL Savitzky-Golay
filter (uses past+future frames -> zero phase lag, smooths noise). It can't run
in real time, but offline it is the best yardstick. Then we run three REAL-TIME
estimators on the same theta and measure how far each is from the reference and
how much it LAGS:

  reference   Savitzky-Golay derivative            (truth, zero-lag)
  EKF         PendulumEKF from ekf.py               (THE filter in use now)
  lowpass     1-frame diff + 1-pole low-pass(fc)    (alt baseline)
  kalman      constant-acceleration KF [th,w,a]     (alt baseline, generic model)

Note: the EKF is re-run OFFLINE on the recorded theta with uniform dt and u=0
(no PWM input, no dropouts/gating). So this grades the EKF *algorithm* on this
data, apples-to-apples with the other estimators — not an exact replay of the
real-time run. Pass --grav-coef/--ekf-sigma-a/--ekf-sigma-theta to match the
values you calibrated in balance_lqr.py.

Metrics vs reference:  lag (ms, via cross-correlation), RMS error, jitter.

Honesty: the reference inherits any errors in theta itself (e.g. the angle
mode-switch jumps). Use --t0/--t1 to compare a CLEAN segment (no jumps) for a
trustworthy verdict.

Usage:
    python compare_omega_filters.py ../swingup_20260610_162253.csv
    python compare_omega_filters.py <csv> --t0 16 --t1 23 --lp-fc 10 --kf-q 80
    python compare_omega_filters.py <csv> --no-show     # save PNG only
"""

import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ekf.py lives in the parent src/ dir — make it importable from this subfolder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── load ──────────────────────────────────────────────────────────
def load(fn):
    t, th = [], []
    with open(fn) as f:
        for row in csv.DictReader(f):
            try:
                t.append(float(row["t_s"]))
                th.append(float(row["theta_deg"]))
            except (ValueError, KeyError):
                continue
    t = np.array(t)
    th = np.unwrap(np.radians(np.array(th)))      # rad, unwrapped for diff
    return t, th


# ── real-time estimators ──────────────────────────────────────────
def ekf_omega(theta, dt, grav_coef, sigma_a, sigma_theta):
    """Run the ACTUAL PendulumEKF (ekf.py) — the filter in use — on the theta
    series and return its omega estimate.

    Measurement-only (u=0): this offline tool has no synchronized PWM, and the
    omega lag/jitter is governed by the noise params, not the input term. Feed
    theta in the EKF's convention (0 = upright); wrap is handled internally."""
    from ekf import PendulumEKF
    ekf = PendulumEKF(grav_coef=grav_coef, sigma_a=sigma_a,
                      sigma_theta=sigma_theta, b=0.0)
    ekf.reset(theta0=theta[0], omega0=0.0)
    out = np.zeros_like(theta)
    for k in range(1, len(theta)):
        ekf.step(theta[k], dt=dt, u=0.0)
        out[k] = ekf.x[1]
    rej_frac = ekf.n_rejected / max(1, len(theta) - 1)
    return out, rej_frac


def lowpass_omega(theta, dt, fc):
    """1-frame backward diff + 1-pole low-pass at cutoff fc (Hz)."""
    tau = 1.0 / (2 * np.pi * fc)
    a = dt / (tau + dt)                            # 1-pole coefficient
    out = np.zeros_like(theta)
    for k in range(1, len(theta)):
        raw = (theta[k] - theta[k - 1]) / dt
        out[k] = a * raw + (1 - a) * out[k - 1]
    return out


def kalman_omega(theta, dt, R, q_jerk):
    """Constant-acceleration Kalman filter. State [theta, omega, alpha];
    measurement = theta. Returns the omega estimate."""
    F = np.array([[1, dt, 0.5 * dt * dt],
                  [0, 1, dt],
                  [0, 0, 1.0]])
    # continuous white-noise-jerk process noise
    Q = q_jerk ** 2 * np.array([
        [dt**5 / 20, dt**4 / 8, dt**3 / 6],
        [dt**4 / 8,  dt**3 / 3, dt**2 / 2],
        [dt**3 / 6,  dt**2 / 2, dt]])
    H = np.array([[1.0, 0.0, 0.0]])
    x = np.array([theta[0], 0.0, 0.0])
    P = np.diag([R, 1.0, 10.0])
    I = np.eye(3)
    out = np.zeros_like(theta)
    for k in range(1, len(theta)):
        x = F @ x                                  # predict
        P = F @ P @ F.T + Q
        y = theta[k] - (H @ x)[0]                  # update
        S = (H @ P @ H.T)[0, 0] + R
        K = (P @ H.T).flatten() / S
        x = x + K * y
        P = (I - np.outer(K, H)) @ P
        out[k] = x[1]
    return out


# ── metrics ───────────────────────────────────────────────────────
def lag_ms(est, ref, dt, max_lag=40):
    """Lag of est behind ref, via best cross-correlation shift (ms)."""
    est = est - est.mean()
    ref = ref - ref.mean()
    best_L, best_c = 0, -np.inf
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            a, b = est[L:], ref[:len(ref) - L] if L else ref
        else:
            a, b = est[:L], ref[-L:]
        if len(a) < 10:
            continue
        c = np.corrcoef(a, b)[0, 1]
        if c > best_c:
            best_c, best_L = c, L
    return best_L * dt * 1000.0, best_L


def metrics(est, ref, dt, trim):
    e = est[trim:-trim]
    r = ref[trim:-trim]
    rms = float(np.sqrt(np.mean((e - r) ** 2)))
    lms, L = lag_ms(e, r, dt)
    # jitter = residual std after removing the lag
    if L > 0:
        jit = float(np.std(e[L:] - r[:len(r) - L]))
    elif L < 0:
        jit = float(np.std(e[:L] - r[-L:]))
    else:
        jit = float(np.std(e - r))
    return rms, lms, jit


# ── main ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--t0", type=float, default=None, help="window start (s)")
    ap.add_argument("--t1", type=float, default=None, help="window end (s)")
    ap.add_argument("--grav-coef", type=float, default=31.5,
                    help="EKF gravity coef ω₀² (match balance_lqr GRAV_OVER_LEFF)")
    ap.add_argument("--ekf-sigma-a", type=float, default=8.0,
                    help="EKF process-noise σ_a rad/s² (match balance_lqr)")
    ap.add_argument("--ekf-sigma-theta", type=float, default=0.02,
                    help="EKF meas-noise σ_θ rad (match your measured value)")
    ap.add_argument("--lp-fc", type=float, default=10.0, help="low-pass cutoff Hz")
    ap.add_argument("--kf-R", type=float, default=None, help="meas var rad^2 (auto if unset)")
    ap.add_argument("--kf-q", type=float, default=60.0, help="KF jerk noise std rad/s^3")
    ap.add_argument("--savgol-win", type=int, default=15, help="odd; reference window")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    t, th = load(args.csv)
    if len(t) < 50:
        sys.exit("too few rows")

    # uniform resample (median dt) so Savitzky-Golay & filters are consistent
    dt = float(np.median(np.diff(t)))
    tu = np.arange(t[0], t[-1], dt)
    thu = np.interp(tu, t, th)

    win = args.savgol_win | 1                       # force odd
    ref = savgol_filter(thu, win, 3, deriv=1, delta=dt)     # truth (zero-lag)
    ekfo, ekf_rej = ekf_omega(thu, dt, args.grav_coef, args.ekf_sigma_a, args.ekf_sigma_theta)
    low = lowpass_omega(thu, dt, args.lp_fc)
    R = args.kf_R if args.kf_R is not None else (np.radians(0.5)) ** 2
    kal = kalman_omega(thu, dt, R, args.kf_q)

    # comparison window
    m = np.ones_like(tu, bool)
    if args.t0 is not None:
        m &= tu >= args.t0
    if args.t1 is not None:
        m &= tu <= args.t1
    idx = np.where(m)[0]
    s, e = idx[0], idx[-1] + 1
    trim = win                                       # drop filter/edge transients

    print("=" * 66)
    print(f"  {Path(args.csv).name}   dt={dt*1000:.1f}ms   "
          f"window=[{tu[s]:.1f},{tu[e-1]:.1f}]s   ref=SavGol(win={win})")
    print("=" * 66)
    print(f"  {'method':<14}{'RMS err':>10}{'lag(ms)':>10}{'jitter':>10}   (vs reference)")
    print(f"  {'-'*52}")
    rows = [("EKF (in use)", ekfo), (f"lowpass {args.lp_fc:.0f}Hz", low),
            ("kalman (generic)", kal)]
    for name, est in rows:
        rms, lms, jit = metrics(est[s:e], ref[s:e], dt, trim)
        print(f"  {name:<14}{rms:>10.3f}{lms:>10.1f}{jit:>10.3f}")
    print(f"  {'-'*52}")
    print("  RMS err & jitter in rad/s; lag>0 means it lags the reference.")
    print("  Lower lag AND lower jitter = better. Reference is the zero-lag truth.")
    if ekf_rej > 0.05:
        print(f"  ⚠ EKF gating REJECTED {ekf_rej*100:.0f}% of frames → wrong regime "
              f"(violent swing-up?). Judge the EKF on BALANCE data (near upright),")
        print(f"    or window a calm segment with --t0/--t1. High rejection = the")
        print(f"    EKF ran open-loop here, so its RMS/jitter above are NOT its fault.")
    else:
        print(f"  EKF gating rejected {ekf_rej*100:.0f}% of frames (low = healthy regime).")
    print("=" * 66)

    # plot
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8))
    for ax, (lo, hi, ttl) in zip(
            (a1, a2),
            ((tu[s], tu[e-1], "full window"),
             (tu[s], min(tu[e-1], tu[s] + 4.0), "zoom (first 4 s of window)"))):
        ax.plot(tu, ref, "k", lw=2.2, label="reference (SavGol, zero-lag)")
        ax.plot(tu, ekfo, lw=1.4, label="EKF (in use)")
        ax.plot(tu, low, lw=1.0, label=f"lowpass {args.lp_fc:.0f}Hz")
        ax.plot(tu, kal, lw=1.0, label="kalman (generic)")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("time (s)"); ax.set_ylabel("omega (rad/s)")
        ax.set_title(ttl); ax.grid(True, alpha=0.3)
    a1.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Angular-velocity estimators vs zero-lag reference  —  {Path(args.csv).name}")
    fig.tight_layout()
    out = str(Path(args.csv).with_suffix("")) + "_omega_compare.png"
    fig.savefig(out, dpi=120)
    print(f"  [saved] {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
