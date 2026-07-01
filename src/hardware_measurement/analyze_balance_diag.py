"""
analyze_balance_diag.py
=======================
Offline diagnosis of a balance run recorded by balance_lqr.py's diagnostic CSV
(balance_diag_YYYYMMDD_HHMMSS.csv). Answers ONE question:

    Is the pole falling because of DELAY, PARAMETERS, or HARDWARE?

It does NOT tune anything. It prints a text report + draws a figure, each panel
mapped to one hypothesis:

  DELAY     -> frame_age / t_delay_dyn distribution + pwm↔theta cross-correlation
  PARAMETER -> innovation (theta_meas - theta_ekf): should be ~zero-mean & small
  HARDWARE  -> in_deadband rate, saturation rate, u_pwm histogram
  LOOP      -> dt_ekf jitter (a stuttering loop destabilizes independent of above)

Plus a GOOD-STATE TRACE: it finds the first moment the pole was near-upright and
slow (a state that SHOULD be recoverable) and prints the next frames so you can
see, row by row, WHY it fell from there.

Usage:
    python analyze_balance_diag.py                    # auto: newest balance_diag_*.csv
    python analyze_balance_diag.py <file.csv>         # a specific file
    python analyze_balance_diag.py --no-plot          # newest, text only

By default (no file given) it grabs the MOST RECENT balance_diag_*.csv in src/
(where balance_lqr.py writes them), so you never edit a filename between runs.

All plot labels are English (project rule).
"""

import sys
import os
import glob
from pathlib import Path

import numpy as np

# config (PWM_DEADBAND, PWM_LIMIT) lives in the parent src/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from config import PWM_DEADBAND, PWM_LIMIT
except Exception:
    PWM_DEADBAND, PWM_LIMIT = 22, 200      # fallbacks if run outside the project

# ── good-state thresholds (what counts as "should have been recoverable") ──
GOOD_ANGLE_DEG = 4.0
GOOD_OMEGA     = 0.6
TRACE_ROWS     = 20


def _pct(x):
    return 100.0 * np.mean(x) if len(x) else 0.0


def load(path):
    """Load the diag CSV into a dict of float arrays (blank -> nan)."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    cols = {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}
    return cols


def report(c):
    n = len(c["t"])
    dur = c["t"][-1] - c["t"][0] if n else 0.0
    print("=" * 64)
    print(f"  balance diag report   ({n} rows, {dur:.2f}s)")
    print("=" * 64)

    # ── LOOP timing ────────────────────────────────────────────────
    dt = c["dt_ekf"] * 1000.0                      # ms
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt):
        print("\n  [LOOP]  control-loop step dt_ekf (ms):")
        print(f"     median {np.median(dt):5.1f}   p95 {np.percentile(dt,95):5.1f}"
              f"   max {dt.max():5.1f}   -> {1000.0/np.median(dt):4.0f} Hz median")
        print(f"     jitter (p95-median) = {np.percentile(dt,95)-np.median(dt):4.1f} ms"
              f"   {'(OK)' if np.percentile(dt,95)-np.median(dt) < 5 else '(STUTTERY — suspect)'}")

    # ── DELAY ──────────────────────────────────────────────────────
    fa = c["frame_age"] * 1000.0
    td = c["t_delay_dyn"] * 1000.0
    print("\n  [DELAY]  frame_age (glass->now) & t_delay_dyn (predict horizon), ms:")
    print(f"     frame_age   median {np.nanmedian(fa):5.1f}   p95 {np.nanpercentile(fa,95):5.1f}"
          f"   max {np.nanmax(fa):5.1f}")
    print(f"     t_delay_dyn median {np.nanmedian(td):5.1f}   p95 {np.nanpercentile(td,95):5.1f}"
          f"   max {np.nanmax(td):5.1f}")
    tau = 1000.0 / np.sqrt(31.5)                    # unstable time constant, ms
    print(f"     tau (1/sqrt(31.5)) = {tau:4.0f} ms;  median delay is "
          f"{100*np.nanmedian(td)/tau:3.0f}% of tau  "
          f"{'(HIGH — delay-limited)' if np.nanmedian(td)/tau > 0.2 else '(ok)'}")

    # ── PARAMETER: innovation ──────────────────────────────────────
    innov = c["theta_meas_deg"] - c["theta_ekf_deg"]
    innov = innov[np.isfinite(innov)]
    if len(innov):
        print("\n  [PARAM]  innovation = theta_meas - theta_ekf (deg):")
        print(f"     mean {np.mean(innov):+5.2f}   std {np.std(innov):4.2f}   "
              f"{'(mean~0 → model/sign OK)' if abs(np.mean(innov)) < 0.5 else '(BIASED → grav_coef/sign suspect)'}")
    gate = _pct(c["gated"] == 1)
    print(f"     gating reject rate = {gate:4.1f}%   final n_rejected = {int(np.nanmax(c['n_rejected']))}"
          f"   {'(low, ok)' if gate < 5 else '(HIGH — vision jumpy or model off)'}")

    # ── HARDWARE: deadband / saturation ────────────────────────────
    db  = _pct(c["in_deadband"] == 1)
    sat = _pct(c["saturated"] == 1)
    print("\n  [HARDWARE]  actuator authority:")
    print(f"     in_deadband (|u_pwm|<{PWM_DEADBAND}) = {db:4.1f}%   "
          f"{'(HIGH — commands too small to move the motor → no fine control)' if db > 20 else '(ok)'}")
    print(f"     saturated (|pwm|>={PWM_LIMIT})       = {sat:4.1f}%   "
          f"{'(HIGH — actuator-limited)' if sat > 15 else '(ok)'}")

    # ── DELAY: pwm <-> theta cross-correlation ─────────────────────
    th = c["theta_ekf_deg"]
    pw = c["pwm"]
    m = np.isfinite(th) & np.isfinite(pw)
    if m.sum() > 20:
        a = th[m] - th[m].mean()
        b = pw[m] - pw[m].mean()
        cc = np.correlate(a, b, mode="full")
        cc /= (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        lags = np.arange(-len(a) + 1, len(a))
        k = np.argmax(np.abs(cc))
        lag = lags[k]
        dt_med = np.median(dt) if len(dt) else 10.0
        print("\n  [DELAY]  pwm<->theta cross-correlation:")
        print(f"     peak |corr|={abs(cc[k]):.2f} at lag {lag} frames "
              f"(~{lag*dt_med:+.0f} ms). Sign of peak: "
              f"{'anti-phase (stabilizing)' if cc[k] < 0 else 'in-phase (LAGGED/positive fb — suspect)'}")

    print()


def good_state_trace(c):
    th = c["theta_meas_deg"]
    om = c["omega_ekf"]
    good = np.isfinite(th) & (np.abs(th) < GOOD_ANGLE_DEG) & (np.abs(om) < GOOD_OMEGA)
    idx = np.where(good)[0]
    print("=" * 64)
    print(f"  GOOD-STATE TRACE   (|theta_meas|<{GOOD_ANGLE_DEG}deg & |omega|<{GOOD_OMEGA})")
    print("=" * 64)
    if len(idx) == 0:
        print("  no good state found — the pole never got near-upright & slow.")
        print("  => it's a SWING-UP/CATCH problem, not a balance-hold problem.\n")
        return
    i0 = idx[0]
    print(f"  first good state at row {i0}, t={c['t'][i0]:.3f}s. Next {TRACE_ROWS} frames:\n")
    print(f"  {'t':>6} {'th_meas':>8} {'th_ekf':>7} {'th_ctrl':>8} {'omega':>6} "
          f"{'u_pwm':>7} {'pwm':>5} {'age':>5} {'db':>3} {'sat':>3}")
    for i in range(i0, min(i0 + TRACE_ROWS, len(c["t"]))):
        print(f"  {c['t'][i]:6.3f} {c['theta_meas_deg'][i]:8.2f} {c['theta_ekf_deg'][i]:7.2f} "
              f"{c['theta_ctrl_deg'][i]:8.2f} {c['omega_ekf'][i]:6.2f} "
              f"{c['u_pwm'][i]:7.1f} {int(c['pwm'][i]):5d} {c['frame_age'][i]*1000:5.0f} "
              f"{int(c['in_deadband'][i]):3d} {int(c['saturated'][i]):3d}")
    print("\n  Read it: PWM correct sign & above deadband & cart responding but pole")
    print("  still diverges -> DELAY.  PWM in deadband/saturated -> HARDWARE.")
    print("  PWM wrong sign/magnitude -> PARAMETER.\n")


def plot(c):
    import matplotlib.pyplot as plt
    t = c["t"]
    fig, ax = plt.subplots(3, 2, figsize=(14, 9))

    # 1. angles
    ax[0, 0].plot(t, c["theta_meas_deg"], ".", ms=2, alpha=0.4, label="theta_meas (raw)")
    ax[0, 0].plot(t, c["theta_ekf_deg"], "-", lw=1, label="theta_ekf")
    ax[0, 0].plot(t, c["theta_ctrl_deg"], "-", lw=1, alpha=0.7, label="theta_ctrl (LQR input)")
    ax[0, 0].axhline(0, color="k", lw=0.5)
    ax[0, 0].set_ylabel("angle (deg)"); ax[0, 0].set_title("Angles"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(True)

    # 2. omega
    ax[0, 1].plot(t, c["omega_meas"], ".", ms=2, alpha=0.3, label="omega_meas (camera)")
    ax[0, 1].plot(t, c["omega_ekf"], "-", lw=1, label="omega_ekf")
    ax[0, 1].set_ylabel("omega (rad/s)"); ax[0, 1].set_title("Angular rate"); ax[0, 1].legend(fontsize=8); ax[0, 1].grid(True)

    # 3. force: u_pwm vs pwm with deadband/limit bands
    ax[1, 0].plot(t, c["u_pwm"], "-", lw=1, alpha=0.7, label="u_pwm (wanted, pre-clamp)")
    ax[1, 0].plot(t, c["pwm"], "-", lw=1, label="pwm (sent)")
    for s in (+1, -1):
        ax[1, 0].axhline(s * PWM_DEADBAND, color="orange", lw=0.7, ls="--")
        ax[1, 0].axhline(s * PWM_LIMIT, color="red", lw=0.7, ls="--")
    ax[1, 0].set_ylabel("PWM"); ax[1, 0].set_title("Force: wanted vs sent (dashed=deadband/limit)")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(True)

    # 4. loop timing
    ax[1, 1].plot(t, c["dt_ekf"] * 1000.0, "-", lw=0.8)
    ax[1, 1].axhline(np.median(c["dt_ekf"] * 1000.0), color="g", lw=1, label="median")
    ax[1, 1].set_ylabel("dt_ekf (ms)"); ax[1, 1].set_title("Loop step (jitter = instability source)")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(True)

    # 5. delay
    ax[2, 0].plot(t, c["frame_age"] * 1000.0, "-", lw=0.8, label="frame_age (glass->now)")
    ax[2, 0].plot(t, c["t_delay_dyn"] * 1000.0, "-", lw=0.8, label="t_delay_dyn (horizon)")
    ax[2, 0].axhline(1000.0 / np.sqrt(31.5), color="k", lw=0.7, ls=":", label="tau")
    ax[2, 0].set_ylabel("ms"); ax[2, 0].set_xlabel("time (s)")
    ax[2, 0].set_title("Delay"); ax[2, 0].legend(fontsize=8); ax[2, 0].grid(True)

    # 6. covariance + gated marks
    ax[2, 1].plot(t, c["P_theta"], "-", lw=0.8, label="P_theta")
    ax[2, 1].plot(t, c["P_omega"], "-", lw=0.8, label="P_omega")
    gt = t[c["gated"] == 1]
    if len(gt):
        ax[2, 1].plot(gt, np.full_like(gt, np.nanmax(c["P_omega"])), "rv", ms=4, label="gated frame")
    ax[2, 1].set_yscale("log"); ax[2, 1].set_ylabel("variance"); ax[2, 1].set_xlabel("time (s)")
    ax[2, 1].set_title("EKF confidence (P) + rejected frames"); ax[2, 1].legend(fontsize=8); ax[2, 1].grid(True)

    plt.tight_layout()
    plt.show()


def newest_diag():
    """Most recent balance_diag_*.csv in src/ (parent of this script)."""
    src_dir = str(Path(__file__).resolve().parent.parent)
    files = glob.glob(os.path.join(src_dir, "balance_diag_*.csv"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def main():
    # any non-flag arg is treated as an explicit path; otherwise use the newest.
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        path = args[0]
    else:
        path = newest_diag()
        if path is None:
            print("no balance_diag_*.csv found in src/ — run a balance first,"
                  " or pass a path explicitly.")
            sys.exit(1)
        print(f"  (auto) newest -> {os.path.basename(path)}")
    if not os.path.exists(path):
        print(f"not found: {path}"); sys.exit(1)
    c = load(path)
    report(c)
    good_state_trace(c)
    if "--no-plot" not in sys.argv:
        try:
            plot(c)
        except Exception as e:
            print(f"  (plot skipped: {e})")


if __name__ == "__main__":
    main()
