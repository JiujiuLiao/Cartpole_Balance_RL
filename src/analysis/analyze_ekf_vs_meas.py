"""
analyze_ekf_vs_meas.py
======================
PURPOSE : Visualize how much the EKF cleans up the raw camera signal, and how
          far delay-compensation shifts the state, for one balance run recorded
          by balance_lqr.py's diagnostic CSV (balance_diag_YYYYMMDD_HHMMSS.csv).
RUNS ON : PC (offline post-processing).
DEPENDS ON : numpy, matplotlib; config.BALANCE_DIAG_DIR (same dir balance_lqr.py
          writes to, so paths can't drift).

Two stacked panels sharing the time axis:
  TOP  (angle,     deg  ) : theta_meas  vs  theta_ekf  vs  theta_ctrl
  BOT  (ang. rate, deg/s) : omega_meas  vs  omega_ekf  vs  omega_ctrl

Read it:
  meas vs ekf  -> how much noise the EKF removed
  ekf  vs ctrl -> how far delay-compensation extrapolated the state forward

Note: omega_* columns in the CSV are stored in rad/s; here they are converted to
deg/s so both panels share degree-based units.

Usage:
    python analyze_ekf_vs_meas.py                 # auto: newest balance_diag_*.csv
    python analyze_ekf_vs_meas.py <file.csv>      # a specific file
    python analyze_ekf_vs_meas.py --no-plot       # text summary only

All plot labels are English (project rule).
"""

import sys
import os
import glob
from pathlib import Path

import numpy as np

# config (BALANCE_DIAG_DIR) lives in the parent src/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from config import BALANCE_DIAG_DIR
except Exception:                          # fallback if run outside the project
    BALANCE_DIAG_DIR = str(Path(__file__).resolve().parent.parent / "balance_diag")

RAD2DEG = 180.0 / np.pi


def load(path):
    """Load the diag CSV into a dict of float arrays (blank -> nan)."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}


def summary(c):
    """Text: how much the EKF/prediction moves the signal vs the raw measurement."""
    t = c["t"]
    print("=" * 60)
    print(f"  ekf-vs-meas   ({len(t)} rows, {t[-1] - t[0]:.2f}s)")
    print("=" * 60)

    d_th_ekf  = c["theta_ekf_deg"]  - c["theta_meas_deg"]
    d_th_ctrl = c["theta_ctrl_deg"] - c["theta_ekf_deg"]
    d_om_ekf  = (c["omega_ekf"] - c["omega_meas"]) * RAD2DEG
    d_om_ctrl = (c["omega_ctrl"] - c["omega_ekf"]) * RAD2DEG

    def line(tag, x):
        x = x[np.isfinite(x)]
        if len(x):
            print(f"     {tag:22} mean {np.mean(x):+7.2f}   std {np.std(x):6.2f}"
                  f"   max|.| {np.max(np.abs(x)):6.2f}")

    print("\n  angle (deg):")
    line("ekf  - meas", d_th_ekf)
    line("ctrl - ekf (delay comp)", d_th_ctrl)
    print("\n  angular rate (deg/s):")
    line("ekf  - meas", d_om_ekf)
    line("ctrl - ekf (delay comp)", d_om_ctrl)
    print()


def plot(c):
    import matplotlib.pyplot as plt
    t = c["t"]
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # ── TOP: angles (deg) ─────────────────────────────────────────────
    ax[0].plot(t, c["theta_meas_deg"], ".", ms=2, alpha=0.4, label="theta_meas (raw camera)")
    ax[0].plot(t, c["theta_ekf_deg"],  "-", lw=1,            label="theta_ekf (filtered)")
    ax[0].plot(t, c["theta_ctrl_deg"], "-", lw=1, alpha=0.8, label="theta_ctrl (EKF + delay comp)")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_ylabel("angle (deg)")
    ax[0].set_title("Angle: raw vs EKF vs delay-compensated")
    ax[0].legend(fontsize=8); ax[0].grid(True)

    # ── BOTTOM: angular rate (deg/s) ──────────────────────────────────
    ax[1].plot(t, c["omega_meas"] * RAD2DEG, ".", ms=2, alpha=0.3, label="omega_meas (raw camera)")
    ax[1].plot(t, c["omega_ekf"]  * RAD2DEG, "-", lw=1,            label="omega_ekf (filtered)")
    ax[1].plot(t, c["omega_ctrl"] * RAD2DEG, "-", lw=1, alpha=0.8, label="omega_ctrl (EKF + delay comp)")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel("angular rate (deg/s)"); ax[1].set_xlabel("time (s)")
    ax[1].set_title("Angular rate: raw vs EKF vs delay-compensated")
    ax[1].legend(fontsize=8); ax[1].grid(True)

    plt.tight_layout()
    plt.show()


def newest_diag():
    """Most recent balance_diag_*.csv in the shared BALANCE_DIAG_DIR (config)."""
    files = glob.glob(os.path.join(BALANCE_DIAG_DIR, "balance_diag_*.csv"))
    return max(files, key=os.path.getmtime) if files else None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        path = args[0]
    else:
        path = newest_diag()
        if path is None:
            print(f"no balance_diag_*.csv found in {BALANCE_DIAG_DIR} — run a"
                  " balance first, or pass a path explicitly.")
            sys.exit(1)
        print(f"  (auto) newest -> {os.path.basename(path)}")
    if not os.path.exists(path):
        print(f"not found: {path}"); sys.exit(1)

    c = load(path)
    summary(c)
    if "--no-plot" not in sys.argv:
        try:
            plot(c)
        except Exception as e:
            print(f"  (plot skipped: {e})")


if __name__ == "__main__":
    main()
