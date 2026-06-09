"""
plot_vmax.py
============
Visualise velocity and acceleration evolution from a vmax_*.csv saved by
test_vmax.py. Reads the CSV, fits the first-order model and overlays the
fit on the raw numerical derivative.

Usage:
    python plot_vmax.py vmax_20260605_171614.csv          # just show
    python plot_vmax.py vmax_20260605_171614.csv --save   # show + save PNG

The PNG (when saved) lands next to the CSV with the same basename
(e.g. vmax_20260605_171614_plot.png).
"""

import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import M_PER_PULSE


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _first_order(t, V, tau):
    """v(t) = V * (1 - exp(-t/tau))"""
    return V * (1.0 - np.exp(-t / tau))


def _accel_from_fit(t, V, tau):
    """a(t) = (V/tau) * exp(-t/tau)  — derivative of the first-order model."""
    return (V / tau) * np.exp(-t / tau)


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------
def load_run(csv_path):
    """Return (fwd_samples, rev_samples) as lists of (t_s, pos_pulses)."""
    fwd, rev = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            t   = float(row["t_s"])
            pos = int(row["pos_pulses"])
            (fwd if row["direction"] == "fwd"
             else rev).append((t, pos))
    return fwd, rev


# ---------------------------------------------------------------------------
# Analysis: numerical derivatives + first-order fit
# ---------------------------------------------------------------------------
def _moving_avg(arr, window):
    """Replication-padded moving average (no zero-bias at edges)."""
    pad = window // 2
    padded = np.concatenate([[arr[0]] * pad, arr, [arr[-1]] * pad])
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def analyse(samples, pwm_sign):
    ts  = np.array([s[0] for s in samples], dtype=float)
    pos = np.array([s[1] for s in samples], dtype=float)

    # Direction-agnostic displacement (positive = in PWM direction)
    disp_m = (pos - pos[0]) * M_PER_PULSE
    if pwm_sign < 0:
        disp_m = -disp_m

    # Velocity — light smoothing for plotting
    vel_raw    = np.gradient(disp_m, ts)
    vel_smooth = _moving_avg(vel_raw, 5)

    # Acceleration — heavier smoothing because it's d²(pos)/dt²
    accel_raw    = np.gradient(vel_smooth, ts)
    accel_smooth = _moving_avg(accel_raw, 7)

    # First-order fit on RAW velocity (not the smoothed version, to keep
    # the fit unbiased by the smoother)
    V0   = float(max(np.max(vel_smooth) * 1.02, 0.05))
    popt, _ = curve_fit(
        _first_order, ts, vel_raw,
        p0     = [V0, 0.07],
        bounds = ([0.01, 0.001], [1.0, 2.0]),
        maxfev = 5000,
    )
    V_fit, tau_fit = popt

    return {
        "ts":           ts,
        "vel_smooth":   vel_smooth,
        "accel_smooth": accel_smooth,
        "V_fit":        float(V_fit),
        "tau_fit":      float(tau_fit),
        "pwm_sign":     pwm_sign,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _plot_direction(ax_v, ax_a, data, color, label):
    ts = data["ts"]
    V, tau = data["V_fit"], data["tau_fit"]

    # ---- velocity ---------------------------------------------------------
    ax_v.scatter(ts * 1000, data["vel_smooth"],
                 s=10, alpha=0.35, color=color,
                 label=f"{label} measured")
    t_fine = np.linspace(0, ts[-1], 400)
    ax_v.plot(t_fine * 1000, _first_order(t_fine, V, tau),
              color=color, linewidth=2,
              label=f"{label} fit  V={V:.3f} m/s  τ={tau*1000:.1f} ms")
    ax_v.axhline(V, color=color, linestyle="--", alpha=0.35, linewidth=1)

    # ---- acceleration -----------------------------------------------------
    ax_a.scatter(ts * 1000, data["accel_smooth"],
                 s=10, alpha=0.35, color=color,
                 label=f"{label} measured (smoothed)")
    a_fit = _accel_from_fit(t_fine, V, tau)
    ax_a.plot(t_fine * 1000, a_fit,
              color=color, linewidth=2,
              label=f"{label} fit  a(0)={V/tau:.2f} m/s²")

    # τ and 3τ markers (only on the first direction to avoid clutter)
    if label.startswith("fwd"):
        for t_mark, name, frac in [(tau, "τ", 0.37), (3*tau, "3τ", 0.05)]:
            ax_a.axvline(t_mark * 1000, color="gray",
                         linestyle=":", alpha=0.5, linewidth=1)
            y_at = _accel_from_fit(t_mark, V, tau)
            ax_a.annotate(f"{name} = {t_mark*1000:.0f} ms\n({frac*100:.0f}% of a(0))",
                          xy=(t_mark * 1000, y_at),
                          xytext=(t_mark * 1000 + 8, y_at + 0.4),
                          fontsize=8, color="dimgray")


def make_plot(csv_path):
    fwd, rev = load_run(csv_path)
    if not fwd and not rev:
        print(f"  no samples found in {csv_path}")
        return None

    fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if fwd:
        d = analyse(fwd, +1)
        _plot_direction(ax_v, ax_a, d, color="tab:blue",  label="fwd (+255)")
    if rev:
        d = analyse(rev, -1)
        _plot_direction(ax_v, ax_a, d, color="tab:orange", label="rev (-255)")

    ax_v.set_ylabel("velocity  (m/s)")
    ax_v.set_title(f"Cart velocity & acceleration  —  {Path(csv_path).name}")
    ax_v.legend(loc="lower right", fontsize=9)
    ax_v.grid(True, alpha=0.3)

    ax_a.set_xlabel("time after PWM step  (ms)")
    ax_a.set_ylabel("acceleration  (m/s²)")
    ax_a.axhline(0, color="black", linewidth=0.5)
    ax_a.legend(loc="upper right", fontsize=9)
    ax_a.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Plot velocity/accel from vmax CSV.")
    p.add_argument("csv", help="Path to vmax_*.csv")
    p.add_argument("--save", action="store_true",
                   help="Also save a PNG next to the CSV (default: show only).")
    args = p.parse_args()

    fig = make_plot(args.csv)
    if fig is None:
        return

    if args.save:
        out = Path(args.csv).with_name(Path(args.csv).stem + "_plot.png")
        fig.savefig(out, dpi=120)
        print(f"  saved: {out}")

    plt.show()


if __name__ == "__main__":
    main()
