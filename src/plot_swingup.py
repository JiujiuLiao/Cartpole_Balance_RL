"""
plot_swingup.py - Visualize a swing-up + balance run from swing_up_energy_dc.py.
================================================================================
Reads one swingup_*.csv and draws, on a shared time axis:
    1) theta  (deg)         with catch / give-up reference lines
    2) omega  (rad/s)
    3) cart_pos (pulses)    with the +/- boundary lines
    4) pwm                  with the +/- safe ceiling (200) and hard limit (255)
plus a theta-omega phase portrait (colored by time) so you can see the whole
"swing-up spiral -> catch near origin -> fall" story at a glance.

The balance phase (where the loop runs at ~100 Hz, dt ~= 10 ms, vs the 50 Hz /
20 ms swing-up loop) is shaded, so you can immediately tell which parts are
swing-up and which are balance.

Usage:
    python plot_swingup.py ../swingup_20260610_162253.csv
    python plot_swingup.py <path-to-csv> --no-show     # save PNG only

All plot labels are in English.
"""

import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# --- reference values (edit if your config changes) ---
CATCH_DEG   = 10.0     # |theta| < this -> handed off to balance (CATCH_ANGLE_DEG)
GIVEUP_DEG  = 30.0     # |theta| > this -> balance gives up (GIVEUP_ANGLE_DEG)
MAX_PULSES  = 3237     # cart boundary (~0.37 m). From config.MAX_PULSES
PWM_SAFE    = 200      # safe ceiling
PWM_LIMIT   = 255      # hard PWM limit
BALANCE_DT  = 0.015    # s; dt below this = fast (100 Hz) balance loop
MIN_RUN     = 5        # min consecutive fast samples to count as a balance interval


def load(fn):
    cols = {"t_s": [], "theta_deg": [], "omega_rad_s": [],
            "pwm": [], "cart_pos": []}
    with open(fn) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                for k in cols:
                    cols[k].append(float(row[k]))
            except (ValueError, KeyError):
                continue
    return {k: np.array(v) for k, v in cols.items()}


def upright_intervals(t, theta):
    """Contiguous runs where |theta| < CATCH_DEG (pendulum held near upright).
    Robust to the control loop's dt jitter, unlike a dt-based test."""
    near = np.abs(theta) < CATCH_DEG
    intervals, i, n = [], 0, len(near)
    while i < n:
        if near[i]:
            j = i
            while j < n and near[j]:
                j += 1
            if j - i >= MIN_RUN:
                intervals.append((t[i], t[j - 1]))
            i = j
        else:
            i += 1
    return intervals


def summarize(d, intervals):
    t, th = d["t_s"], d["theta_deg"]
    absth = np.abs(th)
    up_s = sum(e - s for s, e in intervals)
    dt = np.diff(t)
    print(f"  rows={len(t)}  duration={t[-1]:.1f}s")
    print(f"  swing-up reached |theta|min = {absth.min():.1f} deg "
          f"({'caught upright' if absth.min() < CATCH_DEG else 'never caught'})")
    print(f"  near-upright (|theta|<{CATCH_DEG:.0f}deg): {len(intervals)} episode(s), "
          f"total ~{up_s:.2f}s  longest ~{max((e - s for s, e in intervals), default=0):.2f}s")
    cmax = np.abs(d["cart_pos"]).max()
    print(f"  cart |pos|max = {cmax:.0f} / {MAX_PULSES} boundary "
          f"({'HIT BOUNDARY' if cmax >= MAX_PULSES * 0.97 else 'within limits'})")
    print(f"  omega range [{d['omega_rad_s'].min():.1f}, {d['omega_rad_s'].max():.1f}] rad/s")
    print(f"  loop dt: median={np.median(dt)*1000:.1f}ms  max={dt.max()*1000:.0f}ms  "
          f"spikes>50ms={int(np.sum(dt > 0.05))}  (real-time jitter)")


def shade(ax, intervals, label_once=[True]):
    for s, e in intervals:
        ax.axvspan(s, e, color="gold", alpha=0.18,
                   label="near upright (|theta|<10deg)" if label_once[0] else None)
        label_once[0] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--no-show", action="store_true", help="save PNG only")
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    d = load(args.csv)
    if len(d["t_s"]) < 5:
        sys.exit("Too few rows parsed — check the CSV columns.")
    t = d["t_s"]
    intervals = upright_intervals(t, d["theta_deg"])

    print("=" * 60)
    print(f"  {Path(args.csv).name}")
    summarize(d, intervals)
    print("=" * 60)

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(5, 1, height_ratios=[2, 2, 2, 2, 3], hspace=0.30)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    axp = fig.add_subplot(gs[4])

    # 1) theta
    ax1.plot(t, d["theta_deg"], lw=0.8, color="C0")
    for y in (CATCH_DEG, -CATCH_DEG):
        ax1.axhline(y, color="green", ls="--", lw=0.8)
    for y in (GIVEUP_DEG, -GIVEUP_DEG):
        ax1.axhline(y, color="red", ls=":", lw=0.8)
    ax1.set_ylabel("theta (deg)\n0=up, +/-180=down")
    ax1.set_title(f"Swing-up + balance:  {Path(args.csv).name}")

    # 2) omega
    ax2.plot(t, d["omega_rad_s"], lw=0.8, color="C1")
    ax2.axhline(0, color="gray", lw=0.6)
    ax2.set_ylabel("omega (rad/s)")

    # 3) cart position
    ax3.plot(t, d["cart_pos"], lw=0.8, color="C2")
    for y in (MAX_PULSES, -MAX_PULSES):
        ax3.axhline(y, color="red", ls="--", lw=0.9)
    ax3.set_ylabel("cart_pos (pulses)")

    # 4) pwm
    ax4.plot(t, d["pwm"], lw=0.8, color="C3")
    for y in (PWM_SAFE, -PWM_SAFE):
        ax4.axhline(y, color="orange", ls="--", lw=0.8)
    for y in (PWM_LIMIT, -PWM_LIMIT):
        ax4.axhline(y, color="red", ls=":", lw=0.8)
    ax4.set_ylabel("pwm")
    ax4.set_xlabel("time (s)")

    for ax in (ax1, ax2, ax3, ax4):
        shade(ax, intervals)
        ax.grid(True, alpha=0.3)
    if intervals:
        ax1.legend(loc="upper right", fontsize=8)

    # 5) phase portrait theta-omega, colored by time
    sc = axp.scatter(d["theta_deg"], d["omega_rad_s"], c=t, s=4,
                     cmap="viridis")
    axp.axvline(0, color="gray", lw=0.6)
    axp.axhline(0, color="gray", lw=0.6)
    for x in (CATCH_DEG, -CATCH_DEG):
        axp.axvline(x, color="green", ls="--", lw=0.8)
    axp.set_xlabel("theta (deg)")
    axp.set_ylabel("omega (rad/s)")
    axp.set_title("Phase portrait (color = time): spiral out = swing-up, "
                  "near origin = catch")
    axp.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=axp, label="time (s)")

    out = str(Path(args.csv).with_suffix("")) + "_plot.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"  [saved] {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
