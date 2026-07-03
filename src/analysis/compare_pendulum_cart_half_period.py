"""
analyze_period.py
=================
Plan B visualization: measured pendulum T/2 vs cart T/2 schedule.

Top panel : two curves — measured pendulum half-period / cart scheduled half-period
Bottom panel: difference (ms), positive = pendulum slower than cart (cart switches too early)

Usage:
    python analyze_period.py                          # auto-finds latest CSV
    python analyze_period.py angle_20260410_152730.csv

Edit the cart parameters at the top to match swing_up_dc.py settings.
"""

import csv
import math
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks

# ══════════════════════════════════════════════════════
#  Cart parameters — keep in sync with swing_up_dc.py
# ══════════════════════════════════════════════════════
N_SWINGS       = 33      # total number of half-swings
HALF_PERIOD    = 0.550   # initial T/2 (s)
STEP_PER_SWING = 0.020   # T/2 increment per half-swing after warmup (s)
WARMUP_FRAC    = 1/3     # fraction of swings with fixed T/2

# ══════════════════════════════════════════════════════
#  Physical constants
# ══════════════════════════════════════════════════════
G = 9.81
L = 0.34

# ── Load CSV ──────────────────────────────────────────
def latest_csv(folder):
    files = [f for f in os.listdir(folder)
             if (f.startswith("angle_") or f.startswith("swingup_"))
             and f.endswith(".csv")]
    return sorted(files)[-1] if files else None

folder   = os.path.dirname(os.path.abspath(__file__))
filename = sys.argv[1] if len(sys.argv) > 1 else latest_csv(folder)
if filename is None:
    print("No CSV file found. Please specify a filename.")
    sys.exit(1)
filepath = os.path.join(folder, filename)
print(f"  Reading: {filename}")

t_all, theta_all = [], []
motor_stop_t = None

with open(filepath, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            t_all.append(float(row["t_s"]))
            theta_all.append(float(row["theta_deg"]))
            if motor_stop_t is None and row.get("motor_stop_t", ""):
                motor_stop_t = float(row["motor_stop_t"])
        except (ValueError, KeyError):
            continue

t_all     = np.array(t_all)
theta_all = np.array(theta_all)
swing_all = 180.0 - np.abs(theta_all)

# ── Detect pendulum swing peaks ───────────────────────
# Find local maxima of swing amplitude = pendulum at turning point
MOTOR_START_EST = 3.0

active   = t_all > MOTOR_START_EST
t_a      = t_all[active]
sw_a     = swing_all[active]

dt_med   = float(np.median(np.diff(t_a)))
min_dist = max(1, int(0.35 / dt_med))

peaks_idx, _ = find_peaks(sw_a, height=5.0, distance=min_dist)
pk_t  = t_a[peaks_idx]
pk_sw = sw_a[peaks_idx]

# Half-period = time between consecutive peaks
raw_hp  = np.diff(pk_t)
raw_mid = 0.5 * (pk_t[:-1] + pk_t[1:])
raw_sw  = pk_sw[:-1]

# Remove same-side micro-bounces (T/2 < 0.35 s is unphysical)
ok     = raw_hp > 0.35
hp_t   = raw_mid[ok]
hp_val = raw_hp[ok]
hp_sw  = raw_sw[ok]

# ── Reconstruct cart T/2 timeline ────────────────────
# Estimate cart start: first peak minus one initial T/2
cart_start = pk_t[0] - HALF_PERIOD if len(pk_t) > 0 else MOTOR_START_EST
warmup_n   = int(N_SWINGS * WARMUP_FRAC)

cart_t_centers = []
cart_t2_vals   = []
t_cursor = cart_start
for i in range(1, N_SWINGS + 1):
    t2 = HALF_PERIOD if i <= warmup_n else HALF_PERIOD + (i - warmup_n) * STEP_PER_SWING
    cart_t_centers.append(t_cursor + t2 / 2.0)
    cart_t2_vals.append(t2)
    t_cursor += t2

cart_t_centers = np.array(cart_t_centers)
cart_t2_vals   = np.array(cart_t2_vals)

# ── Difference: interpolate cart T/2 at each pendulum measurement ──
cart_interp = np.interp(hp_t, cart_t_centers, cart_t2_vals,
                        left=cart_t2_vals[0], right=cart_t2_vals[-1])
diff_ms = (hp_val - cart_interp) * 1000   # positive = pendulum slower

# ── Console summary ───────────────────────────────────
T_nat = math.pi * math.sqrt(L / G)
print(f"  Theoretical T/2 (small angle): {T_nat:.3f} s")
print(f"  Peaks detected: {len(pk_t)}   Valid half-periods: {len(hp_val)}")
if motor_stop_t:
    print(f"  Motor stop: {motor_stop_t:.2f} s")

act_mask = hp_t < (motor_stop_t or np.inf)
if act_mask.sum() > 2:
    idx_a = np.where(act_mask)[0]
    n4    = max(1, len(idx_a) // 4)
    print()
    print(f"  {'Phase':<14}  {'Pend T/2':>9}  {'Cart T/2':>9}  {'Diff':>10}")
    print(f"  {'─'*48}")
    for qi, lab in enumerate(["Early (Q1)", "Q2", "Q3", "Late (Q4)"]):
        sl = idx_a[qi*n4:(qi+1)*n4] if qi < 3 else idx_a[qi*n4:]
        if len(sl) == 0:
            continue
        med_p = np.median(hp_val[sl])
        med_c = np.median(cart_interp[sl])
        med_d = np.median(diff_ms[sl])
        flag  = f"pend slow +{med_d:.0f}ms" if med_d > 5 else (
                f"pend fast  {med_d:.0f}ms" if med_d < -5 else "matched")
        print(f"  {lab:<14}  {med_p:>9.3f}s  {med_c:>9.3f}s  {flag:>10}")

# ── Plot ─────────────────────────────────────────────
fig = plt.figure(figsize=(13, 7))
gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0.38)

def vlines(ax):
    ax.axvline(cart_start, color="green", linestyle=":", linewidth=1.2,
               alpha=0.7, label="Motor start")
    if motor_stop_t:
        ax.axvline(motor_stop_t, color="red", linestyle="--", linewidth=1.3,
                   alpha=0.85, label=f"Motor off ({motor_stop_t:.1f}s)")

# ── Top: T/2 comparison ──────────────────────────────
ax1 = fig.add_subplot(gs[0])

ax1.step(cart_t_centers, cart_t2_vals, where="mid",
         color="steelblue", linewidth=2.0, label="Cart T/2 (scheduled)", zorder=3)
ax1.plot(hp_t, hp_val,
         color="darkorange", linewidth=1.3,
         marker="o", markersize=4, zorder=4, label="Pendulum T/2 (measured)")
ax1.axhline(T_nat, color="gray", linestyle="--", linewidth=1.0, alpha=0.7,
            label=f"Theoretical T/2 small-angle = {T_nat:.3f}s")

vlines(ax1)
ax1.set_ylabel("Half-period T/2 (s)", fontsize=11)
ax1.set_title(f"Pendulum T/2 vs Cart T/2  —  {filename}", fontsize=12)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(t_all[0], t_all[-1])
ax1.set_xlabel("Time (s)")

# ── Bottom: difference (ms) ───────────────────────────
ax2 = fig.add_subplot(gs[1])

bar_w  = np.diff(np.append(hp_t, hp_t[-1] + 0.5)) * 0.55
colors = ["crimson" if d > 0 else "steelblue" for d in diff_ms]
ax2.bar(hp_t, diff_ms, width=bar_w, color=colors, alpha=0.75, zorder=3)
ax2.plot(hp_t, diff_ms, color="black", linewidth=0.8, alpha=0.5, zorder=4)

ax2.axhline(0,    color="green",  linewidth=1.8, label="Perfect match (0 ms)")
ax2.axhline( 50,  color="orange", linewidth=0.8, linestyle=":", alpha=0.7, label="±50 ms")
ax2.axhline(-50,  color="orange", linewidth=0.8, linestyle=":", alpha=0.7)

ax2.fill_between(hp_t, diff_ms, 0,
                 where=diff_ms > 0, alpha=0.15, color="crimson",
                 label="Pendulum slower — cart switches too early")
ax2.fill_between(hp_t, diff_ms, 0,
                 where=diff_ms < 0, alpha=0.15, color="steelblue",
                 label="Pendulum faster — cart switches too late")

vlines(ax2)
ax2.set_ylabel("Diff (ms)\nPend T/2 - Cart T/2", fontsize=10)
ax2.set_xlabel("Time (s)", fontsize=11)
ax2.set_title("Period mismatch  (positive = cart switches too early)", fontsize=11)
ax2.legend(loc="upper left", fontsize=8, ncol=2)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(t_all[0], t_all[-1])

out_png = filepath.replace(".csv", "_period_compare.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\n  Plot saved: {os.path.basename(out_png)}")
plt.show()
