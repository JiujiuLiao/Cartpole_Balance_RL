"""
analyze_swingup.py
==================
Visualize and analyze swing-up experiment data.
This analyzer requires an oriented 2pi angle (0=upright, pi=hanging). It is
not valid for new logs from camerapreview_detect_yellow_tape_line_V1.py unless
an external up/down reference has first reconstructed the missing branch.
Works with both:
  - angle_*.csv   (from measure_sigma_theta.py — angle + omega only)
  - swingup_*.csv (from swing_up_energy_dc.py — angle + omega + pwm + pos)

Usage:
    python analyze_swingup.py                        # auto-finds latest CSV
    python analyze_swingup.py angle_20260410_110826.csv
"""

import csv
import math
import sys
import os
import numpy as np

G = 9.81
L = 0.34

# ── Load data ─────────────────────────────────────────────────────────
def latest_csv():
    folder = os.path.dirname(os.path.abspath(__file__))
    files = [f for f in os.listdir(folder)
             if (f.startswith("swingup_") or f.startswith("angle_"))
             and f.endswith(".csv")]
    return sorted(files)[-1] if files else None

default  = latest_csv() or "swingup.csv"
filename = sys.argv[1] if len(sys.argv) > 1 else default
filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
print(f"  Reading: {filename}")

t, theta, omega = [], [], []
pwm, pos        = [], []
motor_stop_t    = None   # time when motor was stopped (from marker)

with open(filepath, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            t.append(float(row["t_s"]))
            theta.append(float(row["theta_deg"]))
            omega.append(float(row.get("omega_rad_s", 0)))
            pwm.append(int(float(row.get("pwm", 0))))
            pos.append(int(float(row.get("cart_pos", 0))))
            # motor_stop_t stored only in first row
            if motor_stop_t is None and row.get("motor_stop_t", ""):
                motor_stop_t = float(row["motor_stop_t"])
        except (ValueError, KeyError):
            continue

t     = np.array(t)
theta = np.array(theta)
omega = np.array(omega)
pwm   = np.array(pwm)
pos   = np.array(pos)

# Detect file type
has_pwm = np.any(pwm != 0)
has_pos = np.any(pos != 0)

# ── Compute energy from angle + omega ─────────────────────────────────
# Filter out wrap-around glitch frames (|ω| physically impossible)
omega_max_physical = math.sqrt(4 * G / L)   # ≈ 10.7 rad/s
omega_clean = np.where(np.abs(omega) > omega_max_physical * 1.2,
                       np.nan, omega)

theta_rad = np.radians(theta)
E         = omega_clean**2 - (2*G/L) * np.cos(theta_rad)
E_ref     = -2*G/L
E_gap     = E - E_ref    # → 0 means enough energy to reach top

# ── Swing amplitude ────────────────────────────────────────────────────
swing = 180.0 - np.abs(theta)   # 0 = hanging, 180 = upright

# ── Summary ────────────────────────────────────────────────────────────
print("=" * 58)
print("  Swing-Up Analysis")
print("=" * 58)
print(f"  File:      {filename}")
print(f"  Duration:  {t[-1]:.1f} s   Samples: {len(t)}")
if motor_stop_t:
    print(f"  Motor stopped at: t = {motor_stop_t:.2f} s")
print()

idx_max = swing.argmax()
print(f"  Max swing from bottom: {swing[idx_max]:.1f}°  "
      f"(= {180-swing[idx_max]:.1f}° from upright,  t={t[idx_max]:.2f}s)")
print(f"  Min E-gap at peak:     {E_gap[idx_max]:.1f} rad²/s²  "
      f"(need ≤ 0 to reach top)")
print()

# Rolling max per 2s window
window = 2.0
t_bins = np.arange(0, t[-1], window)
max_swing_bins = []
for tb in t_bins:
    mask = (t >= tb) & (t < tb + window)
    max_swing_bins.append(swing[mask].max() if mask.any() else 0)

print("  Swing amplitude over time (max per 2s window):")
for tb, ms in zip(t_bins, max_swing_bins):
    bar = "█" * int(ms / 3)
    print(f"    {tb:5.1f}s  {ms:5.1f}°  {bar}")
print()

# ── Plot ──────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_rows = 3 if (has_pwm or has_pos) else 2
    fig = plt.figure(figsize=(12, 4 * n_rows))
    gs  = gridspec.GridSpec(n_rows, 2, figure=fig)

    def add_motor_line(ax):
        if motor_stop_t:
            ax.axvline(motor_stop_t, color="red", linestyle="--",
                       linewidth=1.2, label=f"Motor off ({motor_stop_t:.1f}s)")

    # 1) Angle over time  (full width)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, theta, color="steelblue", linewidth=0.7, label="θ (°)")
    ax1.axhline(0,    color="green", linestyle="--", linewidth=1, label="Upright 0°")
    ax1.axhline( 180, color="gray",  linestyle=":",  linewidth=0.6)
    ax1.axhline(-180, color="gray",  linestyle=":",  linewidth=0.6)
    add_motor_line(ax1)
    ax1.set_ylabel("Pendulum angle θ (°)")
    ax1.set_title("Pendulum Angle Over Time  (0° = upright, ±180° = hanging)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Time (s)")

    # 2) Swing amplitude
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, swing, color="darkorange", linewidth=0.7)
    add_motor_line(ax2)

    # Annotate peak swing
    ax2.axhline(swing[idx_max], color="red", linestyle=":", linewidth=1, alpha=0.6)
    ax2.scatter(t[idx_max], swing[idx_max], color="red", zorder=5, s=60)
    offset_x = min(2.0, (t[-1] - t[idx_max]) * 0.4) if t[idx_max] < t[-1] * 0.7 else -3.0
    offset_y = -15 if swing[idx_max] > 90 else 10
    ax2.annotate(
        f"Max: {swing[idx_max]:.1f}° from bottom\n({180 - swing[idx_max]:.1f}° from upright)",
        xy=(t[idx_max], swing[idx_max]),
        xytext=(t[idx_max] + offset_x, swing[idx_max] + offset_y),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
        color="red", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
    )

    ax2.set_ylabel("Swing from bottom (°)")
    ax2.set_title("Swing Amplitude  (0 = hanging, 180 = upright)")
    ax2.set_xlabel("Time (s)")
    if motor_stop_t:
        ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3) Energy gap (computed from theta + omega)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, E_gap, color="purple", linewidth=0.7, label="E − E_ref")
    ax3.axhline(0, color="green", linestyle="--", linewidth=1,
                label="E = E_ref (just reaches top)")
    add_motor_line(ax3)
    ax3.set_ylabel("E − E_ref  (rad²/s²)")
    ax3.set_title("Energy Gap  (→ 0 means enough energy to reach top)")
    ax3.set_xlabel("Time (s)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    if has_pwm or has_pos:
        # 4) PWM
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(t, pwm, color="crimson", linewidth=0.6)
        ax4.axhline(0, color="black", linewidth=0.5)
        add_motor_line(ax4)
        ax4.set_ylabel("Motor PWM")
        ax4.set_title("Motor Command")
        ax4.set_xlabel("Time (s)")
        ax4.grid(True, alpha=0.3)

        # 5) Cart position
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(t, pos, color="teal", linewidth=0.7)
        ax5.axhline(0, color="black", linewidth=0.5)
        add_motor_line(ax5)
        ax5.set_ylabel("Cart position (pulses)")
        ax5.set_title("Cart Position")
        ax5.set_xlabel("Time (s)")
        ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = filepath.replace(".csv", ".png")
    plt.savefig(out_png, dpi=150)
    print(f"  Plot saved: {os.path.basename(out_png)}")
    plt.show()

except ImportError:
    print("  (pip install matplotlib to see plots)")
