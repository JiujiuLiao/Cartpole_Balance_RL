"""
analyze_pendulum.py  —  PC 上运行
================================
读取 measure_pendulum.py 生成的 CSV，计算：
  - 自然周期 T 和自然频率 ω₀
  - 阻尼系数 ζ（zeta）
  - 对数衰减率 δ

用法：
    python analyze_pendulum.py
    python analyze_pendulum.py --file my_data.csv
"""

import csv
import math
import argparse
import os
import numpy as np

# ── 参数 ──────────────────────────────────────────
DEFAULT_FILE    = "pendulum_data.csv"
MIN_PEAK_HEIGHT = 0.05   # 最小峰值（弧度），过滤噪声，约 3°
MIN_PEAK_DIST   = 0.3    # 两个峰之间最短间隔（秒），防止误检
# ──────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default=DEFAULT_FILE)
args = parser.parse_args()

filepath = os.path.join(os.path.dirname(__file__), args.file)

# ── 读取数据 ──────────────────────────────────────
times  = []
angles = []

with open(filepath, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        times.append(float(row["time_s"]))
        angles.append(float(row["angle_rad"]))

times  = np.array(times)
angles = np.array(angles)

# ── 修复 wrap-around，再转换角度 ──────────────────
# np.unwrap() 消除 ±π 跳变，使角度连续
angles_unwrapped = np.unwrap(angles)

# 以录制开始时的静止角度为参考，计算相对位移
resting_angle = np.mean(angles_unwrapped)   # 取前30帧的均值作为平衡位置
phi = angles_unwrapped - resting_angle

print("=" * 55)
print("  单摆数据分析")
print("=" * 55)
print(f"  数据文件: {filepath}")
print(f"  总时长:   {times[-1]:.2f} 秒")
print(f"  采样点数: {len(times)}")
dt_avg = np.mean(np.diff(times))
print(f"  平均采样间隔: {dt_avg*1000:.1f} ms  (~{1/dt_avg:.0f} Hz)")
print()

# ── 平滑信号（消除相机噪声）────────────────────────
# 用移动平均，窗口 ~125ms（约15帧）
smooth_window = max(5, int(0.125 / dt_avg))
phi_smooth = np.convolve(phi, np.ones(smooth_window)/smooth_window, mode='same')

print(f"  phi 范围: {np.degrees(phi.min()):.2f}° ~ {np.degrees(phi.max()):.2f}°")
print(f"  平滑窗口: {smooth_window} 帧 ({smooth_window * dt_avg * 1000:.0f}ms)")
print()

# ── 寻找峰值（在平滑信号上检测）─────────────────────
peaks_time = []
peaks_amp  = []

for i in range(1, len(phi_smooth) - 1):
    if phi_smooth[i] > phi_smooth[i-1] and phi_smooth[i] > phi_smooth[i+1]:
        if phi_smooth[i] > MIN_PEAK_HEIGHT:
            if not peaks_time or (times[i] - peaks_time[-1]) > MIN_PEAK_DIST:
                peaks_time.append(times[i])
                peaks_amp.append(abs(phi_smooth[i]))
    elif phi_smooth[i] < phi_smooth[i-1] and phi_smooth[i] < phi_smooth[i+1]:
        if phi_smooth[i] < -MIN_PEAK_HEIGHT:
            if not peaks_time or (times[i] - peaks_time[-1]) > MIN_PEAK_DIST:
                peaks_time.append(times[i])
                peaks_amp.append(abs(phi_smooth[i]))

peaks_time = np.array(peaks_time)
peaks_amp  = np.array(peaks_amp)

if len(peaks_time) < 4:
    print(f"  峰值太少（<4个），找到 {len(peaks_time)} 个")
    print(f"  请推摆杆幅度大一些再重新录制")
    exit()

print(f"  找到 {len(peaks_time)} 个峰值")
print()

# ── 计算周期 T ────────────────────────────────────
# 相邻同向峰之间距离 = 一个完整周期
# 相邻峰（正负交替）距离 = 半个周期
half_periods = np.diff(peaks_time)
full_periods = half_periods[::2] * 2   # 每隔一个取，近似完整周期

T   = np.mean(full_periods)
T_std = np.std(full_periods)
omega0 = 2 * math.pi / T

print("── 自然频率 ──────────────────────────────────")
print(f"  周期 T     = {T:.4f} ± {T_std:.4f} 秒")
print(f"  半周期 T/2 = {T/2:.4f} 秒")
print(f"  ω₀        = {omega0:.4f} rad/s")
print(f"  频率 f     = {1/T:.4f} Hz")
print()

# ── 计算阻尼系数 ─────────────────────────────────
# 峰值幅度应按指数衰减：A(t) = A0 * exp(-ζ * ω₀ * t)
# 取对数：ln(A) = ln(A0) - ζ * ω₀ * t
# 线性拟合 ln(A) vs t，斜率 = -ζ * ω₀

log_amp = np.log(peaks_amp)
coeffs  = np.polyfit(peaks_time, log_amp, 1)   # 线性拟合
slope   = coeffs[0]   # = -ζ * ω₀

zeta  = -slope / omega0
delta = -slope * T    # 对数衰减率 δ = ζ * ω₀ * T

print("── 阻尼系数 ──────────────────────────────────")
print(f"  阻尼系数 ζ     = {zeta:.4f}")
print(f"  对数衰减率 δ   = {delta:.4f}")
if zeta < 0.05:
    print("  → 轻阻尼（典型单摆）")
elif zeta < 0.3:
    print("  → 中等阻尼")
else:
    print("  → 重阻尼，摩擦较大")
print()

# ── 用于 swing_up_energy.py 的参数 ───────────────
L_eff = 3 * 9.81 / (2 * omega0**2)   # 从 ω₀ 反推等效摆长（均匀杆假设，见下注）
print("── 给 swing_up_energy.py 用的参数 ───────────")
print(f"  HALF_PERIOD   = {T/2:.3f}   # 给 swing_up.py 用")
print(f"  omega0        = {omega0:.4f}  # 自然频率")
print(f"  DAMPING       = {zeta * omega0:.4f}  # 用于 swing_up_energy.py")
print()

# ── 打印峰值表 ────────────────────────────────────
print("── 峰值明细 ──────────────────────────────────")
print(f"  {'时间(s)':<10} {'幅度(°)':<12} {'拟合幅度(°)'}")
for i, (t, a) in enumerate(zip(peaks_time, peaks_amp)):
    fitted = math.exp(coeffs[1] + slope * t)
    print(f"  {t:<10.3f} {math.degrees(a):<12.2f} {math.degrees(fitted):.2f}")
print()
print("=" * 55)

# ── 可选：画图 ────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # Top plot: angle over time
    ax1.plot(times, np.degrees(phi), color="lightsteelblue", linewidth=0.5, alpha=0.5, label="Raw")
    ax1.plot(times, np.degrees(phi_smooth), color="steelblue", linewidth=1.2, label="Smoothed")
    ax1.scatter(peaks_time, np.degrees(peaks_amp), color="red", zorder=5, label="Peaks")
    ax1.scatter(peaks_time, -np.degrees(peaks_amp), color="red", zorder=5)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angle φ (°)")
    ax1.set_title(f"Pendulum Angle  T={T:.3f}s  ω₀={omega0:.3f} rad/s  ζ={zeta:.4f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom plot: peak amplitude decay (log scale)
    t_fit = np.linspace(peaks_time[0], peaks_time[-1], 200)
    amp_fit = np.exp(coeffs[1] + slope * t_fit)
    ax2.semilogy(peaks_time, peaks_amp, "ro", label="Measured peaks")
    ax2.semilogy(t_fit, amp_fit, "b--", label=f"Exponential fit  ζ={zeta:.4f}")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude (rad, log scale)")
    ax2.set_title("Peak Amplitude Decay")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pendulum_analysis.png", dpi=150)
    print("  Plot saved to pendulum_analysis.png")
    plt.show()

except ImportError:
    print("  (Install matplotlib to generate plots: pip install matplotlib)")
