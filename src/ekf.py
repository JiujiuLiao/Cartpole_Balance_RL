"""
ekf.py
======
Extended Kalman Filter for the cart-pole pendulum (state = [theta, omega]).

替换现在的「一帧差分 + EMA」估角速度方案。一次性解决:
  - 干净的 omega（用物理关系推断，不是把 theta 噪声差分放大）
  - 抗丢帧（检测失败时用模型继续推，不冻结、不爆冲）
  - 抗坏帧（创新门控 gating，认错的色块帧自动丢弃）
  - 延迟前推（把状态推到"指令生效那一刻"，补偿 ~48ms 死区）

设计依据见 docs/kalman_ekf_notes.md。配套文献:
  - Welch & Bishop, "An Introduction to the Kalman Filter", UNC TR 95-041
  - Särkkä, "Bayesian Filtering and Smoothing"（单摆 EKF 标准案例）
  - probml/dynamax: ekf_ukf_pendulum（状态 [角度,角速度]，与本实现同构）

────────────────────────────────────────────────────────────────────────
模型（角度从"正上方"量，theta=0 顶部，±pi 垂下）:
    theta_dot = omega
    omega_dot = (g/L)*sin(theta) + b*u*cos(theta)
                └─ 重力, 直立不稳根源 ─┘ └─ 电机推力 ─┘

    u = 电机 PWM；b = PWM→角加速度系数（先设 0，标定后填入；见笔记区块4）

测量: z = theta（相机只测角度），H = [1, 0]，线性 → 更新步不用雅可比。
只有过程模型非线性 → 只有 F 用雅可比 → 这就是"一个 EKF 走全程"的由来。
────────────────────────────────────────────────────────────────────────
"""

import math
import numpy as np

try:
    from config import GRAVITY, PENDULUM_L_M
except ImportError:           # 允许脱离项目单独 import / 跑自测
    GRAVITY, PENDULUM_L_M = 9.81, 0.34


def wrap_pi(a):
    """把角度折回 (-pi, pi]。用于跨 ±pi 边界正确算创新/状态。"""
    return math.atan2(math.sin(a), math.cos(a))


class PendulumEKF:
    """
    用法（典型控制循环）:

        ekf = PendulumEKF()
        ekf.reset(theta0=math.pi, omega0=0.0)   # 摆静止挂在底部

        # 每来一帧相机数据:
        ekf.predict(dt, u=last_pwm)             # dt = 用 ts_ns 算的真实间隔
        if got_valid_frame:
            ekf.update(theta_measured)          # 丢帧就跳过这一步
        theta_hat, omega_hat = ekf.state()      # 干净的状态

        # 喂控制器前做延迟前推（把状态推到指令生效那刻）:
        theta_c, omega_c = ekf.forward(horizon=0.048, u=intended_pwm)
    """

    def __init__(self,
                 grav_coef=31.5,     # 重力项系数 = m·g·l/I_pivot = ω₀²  (单位 1/s²)
                                     #   ⚠️ 不是 g/L! g/0.34=28.85 是点质量近似，偏低。
                                     #   实测定值(2026-06-17 单摆自由摆动):
                                     #     FFT ω₀=5.58→31.2, 过零点 ω₀=5.65→31.9 → 取 31.5
                                     #   与惯量算的 30.8 互相印证(差~3%)。
                                     #   等价 g/l_eff, l_eff≈0.31m（不是几何长 0.34）。
                 b=0.0,              # PWM→角加速度系数（先 0，标定后填；见笔记区块4）
                 sigma_a=8.0,        # 过程噪声：模型"不知道的"角加速度 std (rad/s^2)
                                     #   ↑调大=更信相机(跟得紧但抖)；调小=更信模型(平滑但滞后)
                 sigma_theta=0.02,   # 测量噪声：相机角度 std (rad)≈1.1°
                                     #   ↑摆静止录几百帧 theta 取标准差，客观可测
                 gate_sigma=5.0):    # 坏帧门控：创新超过 gate_sigma 个标准差 → 丢弃该测量
        self.w0sq, self.b = grav_coef, b
        self.q_a = sigma_a ** 2     # 过程噪声功率（DWNA 模型，见 _Q）
        self.R = sigma_theta ** 2   # 测量噪声方差（标量，因只测 1 个量）
        self.gate = gate_sigma ** 2

        self.x = np.array([math.pi, 0.0])          # 默认从底部静止开始
        self.P = np.diag([0.5, 4.0])               # 初始很不确定，几步自动收敛
        self.n_rejected = 0                        # 被门控丢弃的帧计数（诊断用）

    # ── 初始化 ──────────────────────────────────────────────────────
    def reset(self, theta0, omega0=0.0, p_theta=0.5, p_omega=4.0):
        self.x = np.array([wrap_pi(theta0), omega0])
        self.P = np.diag([p_theta, p_omega])
        self.n_rejected = 0

    # ── 过程模型 + 雅可比（私有）────────────────────────────────────
    def _f(self, x, dt, u):
        """非线性一步推进 f(x,u)。"""
        th, om = x
        om_dot = self.w0sq * math.sin(th) + self.b * u * math.cos(th)
        om_new = om + om_dot * dt
        return np.array([th + om_new * dt,
                         om_new])

    def _F(self, x, dt, u):
        """f 对状态的雅可比（笔记区块4/7：F[1,0] 自带 cos(theta)）。"""
        th = x[0]
        df2_dth = (self.w0sq * math.cos(th)
                   - self.b * u * math.sin(th)) * dt
        return np.array([[1.0,      dt],
                         [df2_dth,  1.0]])

    def _Q(self, dt):
        """
        过程噪声（离散白噪声加速度 DWNA 模型）。
        把"未建模的角加速度"当随机量，方差 q_a，正确随 dt 缩放：
            Q = [[dt^4/4, dt^3/2],
                 [dt^3/2, dt^2  ]] * q_a
        好处：只有 sigma_a 一个旋钮，且变 dt 时自动调整。
        """
        q = self.q_a
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        return np.array([[dt4 / 4.0, dt3 / 2.0],
                         [dt3 / 2.0, dt2]]) * q

    # ── 预测步 ──────────────────────────────────────────────────────
    def predict(self, dt, u=0.0):
        """闭眼推算：状态前推 + 不确定性增大。丢帧时只调它、不调 update。"""
        if dt <= 0:
            return
        F = self._F(self.x, dt, u)
        self.x = self._f(self.x, dt, u)
        self.x[0] = wrap_pi(self.x[0])
        self.P = F @ self.P @ F.T + self._Q(dt)

    # ── 更新步 ──────────────────────────────────────────────────────
    def update(self, z_theta):
        """
        睁眼修正：用相机 theta 融合。返回 True=采纳，False=被门控丢弃。
        H = [1,0]，所以 S、K 都能标量化，不用矩阵求逆。
        """
        y = wrap_pi(z_theta - self.x[0])     # 创新（跨 ±pi 边界安全）
        S = self.P[0, 0] + self.R            # 创新方差
        if y * y > self.gate * S:            # 坏帧门控（卡方门）
            self.n_rejected += 1
            return False
        K = self.P[:, 0] / S                 # 增益 [P00/S, P10/S]
        self.x = self.x + K * y              # ⭐ K[1]≠0 → theta 的误差也修正 omega
        self.x[0] = wrap_pi(self.x[0])
        self.P = (np.eye(2) - np.outer(K, [1.0, 0.0])) @ self.P
        return True

    def step(self, z_theta, dt, u=0.0):
        """便捷：predict + update 一把过。z_theta=None 表示丢帧（只 predict）。"""
        self.predict(dt, u)
        if z_theta is not None:
            self.update(z_theta)
        return self.state()

    # ── 延迟前推（不改内部状态）────────────────────────────────────
    def forward(self, horizon, u=0.0, max_substep=0.005):
        """
        把"当前估计"用模型往前推 horizon 秒（典型 ~0.048s = glass→motion）。
        用同一套非线性 f，零新模型。返回 (theta_wrapped, omega)。
        不修改滤波器内部状态——只产出喂给控制器的"未来态"。
        u 应填你这一拍打算发的 PWM（已知输入 → 不是盲猜，见笔记区块7）。
        """
        if horizon <= 0:
            return self.state()
        x = self.x.copy()
        n = max(1, int(math.ceil(horizon / max_substep)))   # 大跨度切小步，减积分误差
        h = horizon / n
        for _ in range(n):
            x = self._f(x, h, u)
        return wrap_pi(x[0]), x[1]

    # ── 取值 ────────────────────────────────────────────────────────
    def state(self):
        """返回 (theta_wrapped, omega)。"""
        return wrap_pi(self.x[0]), self.x[1]


# ════════════════════════════════════════════════════════════════════════
#  自测：不接硬件，仿真一段摆 + 加噪，对比 EKF vs 一帧差分谁更准
#  运行:  python ekf.py            （只打印 RMSE）
#         python ekf.py --plot     （需要 matplotlib，英文标注）
# ════════════════════════════════════════════════════════════════════════
def _selftest(plot=False):
    rng = np.random.default_rng(0)
    L, g = PENDULUM_L_M, GRAVITY

    # --- 真值：高分辨率仿真一段稳定摆动（从底部偏 0.3rad 释放，u=0）---
    dt_true = 0.001
    T = 6.0
    th, om = math.pi - 0.3, 0.0
    truth = []
    for _ in range(int(T / dt_true)):
        om += ((g / L) * math.sin(th)) * dt_true
        th += om * dt_true
        truth.append((th, om))
    truth = np.array(truth)

    # --- 模拟相机：120fps 采样 theta，加高斯噪声 ---
    fps = 120
    cam_step = int((1.0 / fps) / dt_true)
    sigma_meas = 0.02
    idx = np.arange(0, len(truth), cam_step)
    t_cam = idx * dt_true
    theta_true = truth[idx, 0]
    omega_true = truth[idx, 1]
    theta_meas = theta_true + rng.normal(0, sigma_meas, size=idx.shape)

    # --- 方法A：一帧差分 + EMA（现状）---
    diff_omega = np.zeros_like(theta_meas)
    alpha, prev, of = 0.55, None, 0.0
    for i, z in enumerate(theta_meas):
        if prev is not None:
            inst = wrap_pi(z - prev) / (1.0 / fps)
            of = alpha * inst + (1 - alpha) * of
        diff_omega[i] = of
        prev = z

    # --- 方法B：EKF（仿真用同一系数 g/L 测滤波器数学，与真值自洽）---
    ekf = PendulumEKF(grav_coef=g / L, b=0.0, sigma_a=8.0, sigma_theta=sigma_meas)
    ekf.reset(theta0=theta_meas[0], omega0=0.0)
    ekf_omega, ekf_theta = np.zeros_like(theta_meas), np.zeros_like(theta_meas)
    for i, z in enumerate(theta_meas):
        ekf.step(z, dt=1.0 / fps, u=0.0)
        ekf_theta[i], ekf_omega[i] = ekf.x[0], ekf.x[1]

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def rmse_ang(a, b):    # 角度差先折回 (-pi,pi]，否则跨 ±pi 边界会算出假的 ~2pi 误差
        d = np.array([wrap_pi(x - y) for x, y in zip(a, b)])
        return float(np.sqrt(np.mean(d ** 2)))

    print("=" * 56)
    print("  EKF self-test (simulated pendulum, no hardware)")
    print("=" * 56)
    print(f"  measurement noise sigma = {sigma_meas} rad ({math.degrees(sigma_meas):.1f} deg)")
    print(f"  omega RMSE  diff+EMA : {rmse(diff_omega, omega_true):.4f} rad/s")
    print(f"  omega RMSE  EKF      : {rmse(ekf_omega,  omega_true):.4f} rad/s")
    improve = (1 - rmse(ekf_omega, omega_true) / rmse(diff_omega, omega_true)) * 100
    print(f"  -> EKF omega is {improve:.0f}% better")
    print(f"  theta RMSE  EKF      : {rmse_ang(ekf_theta, theta_true):.4f} rad "
          f"(measurement raw: {rmse_ang(theta_meas, theta_true):.4f})")

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(t_cam, omega_true, "k", lw=2, label="true omega")
        ax[0].plot(t_cam, diff_omega, "r", alpha=0.6, label="diff + EMA")
        ax[0].plot(t_cam, ekf_omega, "b", label="EKF")
        ax[0].set_ylabel("omega (rad/s)"); ax[0].legend(); ax[0].grid(True)
        ax[1].plot(t_cam, theta_true, "k", lw=2, label="true theta")
        ax[1].plot(t_cam, theta_meas, "r.", ms=3, alpha=0.4, label="camera (noisy)")
        ax[1].plot(t_cam, ekf_theta, "b", label="EKF")
        ax[1].set_ylabel("theta (rad)"); ax[1].set_xlabel("time (s)")
        ax[1].legend(); ax[1].grid(True)
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    import sys
    _selftest(plot="--plot" in sys.argv)
