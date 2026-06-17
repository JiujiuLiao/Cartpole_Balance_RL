"""
compute_lqr_K.py - Solve LQR gain K for the cart-pole, OFFLINE on PC.
======================================================================
Runs ON: PC with scipy + numpy (NOT on Pi).
Output : 4 numbers to paste into balance_lqr.py.

Usage:
    1. Measure physical parameters in lab tomorrow.
    2. Edit the "PHYSICAL PARAMETERS" block below.
    3. Tune Q diagonal entries to taste (start from Varghese 2017 values).
    4. python3 compute_lqr_K.py
    5. Copy printed K = [k1, k2, k3, k4] into balance_lqr.py.

State vector convention (matches balance_lqr.py exactly):
    x = [theta, theta_dot, x_cart, x_cart_dot]
        theta      rad,   0 = upright, +theta = clockwise tilt (camera POV)
        theta_dot  rad/s
        x_cart     m,     0 = track center, +x = right
        x_cart_dot m/s

Control law: u = -K @ x   (u in Newtons if FORCE_PER_PWM = 1)
"""

import numpy as np
from scipy.linalg import solve_continuous_are


# ======================================================================
# PHYSICAL PARAMETERS  (raw measured values)
# ======================================================================
M_c = 1.014        # cart mass (kg) — measured 2026-05-27
g   = 9.81         # gravity (m/s^2)

# --- Pendulum mass distribution (measured 2026-06-10) ---
# The pendulum is NOT a uniform rod: the mass sits mostly at the two ends, so
# we build the FULL physical model with rotational inertia I (same derivation
# as the reference project tarunkumarnyu/main.cpp), NOT the point-mass
# approximation. All distances are measured FROM THE PIVOT (rotation axis).
# ⚠️ CONFIRM these distances against the real hardware.
L_rod   = 0.34     # rod length (m): pivot -> far end
m_rod   = 0.024    # uniform rod (kg)
m_pivot = 0.053    # connector lump AT the pivot        (r = 0)
m_tip   = 0.052    # weight at the far end              (r = L_rod)
r_pivot = 0.0      # connector distance from pivot (m)
r_tip   = L_rod    # tip-weight distance from pivot (m)

m_p = m_rod + m_pivot + m_tip          # total pendulum mass (kg)

# Center of mass distance from pivot:  l = Σ(m_i·r_i) / Σ m_i
l_p = (m_pivot * r_pivot
       + m_rod  * (L_rod / 2.0)        # uniform rod's CoM is at its middle
       + m_tip  * r_tip) / m_p

# Inertia about the PIVOT
#   point mass:                 I = m·r²
#   uniform rod about its END:  I = (1/3)·m·L²
I_pivot = (m_pivot * r_pivot**2
           + (1.0 / 3.0) * m_rod * L_rod**2
           + m_tip * r_tip**2)

# Inertia about the CoM (parallel-axis theorem):  I_cm = I_pivot − m·l²
I_cm = I_pivot - m_p * l_p**2

# ======================================================================
# LQR WEIGHTS  —  chosen by BRYSON'S RULE (2026-06-17), not guessed.
# ======================================================================
# Bryson's rule: weight = 1 / (max acceptable value)².  Each state/control is
# normalized by "how far it may stray before we're unhappy", so the cost is
# dimensionally consistent and physically meaningful.
#
# 旧值 diag(100,10,10,50)/R=1 是拍脑袋的：Q_x=10 等于"不在乎小车在哪" →
# K_x≈-3.16(几乎不控小车) → 闭环主导极点仅 -0.14(时间常数7s，稳定裕度薄如纸)
# → 现实里噪声/摩擦一扰动就倒。新值把小车真正管起来，裕度→-0.93(好6.6倍)。
#
# max-acceptable values（依据见注释，可按手感微调）:
TH_MAX   = 0.12   # rad (~7°)   倒过这个角基本救不回
THD_MAX  = 1.0    # rad/s       平衡时角速度别超这个
X_MAX    = 0.15   # m           小车最多偏中心15cm(旧控制曾漂到37cm撞墙)
XD_MAX   = 0.20   # m/s         低于 PWM200≈0.23m/s 的速度上限
U_MAX    = 180    # PWM         日常控制别顶到255，留余量
RHO      = 100    # 温和程度总旋钮：↑更温和(增益小、少饱和)；↓更激进(裕度大)
                  #   通过 43ms 延迟 Padé 稳定性检验：ρ=30~300 均稳；ρ=100 选定。
Q = np.diag([1.0/TH_MAX**2, 1.0/THD_MAX**2, 1.0/X_MAX**2, 1.0/XD_MAX**2])
R = np.array([[RHO / U_MAX**2]])


# ======================================================================
# State-space matrices (linearized about upright equilibrium)
# FULL physical model WITH pendulum inertia I_cm — same derivation as the
# reference project tarunkumarnyu/main.cpp (his build_model_and_weights()).
# State order: x = [theta, theta_dot, x_cart, x_cart_dot]
#
#   p = I_cm·(M+m) + M·m·l²              (coupling determinant; his "p")
#   theta_ddot / theta =  m·g·l·(M+m) / p
#   x_ddot     / theta = -m²·g·l² / p
#   theta_ddot / F     = -m·l / p
#   x_ddot     / F     =  (I_cm + m·l²) / p
#
# Sanity check: set I_cm = 0 and every term collapses back to the old
# point-mass model  [ (M+m)g/(M·l), -m·g/M, -1/(M·l), 1/M ]  — same family.
# ======================================================================
M, m, l, I = M_c, m_p, l_p, I_cm
p = I * (M + m) + M * m * l * l

A = np.array([
    [0.0,                       1.0, 0.0, 0.0],
    [m * g * l * (M + m) / p,   0.0, 0.0, 0.0],   # theta_ddot from theta
    [0.0,                       0.0, 0.0, 1.0],
    [-m * m * g * l * l / p,    0.0, 0.0, 0.0],   # x_ddot from theta
])

# B in NEWTONS, then scaled by FORCE_PER_PWM so the resulting K comes out
# directly in PWM units (no extra multiply needed in balance_lqr.py).
FORCE_PER_PWM = 0.01345    # N per PWM. F_eff(PWM=255) ≈ 3.429 N → 3.429/255.
B = np.array([
    [0.0],
    [-m * l / p],            # theta_ddot from F
    [0.0],
    [(I + m * l * l) / p],   # x_ddot from F
]) * FORCE_PER_PWM


# ======================================================================
# Controllability / solve / stability check
# ======================================================================
def main():
    print("=" * 64)
    print("  LQR gain computation for cart-pole")
    print("=" * 64)
    print(f"  M_c = {M_c} kg   m_p = {m_p:.3f} kg   l_p = {l_p:.4f} m (computed)")
    print(f"  I_cm = {I_cm:.6f} kg·m²   I_pivot = {I_pivot:.6f} kg·m²")
    print(f"  FORCE_PER_PWM = {FORCE_PER_PWM}")
    print(f"  Q diag       = {np.diag(Q)}")
    print(f"  R            = {R[0,0]}")
    print()

    # Controllability: rank of [B, AB, A^2 B, A^3 B] must be 4
    ctrl = np.hstack([B, A @ B, A @ A @ B, A @ A @ A @ B])
    rank = np.linalg.matrix_rank(ctrl)
    print(f"  Controllability rank: {rank}/4  "
          f"{'OK' if rank == 4 else 'NOT FULLY CONTROLLABLE — STOP'}")
    if rank != 4:
        return

    # Solve continuous-time ARE
    P = solve_continuous_are(A, B, Q, R)
    K = (np.linalg.inv(R) @ B.T @ P).flatten()

    print(f"\n  K = [{K[0]:+.4f}, {K[1]:+.4f}, {K[2]:+.4f}, {K[3]:+.4f}]")
    print()
    print(f"  Control law: u = -K @ [theta, theta_dot, x, x_dot]")
    print(f"             = -({K[0]:+.3f})·θ "
          f"-({K[1]:+.3f})·θ̇ "
          f"-({K[2]:+.3f})·x "
          f"-({K[3]:+.3f})·ẋ")

    # Closed-loop stability check (ideal, no delay)
    A_cl = A - B @ K.reshape(1, -1)
    eigs = np.linalg.eigvals(A_cl)
    print(f"\n  Closed-loop eigenvalues (no delay):")
    for e in eigs:
        marker = "OK" if e.real < 0 else "UNSTABLE"
        print(f"    {e.real:+.4f} {e.imag:+.4f}j   [{marker}]")
    print(f"\n  Closed-loop stable: {(eigs.real < 0).all()}")
    print(f"  Dominant pole (max Re): {eigs.real.max():+.3f}  "
          f"(更负=裕度更大；-0.14≈薄如纸, <-0.7 较稳)")

    # Delay-robustness check: does it survive the real 43ms glass→motion delay?
    # 1st-order Padé of e^{-sT} as an input delay, augmented closed loop.
    T_DELAY = 0.043
    a = 2.0 / T_DELAY
    BK = B @ K.reshape(1, -1)
    Aug = np.vstack([np.hstack([A + BK, 2 * a * B]),
                     np.hstack([-K.reshape(1, -1), [[-a]]])])
    dmax = np.linalg.eigvals(Aug).real.max()
    print(f"  With {T_DELAY*1000:.0f}ms delay (Padé): max Re = {dmax:+.3f}  "
          f"[{'OK 延迟下仍稳' if dmax < 0 else 'UNSTABLE 延迟下发散'}]")

    # Print copy-paste block for balance_lqr.py
    print()
    print("=" * 64)
    print("  Copy this block into balance_lqr.py:")
    print("=" * 64)
    print(f"K_THETA      = {K[0]:+.4f}    # theta gain")
    print(f"K_THETA_DOT  = {K[1]:+.4f}    # theta_dot gain")
    print(f"K_X          = {K[2]:+.4f}    # cart-position gain")
    print(f"K_X_DOT      = {K[3]:+.4f}    # cart-velocity gain")
    print()


if __name__ == "__main__":
    main()
