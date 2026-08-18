"""
compute_lqr_K.py - Solve LQR gain K for the cart-pole, OFFLINE on PC.
======================================================================
Runs ON: PC with scipy + numpy (NOT on Pi).
Output : an offline model calculation.  It does NOT automatically replace
         the active gain used by balance_lqr.py.

Usage:
    1. Measure physical parameters in lab.
    2. Update values in **config.py** (single source of truth).
    3. Tune Q diagonal entries below (LQR-specific, stays local).
    4. python3 compute_lqr_K.py
    5. Compare the result with the current configuration and hardware data;
       do not paste it into balance_lqr.py without model and hardware review.

State vector convention (matches balance_lqr.py exactly):
    x = [theta, theta_dot, x_cart, x_cart_dot]
        theta      rad,   0 = upright, +theta = clockwise tilt (camera POV)
        theta_dot  rad/s
        x_cart     m,     0 = track center, +x = right
        x_cart_dot m/s

Control law: u = -K @ x   (u in Newtons if FORCE_PER_PWM = 1)
"""

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_continuous_are

# Expose src/ (top-level libs).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ======================================================================
# PHYSICAL PARAMETERS  (single source of truth: config.py)
# ======================================================================
# Single-letter aliases below match textbook notation for the derivations.
# To change any value, edit config.py — NOT here.
from config import (
    CART_MASS_KG            as M_c,
    GRAVITY                 as g,
    PENDULUM_L_M            as L_rod,
    PENDULUM_ROD_MASS_KG    as m_rod,
    PENDULUM_PIVOT_MASS_KG  as m_pivot,
    PENDULUM_TIP_MASS_KG    as m_tip,
    PENDULUM_MASS_KG        as m_p,
    FORCE_PER_PWM,
)

# Lumped-mass distances from the pivot
r_pivot = 0.0
r_tip   = L_rod

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
# LQR WEIGHTS - tune these to shape behaviour
# ======================================================================
# Q diagonals = "how much do I care about each state being non-zero?"
#   Bigger -> controller works harder to drive that state to zero.
# R                = "how much do I penalize control effort?"
#   Bigger -> smoother, weaker control. Smaller -> aggressive, noisy.
#
# Varghese 2017 used diag(1, 1, 500, 250) and R=1 for a 2.4 kg cart.
# For a lighter cart (your hardware), reduce x weights proportionally.
Q = np.diag([
    100.0,   # theta       (rad)     — MOST important, drive to upright
    10.0,    # theta_dot   (rad/s) 
    10.0,    # x_cart      (m)       — small drift OK
    50.0,    # x_cart_dot  (m/s)     — HEAVY penalty: at v=0.30 m/s the
                                    #   motor has 0 N left, so LQR must
                                    #   keep cart well below v_max
])
R = np.array([[1.0]])    # scalar; bigger = gentler control


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

# B in NEWTONS, then scaled by FORCE_PER_PWM (from config) so the resulting K
# comes out directly in PWM units (no extra multiply needed in balance_lqr.py).
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

    # Closed-loop stability check
    A_cl = A - B @ K.reshape(1, -1)
    eigs = np.linalg.eigvals(A_cl)
    print(f"\n  Closed-loop eigenvalues:")
    for e in eigs:
        marker = "OK" if e.real < 0 else "UNSTABLE"
        print(f"    {e.real:+.4f} {e.imag:+.4f}j   [{marker}]")
    print(f"\n  Closed-loop stable: {(eigs.real < 0).all()}")

    # Print the offline result.  It is not the runtime-gain source.
    print()
    print("=" * 64)
    print("  OFFLINE MODEL RESULT — DO NOT COPY WITHOUT REVIEW:")
    print("=" * 64)
    print(f"K_THETA      = {K[0]:+.4f}    # theta gain")
    print(f"K_THETA_DOT  = {K[1]:+.4f}    # theta_dot gain")
    print(f"K_X          = {K[2]:+.4f}    # cart-position gain")
    print(f"K_X_DOT      = {K[3]:+.4f}    # cart-velocity gain")
    print()


if __name__ == "__main__":
    main()
