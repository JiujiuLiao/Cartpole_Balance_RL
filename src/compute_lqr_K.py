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
# PHYSICAL PARAMETERS - !!! PLACEHOLDERS - REPLACE WITH LAB MEASUREMENTS !!!
# ======================================================================
M_c = 1.50         # cart mass (kg)        — weigh cart + Pi + camera + plate
m_p = 0.050        # pendulum mass (kg)    — weigh whole rod assembly
l_p = 0.20         # pivot-to-CoM distance (m)
                   #   NOTE: NOT the same as the rod length 0.34 m.
                   #   Find by balancing rod on a thin support; mark the
                   #   balance point; measure from there to the pivot.
g   = 9.81         # gravity (m/s^2)

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
    1.0,     # theta       (rad)
    1.0,     # theta_dot   (rad/s)
    50.0,    # x_cart      (m)         — Varghese 500 / lighter cart -> 50
    25.0,    # x_cart_dot  (m/s)       — Varghese 250 / lighter cart -> 25
])
R = np.array([[1.0]])    # scalar; bigger = gentler control


# ======================================================================
# State-space matrices (linearized about upright equilibrium)
# Derivation: Varghese 2017 eqs 17-19; sign-corrected.
# ======================================================================
A = np.array([
    [0.0,                                1.0, 0.0, 0.0],
    [(M_c + m_p) * g / (M_c * l_p),      0.0, 0.0, 0.0],
    [0.0,                                0.0, 0.0, 1.0],
    [-m_p * g / M_c,                     0.0, 0.0, 0.0],
])

# B assumes u is in NEWTONS. If you want K directly in PWM units, scale
# B by your measured FORCE_PER_PWM (Newton per PWM count, typically 0.01 - 0.1).
FORCE_PER_PWM = 1.0    # set to 1.0 -> K is in N-units;
                       # set to ~0.05 (if you measured it) -> K is in PWM units
B = np.array([
    [0.0],
    [-1.0 / (M_c * l_p)],
    [0.0],
    [1.0 / M_c],
]) * FORCE_PER_PWM


# ======================================================================
# Controllability / solve / stability check
# ======================================================================
def main():
    print("=" * 64)
    print("  LQR gain computation for cart-pole")
    print("=" * 64)
    print(f"  M_c = {M_c} kg   m_p = {m_p} kg   l_p = {l_p} m")
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
