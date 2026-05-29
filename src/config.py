"""
Centralized configuration for the cartpole-pid project.

SINGLE SOURCE OF TRUTH for all hardware physics constants and global safety
limits. Algorithm-specific tuning (PID gains, swing-up amplitude, control
loop frequency) stays inside each individual script.

Rule of thumb when adding something here:
  - "Did I need to go to the lab to measure this?"
        → put it here. One file, one change, all scripts updated.
  - "Am I just tuning a knob until this controller feels right?"
        → keep it inside that script. Different controllers tune differently.
"""

import math

# ════════════════════════════════════════════════════════════════════════
#  Hardware physics  (measured — change here, propagates everywhere)
# ════════════════════════════════════════════════════════════════════════

# ── Encoder & drivetrain ─────────────────────────────
# GT2 synchronous belt + 40-tooth pulley directly on motor output shaft.
# WHEEL_DIAMETER_M is the pulley pitch diameter (= 40 × 2mm / π = 25.46mm);
# the "wheel" naming is legacy and just means "drive geometry".
PULSES_PER_REV   = 700          # measured 2026-05-27 (new motor + GT2 40T)
WHEEL_DIAMETER_M = 0.02546      # GT2 40T pulley pitch diameter
M_PER_PULSE      = math.pi * WHEEL_DIAMETER_M / PULSES_PER_REV
# M_PER_PULSE ≈ 0.1143 mm/pulse — cross-validated against
# pwm_range_20260527_150747.csv (speed/pps = 0.0001143 across all PWMs).

# ── Pendulum geometry & masses ───────────────────────
PENDULUM_L_M     = 0.34         # effective rod length (pivot → CoM)
CART_MASS_KG     = 1.014        # cart + motor + holder + wheels (no pole)
PENDULUM_MASS_KG = 0.127        # rod only

# ── World ────────────────────────────────────────────
GRAVITY = 9.81                  # m/s²

# ── Track geometry & safety margin ───────────────────
TRACK_LENGTH_M  = 0.84                          # full physical track length
TRACK_HALF_M    = TRACK_LENGTH_M / 2            # 0.42 m, center → end
SAFETY_MARGIN_M = 0.05                          # never get within this of the end
USABLE_HALF_M   = TRACK_HALF_M - SAFETY_MARGIN_M  # 0.37 m (controllers' boundary)


# ── Conversion helpers ───────────────────────────────
def pulses_for_distance(meters: float) -> int:
    """Convert a track distance (m) to encoder pulses."""
    return round(meters / M_PER_PULSE)


def distance_for_pulses(pulses: int) -> float:
    """Convert encoder pulses to track distance (m)."""
    return pulses * M_PER_PULSE


# Derived AFTER helpers are defined.
MAX_PULSES = pulses_for_distance(USABLE_HALF_M)
# MAX_PULSES ≈ 3238 (USABLE_HALF_M=0.37 m at M_PER_PULSE=0.1143 mm).


# ════════════════════════════════════════════════════════════════════════
#  Global safety limits  (any controller must respect these)
# ════════════════════════════════════════════════════════════════════════

PWM_LIMIT        = 255          # IBT-2 absolute max — DO NOT exceed.
PWM_MAX_SAFE     = 200          # recommended ceiling: cart speed ≈ 0.23 m/s
                                # at this PWM (measured 2026-05-27). Above 200,
                                # cart traverses half-track in <1.5 s — too
                                # little control margin.
PWM_DEADBAND     = 22           # measured 2026-05-27: below |PWM|=22 cart
                                # does not move. Asymmetric (fwd ≈ 103 pps,
                                # rev ≈ 70 pps at PWM=22).

GIVEUP_ANGLE_DEG = 25.0         # ANY controller must abort balance/swing if
                                # |θ| exceeds this — pole has fallen too far.


# ════════════════════════════════════════════════════════════════════════
#  Communications
# ════════════════════════════════════════════════════════════════════════

DEFAULT_SERIAL_PORT = "/dev/ttyUSB0"   # Pi is the host now (2026-05-29); override per-script if needed
SERIAL_BAUD         = 115200
DEFAULT_ZMQ_ADDR    = "tcp://127.0.0.1:9999"   # V3 PUB runs on same Pi (2026-05-29)


# ════════════════════════════════════════════════════════════════════════
#  pi/pi_balance.py gains  (owned by that script only — do not reuse)
# ════════════════════════════════════════════════════════════════════════
# These exist here because pi_balance.py imports them. The newer balance_pid.py
# and balance_lqr.py keep their own gains locally — different controllers
# tune independently. New scripts: do NOT import these.
Kp = 10.0                       # angle proportional gain (PWM per degree)
                                # WAS 80 — saturated PWM at any |θ|>3°, making
                                # balance slam the cart into max force as soon as
                                # caught. 2026-05-29: dropped to 10 (so at |θ|=25°
                                # giveup we hit ~250, just below saturation).
Kd =  5.0                       # angle derivative   gain (PWM per rad/s)
Kx =  0.0                       # cart-position pull  gain (PWM per pulse)

ANGLE_SIGN = -1   # verified 2026-05-29: +PWM=cart right, +θ=right tilt;
                  # PD needs pwm>0 when θ>0, so the "-Kp·err" formula needs SIGN=-1.
OMEGA_SIGN = +1   # not yet validated independently — flip if Kd makes oscillation worse
CART_SIGN  = +1   # not yet validated — flip if cart drifts away from 0 instead of returning
