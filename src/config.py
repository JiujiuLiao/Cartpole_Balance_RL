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
#  Auto-recenter on MotorComm.close()  (2026-06-03)
# ════════════════════════════════════════════════════════════════════════
# When True, MotorComm.close() drives the cart back to encoder 0 before
# disconnecting. Scripts that already do positioning (move_cart2_center.py)
# should pass auto_recenter_on_close=False to opt out (avoid recursion).

AUTO_RECENTER_ON_CLOSE = True
RECENTER_PWM_MAX       = 120    # full-speed PWM when cart is far from center
RECENTER_PWM_FLOOR     = 80     # min PWM near center; must exceed PWM_DEADBAND (22)
RECENTER_SLOW_ZONE     = 300    # pulses — inside this, PWM scales proportionally
RECENTER_TOLERANCE     = 10     # pulses — stop within this many of 0
RECENTER_TIMEOUT_S     = 10.0   # give up after this many seconds


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
Kp = 40.0                       # angle proportional gain (PWM per degree)
                                # 80 (too aggressive, slammed at small θ)
                                # → 10 (too gentle — at θ=8° pwm only +80, cart
                                #   accel ~0.47 m/s², can't fight gravity 1.74 m/s²)
                                # → 40 (2026-06-01): saturates pwm at θ≈6°,
                                #   giving cart full force in the catch zone.
                                #   Earlier "Kp=80 over-aggressive" diagnosis was
                                #   wrong — real problem was sign + catch state.
Kd =  8.0                       # angle derivative   gain (PWM per rad/s)
                                # 2026-06-02: 5 → 8 for stronger damping in
                                # balance D term. Counters the angular velocity
                                # buildup that causes 200-500ms balance fall.
                                # If cart shows high-freq jitter, dial back to 6-7.
Kx =  0.02                      # cart-position pull  gain (PWM per pulse)
                                # 2026-05-29: bumped 0 → 0.02 because swing-up
                                # leaves cart near boundary (~2900 pulses) and
                                # without this balance just lets cart drift
                                # the rest of the way into the wall. With 0.02,
                                # cart_pos=2900 contributes -58 PWM, which
                                # easily dominates the small θ correction.

ANGLE_SIGN = -1   # verified 2026-05-29: +PWM=cart right, +θ=right tilt;
                  # PD needs pwm>0 when θ>0, so the "-Kp·err" formula needs SIGN=-1.
OMEGA_SIGN = +1   # not yet validated independently — flip if Kd makes oscillation worse
CART_SIGN  = +1   # not yet validated — flip if cart drifts away from 0 instead of returning
