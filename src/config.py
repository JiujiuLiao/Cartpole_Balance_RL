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
import os

# ════════════════════════════════════════════════════════════════════════
#  Hardware physics  (measured — change here, propagates everywhere)
# ════════════════════════════════════════════════════════════════════════

# ── Encoder & drivetrain ─────────────────────────────
PULLEY_TEETH      = 80          # GT2 pulley tooth count (upgraded 2026-07-22)
BELT_PITCH_M      = 0.002       # GT2 tooth pitch = 2 mm
PULSES_PER_REV    = 700         # measured 2026-05-27 (encoder per OUTPUT-shaft rev, pulley-independent)
WHEEL_DIAMETER_M  = PULLEY_TEETH * BELT_PITCH_M / math.pi
M_PER_PULSE      = math.pi * WHEEL_DIAMETER_M / PULSES_PER_REV

GRAVITY = 9.81                  # m/s²
# ── Track geometry & safety margin ───────────────────
TRACK_LENGTH_M  = 0.52                          # full physical track length
TRACK_HALF_M    = TRACK_LENGTH_M / 2            # 0.26 m, center → end
SAFETY_MARGIN_M = 0.02                          # never get within this of the end
USABLE_HALF_M   = TRACK_HALF_M - SAFETY_MARGIN_M  # 0.24 m (controllers' boundary)


# ── Pendulum geometry & masses ───────────────────────
PENDULUM_L_M     = 0.385         # rod length pivot → far end (m)
CART_MASS_KG     = 1.014        # cart + motor + holder + wheels (no pole)

# Pendulum mass distribution
PENDULUM_ROD_MASS_KG    = 0.100     # uniform rod
PENDULUM_PIVOT_MASS_KG  = 0.057     # connector lump at pivot (r = 0)
PENDULUM_TIP_MASS_KG    = 0.0     # weight at far end       (r = PENDULUM_L_M)
PENDULUM_MASS_KG        = (PENDULUM_ROD_MASS_KG
                           + PENDULUM_PIVOT_MASS_KG
                           + PENDULUM_TIP_MASS_KG)   # = 0.157 kg (derived)

# ── Motor force calibration (used by LQR / RL) ───────
FORCE_PER_PWM = 0.04        # N/PWM, provisional nominal value from three
                            # 80T tests on 2026-08-11; measured c range
                            # approximately 0.0389-0.0443 N/PWM.

# ── Conversion helpers ───────────────────────────────
def pulses_for_distance(meters: float) -> int:
    """Convert a track distance (m) to encoder pulses."""
    return round(meters / M_PER_PULSE)


def distance_for_pulses(pulses: int) -> float:
    """Convert encoder pulses to track distance (m)."""
    return pulses * M_PER_PULSE


# Derived AFTER helpers are defined.  The 52 cm track gives 26 cm from
# encoder center to a mechanical end; keep 2 cm margin for all controllers.
MAX_PULSES = pulses_for_distance(USABLE_HALF_M)  # 0.24 m -> 1050 pulses @ 80T


# ════════════════════════════════════════════════════════════════════════
#  Global safety limits  (any controller must respect these)
# ════════════════════════════════════════════════════════════════════════

PWM_LIMIT        = 255          # IBT-2 absolute max — DO NOT exceed.
                                # (A softer ceiling of ~200 keeps cart speed
                                # ≈ 0.23 m/s and gives more control margin;
                                # tighten PWM_LIMIT here if that becomes needed.)
PWM_DEADBAND     = 52           # measured with 80T pulley: fwd starts at 52,
                                # reverse starts at 50; use the larger value
                                # for one symmetric conservative compensation.

GIVEUP_ANGLE_DEG = 10.0         # ANY controller must abort balance/swing if
                                # |θ| exceeds this — pole has fallen too far.


# ════════════════════════════════════════════════════════════════════════
#  Auto-recenter on MotorComm.close()  (2026-06-03)
# ════════════════════════════════════════════════════════════════════════
# When True, MotorComm.close() drives the cart back to encoder 0 before
# disconnecting. Scripts that already do positioning (move_cart2_center.py)
# should pass auto_recenter_on_close=False to opt out (avoid recursion).

AUTO_RECENTER_ON_CLOSE = True
RECENTER_PWM_MAX       = 120    # full-speed PWM when cart is far from center
RECENTER_PWM_FLOOR     = 80     # min PWM near center; must exceed PWM_DEADBAND (52)
RECENTER_SLOW_ZONE     = 300    # pulses — inside this, PWM scales proportionally
RECENTER_TOLERANCE     = 10     # pulses — stop within this many of 0
RECENTER_TIMEOUT_S     = 10.0   # give up after this many seconds


# ════════════════════════════════════════════════════════════════════════
#  Communications
# ════════════════════════════════════════════════════════════════════════

DEFAULT_SERIAL_PORT = "/dev/ttyUSB0"   # Pi is the host now (2026-05-29); override per-script if needed
SERIAL_BAUD         = 115200
CURRENT_VISION_SCRIPT = "camerapreview_detect_yellow_tape_line_V1.py"
DEFAULT_ZMQ_ADDR      = "tcp://127.0.0.1:9999"  # current publisher runs on same Pi

# The current yellow-tape detector measures an unoriented LINE, not a directed
# pivot-to-tip vector.  Its angle repeats every pi radians and therefore cannot
# distinguish upright from hanging.  Consumers that need an absolute up/down
# state (especially automatic swing-up) must refuse this source or add another
# asymmetric marker/sensor first.
VISION_ANGLE_PERIOD_RAD    = math.pi
VISION_SUPPORTS_UP_DOWN    = False


# ════════════════════════════════════════════════════════════════════════
#  Diagnostics / logging  (single source of truth for output paths)
# ════════════════════════════════════════════════════════════════════════
# All time-stamped run outputs live in <project_root>/data/. Writers and
# analyzers both import from here so they can never drift apart.
_PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR       = os.path.join(_PROJECT_ROOT, "data")

BALANCE_DIAG_DIR = os.path.join(_DATA_DIR, "balance_diag")   # balance_lqr writes, analyze_balance_diag reads
BALANCE_LOGS_DIR = os.path.join(_DATA_DIR, "balance_logs")   # tune_balance_pd writes
SWINGUP_LOGS_DIR = os.path.join(_DATA_DIR, "swingup_logs")   # swing_up_* writes
CALIBRATION_DIR  = os.path.join(_DATA_DIR, "calibration")    # measure_*, test_* write


# ════════════════════════════════════════════════════════════════════════
#  controllers/balance_pd.py gains  (historical exception — do not reuse)
# ════════════════════════════════════════════════════════════════════════
# balance_pd.py imports these because tune_balance_pd.py tweaks them here
# during lab sessions. balance_lqr.py keeps its K locally in its own file —
# different controllers tune independently. New controllers: do NOT reuse
# these names; define your own local gains and cite the tuning date.
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
