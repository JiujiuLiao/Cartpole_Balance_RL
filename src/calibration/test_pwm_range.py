"""
test_pwm_range.py
=================
Sweep PWM and measure steady-state cart speed at each step.

Changes from v2:
  - Proportional return-to-origin (slows down near center, less overshoot)
  - Bidirectional sweep: forward + reverse, results averaged per PWM

Usage:
    python test_pwm_range.py 

Output:
    pwm_range_YYYYMMDD_HHMMSS.csv      — summary: fwd/rev/avg per PWM
    pwm_range_YYYYMMDD_HHMMSS_raw.csv  — raw position samples, both directions
"""

import time
import sys
import os
import csv
from pathlib import Path

# Allow `from motor_comm import ...` / `from config import ...` when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor_comm import MotorComm
from config import (DEFAULT_SERIAL_PORT, SERIAL_BAUD,
                    PULSES_PER_REV, WHEEL_DIAMETER_M, M_PER_PULSE,
                    MAX_PULSES, pulses_for_distance)

# ── Ports ─────────────────────────────────────────
PORT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERIAL_PORT

# ── PWM sweep — dense near deadband ───────────────
DENSE_UP_TO = 150   # PWM threshold: fine step below, coarse step above
STEP_DENSE  = 2     # PWM step in dense region  (0 → DENSE_UP_TO)
STEP_SPARSE = 10    # PWM step in sparse region (DENSE_UP_TO → 255)

# ── Timing ────────────────────────────────────────
STABILIZE_S  = 0.5    # s — motor runs this long before sampling starts (speed settles)
MEASURE_S    = 1.0    # s — window during which encoder samples are collected
SAMPLE_INT_S = 0.05   # s — interval between encoder reads (50 ms → ~20 samples/s)

# ── Standstill detection ──────────────────────────
STANDSTILL_THR = 3    # pulses/s — below this the cart is considered stopped
STANDSTILL_S   = 0.4  # s        — must stay below threshold for this long

# ── Return-to-origin (proportional) ───────────────
RETURN_SLOW_ZONE  = 300   # pulses    — braking zone: proportional scale-down starts here
RETURN_PWM_MAX    = 120   # PWM 0–255 — full-speed return when cart is far from origin
RETURN_PWM_FLOOR  = 80    # PWM 0–255 — minimum return PWM near origin; must be > deadband
                           # update this after first run if motor doesn't move

# ── Safety ────────────────────────────────────────
# Keep this limit in physical units. The old hard-coded 3000-pulse value was
# based on the 40T pulley and is too far out for the current 80T setup.
BOUNDARY_DISTANCE_CM = 27.7  # center-to-limit; matches the configured safe stop
BOUNDARY = pulses_for_distance(BOUNDARY_DISTANCE_CM / 100.0)
if BOUNDARY >= MAX_PULSES:
    raise ValueError(
        f"Test boundary ({BOUNDARY} pulses) must be inside "
        f"MAX_PULSES ({MAX_PULSES} pulses)"
    )

# ── Return to origin (proportional) ───────────────
def return_to_origin(comm, timeout=8.0):
    """
    Drive cart back to encoder 0.
    Speed scales with distance: full speed when far, slows near origin.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        pos = comm.get_position()
        if abs(pos) <= 5:
            break

        dist = abs(pos)
        if dist >= RETURN_SLOW_ZONE:
            pwm = RETURN_PWM_MAX
        else:
            # Linear scale: RETURN_PWM_MAX at SLOW_ZONE → RETURN_PWM_FLOOR at 0
            ratio = dist / RETURN_SLOW_ZONE
            pwm   = int(RETURN_PWM_FLOOR + ratio * (RETURN_PWM_MAX - RETURN_PWM_FLOOR))
            pwm   = max(RETURN_PWM_FLOOR, pwm)

        comm.send("v", pwm if pos < 0 else -pwm)
        time.sleep(0.02)

    comm.stop()
    time.sleep(0.3)


# ── Standstill check ──────────────────────────────
def wait_for_standstill(comm, timeout=3.0):
    """Wait until cart velocity stays below STANDSTILL_THR for STANDSTILL_S."""
    still_since = None
    deadline    = time.time() + timeout
    prev_pos    = comm.get_position()
    prev_t      = time.time()

    while time.time() < deadline:
        time.sleep(0.05)
        now = time.time()
        pos = comm.get_position()
        dt  = now - prev_t
        vel = abs(pos - prev_pos) / dt if dt > 0 else 0   # pulses/s — instantaneous speed estimate

        if vel < STANDSTILL_THR:
            if still_since is None:
                still_since = now
            elif now - still_since >= STANDSTILL_S:
                return True
        else:
            still_since = None

        prev_pos = pos
        prev_t   = now

    return False


# ── PWM list ──────────────────────────────────────
def build_pwm_list():
    dense  = list(range(0, DENSE_UP_TO + 1, STEP_DENSE))
    sparse = list(range(DENSE_UP_TO + STEP_SPARSE, 256, STEP_SPARSE))
    if 255 not in sparse:
        sparse.append(255)
    return dense + sparse


# ── Single-direction sweep ─────────────────────────
def run_sweep_one_direction(comm, direction):
    """
    Sweep all PWM values in one direction (+1 = forward, -1 = reverse).
    Returns (results_list, raw_rows_list).
      results: [{pwm, speed_pps, speed}, ...]   speed in m/s, speed_pps in pulses/s
      raw:     [{direction, pwm, t_s, pos_pulses}, ...]
    """
    label    = "Forward" if direction == +1 else "Reverse"
    results  = []
    raw_rows = []
    pwm_list = build_pwm_list()

    print()
    print(f"  ── {label} sweep ──────────────────────────────────")
    print(f"  {'PWM':>5}  {'Pulses/s':>10}  {'m/s':>8}  note")
    print(f"  {'─'*45}")

    for pwm in pwm_list:

        if pwm == 0:
            results.append({"pwm": 0, "speed_pps": 0.0, "speed": 0.0})
            print(f"  {0:>5}  {0:>10.1f}  {0:>8.4f}")
            continue

        # Return to center and confirm standstill
        return_to_origin(comm)
        comm.send("h", 0)
        time.sleep(0.1)
        wait_for_standstill(comm)

        # Pre-test boundary check
        pos = comm.get_position()
        if abs(pos) > BOUNDARY:
            print(f"  PWM={pwm}: boundary exceeded before test ({pos}) — stopping")
            break

        # Stabilize
        comm.send("v", direction * pwm)
        t_stab_end = time.time() + STABILIZE_S
        hit = False
        while time.time() < t_stab_end:
            if abs(comm.get_position()) > BOUNDARY:
                comm.stop()
                print(f"  PWM={pwm}: boundary hit during stabilize — stopping")
                hit = True
                break
            time.sleep(0.05)
        if hit:
            break

        # Measure: collect raw samples
        raw_this = []
        t0 = time.time()
        t_end = t0 + MEASURE_S

        while time.time() < t_end:
            now = time.time()
            pos = comm.get_position()
            raw_this.append({
                "direction":  direction,
                "pwm":        pwm,
                "t_s":        round(now - t0, 4),
                "pos_pulses": pos,
            })
            if abs(pos) > BOUNDARY:
                comm.stop()
                print(f"  PWM={pwm}: boundary hit during measure — stopping")
                hit = True
                break
            time.sleep(SAMPLE_INT_S)

        comm.stop()
        time.sleep(0.2)

        if hit:
            break

        raw_rows.extend(raw_this)

        # Speed from first/last raw sample
        if len(raw_this) >= 2:
            p0 = raw_this[0]["pos_pulses"]               # pulses — position at window start
            p1 = raw_this[-1]["pos_pulses"]              # pulses — position at window end
            dt = raw_this[-1]["t_s"] - raw_this[0]["t_s"]           # s — elapsed time
            speed_pps = abs((p1 - p0) / dt) if dt > 0 else 0.0     # pulses/s — net displacement / time
        else:
            speed_pps = 0.0

        speed = speed_pps * M_PER_PULSE   # m/s

        note = ""
        if speed_pps > 5 and len(results) > 0 and results[-1]["speed_pps"] < 5:
            note = f"← {label.lower()} motor starts"

        results.append({"pwm": pwm, "speed_pps": round(speed_pps, 1),
                        "speed": round(speed, 4)})
        print(f"  {pwm:>5}  {speed_pps:>10.1f}  {speed:>8.4f}  {note}")

    return results, raw_rows


# ── Combined sweep ─────────────────────────────────
def run_sweep(comm):
    """Forward + reverse sweeps; average speed per PWM."""
    fwd_results, fwd_raw = run_sweep_one_direction(comm, +1)

    print("\n  Returning to center before reverse sweep...")
    return_to_origin(comm)
    wait_for_standstill(comm)

    rev_results, rev_raw = run_sweep_one_direction(comm, -1)

    # Build averaged summary (match by pwm)
    rev_map = {r["pwm"]: r for r in rev_results}
    summary = []
    for f in fwd_results:
        pwm = f["pwm"]
        r   = rev_map.get(pwm, {"speed_pps": 0.0, "speed": 0.0})
        avg_speed_pps = (f["speed_pps"] + r["speed_pps"]) / 2   # pulses/s
        avg_speed     = (f["speed"]     + r["speed"])     / 2   # m/s
        summary.append({
            "pwm":           pwm,
            "fwd_speed":     f["speed"],           # m/s
            "rev_speed":     r["speed"],           # m/s
            "avg_speed":     round(avg_speed, 4),  # m/s
            "fwd_speed_pps": f["speed_pps"],       # pulses/s
            "rev_speed_pps": r["speed_pps"],       # pulses/s
            "avg_speed_pps": round(avg_speed_pps, 1),  # pulses/s
        })

    raw_rows = fwd_raw + rev_raw
    return summary, raw_rows


# ── Main ──────────────────────────────────────────
def main():
    pwm_list = build_pwm_list()
    print("=" * 60)
    print("  PWM Range Sweep Test  (bidirectional + proportional return)")
    print("=" * 60)
    print(f"  Port:        {PORT}")
    print(f"  PWM steps:   0→{DENSE_UP_TO} step {STEP_DENSE},  "
          f"{DENSE_UP_TO}→255 step {STEP_SPARSE}  ({len(pwm_list)} steps per direction)")
    print(f"  Timing:      {STABILIZE_S}s stabilize + {MEASURE_S}s measure per step")
    print(f"  Return:      proportional  (floor={RETURN_PWM_FLOOR}, "
          f"max={RETURN_PWM_MAX}, slow zone={RETURN_SLOW_ZONE} pulses)")
    print(f"  Encoder:     {PULSES_PER_REV} pulses/rev, "
          f"{WHEEL_DIAMETER_M*1000:.0f}mm pulley")
    print(f"  Boundary:    ±{BOUNDARY_DISTANCE_CM:.1f} cm "
          f"(derived ±{BOUNDARY} pulses)")

    try:
        comm = MotorComm(PORT, SERIAL_BAUD, verbose=True)
    except Exception as e:
        print(f"\n  Failed to connect: {e}")
        return

    input("\n  Press ENTER when cart is near center and free to move...")

    try:
        summary, raw_rows = run_sweep(comm)
    except KeyboardInterrupt:
        print("\n  Interrupted")
        summary, raw_rows = [], []
    finally:
        comm.stop()

    if not summary:
        comm.close()
        return

    # ── Results ───────────────────────────────────
    moving_fwd = [r for r in summary if r["fwd_speed_pps"] > 5]
    moving_rev = [r for r in summary if r["rev_speed_pps"] > 5]
    deadband_fwd = moving_fwd[0]["pwm"] if moving_fwd else "unknown"
    deadband_rev = moving_rev[0]["pwm"] if moving_rev else "unknown"
    max_row      = max(summary, key=lambda r: r["avg_speed"])

    print()
    print("=" * 60)
    print(f"  Deadband (forward): PWM < {deadband_fwd}")
    print(f"  Deadband (reverse): PWM < {deadband_rev}")
    print(f"  Max avg speed:      {max_row['avg_speed']:.4f} m/s  "
          f"at PWM={max_row['pwm']}")
    print(f"  (fwd={max_row['fwd_speed']:.4f}  "
          f"rev={max_row['rev_speed']:.4f})")
    print("=" * 60)

    # ── Save CSVs ─────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    folder    = os.path.dirname(os.path.abspath(__file__))

    summary_file = os.path.join(folder, f"pwm_range_{timestamp}.csv")
    fields = ["pwm", "fwd_speed", "rev_speed", "avg_speed",
              "fwd_speed_pps", "rev_speed_pps", "avg_speed_pps"]
    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    print(f"  Summary saved: pwm_range_{timestamp}.csv")

    if raw_rows:
        raw_file = os.path.join(folder, f"pwm_range_{timestamp}_raw.csv")
        with open(raw_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["direction", "pwm", "t_s", "pos_pulses"])
            writer.writeheader()
            writer.writerows(raw_rows)
        print(f"  Raw saved:     pwm_range_{timestamp}_raw.csv")

    comm.close()


if __name__ == "__main__":
    main()
