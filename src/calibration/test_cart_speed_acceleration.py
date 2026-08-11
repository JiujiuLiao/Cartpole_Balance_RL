"""
test_cart_speed_acceleration.py
==============================
Unified calibration for the cart's loaded maximum speed and acceleration.

Modes:
    python test_cart_speed_acceleration.py --mode vmax
    python test_cart_speed_acceleration.py --mode accel
    python test_cart_speed_acceleration.py --mode all

The current 52 cm track is handled conservatively:
    - measurement endpoint: +/-18 cm from encoder center
    - emergency boundary:  +/-22 cm from encoder center

The 80T pulley conversion comes from src/config.py. The acceleration burst
also requires the ESP32 firmware's <a,PWM> command and matching firmware-side
safe limit (ACCEL_MAX_PULSES).
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor_comm import MotorComm
from config import (
    DEFAULT_SERIAL_PORT, SERIAL_BAUD,
    PULLEY_TEETH, BELT_PITCH_M, PULSES_PER_REV, M_PER_PULSE,
    MAX_PULSES, PWM_DEADBAND, GRAVITY,
    CART_MASS_KG, PENDULUM_MASS_KG,
    pulses_for_distance, CALIBRATION_DIR,
)


# ── Physical test geometry ──────────────────────────────────────────────
RUNWAY_HALF_DISTANCE_CM = 18.0  # measurement endpoint from encoder center
SAFE_STOP_DISTANCE_CM   = 22.0  # emergency boundary; track half-length is 26 cm
RUNWAY_PULSES    = pulses_for_distance(RUNWAY_HALF_DISTANCE_CM / 100.0)
SAFE_STOP_PULSES = pulses_for_distance(SAFE_STOP_DISTANCE_CM / 100.0)

if not (0 < RUNWAY_PULSES < SAFE_STOP_PULSES < MAX_PULSES):
    raise ValueError(
        "Require 0 < RUNWAY_PULSES < SAFE_STOP_PULSES < MAX_PULSES; "
        f"got {RUNWAY_PULSES}, {SAFE_STOP_PULSES}, {MAX_PULSES}"
    )


# ── Maximum-speed test parameters ────────────────────────────────────────
MAX_SPEED_PWM       = 255
MAX_SPEED_RUN_TIME_S = 1.5
MAX_SPEED_SAMPLE_S  = 0.005
V_MAX_REFERENCE_RPM = 251.0  # reference only; not used as the measured result
PULLEY_CIRCUMFERENCE_M = PULLEY_TEETH * BELT_PITCH_M
V_MAX_THEORETICAL = V_MAX_REFERENCE_RPM * PULLEY_CIRCUMFERENCE_M / 60.0


# ── Acceleration-burst parameters ────────────────────────────────────────
ACCEL_PWM_LEVELS = (120, 160, 200, 255)
ACCEL_RUNS = 5
ACCEL_SAMPLE_US = 1000  # must match firmware sampling grid
RETURN_PWM = 120        # safely above the measured PWM deadband of 52
STANDSTILL_THR = 3      # pulses/s
STANDSTILL_S = 0.4

MASS_TOTAL_KG = CART_MASS_KG + PENDULUM_MASS_KG
OUTPUT_DIR = Path(CALIBRATION_DIR)


# ── Common positioning helpers ────────────────────────────────────────────
def read_position(comm):
    pos = comm.get_position(default=None)
    if pos is None:
        raise RuntimeError("encoder position read timed out")
    return pos


def drive_to(comm, target_pulses, timeout=8.0, tolerance=15):
    """Move to a target that is already known to be inside SAFE_STOP_PULSES."""
    if abs(target_pulses) >= SAFE_STOP_PULSES:
        raise ValueError(f"target {target_pulses} is outside the test boundary")

    deadline = time.time() + timeout
    while time.time() < deadline:
        pos = read_position(comm)
        error = target_pulses - pos
        if abs(error) <= tolerance:
            comm.stop()
            time.sleep(0.25)
            return True
        comm.send_velocity(RETURN_PWM if error > 0 else -RETURN_PWM)
        time.sleep(0.02)

    comm.stop()
    return False


def return_to_center(comm):
    if not drive_to(comm, 0):
        raise RuntimeError("could not return cart to encoder center")


def wait_for_standstill(comm, timeout=3.0):
    still_since = None
    deadline = time.time() + timeout
    previous_pos = read_position(comm)
    previous_t = time.time()

    while time.time() < deadline:
        time.sleep(0.05)
        now = time.time()
        pos = read_position(comm)
        dt = now - previous_t
        velocity = abs(pos - previous_pos) / dt if dt > 0 else 0.0

        if velocity < STANDSTILL_THR:
            if still_since is None:
                still_since = now
            elif now - still_since >= STANDSTILL_S:
                return True
        else:
            still_since = None

        previous_pos, previous_t = pos, now

    return False


def prepare_center(comm):
    """Home the encoder at the user's manually confirmed physical center."""
    comm.stop()
    if not comm.send("h", 0):
        raise RuntimeError("ESP32 did not acknowledge encoder home command")
    time.sleep(0.2)


def prepare_start(comm, target_pulses):
    if not drive_to(comm, target_pulses):
        raise RuntimeError(f"could not reach start position {target_pulses}")
    if not wait_for_standstill(comm):
        raise RuntimeError("cart did not reach standstill before the test")


# ── Maximum speed ─────────────────────────────────────────────────────────
def run_speed_once(comm, direction):
    """Run PWM=255 from one runway endpoint to the opposite endpoint."""
    start = -RUNWAY_PULSES if direction > 0 else RUNWAY_PULSES
    target = RUNWAY_PULSES if direction > 0 else -RUNWAY_PULSES
    label = "fwd" if direction > 0 else "rev"

    prepare_start(comm, start)
    print(f"  {label}: start={start} target={target} PWM={MAX_SPEED_PWM}")

    samples = []
    comm.send_velocity(direction * MAX_SPEED_PWM)
    t0 = time.perf_counter()
    reached_target = False

    while True:
        t = time.perf_counter() - t0
        if t >= MAX_SPEED_RUN_TIME_S:
            break

        pos = read_position(comm)
        samples.append((t, pos))

        if abs(pos) >= SAFE_STOP_PULSES:
            print(f"  ! emergency boundary reached at {pos} pulses")
            break
        if (direction > 0 and pos >= target) or (direction < 0 and pos <= target):
            reached_target = True
            break

        time.sleep(MAX_SPEED_SAMPLE_S)

    comm.stop()
    time.sleep(0.3)
    if not reached_target:
        print("  warning: target endpoint was not reached before the run ended")
    return samples


def _smooth(values, width=5):
    if len(values) < width:
        return values
    half = width // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def _first_order(t, terminal_speed, tau):
    return terminal_speed * (1.0 - np.exp(-t / tau))


def analyze_speed(samples, direction):
    if len(samples) < 10:
        return None

    t = np.asarray([row[0] for row in samples], dtype=float)
    pos = np.asarray([row[1] for row in samples], dtype=float)
    displacement = (pos - pos[0]) * M_PER_PULSE
    if direction < 0:
        displacement = -displacement

    velocity = np.gradient(displacement, t)
    velocity_smooth = _smooth(velocity)

    plateau_mask = (t >= 0.20) & (t <= 0.50)
    if plateau_mask.sum() >= 3:
        v_plateau = float(np.median(velocity_smooth[plateau_mask]))
    else:
        tail = velocity_smooth[max(0, int(0.8 * len(velocity_smooth))):]
        v_plateau = float(np.median(tail))

    v_peak = float(np.max(velocity_smooth))
    fit_ok = False
    v_fit = tau_fit = r_squared = float("nan")
    try:
        popt, _ = curve_fit(
            _first_order, t, velocity_smooth,
            p0=[max(v_peak * 1.02, 0.05), 0.07],
            bounds=([0.01, 0.001], [2.0, 2.0]),
            maxfev=5000,
        )
        v_fit, tau_fit = [float(x) for x in popt]
        model = _first_order(t, *popt)
        ss_res = float(np.sum((velocity_smooth - model) ** 2))
        ss_tot = float(np.sum((velocity_smooth - np.mean(velocity_smooth)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        fit_ok = True
    except Exception:
        pass

    early_mask = t <= 0.05
    if early_mask.sum() >= 3:
        accel_50ms = float(np.polyfit(t[early_mask], velocity_smooth[early_mask], 1)[0])
    else:
        accel_50ms = float("nan")

    return {
        "direction": "fwd" if direction > 0 else "rev",
        "v_plateau": v_plateau,
        "v_peak": v_peak,
        "v_fit": v_fit,
        "tau_s": tau_fit,
        "accel_50ms": accel_50ms,
        "r_squared": r_squared,
        "fit_ok": fit_ok,
    }


def save_speed_csv(rows, timestamp):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"vmax_{timestamp}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run", "direction", "t_s", "pos_pulses", "disp_m"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_speed_test(comm, runs):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw_rows = []
    stats = []

    for run in range(1, runs + 1):
        for direction in (+1, -1):
            samples = run_speed_once(comm, direction)
            result = analyze_speed(samples, direction)
            if result is None:
                print(f"  {('fwd' if direction > 0 else 'rev')} run {run}: insufficient samples")
                continue
            stats.append(result)
            print(
                f"    {result['direction']} run {run}: "
                f"v_plateau={result['v_plateau']:.3f} m/s, "
                f"v_fit={result['v_fit']:.3f} m/s, "
                f"tau={result['tau_s']*1000:.1f} ms, "
                f"R^2={result['r_squared']:.3f}"
            )
            start_pos = samples[0][1]
            sign = direction
            for t, pos in samples:
                raw_rows.append({
                    "run": run,
                    "direction": result["direction"],
                    "t_s": f"{t:.4f}",
                    "pos_pulses": pos,
                    "disp_m": f"{sign * (pos - start_pos) * M_PER_PULSE:.6f}",
                })

    if raw_rows:
        print(f"  raw speed data saved: {save_speed_csv(raw_rows, timestamp)}")

    plateau_by_dir = {}
    for label in ("fwd", "rev"):
        values = [s["v_plateau"] for s in stats if s["direction"] == label]
        if values:
            plateau_by_dir[label] = float(np.median(values))

    print("\n  SPEED SUMMARY")
    print(f"    reference no-load speed: {V_MAX_THEORETICAL:.3f} m/s")
    for label, value in plateau_by_dir.items():
        print(f"    {label}: measured plateau median = {value:.3f} m/s")
    if plateau_by_dir:
        print(
            f"    conservative loaded v_max = "
            f"{min(plateau_by_dir.values()):.3f} m/s"
        )

    return stats


def analyze_speed_csv(path):
    grouped = {}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row.get("run", 1)), row["direction"])
            grouped.setdefault(key, []).append(
                (float(row["t_s"]), int(row["pos_pulses"]))
            )

    print(f"\nRE-ANALYZE SPEED: {path}")
    for (run, label), samples in sorted(grouped.items()):
        result = analyze_speed(samples, +1 if label == "fwd" else -1)
        print(f"  run {run} {label}: {result}")


# ── Maximum acceleration ─────────────────────────────────────────────────
def run_accel_burst(comm, pwm):
    """Capture an ESP32-timed <a,PWM> burst and return time/pulse arrays."""
    return_to_center(comm)
    if not wait_for_standstill(comm):
        raise RuntimeError("cart did not stop before acceleration burst")
    if not comm.send("h", 0):
        raise RuntimeError("ESP32 did not acknowledge burst home command")

    comm.ser.reset_input_buffer()
    comm.ser.write(f"<a,{pwm}>".encode())

    lines = []
    recording = False
    ended = False
    deadline = time.time() + 8.0
    while time.time() < deadline:
        raw = comm.ser.readline().decode(errors="replace").strip()
        if not raw:
            continue
        if raw.startswith("ACCEL_BEGIN"):
            recording = True
            lines = []
            continue
        if raw.startswith("ACCEL_END"):
            ended = True
            break
        if recording and not raw.startswith("t_us"):
            lines.append(raw)

    comm.stop()
    if not ended:
        raise RuntimeError("did not receive ACCEL_END; check ESP32 firmware")

    t_us, positions = [], []
    for line in lines:
        parts = line.split(",")
        if len(parts) != 2:
            continue
        try:
            t_us.append(float(parts[0]))
            positions.append(float(parts[1]))
        except ValueError:
            continue

    t = np.asarray(t_us) / 1e6
    pos = np.asarray(positions)
    if len(t) < 5:
        raise RuntimeError("acceleration burst returned too few samples")
    if np.max(np.abs(pos)) > SAFE_STOP_PULSES + 5:
        raise RuntimeError(
            f"firmware safety limit exceeded: {np.max(np.abs(pos)):.0f} pulses "
            f"> {SAFE_STOP_PULSES}"
        )
    return t, pos


def analyze_acceleration(t, pos_pulses):
    x = (pos_pulses - pos_pulses[0]) * M_PER_PULSE
    if x[-1] < 0:
        x = -x

    velocity = np.gradient(x, t)
    velocity_smooth = _smooth(velocity, width=5)
    tail = velocity_smooth[max(0, int(0.7 * len(velocity_smooth))):]
    v_term = float(np.median(tail)) if len(tail) else float(velocity_smooth[-1])

    if v_term > 1e-6:
        above = np.where(velocity_smooth >= 0.6 * v_term)[0]
        cut = int(above[0]) if len(above) else len(t)
    else:
        cut = len(t)
    cut = int(np.clip(cut, 5, len(t)))

    coeff = np.polyfit(t[:cut], x[:cut], 2)
    a_initial = float(2.0 * coeff[0])
    a_velocity = float(np.polyfit(t[:cut], velocity_smooth[:cut], 1)[0])
    return {
        "a_initial": a_initial,
        "a_velocity": a_velocity,
        "v_term": v_term,
        "cut": cut,
    }


def save_accel_csv(t, pos, pwm, direction, run, timestamp):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"accel_{timestamp}_pwm{abs(pwm)}_{direction}_run{run}.csv"
    np.savetxt(
        path,
        np.column_stack([t * 1e6, pos]),
        fmt="%.1f",
        delimiter=",")
    # Add a readable header without relying on a second write path.
    text = path.read_text()
    path.write_text("t_us,pos_pulses\n" + text)
    return path


def run_acceleration_test(comm, pwm_levels, runs):
    summary = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for pwm in pwm_levels:
        if abs(pwm) <= PWM_DEADBAND:
            raise ValueError(f"acceleration PWM {pwm} must exceed deadband {PWM_DEADBAND}")
        for direction, sign in (("fwd", +1), ("rev", -1)):
            values = []
            for run in range(1, runs + 1):
                t, pos = run_accel_burst(comm, sign * pwm)
                result = analyze_acceleration(t, pos)
                values.append(result["a_initial"])
                path = save_accel_csv(t, pos, sign * pwm, direction, run, timestamp)
                print(
                    f"    PWM={pwm} {direction} run {run}: "
                    f"a_initial={result['a_initial']:.3f} m/s^2, "
                    f"a_velocity={result['a_velocity']:.3f}, "
                    f"v_term={result['v_term']:.3f} m/s -> {path.name}"
                )
            median = float(np.median(values))
            summary.append({
                "pwm": pwm,
                "direction": direction,
                "a_initial_mps2": median,
            })
            print(f"    median: PWM={pwm} {direction} a_initial={median:.3f} m/s^2")

    summary_path = OUTPUT_DIR / f"acceleration_summary_{timestamp}.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pwm", "direction", "a_initial_mps2"])
        writer.writeheader()
        writer.writerows(summary)

    print("\n  ACCELERATION SUMMARY")
    for row in summary:
        print(
            f"    PWM={row['pwm']} {row['direction']}: "
            f"{row['a_initial_mps2']:.3f} m/s^2"
        )
    if summary:
        worst = min(row["a_initial_mps2"] for row in summary)
        print(f"    conservative weakest-direction value: {worst:.3f} m/s^2")
        print("    required acceleration for 5 deg: %.3f m/s^2" %
              (GRAVITY * np.tan(np.radians(5.0))))
        print(f"    summary saved: {summary_path}")
    return summary


def analyze_accel_csv(path):
    arr = []
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("t_us") or line.startswith("#"):
            continue
        try:
            t_us, pos = line.split(",")[:2]
            arr.append((float(t_us), float(pos)))
        except ValueError:
            continue
    if len(arr) < 5:
        raise ValueError("acceleration CSV contains too few numeric samples")
    t = np.asarray([row[0] for row in arr]) / 1e6
    pos = np.asarray([row[1] for row in arr])
    print(analyze_acceleration(t, pos))


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", nargs="?", default=DEFAULT_SERIAL_PORT)
    parser.add_argument("--mode", choices=("vmax", "accel", "all"), default="vmax")
    parser.add_argument("--vmax-runs", type=int, default=3)
    parser.add_argument("--accel-pwm", type=int, nargs="+", default=list(ACCEL_PWM_LEVELS))
    parser.add_argument("--accel-runs", type=int, default=ACCEL_RUNS)
    parser.add_argument("--vmax-csv", help="offline re-analysis of a speed CSV")
    parser.add_argument("--accel-csv", help="offline re-analysis of an acceleration CSV")
    args = parser.parse_args()

    if args.vmax_csv:
        analyze_speed_csv(args.vmax_csv)
        return
    if args.accel_csv:
        analyze_accel_csv(args.accel_csv)
        return

    print("=" * 68)
    print("  Cart speed and acceleration calibration")
    print("=" * 68)
    print(f"  mode                 : {args.mode}")
    print(f"  pulley               : {PULLEY_TEETH}T GT2 ({PULLEY_CIRCUMFERENCE_M*1000:.1f} mm/rev)")
    print(f"  encoder              : {PULSES_PER_REV} pulses/rev")
    print(f"  runway               : +/-{RUNWAY_HALF_DISTANCE_CM:.1f} cm ({RUNWAY_PULSES} pulses)")
    print(f"  emergency boundary   : +/-{SAFE_STOP_DISTANCE_CM:.1f} cm ({SAFE_STOP_PULSES} pulses)")
    print(f"  track length         : 52.0 cm; deadband={PWM_DEADBAND} PWM")
    print(f"  reference no-load v  : {V_MAX_THEORETICAL:.3f} m/s @ {V_MAX_REFERENCE_RPM:.0f} RPM")
    print()

    with MotorComm(args.port, SERIAL_BAUD, verbose=True) as comm:
        input("  Press ENTER with cart near physical center and free to move...")
        prepare_center(comm)

        if args.mode in ("vmax", "all"):
            print("\n  === MAXIMUM SPEED TEST ===")
            run_speed_test(comm, max(1, args.vmax_runs))
            return_to_center(comm)

        if args.mode in ("accel", "all"):
            print("\n  === MAXIMUM ACCELERATION TEST ===")
            run_acceleration_test(comm, args.accel_pwm, max(1, args.accel_runs))
            return_to_center(comm)


if __name__ == "__main__":
    main()
