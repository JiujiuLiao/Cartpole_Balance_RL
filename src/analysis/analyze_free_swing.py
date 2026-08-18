"""Analyze fixed-cart free-swing CSV files."""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from config import (CALIBRATION_DIR, M_PER_PULSE, PENDULUM_L_M,
                    PENDULUM_PIVOT_MASS_KG, PENDULUM_ROD_MASS_KG,
                    PENDULUM_TIP_MASS_KG)


GAP_LIMIT_S = 0.10
MIN_PERIOD_S = 0.40
MAX_PERIOD_S = 2.50
MIN_CYCLES = 15
MAX_CART_MM = 0.50
MIN_VALID_PCT = 98.0
MAX_METHOD_DIFF_PCT = 2.0
MAX_CV_PCT = 2.0

# In the current configuration the pivot lump is at r=0, so it contributes
# neither gravitational torque nor pivot inertia.  Keep the formula explicit
# so it is easy to update if the top assembly changes later.
I_PIVOT = (PENDULUM_ROD_MASS_KG * PENDULUM_L_M ** 2 / 3.0
           + PENDULUM_TIP_MASS_KG * PENDULUM_L_M ** 2)
M_LC = (PENDULUM_ROD_MASS_KG * PENDULUM_L_M / 2.0
        + PENDULUM_TIP_MASS_KG * PENDULUM_L_M)


def finite_float(value, default=np.nan):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def load_csv(path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"sensor_ts_ns", "theta_raw_rad", "valid",
                    "cart_pos_pulses", "release_angle_deg"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            try:
                ts_ns = int(row["sensor_ts_ns"] or 0)
                valid = int(row["valid"] or 0) == 1
                cart = int(row["cart_pos_pulses"] or 0)
            except (TypeError, ValueError):
                continue
            theta = finite_float(row.get("theta_raw_rad", ""))
            frame_id = int(row.get("frame_id", -1) or -1)
            release_event = int(row.get("release_event", 0) or 0)
            angle_deg = finite_float(row.get("release_angle_deg", ""))
            rows.append((ts_ns, theta, valid, cart, frame_id,
                         release_event, angle_deg))

    if not rows:
        raise ValueError("CSV contains no usable rows")
    return rows


def circular_mean(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan
    return math.atan2(float(np.mean(np.sin(values))),
                      float(np.mean(np.cos(values))))


def relative_angle(theta, theta_down):
    return np.arctan2(np.sin(theta - theta_down),
                      np.cos(theta - theta_down))


def valid_ratio(rows):
    valid_count = sum(1 for r in rows if r[2])
    frame_ids = [r[4] for r in rows if r[4] >= 0]
    if len(frame_ids) >= 2:
        expected = max(frame_ids) - min(frame_ids) + 1
    else:
        expected = len(rows)
    ratio = 100.0 * valid_count / expected if expected else 0.0
    received_ratio = 100.0 * len(rows) / expected if expected else 0.0
    return valid_count, expected, ratio, received_ratio


def crossing_period(t, phi):
    """Return period and number of full cycles from positive-going crossings."""
    if len(t) < 3:
        return np.nan, 0, np.array([])
    dt_med = float(np.median(np.diff(t)))
    max_gap = max(GAP_LIMIT_S, 3.0 * dt_med)
    crossings = []
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        if dt <= 0 or dt > max_gap:
            continue
        y0, y1 = phi[i], phi[i + 1]
        if y0 <= 0.0 < y1 and y1 != y0:
            frac = -y0 / (y1 - y0)
            crossings.append(t[i] + frac * dt)

    crossings = np.asarray(crossings, dtype=float)
    periods = np.diff(crossings)
    periods = periods[(periods >= MIN_PERIOD_S) & (periods <= MAX_PERIOD_S)]
    if len(periods) == 0:
        return np.nan, 0, crossings
    return float(np.median(periods)), int(len(periods)), crossings


def fft_frequency(t, phi):
    """Estimate the dominant frequency with zero-padded FFT peak interpolation."""
    if len(t) < 32:
        return np.nan
    dt = float(np.median(np.diff(t)))
    if not math.isfinite(dt) or dt <= 0:
        return np.nan

    uniform_t = np.arange(t[0], t[-1], dt)
    if len(uniform_t) < 32:
        return np.nan
    uniform_phi = np.interp(uniform_t, t, phi)

    # Remove slow bias/drift but do not apply an EKF or a dynamic filter.
    centered_t = uniform_t - np.mean(uniform_t)
    trend = np.polyfit(centered_t, uniform_phi, 1)
    signal = uniform_phi - np.polyval(trend, centered_t)
    window = np.hanning(len(signal))
    signal *= window

    nfft = 1
    while nfft < max(16384, 16 * len(signal)):
        nfft <<= 1
    spectrum = np.abs(np.fft.rfft(signal, n=nfft))
    frequencies = np.fft.rfftfreq(nfft, dt)
    band = (frequencies >= 0.30) & (frequencies <= 2.50)
    indices = np.flatnonzero(band)
    if len(indices) == 0:
        return np.nan
    k = int(indices[np.argmax(spectrum[indices])])
    if k <= 0 or k >= len(spectrum) - 1:
        return float(frequencies[k])

    # Quadratic interpolation in log magnitude gives sub-bin peak location.
    a, b, c = np.log(np.maximum(spectrum[k - 1:k + 2], 1e-30))
    denominator = a - 2.0 * b + c
    if abs(denominator) < 1e-12:
        offset = 0.0
    else:
        offset = 0.5 * (a - c) / denominator
        offset = float(np.clip(offset, -1.0, 1.0))
    return float(frequencies[k] + offset * (frequencies[1] - frequencies[0]))


def gravity_from_period(period_s):
    if not math.isfinite(period_s) or period_s <= 0 or M_LC <= 0:
        return np.nan, np.nan, np.nan
    omega = 2.0 * math.pi / period_s
    k_g = omega * omega
    g_hat = I_PIVOT * k_g / M_LC
    return omega, k_g, g_hat


def analyze_one(path):
    rows = load_csv(path)
    ts = np.asarray([r[0] for r in rows], dtype=np.int64)
    theta = np.asarray([r[1] for r in rows], dtype=float)
    valid = np.asarray([r[2] for r in rows], dtype=bool)
    cart = np.asarray([r[3] for r in rows], dtype=float)
    release_event = np.asarray([r[5] for r in rows], dtype=int)
    release_deg = float(np.nanmedian([r[6] for r in rows]))

    valid_count, expected, valid_pct, received_pct = valid_ratio(rows)
    ts_ok = ts > 0
    ts_valid = ts[ts_ok]
    monotonic = len(ts_valid) > 1 and bool(np.all(np.diff(ts_valid) > 0))
    valid_with_ts_pct = (100.0 * np.sum(valid & ts_ok) / valid_count
                         if valid_count else 0.0)

    # Encoder movement is checked over the entire captured interval, relative
    # to the median pre-release position.
    event_indices = np.flatnonzero(release_event == 1)
    event_index = int(event_indices[0]) if len(event_indices) else 0
    pre_cart = cart[:max(1, event_index)]
    baseline_pulses = float(np.median(pre_cart))
    max_cart_mm = float(np.max(np.abs(cart - baseline_pulses))
                        * M_PER_PULSE * 1000.0)

    # Downward reference from the pre-release valid frames.  A circular mean
    # avoids the +/-pi branch at the hanging position.
    pre_mask = valid & np.isfinite(theta) & (np.arange(len(rows)) < event_index)
    if not np.any(pre_mask):
        pre_mask = valid & np.isfinite(theta)
    theta_down = circular_mean(theta[pre_mask])

    analysis_mask = valid & np.isfinite(theta) & (ts > 0)
    if len(event_indices):
        release_ts = ts[event_index]
        analysis_mask &= ts >= release_ts
    else:
        release_ts = int(ts[analysis_mask][0]) if np.any(analysis_mask) else 0

    t = (ts[analysis_mask].astype(float) - release_ts) * 1e-9
    phi = relative_angle(theta[analysis_mask], theta_down)
    order = np.argsort(t)
    t, phi = t[order], phi[order]
    keep = np.r_[True, np.diff(t) > 0] if len(t) else np.array([], dtype=bool)
    t, phi = t[keep], phi[keep]

    T_zero, cycles, crossings = crossing_period(t, phi)
    omega_zero, k_zero, g_zero = gravity_from_period(T_zero)
    f_fft = fft_frequency(t, phi)
    T_fft = 1.0 / f_fft if math.isfinite(f_fft) and f_fft > 0 else np.nan
    omega_fft, k_fft, g_fft = gravity_from_period(T_fft)
    method_diff = (100.0 * abs(g_zero - g_fft) /
                   ((g_zero + g_fft) / 2.0)
                   if math.isfinite(g_zero) and math.isfinite(g_fft)
                   and (g_zero + g_fft) > 0 else np.nan)

    duration_s = float(t[-1] - t[0]) if len(t) else 0.0
    pass_valid = valid_pct >= MIN_VALID_PCT and valid_with_ts_pct >= MIN_VALID_PCT
    pass_cart = max_cart_mm <= MAX_CART_MM
    pass_cycles = cycles >= MIN_CYCLES
    pass_method = math.isfinite(method_diff) and method_diff <= MAX_METHOD_DIFF_PCT
    run_pass = pass_valid and monotonic and pass_cart and pass_cycles and pass_method

    return {
        "file": str(path),
        "run_id": path.stem,
        "release_angle_deg": release_deg,
        "rows": len(rows),
        "valid_count": valid_count,
        "expected_frames": expected,
        "valid_pct": valid_pct,
        "received_pct": received_pct,
        "valid_with_ts_pct": valid_with_ts_pct,
        "timestamp_monotonic": monotonic,
        "max_cart_mm": max_cart_mm,
        "theta_down_rad": theta_down,
        "duration_s": duration_s,
        "cycles": cycles,
        "T_zero_s": T_zero,
        "T_fft_s": T_fft,
        "g_zero": g_zero,
        "g_fft": g_fft,
        "k_zero": k_zero,
        "k_fft": k_fft,
        "method_diff_pct": method_diff,
        "pass_valid": pass_valid,
        "pass_cart": pass_cart,
        "pass_cycles": pass_cycles,
        "pass_method": pass_method,
        "pass_run": run_pass,
    }


def fmt(value, digits=3):
    return "--" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def print_result(result):
    status = "PASS" if result["pass_run"] else "FAIL"
    checks = ("valid" if result["pass_valid"] else "VALID_FAIL") + "," \
             + ("cart" if result["pass_cart"] else "CART_FAIL") + "," \
             + ("cycles" if result["pass_cycles"] else "CYCLES_FAIL") + "," \
             + ("fft-zero" if result["pass_method"] else "FFT_ZERO_FAIL")
    print(f"{status:4s}  {Path(result['file']).name}")
    print(f"      valid={result['valid_pct']:.2f}%  "
          f"cart_max={result['max_cart_mm']:.3f} mm  "
          f"cycles={result['cycles']}  "
          f"Tzero={fmt(result['T_zero_s'], 5)} s  "
          f"Tfft={fmt(result['T_fft_s'], 5)} s")
    print(f"      g_zero={fmt(result['g_zero'], 4)}  "
          f"g_fft={fmt(result['g_fft'], 4)}  "
          f"method_diff={fmt(result['method_diff_pct'], 2)}%  [{checks}]")


def cv(values):
    values = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if len(values) < 2 or np.mean(values) == 0:
        return np.nan
    return float(100.0 * np.std(values, ddof=1) / abs(np.mean(values)))


def find_inputs(input_path):
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("free_swing_*_run*.csv"))
    raise FileNotFoundError(path)


def write_summary(results, directory):
    output = Path(directory) / "free_swing_analysis_summary.csv"
    fields = [
        "file", "release_angle_deg", "valid_pct", "max_cart_mm", "cycles",
        "T_zero_s", "T_fft_s", "g_zero", "g_fft", "method_diff_pct",
        "pass_run",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in fields})
    return output


def main():
    parser = argparse.ArgumentParser(description="Analyze fixed-cart free-swing CSVs")
    parser.add_argument("--input", required=True,
                        help="One CSV file or the free_swing output directory")
    args = parser.parse_args()

    try:
        paths = find_inputs(args.input)
    except (FileNotFoundError, OSError) as exc:
        parser.error(str(exc))
    if not paths:
        parser.error("no free_swing_*_run*.csv files found")

    print("=" * 78)
    print("  Fixed-cart pendulum free-swing acceptance analysis")
    print("=" * 78)
    print(f"  Expected current-model T ≈ 1.016 s; M_PER_PULSE={M_PER_PULSE*1000:.6f} mm")
    print(f"  Model I={I_PIVOT:.8f} kg m², m*l_c={M_LC:.8f} kg m")
    print()

    results = []
    for path in paths:
        try:
            result = analyze_one(path)
        except (OSError, ValueError, KeyError) as exc:
            print(f"FAIL  {path.name}: {exc}")
            continue
        results.append(result)
        print_result(result)

    if not results:
        raise SystemExit("No analyzable runs.")

    print()
    print("── Repeatability by absolute release angle ──")
    groups = {}
    for result in results:
        key = round(abs(result["release_angle_deg"]), 3)
        groups.setdefault(key, []).append(result)

    for angle, group in sorted(groups.items()):
        cv_zero = cv([r["g_zero"] for r in group])
        cv_fft = cv([r["g_fft"] for r in group])
        enough = len(group) >= 5
        pass_group = (enough and math.isfinite(cv_zero) and math.isfinite(cv_fft)
                      and cv_zero <= MAX_CV_PCT and cv_fft <= MAX_CV_PCT)
        print(f"  {angle:g} deg: n={len(group)}  "
              f"CV_zero={fmt(cv_zero, 2)}%  CV_fft={fmt(cv_fft, 2)}%  "
              f"{'PASS' if pass_group else 'FAIL (need 5 runs and CV<=2%)'}")

    summary_path = write_summary(results, Path(args.input).parent
                                 if Path(args.input).is_file() else args.input)
    print()
    print(f"Summary CSV: {summary_path}")
    print("Note: g_hat is the equivalent gravity estimate; k_hat=omega² is also reported.")


if __name__ == "__main__":
    main()
