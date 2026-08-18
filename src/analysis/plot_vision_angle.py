"""
plot_vision_angle.py
====================
Offline visualization for yellow-tape vision angle recordings.

The real-time detector writes one row per processed camera request.  This
script deliberately stays outside src/vision: it does not participate in the
camera, ZMQ, or control loop.  It reads one or more diagnostic CSV files and
plots:

  1. wrapped line angle and a reconstructed continuous line angle;
  2. angular velocity from the detector;
  3. fitted-line geometry (pivot gap and visible line length).

The CSV stores the line angle modulo 180 degrees, so theta naturally wraps at
 +/-90 degrees.  The continuous curve is reconstructed from consecutive
 valid samples; it is useful for viewing motion, but it must not be treated as
 a new measurement source.

Usage examples:

    python src/analysis/plot_vision_angle.py \
        data/calibration/vision_static.csv \
        data/calibration/vision_slow.csv \
        data/calibration/vision_medium.csv \
        data/calibration/vision_fast.csv \
        --save --no-show --axis-limit 150

For one recording, the default output is written to the sibling ``plots``
directory.  For multiple recordings, the default output is:

    data/calibration/plots/vision_angle_overview.png
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib


REQUIRED_COLUMNS = {
    "t_s",
    "theta_rad",
    "omega_rad_s",
    "valid",
}


def load_recording(path: Path) -> dict[str, object]:
    """Load one vision CSV and return numeric arrays plus the file path."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{path.name}: missing columns: {missing_text}")

        values: dict[str, list[float]] = {
            "t_s": [],
            "theta_rad": [],
            "omega_rad_s": [],
            "valid": [],
        }
        optional_names = (
            "axis_line_distance_px",
            "line_length_px",
            "line_width_px",
            "linearity",
            "fps",
        )
        optional: dict[str, list[float]] = {
            name: [] for name in optional_names if name in columns
        }

        for line_number, row in enumerate(reader, start=2):
            try:
                values["t_s"].append(float(row["t_s"]))
                values["theta_rad"].append(float(row["theta_rad"]))
                values["omega_rad_s"].append(float(row["omega_rad_s"]))
                values["valid"].append(float(row["valid"]))
                for name, output in optional.items():
                    raw = row.get(name, "")
                    output.append(float(raw) if raw not in ("", None) else math.nan)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path.name}: invalid numeric value on CSV line {line_number}"
                ) from exc

    if not values["t_s"]:
        raise ValueError(f"{path.name}: no data rows found")

    data: dict[str, object] = {
        "path": path,
        "t_s": np.asarray(values["t_s"], dtype=float),
        "theta_rad": np.asarray(values["theta_rad"], dtype=float),
        "omega_rad_s": np.asarray(values["omega_rad_s"], dtype=float),
        "valid": np.asarray(values["valid"], dtype=int).astype(bool),
    }
    for name, output in optional.items():
        data[name] = np.asarray(output, dtype=float)
    return data


def wrap_line_delta(delta: float) -> float:
    """Return the smallest signed delta for a line orientation modulo pi."""
    half_pi = 0.5 * math.pi
    return (delta + half_pi) % math.pi - half_pi


def reconstruct_unwrapped_angle(theta_rad: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Reconstruct a continuous line orientation from wrapped valid samples."""
    result = np.full(theta_rad.shape, np.nan, dtype=float)
    previous = None

    for index, (current, is_valid) in enumerate(zip(theta_rad, valid)):
        if not is_valid:
            if previous is not None:
                result[index] = previous
            continue

        if previous is None:
            previous = float(current)
        else:
            previous += wrap_line_delta(float(current) - previous)
        result[index] = previous

    return result


def invalid_spans(t_s: np.ndarray, valid: np.ndarray) -> list[tuple[float, float]]:
    """Return time spans containing invalid detector rows."""
    spans: list[tuple[float, float]] = []
    start = None
    for index, is_valid in enumerate(valid):
        if not is_valid and start is None:
            start = index
        if is_valid and start is not None:
            end = index - 1
            spans.append((float(t_s[start]), float(t_s[end])))
            start = None
    if start is not None:
        spans.append((float(t_s[start]), float(t_s[-1])))
    return spans


def percentile(values: np.ndarray, fraction: float) -> float:
    """Return a percentile for a non-empty finite array."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.percentile(finite, fraction))


def print_summary(data: dict[str, object]) -> None:
    """Print compact, reproducible statistics for one recording."""
    path = data["path"]
    t_s = data["t_s"]
    theta_deg = np.degrees(data["theta_rad"])
    omega = data["omega_rad_s"]
    valid = data["valid"]
    dt = np.diff(t_s)
    valid_theta = theta_deg[valid]
    valid_omega = omega[valid]

    duration = float(t_s[-1] - t_s[0])
    loop_hz = len(t_s) / duration if duration > 0 else math.nan
    valid_rate = 100.0 * float(np.mean(valid))
    print(
        f"  {Path(path).name}: rows={len(t_s)}, duration={duration:.2f}s, "
        f"loop={loop_hz:.1f}Hz, valid={valid_rate:.2f}%"
    )
    if dt.size:
        print(
            f"    dt median={np.median(dt) * 1000:.2f}ms, "
            f"p95={percentile(dt, 95) * 1000:.2f}ms"
        )
    if valid_theta.size:
        print(
            f"    theta valid range={valid_theta.min():.2f}.."
            f"{valid_theta.max():.2f}deg, "
            f"|omega|max={np.max(np.abs(valid_omega)):.2f}rad/s"
        )


def add_invalid_shading(ax, data: dict[str, object]) -> None:
    """Mark detector-invalid intervals on a plot."""
    for start, end in invalid_spans(data["t_s"], data["valid"]):
        ax.axvspan(start, end, color="tab:red", alpha=0.12, linewidth=0)


def make_figure(data_sets: list[dict[str, object]], axis_limit: float | None):
    """Create the comparison figure for all recordings."""
    import matplotlib.pyplot as plt

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown"]
    rows = len(data_sets)
    figure, axes = plt.subplots(
        rows,
        3,
        figsize=(16, max(4.0 * rows, 6.0)),
        squeeze=False,
        sharex=False,
    )
    figure.suptitle(
        "Yellow-tape vision angle recordings",
        fontsize=15,
        fontweight="bold",
    )

    for row_index, data in enumerate(data_sets):
        color = colors[row_index % len(colors)]
        path = data["path"]
        t_s = data["t_s"]
        valid = data["valid"]
        theta_wrapped_deg = np.degrees(data["theta_rad"])
        theta_unwrapped_deg = np.degrees(
            reconstruct_unwrapped_angle(data["theta_rad"], valid)
        )
        omega = data["omega_rad_s"]

        ax_theta, ax_omega, ax_geometry = axes[row_index]

        # Angle: show the stored modulo-180 representation and the reconstructed
        # continuous line orientation separately.
        ax_theta.plot(
            t_s,
            theta_wrapped_deg,
            color="0.65",
            linewidth=0.8,
            alpha=0.75,
            label="stored wrapped angle",
        )
        ax_theta.plot(
            t_s[valid],
            theta_unwrapped_deg[valid],
            color=color,
            linewidth=1.2,
            label="reconstructed continuous angle",
        )
        if np.any(~valid):
            ax_theta.scatter(
                t_s[~valid],
                theta_wrapped_deg[~valid],
                color="tab:red",
                marker="x",
                s=18,
                label="invalid frame",
                zorder=4,
            )
        add_invalid_shading(ax_theta, data)
        ax_theta.axhline(0.0, color="0.35", linewidth=0.7)
        ax_theta.axhline(90.0, color="0.65", linestyle=":", linewidth=0.7)
        ax_theta.axhline(-90.0, color="0.65", linestyle=":", linewidth=0.7)
        ax_theta.set_ylabel("angle (deg)")
        ax_theta.set_title(Path(path).name)
        ax_theta.grid(True, alpha=0.25)
        ax_theta.legend(loc="best", fontsize=8)

        # Angular velocity is meaningful only for valid detector rows.
        omega_valid = np.where(valid, omega, np.nan)
        ax_omega.plot(t_s, omega_valid, color=color, linewidth=1.0)
        add_invalid_shading(ax_omega, data)
        ax_omega.axhline(0.0, color="0.35", linewidth=0.7)
        ax_omega.set_ylabel("omega (rad/s)")
        ax_omega.set_title("detector angular velocity")
        ax_omega.grid(True, alpha=0.25)

        # Geometry makes the effect of --axis-line-gap visible.  Both fields
        # are optional so older CSVs remain usable.
        gap = data.get("axis_line_distance_px")
        length = data.get("line_length_px")
        has_gap = gap is not None and np.isfinite(gap).any()
        has_length = length is not None and np.isfinite(length).any()
        if has_gap:
            ax_geometry.plot(
                t_s,
                gap,
                color="tab:red",
                linewidth=1.0,
                label="pivot-line gap",
            )
            if axis_limit is not None:
                ax_geometry.axhline(
                    axis_limit,
                    color="tab:red",
                    linestyle="--",
                    linewidth=0.9,
                    label=f"configured limit ({axis_limit:g}px)",
                )
        if has_length:
            ax_length = ax_geometry.twinx()
            ax_length.plot(
                t_s,
                length,
                color="tab:purple",
                linewidth=1.0,
                alpha=0.85,
                label="fitted line length",
            )
            ax_length.set_ylabel("line length (px)", color="tab:purple")
            ax_length.tick_params(axis="y", labelcolor="tab:purple")
        if not has_gap and not has_length:
            ax_geometry.text(
                0.5,
                0.5,
                "No geometry columns in CSV",
                ha="center",
                va="center",
                transform=ax_geometry.transAxes,
            )
        else:
            ax_geometry.set_ylabel("pivot-line gap (px)", color="tab:red")
            ax_geometry.tick_params(axis="y", labelcolor="tab:red")
            ax_geometry.legend(loc="upper left", fontsize=8)
        add_invalid_shading(ax_geometry, data)
        ax_geometry.set_title("detection geometry")
        ax_geometry.grid(True, alpha=0.25)

        ax_theta.set_xlabel("time (s)")
        ax_omega.set_xlabel("time (s)")
        ax_geometry.set_xlabel("time (s)")

    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return figure


def default_output(paths: list[Path]) -> Path:
    """Choose a stable output location without overwriting raw CSV files."""
    if len(paths) == 1:
        output_dir = paths[0].parent / "plots"
        return output_dir / f"{paths[0].stem}_angle_plot.png"
    output_dir = paths[0].parent / "plots"
    return output_dir / "vision_angle_overview.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot angle and quality diagnostics from vision CSV files."
    )
    parser.add_argument("csv", nargs="+", type=Path, help="Vision CSV file(s)")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save a PNG in data/calibration/plots (or at --output)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG path; implies --save",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a GUI window; use this on headless Raspberry Pi",
    )
    parser.add_argument(
        "--axis-limit",
        type=float,
        default=None,
        help="Optional MAX_LINE_AXIS_DISTANCE_PX to draw on geometry plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    paths = [path.expanduser().resolve() for path in args.csv]
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"CSV file not found: {path}")

    try:
        data_sets = [load_recording(path) for path in paths]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print("VISION ANGLE PLOT")
    for data in data_sets:
        print_summary(data)

    figure = make_figure(data_sets, args.axis_limit)
    output_path = args.output or default_output(paths)
    should_save = args.save or args.output is not None
    if should_save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=140)
        print(f"  saved: {output_path}")

    if args.no_show:
        return

    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()
