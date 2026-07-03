# Inverted Pendulum (cartpole-pid) — Code-level Navigation

> **Project-wide rules, hardware specs, wiring, and calibration values live in [`../CLAUDE.md`](../CLAUDE.md).** Read that first.
>
> This file only documents code-level details specific to `cartpole-pid/`.
> Never duplicate hardware numbers here — they go stale. Point readers to
> `../CLAUDE.md` or to `src/config.py` instead.

---

## Current Architecture (cartpole-pid only)

```
Pi 4 (camera + PID) ──USB──▶ ESP32 (IBT-2 + DC motor + encoder)
```

The PC is **no longer in the control loop**. The Pi runs the full pipeline
in a single process:

```
picamera2 frame  →  HSV mask  →  marker angle  →  PID  →  serial  →  ESP32
```

The PC is used only for SSH-based editing, monitoring, and offline analysis.

**Implication for AI:** scripts that depend on the old cross-machine ZMQ
setup (`src/camera_comm.py`, plus anything importing from it) are **legacy**.
Current entry points:
- Swing-up + balance handoff: `scripts/swing_up_energy_dc.py`
- Standalone PD tuning:       `scripts/tune_balance_pd.py`
Vision (`vision/camerapreview_detect_color_V3.py`) runs on the Pi and
publishes over ZMQ to the same Pi — no cross-machine socket.

---

## Directory Layout (inside cartpole-classical-control/)

```
cartpole-classical-control/
├── CLAUDE.md                        ← this file (code-level)
├── data/                            ← ALL time-stamped CSV outputs live here
│   ├── balance_diag/                    balance_lqr writes; analyze_balance_diag reads
│   ├── balance_logs/                    tune_balance_pd writes
│   ├── swingup_logs/                    swing_up_* writes
│   └── calibration/                     measurement scripts write here
├── src/                             ← Python code
│   │  ── core libraries (top level; shared, imported by everything) ──
│   ├── config.py                    ← single source of truth (parameters + data paths)
│   ├── motor_comm.py                ← MotorComm class — ESP32 serial protocol
│   ├── camera_comm.py               ← LEGACY (ZMQ subscriber, old arch)
│   ├── ekf.py                       ← state estimator (angle + omega)
│   │  ── subdirectories (organized by role, 2026-07 refactor) ──
│   ├── controllers/                 ← control algorithm libraries + tools
│   │   ├── balance_pd.py            ← PD balance loop  (called by swing-up)
│   │   ├── balance_lqr.py           ← LQR balance loop (called by swing-up)
│   │   ├── compute_lqr_K.py         ← LQR K solver (offline, pairs with balance_lqr)
│   │   └── test_balance_lqr.py      ← simulation test for LQR
│   ├── scripts/                     ← runnable entry-point scripts (`python xxx.py`)
│   │   ├── swing_up_energy_dc.py    ← energy-method swing-up + dispatch to balance
│   │   ├── swing_up_dc_without_camera.py    ← swing-up without vision
│   │   ├── tune_balance_pd.py       ← standalone PD tuner (pairs with `balance_pd.py`)
│   │   ├── move_cart2_center.py     ← helper: return cart to center
│   │   └── measure_sigma_theta.py   ← static-pole → prints EKF_SIGMA_THETA
│   ├── calibration/                 ← one-off hardware measurement
│   │   ├── measure_ppr.py, test_pwm_range.py, test_accel_burst.py,
│   │   ├── test_vmax.py, fit_force_curve.py, collect_b_data.py,
│   │   ├── fit_ekf_b.py, record_validation_data.py,
│   │   ├── acquire_pendulum_dynamics.py, test_frame_delay.py
│   ├── analysis/                    ← offline CSV analysis + plotting
│   │   ├── analyze_pendulum.py, analyze_balance_diag.py, analyze_ekf_vs_meas.py,
│   │   ├── analyze_swingup.py, plot_swingup.py, plot_vmax.py,
│   │   └── compare_omega_filters.py, compare_pendulum_cart_half_period.py
│   ├── latency_tests/               ← end-to-end latency probing
│   │   ├── pc_end_to_end_latency.py, pi_latency_full.py, pi_vision_latency.py
│   └── vision/                      ← Pi-side real-time vision (currently active)
│       ├── camerapreview_detect_color_V3.py     ← current vision publisher
│       └── axis_calib.json                       ← pivot-axis persistence
├── dc_motor_controller/             ← ESP32 firmware (Arduino sketch)
└── achive/                          ← archived, do not reference
```

Scripts inside ANY subdirectory of `src/` (`controllers/`, `scripts/`,
`calibration/`, `analysis/`, `latency_tests/`, `vision/`) add `src/` to
`sys.path` at the top so they can `from config import ...`
and `from motor_comm import ...`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

`scripts/swing_up_energy_dc.py` additionally puts `src/controllers/` on
`sys.path` so it can `from balance_pd import ...` / `from balance_lqr import ...`.
Add the same second line to any new script that imports a controller.

---

## Where to Find / Change Things

| What                                        | Where                                  |
|---------------------------------------------|----------------------------------------|
| Any tuneable parameter (PID, deadband, ...) | `src/config.py`                        |
| Serial protocol with ESP32                  | `src/motor_comm.py` docstring          |
| Camera setup (FPS, resolution)              | inside the script that owns the camera |
| HSV color thresholds                        | inside the script that uses them       |
| ESP32 firmware                              | `dc_motor_controller/`                 |
| **Hardware specs / wiring / GPIO map**      | **[`../CLAUDE.md`](../CLAUDE.md)**     |

---

## Running Things

Every entry point (swing-up, tuner, calibration, analysis) is a script
inside a `src/*/` subdirectory. Run from the project root:

```bash
# Swing-up + auto-handoff to balance (LQR by default; PD if USE_LQR_BALANCE=False):
python3 src/scripts/swing_up_energy_dc.py /dev/ttyUSB0

# Standalone PD tuning (find Kp/Kd/Kx by hand, then copy winners into config.py):
python3 src/scripts/tune_balance_pd.py /dev/ttyUSB0

# Utilities:
python3 src/scripts/move_cart2_center.py /dev/ttyUSB0
python3 src/analysis/analyze_swingup.py                 # offline analysis
```

The Pi is where hardware scripts run today (its `/dev/ttyUSB0` is the
ESP32). Vision publisher (`vision/camerapreview_detect_color_V3.py`) must
be running in a separate terminal before any controller starts.

---

## Conventions When Adding Code

- **Top of every script**: docstring with `PURPOSE`, `RUNS ON` (Pi or PC),
  `DEPENDS ON`, and a `Usage` example.
- **Function signatures**: use type hints. Document angle units
  (rad vs deg), PWM range, and position units (pulses vs meters) in
  the docstring.
- **No magic numbers**: any constant that might be retuned goes into
  `config.py`. Local-only constants get a comment explaining their
  derivation.
- **Logging during experiments**: write CSV to `data/` (create the
  folder if missing). Header on row 1, one sample per row, English
  column names.
- **File naming**:
  - `pi_*`       → runs on the Raspberry Pi
  - `pc_*`       → runs on the PC (optional prefix for new PC scripts)
  - `test_*`     → one-off probe / smoke test
  - `analyze_*`  → offline post-processing
  - `measure_*`  → hardware calibration
  - **Controllers (algorithm bare-name = library; verb-prefix = tool)**:
    - `balance_<algo>.py`         → controller library, called by swing-up
                                     (e.g. `balance_pd.py`, `balance_lqr.py`)
    - `tune_balance_<algo>.py`    → standalone interactive tuner for that controller
                                     (e.g. `tune_balance_pd.py`)
    - `compute_<algo>_*.py`       → offline math tool that produces gains/matrices
                                     (e.g. `compute_lqr_K.py`)

---

## Currently Open Items (PID work in progress)

- [ ] `omega` units: convert to deg/s for consistency with `theta_deg`
- [ ] First on-track balance run with LQR to confirm K values from `compute_lqr_K.py`
- [ ] Merge Pi-side tuning branch back into main

---

## Legacy / Deprecated — Do Not Extend

These files still build but belong to the old PC-loop architecture.
Do not add features to them. New work goes in `scripts/` or `controllers/`.

- `src/camera_comm.py`
- `src/camerapreview_detect_color2.py`
- everything under `achive/`

---

## When You Don't Know Something

The "ask, don't guess" rule from [`../CLAUDE.md`](../CLAUDE.md) applies.
Particularly:

- a physical spec not yet in `config.py`
- which of two similar scripts is canonical
- whether a parameter is hardware-fixed or task-tuneable
- which direction is "positive" for the motor / encoder / angle

A clarifying question is always cheaper than a wrong assumption baked into
committed code.
