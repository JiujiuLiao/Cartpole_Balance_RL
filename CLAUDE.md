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

The PC is **no longer in the control loop**. The Pi runs two local processes:

```
picamera2 → yellow-line vision publisher → local ZMQ :9999
                                      ↓
                         controller → USB serial → ESP32
```

The PC is used only for SSH-based editing, monitoring, and offline analysis.

`src/camera_comm.py` is the current shared local-ZMQ subscriber. Cross-machine
addresses are legacy, but the subscriber itself is not.
Current entry points:
- Isolated LQR balance:       `controllers/test_balance_lqr.py`
- Standalone PD tuning:       `scripts/tune_balance_pd.py`
Vision (`vision/camerapreview_detect_yellow_tape_line_V1.py`) runs on the Pi
and publishes over ZMQ to the same Pi. The line angle is modulo pi and cannot
distinguish upright from hanging, so automatic energy swing-up is currently
disabled; `scripts/swing_up_energy_dc.py` refuses to arm.

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
│   ├── camera_comm.py               ← current local-ZMQ vision subscriber
│   ├── ekf.py                       ← state estimator (angle + omega)
│   │  ── subdirectories (organized by role, 2026-07 refactor) ──
│   ├── controllers/                 ← control algorithm libraries + tools
│   │   ├── balance_pd.py            ← PD balance loop  (called by swing-up)
│   │   ├── balance_lqr.py           ← LQR balance loop (called by swing-up)
│   │   ├── compute_lqr_K.py         ← baseline LQR solver (offline; review before promotion)
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
│   │   ├── acquire_pendulum_dynamics.py, free_swing_recorder.py,
│   │   └── test_frame_delay.py
│   ├── analysis/                    ← offline CSV analysis + plotting
│   │   ├── analyze_pendulum.py, analyze_balance_diag.py, analyze_ekf_vs_meas.py,
│   │   ├── analyze_swingup.py, plot_swingup.py, plot_vmax.py,
│   │   └── compare_omega_filters.py, compare_pendulum_cart_half_period.py
│   ├── latency_tests/               ← end-to-end latency probing
│   │   ├── pi_latency_full.py                       ← current pipeline
│   │   └── pc_end_to_end_latency.py, pi_vision_latency.py ← legacy protocols
│   └── vision/                      ← Pi-side real-time vision (currently active)
│       ├── camerapreview_detect_yellow_tape_line_V1.py ← current publisher
│       ├── camerapreview_detect_color_V3.py             ← legacy dual-colour publisher
│       ├── axis_calib.json                              ← pivot persistence
│       └── hsv_calib.json                               ← yellow HSV persistence
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
# Current vision publisher (start first in its own Pi terminal):
python3 src/vision/camerapreview_detect_yellow_tape_line_V1.py --no-gui

# Swing-up + auto-handoff to balance (LQR by default; PD if USE_LQR_BALANCE=False):
# Currently refuses to arm because the line detector has no up/down state.
python3 src/scripts/swing_up_energy_dc.py /dev/ttyUSB0

# Isolated hand-lift LQR balance:
python3 src/controllers/test_balance_lqr.py /dev/ttyUSB0

# Fixed-cart free swing (vision must use --publish-invalid):
python3 src/calibration/free_swing_recorder.py --release-angle-deg 5 --run-id 5deg_run01

# Standalone PD tuning (find Kp/Kd/Kx by hand, then copy winners into config.py):
python3 src/scripts/tune_balance_pd.py /dev/ttyUSB0

# Utilities:
python3 src/scripts/move_cart2_center.py /dev/ttyUSB0
python3 src/analysis/analyze_swingup.py                 # offline analysis
```

The Pi is where hardware scripts run today (its `/dev/ttyUSB0` is the ESP32).
The yellow-tape line publisher must be running in a separate terminal before
any controller or camera-data logger starts.

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
- [ ] Validate yellow-line static zero/sign, valid-frame rate and dynamic latency
- [ ] Add an asymmetric visual feature/sensor before re-enabling automatic swing-up
- [ ] First on-track balance run with the active K documented in `../docs/lqr_gain_status.md`
- [ ] Merge Pi-side tuning branch back into main

---

## Legacy / Deprecated — Do Not Extend

These files still build but belong to the old PC-loop architecture.
Do not add features to them. New work goes in `scripts/` or `controllers/`.

- `src/camerapreview_detect_color2.py`
- `src/vision/camerapreview_detect_color_V3.py`
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
