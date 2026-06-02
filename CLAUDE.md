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

**Implication for AI:** scripts that depend on ZMQ between Pi and PC
(`src/camerapreview_detect_color2.py`, `src/camera_comm.py`,
`src/balance_pid.py`) are **legacy**. The new entrypoint is `pi/pi_balance.py`.

---

## Directory Layout (inside cartpole-pid/)

```
cartpole-pid/
├── CLAUDE.md                        ← this file (code-level)
├── pi/                              ← runs ON the Raspberry Pi
│   ├── pi_balance.py                ← MAIN: balance PID controller
│   └── pi_latency.py                ← latency measurement
├── src/                             ← runs on PC + shared library
│   ├── config.py                    ← single source of truth (parameters)
│   ├── motor_comm.py                ← MotorComm class — ESP32 serial protocol
│   ├── camera_comm.py               ← LEGACY (ZMQ subscriber, old arch)
│   ├── balance_pid.py               ← LEGACY (PC-loop balance)
│   ├── camerapreview_detect_color2.py  ← LEGACY (Pi ZMQ publisher)
│   ├── swing_up_dc.py               ← swing-up controller (active)
│   ├── swing_up_energy_dc.py        ← energy-based swing-up (alt)
│   ├── move_cart2_center.py         ← helper: return cart to center
│   ├── log_angle.py                 ← angle recorder during experiments
│   ├── analyze_*.py                 ← offline analysis scripts
│   └── hardware_measurement/        ← calibration / standalone tests
├── dc_motor_controller/             ← ESP32 firmware (Arduino sketch)
├── cartpole/                        ← legacy package, do not extend
└── achive/                          ← archived, do not reference
```

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

## How `pi/` Scripts Import from `src/`

Files under `pi/` add `src/` to `sys.path` at the top:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from motor_comm import MotorComm
from config import Kp, Kd, Kx, PWM_LIMIT
```

Keep this pattern when adding new scripts in `pi/`.

---

## Running Things

```bash
# On the Raspberry Pi (over SSH):
python3 ~/cartpole-pid/pi/pi_balance.py /dev/ttyUSB0
python3 ~/cartpole-pid/pi/pi_latency.py /dev/ttyUSB0

# On the PC:
python  src/swing_up_dc.py            # swing-up still runs from PC for now
python  src/analyze_swingup.py        # offline analysis
```

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

---

## Currently Open Items (PID work in progress)

- [ ] PWM deadband compensation in `pi/pi_balance.py`
- [ ] `omega` units: convert to deg/s for consistency with `theta_deg`
- [ ] CSV logging during balance runs
- [ ] First on-track balance run to find a working (Kp, Kd) pair

---

## Legacy / Deprecated — Do Not Extend

These files still build but belong to the old PC-loop architecture.
Do not add features to them. New work goes in `pi/pi_balance.py`.

- `src/camera_comm.py`
- `src/balance_pid.py`
- `src/camerapreview_detect_color2.py`
- `cartpole/Communicator.py` (superseded by `src/motor_comm.py`)
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
