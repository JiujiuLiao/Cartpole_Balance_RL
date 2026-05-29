# Inverted Pendulum (cartpole-pid) — AI Navigation

> **Read this file FIRST** when working on this project. It tells you what
> exists, what the current architecture is, and how to extend the code
> without breaking conventions.

---

## ⚠️ Hard Rules (must follow at all times)

1. **English only — everywhere.** All code, comments, docstrings, log
   messages, plot labels, file names, variable names, commit messages,
   CSV column headers, and any new documentation must be in English.
   No Chinese, no German. The user may speak Chinese in chat, but
   **anything written into the repo is English only**.
2. **Single source of truth for parameters: `src/config.py`.**
   Never hardcode tuneable numbers (PID gains, safety limits, hardware
   constants, PWM characteristics) inside a script. Import them from
   `config`. When you add a new tuneable value, add it to `config.py`
   first, then import it.
3. **One `MotorComm`.** Use `from motor_comm import MotorComm`. Never
   re-define it inline in another script. If the class is missing a
   method you need, add it to `src/motor_comm.py` — do not copy-paste.
4. **Discuss before writing code** for any non-trivial change.
   The user prefers to review the approach first.
5. **Do not assume unknown physical quantities.** If you need a length,
   mass, angle, speed, deadband, etc. and it is not in `config.py`,
   ask the user — do not guess.
6. **Visualizations use English labels** — no non-ASCII characters in
   matplotlib titles, axis labels, legends, or annotations.

---

## Current Architecture (as of 2026-05)

```
Pi 4 (camera + PID) ──USB──▶ ESP32 (IBT-2 + DC motor + encoder)
```

The PC is **no longer in the control loop**. The Pi runs the full
pipeline in a single process:

```
picamera2 frame  →  HSV mask  →  marker angle  →  PID  →  serial  →  ESP32
```

The PC is used only for SSH-based editing, monitoring, and offline
analysis. There is no ZMQ, no NTP, no socket between Pi and PC in the
real-time path.

**Implication for AI:** scripts that depend on ZMQ between Pi and PC
(`src/camerapreview_detect_color2.py`, `src/camera_comm.py`,
`src/balance_pid.py`) are **legacy**. Do not assume ZMQ is part of the
control loop. The new entrypoint is `pi/pi_balance.py`.

---

## Directory Layout

```
cartpole-pid/
├── CLAUDE.md                        ← this file (AI entrypoint)
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

| What                              | Where                                  |
|-----------------------------------|----------------------------------------|
| Track length, wheel diameter      | `src/config.py`                        |
| Encoder pulses per revolution     | `src/config.py`                        |
| PID gains (Kp / Kd / Kx)          | `src/config.py`                        |
| Sign correctors (ANGLE/OMEGA/...) | `src/config.py`                        |
| PWM limit, PWM deadband           | `src/config.py`                        |
| Balance safety thresholds         | `src/config.py`                        |
| Serial protocol with ESP32        | `src/motor_comm.py` docstring          |
| Camera setup (FPS, resolution)    | inside the script that owns the camera |
| HSV color thresholds              | inside the script that uses them       |
| ESP32 firmware                    | `dc_motor_controller/`                 |

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

## Measured Hardware Constants (do not guess these)

| Quantity                  | Value             | Source                       |
|---------------------------|-------------------|------------------------------|
| Track length              | 0.84 m            | measured                     |
| Pulley / wheel diameter   | 0.016 m           | calipers                     |
| Encoder pulses per rev    | 616               | measured 2026-04-14          |
| Pendulum length           | 0.34 m            | measured                     |
| Pendulum mass             | 0.128 kg          | scale                        |
| Cart mass                 | 1.014 kg          | scale                        |
| Catch angle (target)      | ±5°               | tuning target                |
| Motor PWM deadband        | ~26 (estimated)   | from latency test; recalibrate |
| Vision latency (120 fps)  | 8.32 ms avg       | measured 2026-05-04          |

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

## Currently Open Items (work in progress)

- [ ] PWM deadband compensation in `pi/pi_balance.py`
- [ ] `omega` units: convert to deg/s for consistency with `theta_deg`
- [ ] CSV logging during balance runs
- [ ] Calibrate exact deadband (sweep PWM from 0 upward, log first motion)
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

**Ask the user.** Especially when you would otherwise need to invent:

- a physical specification (length, mass, angle, speed, deadband)
- which of two similar scripts is canonical
- whether a parameter is hardware-fixed or task-tuneable
- which direction is "positive" for the motor / encoder / angle

Do not guess and proceed. A short clarifying question is always cheaper
than a wrong assumption baked into committed code.
