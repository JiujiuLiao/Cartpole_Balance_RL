"""
pi_latency_full.py
==================
Full glass-to-motion latency — runs ON THE RASPBERRY PI (new direct architecture:
PID on Pi, USB direct to ESP32, no PC / no ZMQ / no NTP).

Unlike pi_latency.py, this version:
  1. Isolates the *hidden* camera capture latency (sensor exposure + CSI transfer
     + buffering) using the hardware SensorTimestamp that libcamera/picamera2
     stamps at capture time. pi_latency.py started its clock at capture_array()
     and therefore could never see that segment.
  2. Uses the REAL V3 vision pipeline by importing detect_centroid / compute_angle
     straight from camerapreview_detect_color_V3.py, and replicates V3's camera
     config (768x432, fixed exposure/gain, AE/AWB off). So ⓪/① reflect what the
     controller actually runs — not an outdated v2 approximation.

Segments measured (all on the Pi's own clock — no cross-device sync needed):

  ⓪ Sensor capture   = now_recv - SensorTimestamp
        photon hits sensor -> frame available in software.
        SensorTimestamp is referenced to CLOCK_BOOTTIME, the same clock we read
        with clock_gettime_ns(CLOCK_BOOTTIME). This is the "LED-free" way to get
        the true camera latency on a CSI camera.

  ① Image processing = V3 pipeline: BGR2HSV -> detect_centroid(yellow) +
        detect_centroid(blue) -> compute_angle  (perf_counter delta)

  ② PID compute      = the control arithmetic itself (normally < 0.1 ms)

  ③ Comm RTT         = Pi -> ESP32 -> echo back   (pure serial round-trip, <p,0>)

  ④ Motor response   = command sent -> encoder first moves
        includes one-way comm + ESP32 parse + mechanical start.
        mechanical-only  ≈  ④ - ③/2.

  Total glass-to-motion ≈ ⓪ + ① + ② + (④ - ③/2)
        (④ is measured via an encoder read-back, so we subtract the return
         comm leg ③/2 to keep it a forward-only estimate.)

Usage on the Pi:
    python3 pi_latency_full.py                 # uses /dev/ttyUSB0
    python3 pi_latency_full.py /dev/ttyUSB1    # specify port
    python3 pi_latency_full.py /dev/ttyUSB0 --no-motor   # skip ④ (no cart movement)

Notes:
  - BOTH the yellow and blue markers must be visible for ① to run (V3 needs both
    for the dual-color vector; a single marker falls back to the axis but that
    path is not what we want to time).
  - ④ briefly pulses the motor each sample — keep the cart near center.
  - At 120 fps the frame interval is 8.33 ms, so ⓪ is quantized to frame
    boundaries. Run the full N samples and read the MEDIAN / P95, not one shot.
"""

from picamera2 import Picamera2
import numpy as np
import time
import sys
from pathlib import Path

# This folder (for the V3 detector) and its parent (for motor_comm) on the path.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import cv2  # noqa: E402  (kept explicit; V3 also imports it)

# ── Reuse the REAL V3 vision logic and camera config ──────────────────────────
# Importing the module only defines functions/constants — main() is guarded by
# __name__ == "__main__", so no camera/ZMQ/argparse runs on import.
from camerapreview_detect_color_V3 import (   # noqa: E402
    detect_centroid, compute_angle, load_hsv, load_axis,
    W, H, RAW_W, RAW_H, FRAME_DURATION_US, EXPOSURE_US,
    ANALOGUE_GAIN, LENS_POSITION,
)

try:
    from motor_comm import MotorComm   # noqa: E402
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False


# ── Configuration ─────────────────────────────────────────────────────────────
_args       = [a for a in sys.argv[1:] if not a.startswith("--")]
_flags      = [a for a in sys.argv[1:] if a.startswith("--")]
SERIAL_PORT = _args[0] if _args else "/dev/ttyUSB0"
SERIAL_BAUD = 115200
N_SAMPLES   = 200
TEST_PWM    = 240        # must be above motor deadband (~22)
MAX_CART_POS = 1500      # pulses — skip ④ if cart is too far from center
ENC_TIMEOUT = 0.15       # s — max wait for the encoder to first move
DO_MOTOR    = "--no-motor" not in _flags

# CLOCK_BOOTTIME is what libcamera uses for SensorTimestamp on the Raspberry Pi.
CLOCK = time.CLOCK_BOOTTIME


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Pi Direct — FULL glass-to-motion latency (V3 pipeline + SensorTimestamp)")
    print("=" * 70)
    print(f"  Serial : {SERIAL_PORT}{'  (motor test ON)' if DO_MOTOR else '  (motor test OFF)'}")
    print(f"  Frame  : {W}x{H}  exposure={EXPOSURE_US}us  gain={ANALOGUE_GAIN}")
    print(f"  Samples: {N_SAMPLES}")
    print()

    yl, yh, bl, bh = load_hsv()
    axis = load_axis()

    # ── Camera: replicate V3's config exactly, so ⓪/① match runtime. ──────────
    picam = Picamera2()
    cfg = picam.create_video_configuration(
        main={"size": (W, H), "format": "RGB888"},
        raw={"size": (RAW_W, RAW_H)},
        controls={
            "FrameDurationLimits": (FRAME_DURATION_US, FRAME_DURATION_US),
            "ExposureTime": EXPOSURE_US,
            "AnalogueGain": ANALOGUE_GAIN,
            "AwbEnable": False,
            "AeEnable": False,
            "LensPosition": LENS_POSITION,
        },
        buffer_count=4,
    )
    picam.configure(cfg)
    picam.start()
    time.sleep(0.5)

    # ── Serial (optional) ─────────────────────────────────────────
    comm = None
    if DO_MOTOR and SERIAL_OK:
        try:
            comm = MotorComm(SERIAL_PORT, SERIAL_BAUD, timeout=0.05, echo_timeout=0.08)
            print("  Serial connected.")
        except Exception as e:
            print(f"  Serial not available ({e}) — ③④ skipped.")
    elif DO_MOTOR and not SERIAL_OK:
        print("  pyserial / motor_comm not importable — ③④ skipped.")

    print()
    print("  Keep BOTH yellow and blue markers visible throughout the test.")
    if comm:
        print("  Cart pulses briefly each sample for ④ — keep it near center.")
    print()

    # Discard first 10 frames (pipeline settling)
    for _ in range(10):
        picam.capture_array()

    print(f"  {'#':>4}  {'⓪sens':>8}  {'①proc':>8}  {'②pid':>7}  {'③rtt':>7}  {'④motor':>8}  {'exp ms':>7}")
    print(f"  {'─'*60}")

    lat0, lat1, lat2, lat3, lat4, exps = [], [], [], [], [], []
    skipped = 0
    n = 0

    try:
        while n < N_SAMPLES:

            # ── ⓪ Sensor capture latency ──────────────────────────
            req = picam.capture_request()
            now_ns = time.clock_gettime_ns(CLOCK)
            md = req.get_metadata()
            ts_sensor = md.get("SensorTimestamp")        # ns, CLOCK_BOOTTIME
            exp_us    = md.get("ExposureTime", 0)        # microseconds
            frame = req.make_array("main")
            req.release()

            if ts_sensor is None:
                print("  [!] No SensorTimestamp in metadata — picamera2 too old?")
                break
            l0 = (now_ns - ts_sensor) / 1e6              # ms

            # ── ① Image processing (REAL V3 pipeline) ─────────────
            s0 = time.perf_counter()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            yc = detect_centroid(hsv, yl, yh)
            bc = detect_centroid(hsv, bl, bh)
            angle_raw, mode = compute_angle(yc, bc, axis)
            s1 = time.perf_counter()
            if angle_raw is None:
                skipped += 1
                continue
            l1 = (s1 - s0) * 1000

            # ── ② PID compute (representative arithmetic) ──────────
            p0 = time.perf_counter()
            theta_deg = angle_raw * 57.2958
            err = theta_deg                               # error vs upright (0°)
            pwm = -8.0 * err                              # placeholder Kp; compute cost only
            pwm = max(-240.0, min(240.0, pwm))
            l2 = (time.perf_counter() - p0) * 1000

            # ── ③ Comm RTT (pure Pi <-> ESP32 serial round-trip) ──
            l3 = None
            if comm is not None:
                r0 = time.perf_counter()
                comm.send("p", 0)                         # waits for the <p,0> echo
                l3 = (time.perf_counter() - r0) * 1000

            # ── ④ Motor response (command -> encoder first moves) ──
            l4 = None
            if comm is not None:
                pos0 = comm.get_position()
                if abs(pos0) < MAX_CART_POS:
                    direction = 1 if pwm >= 0 else -1
                    t_cmd = time.perf_counter()
                    comm.send_velocity(direction * TEST_PWM)
                    deadline = t_cmd + ENC_TIMEOUT
                    while time.perf_counter() < deadline:
                        if abs(comm.get_position() - pos0) >= 2:
                            l4 = (time.perf_counter() - t_cmd) * 1000
                            break
                        time.sleep(0.001)
                    comm.stop()
                    time.sleep(0.05)

            # ── store ──────────────────────────────────────────────
            lat0.append(l0)
            lat1.append(l1)
            lat2.append(l2)
            exps.append(exp_us / 1000.0)
            if l3 is not None:
                lat3.append(l3)
            if l4 is not None:
                lat4.append(l4)

            l3s = f"{l3:7.1f}" if l3 is not None else "    n/a"
            l4s = f"{l4:8.1f}" if l4 is not None else "     n/a"
            print(f"  {n+1:>4}  {l0:8.1f}  {l1:8.1f}  {l2:7.3f}  {l3s}  {l4s}  {exp_us/1000.0:7.2f}")
            n += 1

    except KeyboardInterrupt:
        print("\n  Stopped early.")
    finally:
        if comm:
            comm.close()
        picam.stop()

    # ── Statistics ────────────────────────────────────────────────
    if not lat0:
        print("\n  No valid samples collected (markers not both visible?).")
        return

    def stats(name, data, unit="ms"):
        if not data:
            return
        a = np.array(data)
        print(f"  {name:<26}  {np.mean(a):>7.2f}  {np.median(a):>7.2f}"
              f"  {np.percentile(a, 95):>7.2f}  {np.max(a):>7.2f}  {unit}")

    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  {'Segment':<26}  {'Mean':>7}  {'Median':>7}  {'P95':>7}  {'Max':>7}")
    print(f"  {'─'*62}")
    stats("⓪ Sensor capture",        lat0)
    stats("① Image processing",      lat1)
    stats("② PID compute",           lat2)
    stats("③ Comm RTT",              lat3)
    stats("④ Motor response",        lat4)
    stats("   exposure time",        exps)

    # Forward glass-to-motion estimate.
    print(f"  {'─'*62}")
    n_t = min(len(lat0), len(lat1), len(lat2))
    total = np.array(lat0[:n_t]) + np.array(lat1[:n_t]) + np.array(lat2[:n_t])
    if lat4:
        n_t = min(n_t, len(lat4))
        rtt_half = (np.median(lat3) / 2.0) if lat3 else 0.0
        total = (np.array(lat0[:n_t]) + np.array(lat1[:n_t]) + np.array(lat2[:n_t])
                 + np.array(lat4[:n_t]) - rtt_half)
        label = "Total glass→motion (-rtt/2)"
    else:
        label = "Total ⓪+①+② (no motor)"
    stats(label, total.tolist())

    print()
    print(f"  Skipped frames (no valid angle): {skipped}")
    print("  Reminder: ⓪ is frame-referenced to start of readout; if you want the")
    print("  exposure midpoint, subtract ~exposure/2 from ⓪.")
    print("  At 120 fps ⓪ is quantized to 8.33 ms steps — trust the median/P95.")
    print("=" * 70)


if __name__ == "__main__":
    main()
