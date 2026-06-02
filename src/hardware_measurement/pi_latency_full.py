"""
pi_latency_full.py
==================
Full glass-to-motion latency — runs ON THE RASPBERRY PI (new direct architecture:
PID on Pi, USB direct to ESP32, no PC / no ZMQ / no NTP).

Unlike pi_latency.py, this version isolates the *hidden* camera capture latency
(sensor exposure + CSI transfer + buffering) using the hardware SensorTimestamp
that libcamera/picamera2 stamps at capture time. pi_latency.py started its clock
at capture_array() and therefore could never see that segment.

Segments measured (all on the Pi's own clock — no cross-device sync needed):

  ⓪ Sensor capture   = now_recv - SensorTimestamp
        photon hits sensor -> frame available in software.
        SensorTimestamp is referenced to CLOCK_BOOTTIME, the same clock we read
        with clock_gettime_ns(CLOCK_BOOTTIME). This is the "LED-free" way to get
        the true camera latency on a CSI camera.

  ① Image processing = resize -> HSV -> mask -> contour -> angle  (perf_counter delta)

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
  - A yellow or blue marker MUST be visible for ① to run.
  - ④ briefly pulses the motor each sample — keep the cart near center.
  - At 120 fps the frame interval is 8.33 ms, so ⓪ is quantized to frame
    boundaries. Run the full N samples and read the MEDIAN / P95, not one shot.
"""

from picamera2 import Picamera2
import cv2
import numpy as np
import math
import time
import sys

# Allow `from motor_comm import MotorComm` when run from this subfolder.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from motor_comm import MotorComm
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False


# ── Configuration ─────────────────────────────────────────────────────────────
args        = [a for a in sys.argv[1:] if not a.startswith("--")]
flags       = [a for a in sys.argv[1:] if a.startswith("--")]
SERIAL_PORT = args[0] if args else "/dev/ttyUSB0"
SERIAL_BAUD = 115200
N_SAMPLES   = 200
TEST_PWM    = 240        # must be above motor deadband (~22)
MAX_CART_POS = 1500      # pulses — skip ④ if cart is too far from center
ENC_TIMEOUT = 0.15       # s — max wait for the encoder to first move
DO_MOTOR    = "--no-motor" not in flags

# CLOCK_BOOTTIME is what libcamera uses for SensorTimestamp on the Raspberry Pi.
CLOCK = time.CLOCK_BOOTTIME


# ── Camera geometry / HSV (identical to camerapreview_detect_color2.py) ─────────
skalierung = 0.5
width  = int(640 * skalierung)
height = int(480 * skalierung)
x_ori  = int(280 * skalierung)
y_ori  = int(480 * skalierung)

gelb_min = np.array([ 20,  90,  90], np.uint8)
gelb_max = np.array([ 30, 255, 255], np.uint8)
blau_min = np.array([100, 150,  30], np.uint8)
blau_max = np.array([120, 255, 255], np.uint8)


def compute_angle(kind, x_m, y_m):
    if y_m == y_ori and x_m > x_ori:
        return -math.pi / 2 if kind == "gelb" else  math.pi / 2
    if y_m == y_ori and x_m < x_ori:
        return  math.pi / 2 if kind == "gelb" else -math.pi / 2
    if y_m > y_ori and x_m == x_ori:
        return 0.0           if kind == "gelb" else  math.pi
    if y_m < y_ori and x_m == x_ori:
        return  math.pi      if kind == "gelb" else  0.0
    if y_m < y_ori and x_m > x_ori:
        a = math.atan(abs(x_m - x_ori) / abs(y_m - y_ori))
        return -math.pi + a  if kind == "gelb" else a
    if y_m > y_ori and x_m > x_ori:
        a = math.atan(abs(x_m - x_ori) / abs(y_ori - y_m))
        return -a             if kind == "gelb" else math.pi - a
    if y_m > y_ori and x_m < x_ori:
        a = math.atan(abs(x_ori - x_m) / abs(y_ori - y_m))
        return -math.pi + a   if kind == "gelb" else a
    a = math.atan(abs(x_ori - x_m) / abs(y_m - y_ori))
    return  math.pi - a       if kind == "gelb" else -a


def process_frame(frame):
    """resize -> HSV -> mask -> contour -> angle. Returns angle or None (no marker)."""
    frame = cv2.resize(frame, (width, height))
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    gelb_maske = cv2.inRange(hsv, gelb_min, gelb_max)
    blau_maske = cv2.inRange(hsv, blau_min, blau_max)

    gelb_k, _ = cv2.findContours(gelb_maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blau_k, _ = cv2.findContours(blau_maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if gelb_k:
        gi = int(np.argmax([cv2.contourArea(k) for k in gelb_k]))
        gf = cv2.contourArea(gelb_k[gi])
    else:
        gf, gi = 0, -1
    if blau_k:
        bi = int(np.argmax([cv2.contourArea(k) for k in blau_k]))
        bf = cv2.contourArea(blau_k[bi])
    else:
        bf, bi = 0, -1

    if gf == 0 and bf == 0:
        return None

    kind   = "gelb" if gf >= bf else "blau"
    kontur = gelb_k[gi] if kind == "gelb" else blau_k[bi]
    M = cv2.moments(kontur)
    if M["m00"] <= 0:
        return None
    x_m = int(M["m10"] / M["m00"])
    y_m = int(M["m01"] / M["m00"])
    return compute_angle(kind, x_m, y_m)


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Pi Direct — FULL glass-to-motion latency (with SensorTimestamp)")
    print("=" * 70)
    print(f"  Serial : {SERIAL_PORT}{'  (motor test ON)' if DO_MOTOR else '  (motor test OFF)'}")
    print(f"  Samples: {N_SAMPLES}")
    print()

    # ── Camera (120 fps). Use capture_request() so we can read SensorTimestamp. ──
    picam = Picamera2()
    cfg = picam.create_video_configuration(
        main={"format": "RGB888", "size": (640, 480)},
        controls={"FrameDurationLimits": (8333, 8333)},   # ≈120 fps
    )
    picam.configure(cfg)
    picam.start()
    time.sleep(1)
    try:
        picam.set_controls({"AfMode": 0, "LensPosition": 4.5})
    except Exception:
        pass

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
    print("  Keep a yellow/blue marker visible throughout the test.")
    if comm:
        print("  Cart pulses briefly each sample for ④ — keep it near center.")
    print()

    # Discard first 10 frames (AE/AWB settling)
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

            # ── ① Image processing ────────────────────────────────
            s0 = time.perf_counter()
            angle = process_frame(frame)
            s1 = time.perf_counter()
            if angle is None:
                skipped += 1
                continue
            l1 = (s1 - s0) * 1000

            # ── ② PID compute (representative arithmetic) ──────────
            p0 = time.perf_counter()
            theta_deg = angle * 57.2958
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
        print("\n  No valid samples collected.")
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
    print(f"  Skipped frames (no marker): {skipped}")
    print("  Reminder: ⓪ is frame-referenced to start of readout; if you want the")
    print("  exposure midpoint, subtract ~exposure/2 from ⓪.")
    print("  At 120 fps ⓪ is quantized to 8.33 ms steps — trust the median/P95.")
    print("=" * 70)


if __name__ == "__main__":
    main()
