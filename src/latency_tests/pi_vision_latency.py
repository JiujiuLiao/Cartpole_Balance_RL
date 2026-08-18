"""
LEGACY vision benchmark for the retired color2 pipeline.

It does not represent camerapreview_detect_yellow_tape_line_V1.py. Use
`python3 src/latency_tests/pi_latency_full.py --no-motor`, which imports and
times the current detector directly.

Vision pipeline latency benchmark — runs ON THE RASPBERRY PI.

Measures only the camera + OpenCV processing chain locally
(capture -> resize -> HSV -> mask -> contour -> angle).
No network, no serial, no motor — pure vision timing.

Replicates the same processing as camerapreview_detect_color2.py but times
every stage separately, so we know exactly where the time goes and what
the upper bound on the PID control loop frequency is from the vision side.

Use this to verify the Pi can keep up with the target control loop
frequency (e.g. 50 Hz) before integrating the rest of the system.
For the full chain (Pi -> PC -> ESP32 -> motor), see pc_end_to_end_latency.py.

Usage on the Raspberry Pi:
    python3 pi_vision_latency.py

Note: a yellow or blue marker MUST be visible in the frame, otherwise
findContours returns nothing and the iteration is skipped.
"""

raise SystemExit(
    "pi_vision_latency.py is the retired color2 benchmark; run "
    "pi_latency_full.py --no-motor for the current yellow-line pipeline"
)

from picamera2 import Picamera2
import cv2
import numpy as np
import math
import time

# === Camera setup (120 fps, RGB888 -> numpy gives BGR-ordered array) ===
picam = Picamera2()
config = picam.create_video_configuration(
    main={"format": "RGB888", "size": (640, 480)},
    controls={"FrameDurationLimits": (8333, 8333)},
)
picam.configure(config)
picam.start()
time.sleep(1)
try:
    picam.set_controls({"AfMode": 0, "LensPosition": 4.5})
except Exception:
    pass

# === HSV thresholds and geometry (same values as detection script) ===
gelb_min = np.array([20, 90, 90], np.uint8)
gelb_max = np.array([30, 255, 255], np.uint8)
blau_min = np.array([100, 150, 30], np.uint8)
blau_max = np.array([120, 255, 255], np.uint8)

skalierung = 0.5
width = int(640 * skalierung)
height = int(480 * skalierung)
x_ori = int(280 * skalierung)
y_ori = int(480 * skalierung)


def find_largest_contour(gelb_konturen, blau_konturen):
    if len(gelb_konturen) > 0:
        gelb_flaechen = [cv2.contourArea(k) for k in gelb_konturen]
        gi = int(np.argmax(gelb_flaechen))
        gf = gelb_flaechen[gi]
    else:
        gf = 0
        gi = -1
    if len(blau_konturen) > 0:
        blau_flaechen = [cv2.contourArea(k) for k in blau_konturen]
        bi = int(np.argmax(blau_flaechen))
        bf = blau_flaechen[bi]
    else:
        bf = 0
        bi = -1
    if gf == 0 and bf == 0:
        return None, None
    if gf > bf:
        return "gelb", gelb_konturen[gi]
    return "blau", blau_konturen[bi]


def compute_angle(kind, x_m, y_m):
    """Same 8-octant angle calculation as the detection script."""
    if y_m == y_ori and x_m > x_ori:
        return -math.pi / 2 if kind == "gelb" else math.pi / 2
    if y_m == y_ori and x_m < x_ori:
        return math.pi / 2 if kind == "gelb" else -math.pi / 2
    if y_m > y_ori and x_m == x_ori:
        return 0.0 if kind == "gelb" else math.pi
    if y_m < y_ori and x_m == x_ori:
        return math.pi if kind == "gelb" else 0.0
    if y_m < y_ori and x_m > x_ori:
        a = math.atan(abs(x_m - x_ori) / abs(y_m - y_ori))
        return -math.pi + a if kind == "gelb" else a
    if y_m > y_ori and x_m > x_ori:
        a = math.atan(abs(x_m - x_ori) / abs(y_ori - y_m))
        return -a if kind == "gelb" else math.pi - a
    if y_m > y_ori and x_m < x_ori:
        a = math.atan(abs(x_ori - x_m) / abs(y_ori - y_m))
        return -math.pi + a if kind == "gelb" else a
    a = math.atan(abs(x_ori - x_m) / abs(y_m - y_ori))
    return math.pi - a if kind == "gelb" else -a


# === Run the test ===
N = 2000
print(f"Running {N} iterations... keep a yellow/blue marker in view.")

t_capture = []
t_resize = []
t_hsv = []
t_mask = []
t_contour = []
t_angle = []
t_total = []
skipped = 0

# Discard a few frames so AE/AWB has settled
for _ in range(10):
    picam.capture_array()

while len(t_total) < N:
    s0 = time.perf_counter()
    frame = picam.capture_array()
    s1 = time.perf_counter()

    frame = cv2.resize(frame, (width, height))
    s2 = time.perf_counter()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s3 = time.perf_counter()

    gelb_maske = cv2.inRange(hsv, gelb_min, gelb_max)
    blau_maske = cv2.inRange(hsv, blau_min, blau_max)
    s4 = time.perf_counter()

    gelb_konturen, _ = cv2.findContours(gelb_maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blau_konturen, _ = cv2.findContours(blau_maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    s5 = time.perf_counter()

    kind, kontur = find_largest_contour(gelb_konturen, blau_konturen)
    if kind is None:
        skipped += 1
        continue
    M = cv2.moments(kontur)
    if M["m00"] <= 0:
        skipped += 1
        continue
    x_m = M["m10"] / M["m00"]
    y_m = M["m01"] / M["m00"]
    angle = compute_angle(kind, x_m, y_m)
    s6 = time.perf_counter()

    t_capture.append((s1 - s0) * 1000)
    t_resize.append((s2 - s1) * 1000)
    t_hsv.append((s3 - s2) * 1000)
    t_mask.append((s4 - s3) * 1000)
    t_contour.append((s5 - s4) * 1000)
    t_angle.append((s6 - s5) * 1000)
    t_total.append((s6 - s0) * 1000)

picam.stop()


# === Stats ===
def stats(name, arr):
    a = np.array(arr)
    print(f"  {name:10s}  avg={a.mean():6.2f}  max={a.max():6.2f}  "
          f"p99={np.percentile(a, 99):6.2f}  std={a.std():5.2f}  ms")


total_arr = np.array(t_total)
print(f"\n=== {N} iterations done (skipped {skipped} frames with no marker) ===")
stats("capture",  t_capture)
stats("resize",   t_resize)
stats("hsv",      t_hsv)
stats("mask",     t_mask)
stats("contour",  t_contour)
stats("angle",    t_angle)
stats("TOTAL",    t_total)

print(f"\nMean control loop rate: {1000 / total_arr.mean():.1f} Hz")
print(f"Worst-case loop rate (1/p99): {1000 / np.percentile(total_arr, 99):.1f} Hz")
print(f"Suggested PID period: {int(np.ceil(np.percentile(total_arr, 99)))} ms "
      f"(≈ {1000 / np.ceil(np.percentile(total_arr, 99)):.0f} Hz)")
