"""
pi_latency.py
=============
Latency measurement for the Pi-direct architecture — runs ON THE RASPBERRY PI.

Measures:
  ① Camera + image processing   (Pi-local, no serial needed)
  ② Serial + motor response     (Pi -> ESP32 -> encoder responds)

Compare these results against pc_end_to_end_latency.py to quantify
the improvement from removing the PC and ZMQ from the control loop.

Expected results after architecture change:
  ①  ~10 ms   (same as before)
  ②  ~16 ms   (same as old ④, now measured on Pi directly)
  Total ~26 ms  vs ~232 ms before

Usage:
    python3 pi_latency.py                   # uses /dev/ttyUSB0
    python3 pi_latency.py /dev/ttyUSB1      # specify port

Note: a yellow or blue marker MUST be visible for ① to be measured.
      ESP32 optional — ② is skipped if serial is not available.
"""

from picamera2 import Picamera2
import cv2
import numpy as np
import math
import time
import sys
import serial


# ── Configuration ─────────────────────────────────────────────────────────────
SERIAL_PORT  = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
SERIAL_BAUD  = 115200
N_SAMPLES    = 200
TEST_PWM     = 240       # must be above motor deadband (~26)
MAX_CART_POS = 1500      # pulses — skip ② if cart is too far from center
ENC_TIMEOUT  = 0.15      # s — max wait for encoder to respond


# ── Inline serial comm ────────────────────────────────────────────────────────

class MotorComm:
    def __init__(self, port, baud=115200, timeout=0.05, echo_timeout=0.08):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2.0)
        self.ser.reset_input_buffer()
        self.echo_timeout = echo_timeout

    def send_velocity(self, pwm):
        pwm = max(-255, min(255, int(pwm)))
        self.ser.reset_input_buffer()
        self.ser.write(f"<v,{pwm}>".encode())

    def get_position(self, default=0):
        msg = b"<p,0>"
        self.ser.reset_input_buffer()
        self.ser.write(msg)
        deadline = time.time() + self.echo_timeout
        buf = b""
        while time.time() < deadline:
            if self.ser.in_waiting:
                buf += self.ser.read(self.ser.in_waiting)
                if msg in buf:
                    payload = buf[:buf.find(msg)].decode(errors="ignore").strip()
                    try:
                        return int(payload.split()[-1])
                    except (ValueError, IndexError):
                        return default
            time.sleep(0.001)
        return default

    def stop(self):
        self.send_velocity(0)

    def close(self):
        self.stop()
        time.sleep(0.1)
        self.ser.close()


# ── Camera parameters (same as camerapreview_detect_color2.py) ───────────────
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Pi Direct — Latency Measurement")
    print("=" * 60)
    print(f"  Serial : {SERIAL_PORT}")
    print(f"  Samples: {N_SAMPLES}")
    print()

    # ── Camera setup ──────────────────────────────────────────
    picam = Picamera2()
    cfg = picam.create_video_configuration(
        main={"format": "RGB888", "size": (640, 480)},
        controls={"FrameRate": 120.0},
    )
    picam.configure(cfg)
    picam.start()
    time.sleep(1)
    try:
        picam.set_controls({"AfMode": 0, "LensPosition": 4.5})
    except Exception:
        pass

    # ── Serial setup (optional) ───────────────────────────────
    comm = None
    try:
        comm = MotorComm(SERIAL_PORT, SERIAL_BAUD)
        print("  Serial connected.")
    except Exception as e:
        print(f"  Serial not available ({e}) — ② will be skipped.")

    print()
    print("  Keep a yellow/blue marker visible throughout the test.")
    if comm:
        print("  Cart will briefly move each sample to measure ② — keep it near center.")
    print()

    # Discard first 10 frames (AE/AWB settling)
    for _ in range(10):
        picam.capture_array()

    print(f"  {'#':>4}  {'① ms':>8}  {'② ms':>8}")
    print(f"  {'─'*28}")

    lat1, lat2 = [], []
    skipped = 0
    n = 0

    try:
        while n < N_SAMPLES:

            # ── ① Camera + image processing ───────────────
            s0 = time.perf_counter()
            frame = picam.capture_array()

            frame = cv2.resize(frame, (width, height))
            hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            gelb_maske = cv2.inRange(hsv, gelb_min, gelb_max)
            blau_maske = cv2.inRange(hsv, blau_min, blau_max)

            gelb_k, _ = cv2.findContours(gelb_maske, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            blau_k, _ = cv2.findContours(blau_maske, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

            # Find largest contour
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
                skipped += 1
                continue

            kind   = "gelb" if gf >= bf else "blau"
            kontur = gelb_k[gi] if kind == "gelb" else blau_k[bi]

            M = cv2.moments(kontur)
            if M["m00"] <= 0:
                skipped += 1
                continue

            x_m   = int(M["m10"] / M["m00"])
            y_m   = int(M["m01"] / M["m00"])
            angle = compute_angle(kind, x_m, y_m)
            s1    = time.perf_counter()

            l1 = (s1 - s0) * 1000   # ms

            # ── ② Serial + motor response ─────────────────
            l2 = None
            if comm is not None:
                pos0 = comm.get_position()
                if abs(pos0) < MAX_CART_POS:
                    direction = 1 if angle >= 0 else -1
                    t_cmd = time.perf_counter()
                    comm.send_velocity(direction * TEST_PWM)

                    deadline = t_cmd + ENC_TIMEOUT
                    while time.perf_counter() < deadline:
                        if abs(comm.get_position() - pos0) >= 2:
                            l2 = (time.perf_counter() - t_cmd) * 1000
                            break
                        time.sleep(0.001)

                    comm.stop()
                    time.sleep(0.05)

            lat1.append(l1)
            if l2 is not None:
                lat2.append(l2)

            l2_str = f"{l2:8.1f}" if l2 is not None else "     n/a"
            print(f"  {n+1:>4}  {l1:8.1f}  {l2_str}")
            n += 1

    except KeyboardInterrupt:
        print("\n  Stopped early.")
    finally:
        if comm:
            comm.close()
        picam.stop()

    # ── Statistics ────────────────────────────────────────────
    if not lat1:
        print("\n  No valid samples collected.")
        return

    def stats(name, data):
        a = np.array(data)
        print(f"  {name:<28}  {np.mean(a):>7.1f}ms"
              f"  {np.median(a):>7.1f}ms"
              f"  {np.percentile(a, 95):>7.1f}ms"
              f"  {np.max(a):>7.1f}ms")

    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  {'Segment':<28}  {'Mean':>8}  {'Median':>8}  {'P95':>8}  {'Max':>8}")
    print(f"  {'─'*62}")
    stats("① Camera + image processing", lat1)
    if lat2:
        stats("② Serial + motor response", lat2)
        total = np.array(lat1[:len(lat2)]) + np.array(lat2)
        stats("Total (① + ②)", total.tolist())
    print()
    print(f"  Skipped frames (no marker): {skipped}")
    if lat2:
        a = np.array(lat2)
        print(f"\n  Control loop upper bound:  {1000/np.mean(np.array(lat1[:len(lat2)])+a):.0f} Hz  (mean)")
        print(f"                             {1000/np.percentile(np.array(lat1[:len(lat2)])+a,95):.0f} Hz  (P95 worst case)")
    print("=" * 70)


if __name__ == "__main__":
    main()
