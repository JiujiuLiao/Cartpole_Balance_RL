"""
pc_end_to_end_latency.py
========================
End-to-end latency measurement — runs ON THE PC.

Measures the full control chain across 3 devices:
  Pi (vision) -> ZMQ -> PC (PID) -> Serial -> ESP32 (motor) -> encoder.

For vision-only timing on the Pi, see pi_vision_latency.py instead.

Chain measured:
  Pi captures frame
      ↓ [①] camera + image processing  =  parts[8] - parts[7]   (Pi internal, no NTP needed)
  Pi sends ZMQ
      ↓ [②] network transmission       =  t_recv   - parts[8]   (needs NTP sync)
  PC receives ZMQ
      ↓ [③] PID computation            =  measured here (~0ms)
             + loop phase delay        =  estimated 0~20ms in real controller (50Hz)
  PC sends serial command
      ↓ [④] serial + motor response    =  t_encoder - t_cmd      (PC internal, no NTP needed)
  Cart moves

Requirements:
  1. Pi: run camerapreview_detect_color2.py (current_time added as parts[8])
  2. NTP sync on both Pi and PC (for ② measurement)
  3. ESP32 connected via serial (optional, for ④ measurement)

Usage:
  python pc_end_to_end_latency.py COM6 tcp://192.168.0.109:9999
  python pc_end_to_end_latency.py               # uses defaults
  python pc_end_to_end_latency.py COM6          # serial only, default ZMQ
"""

import time
import sys
import numpy as np
from pathlib import Path

# Allow `from motor_comm import ...` / `from config import ...` when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from motor_comm import MotorComm
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False

from camera_comm import CameraSubscriber
from config import (DEFAULT_SERIAL_PORT, SERIAL_BAUD, DEFAULT_ZMQ_ADDR,
                    Kp, Kd, Kx, ANGLE_SIGN, OMEGA_SIGN, CART_SIGN, PWM_LIMIT)

# ── Configuration ─────────────────────────────────
SERIAL_PORT  = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERIAL_PORT
ZMQ_ADDR     = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ZMQ_ADDR
N_SAMPLES    = 200       # number of ZMQ messages to measure
TEST_PWM     = 240       # PWM for ④ test — must be above deadband (~26)
MAX_CART_POS = 1500       # pulses — skip ④ if cart is too far from center
ENC_TIMEOUT  = 0.15      # s — max wait time for encoder to respond in ④


# ── Main ──────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Inverted Pendulum — Latency Measurement")
    print("=" * 65)
    print(f"  ZMQ    : {ZMQ_ADDR}")
    print(f"  Serial : {SERIAL_PORT}")
    print(f"  Samples: {N_SAMPLES}")
    print()

    # ── ZMQ setup ─────────────────────────────────
    cam = CameraSubscriber(ZMQ_ADDR, recv_timeout_ms=3000, conflate=True)

    print("  Flushing stale ZMQ frames...", end="", flush=True)
    flushed = cam.drain()
    print(f" {flushed} flushed.")

    # ── Serial setup ──────────────────────────────
    comm = None
    if SERIAL_OK:
        try:
            comm = MotorComm(SERIAL_PORT, SERIAL_BAUD,
                             timeout=0.05, echo_timeout=0.04)
            print("  Serial connected.")
        except Exception as e:
            print(f"  Serial not available ({e}) — ④ will be skipped.")
    else:
        print("  pyserial not installed — ④ will be skipped.")

    print()
    print("  [Note] NTP sync is required for ② to be accurate.")
    print("         If ② shows negative values, the clocks are not synced.")
    print()
    print(f"  {'#':>4}  {'① ms':>8}  {'② ms':>8}  {'①+② ms':>9}  {'③ ms':>8}  {'④ ms':>8}  {'FPS':>7}")
    print(f"  {'─'*68}")

    lat1, lat2, lat12, lat3, lat4, lat_fps = [], [], [], [], [], []
    prev_pi_start = None
    neg_count = 0
    n = 0

    try:
        while n < N_SAMPLES:

            # ── Receive ZMQ message ───────────────
            msg = cam.recv()
            if msg is None:
                print("  [!] ZMQ timeout — is the Pi camera script running?")
                continue

            t_recv = time.time()

            parts = msg.split(",")

            # Verify that the message has 9 fields (Pi code updated)
            if len(parts) < 9:
                print("  [!] Message has fewer than 9 fields.")
                print("      Check that current_time was added to the Pi ZMQ message.")
                continue

            try:
                t_pi_start = float(parts[7])   # loop start on Pi
                t_pi_done  = float(parts[8])   # angle computed, ready to send
            except ValueError:
                continue

            # ── Camera FPS ────────────────────────
            fps = None
            if prev_pi_start is not None:
                interval = t_pi_start - prev_pi_start
                if 0 < interval < 1.0:   # Sanity check: ignore gaps longer than 1s
                    fps = 1.0 / interval
            prev_pi_start = t_pi_start

            # ── ① Camera + image processing ───────
            l1 = (t_pi_done - t_pi_start) * 1000   # ms

            # ── ② ZMQ network transmission ────────
            l2 = (t_recv - t_pi_done) * 1000        # ms

            # ── ①+② Combined ─────────────────────
            l12 = (t_recv - t_pi_start) * 1000      # ms

            # ── ③ PID computation time ────────────
            t_pid_start = time.time()
            theta_deg = float(parts[4]) * 57.2958
            omega     = float(parts[5])
            err  = theta_deg     # camera convention: upright = 0°
            _pwm = (ANGLE_SIGN * (-Kp * err)
                  + OMEGA_SIGN * ( Kd * omega))
            # Kx * cart_pos omitted: cart position not read at this point
            _pwm = max(-float(PWM_LIMIT), min(float(PWM_LIMIT), _pwm))
            l3 = (time.time() - t_pid_start) * 1000  # ms

            # ── ④ Serial + motor response ─────────
            l4 = None
            if comm is not None:
                pos0 = comm.get_position()
                if abs(pos0) < MAX_CART_POS:
                    direction = 1 if _pwm >= 0 else -1
                    t_cmd = time.time()
                    comm.send_velocity(direction * TEST_PWM)

                    deadline = t_cmd + ENC_TIMEOUT
                    while time.time() < deadline:
                        if abs(comm.get_position() - pos0) >= 2:
                            l4 = (time.time() - t_cmd) * 1000
                            break
                        time.sleep(0.001)

                    comm.stop()
                    time.sleep(0.05)

            # ── Validate and store ─────────────────
            if l1 < 0 or l1 > 500:
                continue

            if l2 < -50:
                neg_count += 1
                if neg_count <= 5:
                    print(f"  [!] ② negative ({l2:.1f}ms) — clock offset too large, run NTP sync again.")
                continue

            lat1.append(l1)
            lat2.append(l2)
            lat12.append(l12)
            lat3.append(l3)
            if l4 is not None:
                lat4.append(l4)
            if fps is not None:
                lat_fps.append(fps)

            l4_str  = f"{l4:8.1f}"  if l4  is not None else "     n/a"
            fps_str = f"{fps:7.1f}" if fps is not None else "    n/a"
            print(f"  {n+1:>4}  {l1:8.1f}  {l2:8.1f}  {l12:9.1f}  {l3:8.3f}  {l4_str}  {fps_str}")
            n += 1

    except KeyboardInterrupt:
        print("\n  Stopped early.")
    finally:
        if comm:
            comm.close()
        cam.close()

    # ── Statistics ────────────────────────────────
    if not lat1:
        print("\n  No valid samples collected.")
        return

    print()
    print("=" * 65)
    print("  Summary")
    print("=" * 65)
    print(f"  {'Segment':<20}  {'Mean':>8}  {'Median':>8}  {'P95':>8}  {'Max':>8}")
    print(f"  {'─'*58}")

    def row(name, data):
        if not data:
            return
        a = np.array(data)
        print(f"  {name:<20}  {np.mean(a):>7.1f}ms"
              f"  {np.median(a):>7.1f}ms"
              f"  {np.percentile(a,95):>7.1f}ms"
              f"  {np.max(a):>7.1f}ms")

    row("① Camera + image processing", lat1)
    row("② ZMQ transmission", lat2)
    row("①+② Total", lat12)
    row("③ PID computation", lat3)
    if lat4:
        row("④ Serial + motor", lat4)
    if lat_fps:
        a = np.array(lat_fps)
        print(f"  {'Camera FPS':<20}  {np.mean(a):>7.1f}fps"
              f"  {np.median(a):>7.1f}fps"
              f"  {np.percentile(a,5):>7.1f}fps"
              f"  {np.min(a):>7.1f}fps  ← min")

    # Total latency
    print(f"  {'─'*58}")
    n_total = min(len(lat1), len(lat2), len(lat3))
    total = (np.array(lat1[:n_total])
           + np.array(lat2[:n_total])
           + np.array(lat3[:n_total]))
    if lat4:
        n_total = min(n_total, len(lat4))
        total = total[:n_total] + np.array(lat4[:n_total])

    a = total
    print(f"  {'Total latency':<20}  {np.mean(a):>7.1f}ms"
          f"  {np.median(a):>7.1f}ms"
          f"  {np.percentile(a,95):>7.1f}ms"
          f"  {np.max(a):>7.1f}ms")

    print()
    print("  Note: ③ here measures only PID computation time (usually <0.1 ms).")
    print("        The real controller (50 Hz loop) adds an extra 0~20 ms loop wait, averaging about 10 ms.")
    if neg_count > 0:
        print(f"\n  Warning: {neg_count} samples were dropped because ② was negative.")
        print("        Please resync clocks with NTP and measure again.")
    print("=" * 65)


if __name__ == "__main__":
    main()
