"""
test_dc_connection.py
=====================
Verify serial communication between Raspberry Pi and ESP32 DC motor controller.

Usage:
    python test_dc_connection.py
    python test_dc_connection.py COM4       # Windows
    python test_dc_connection.py /dev/ttyUSB0   # Linux / RPi
"""

import serial
import time
import sys


PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
BAUD = 115200

# Encoder calibration
PULSES_PER_REV   = 616        # measured 2026-04-14 (Method A x2: 616.7, 615.2)
WHEEL_DIAMETER_M = 0.016      # 16 mm pulley
M_PER_PULSE      = (3.14159 * WHEEL_DIAMETER_M) / PULSES_PER_REV


class Comm:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        self.ser.reset_input_buffer()
        print(f"  Connected: {port} @ {baud}")

    def send(self, msg_type, value):
        msg = f"<{msg_type},{value}>".encode()
        self.ser.reset_input_buffer()
        self.ser.write(msg)
        start = time.time()
        buf = b""
        while time.time() - start < 3:
            if self.ser.in_waiting:
                buf += self.ser.read(self.ser.in_waiting)
                if msg in buf:
                    return True
            time.sleep(0.001)
        print(f"  WARNING: no echo for {msg}")
        return False

    def send_and_read(self, msg_type, value):
        msg = f"<{msg_type},{value}>".encode()
        self.ser.reset_input_buffer()
        self.ser.write(msg)
        start = time.time()
        buf = b""
        while time.time() - start < 3:
            if self.ser.in_waiting:
                buf += self.ser.read(self.ser.in_waiting)
                if msg in buf:
                    idx = buf.find(msg)
                    payload = buf[:idx].decode(errors="ignore").strip()
                    return payload if payload else None
            time.sleep(0.001)
        return None

    def get_position(self):
        payload = self.send_and_read("p", 0)
        try:
            return int(payload) if payload else 0
        except (ValueError, TypeError):
            return 0

    def close(self):
        self.send("v", 0)
        self.ser.close()


def test_communication(comm):
    print("\n" + "=" * 50)
    print("Test 1: Communication")
    print("=" * 50)
    ok = comm.send("h", 0)
    print("  PASS" if ok else "  FAIL - check firmware and wiring")
    return ok


def test_encoder(comm):
    print("\n" + "=" * 50)
    print("Test 2: Encoder")
    print("=" * 50)
    comm.send("h", 0)
    time.sleep(0.2)
    print("  Manually turn the motor shaft for 5 seconds...")
    start = time.time()
    while time.time() - start < 5:
        pos = comm.get_position()
        cm = pos * M_PER_PULSE * 100
        print(f"\r  Encoder: {pos:>8} pulses  ({cm:+.1f} cm)", end="", flush=True)
        time.sleep(0.1)
    print()
    pos = comm.get_position()
    print(f"  Final: {pos} pulses")
    print("  PASS" if pos != 0 else "  NOTE - stayed at 0, check encoder wiring")


def test_forward(comm):
    print("\n" + "=" * 50)
    print("Test 3: Forward motion")
    print("=" * 50)
    comm.send("h", 0)
    time.sleep(0.2)
    for pwm in [80, 150, 200, 255]:
        comm.send("v", pwm)
        time.sleep(0.8)
        pos = comm.get_position()
        speed = pos * M_PER_PULSE / 0.8
        print(f"  PWM={pwm:>3}  pos={pos:>8}  ~{speed:.3f} m/s")
    comm.send("v", 0)
    time.sleep(0.3)
    # Return to origin
    _return_to_origin(comm)


def test_reverse(comm):
    print("\n" + "=" * 50)
    print("Test 4: Reverse motion")
    print("=" * 50)
    comm.send("h", 0)
    time.sleep(0.2)
    for pwm in [-80, -150, -200, -255]:
        comm.send("v", pwm)
        time.sleep(0.8)
        pos = comm.get_position()
        speed = pos * M_PER_PULSE / 0.8
        print(f"  PWM={pwm:>4}  pos={pos:>8}  ~{speed:.3f} m/s")
    comm.send("v", 0)
    time.sleep(0.3)
    _return_to_origin(comm)


def _return_to_origin(comm, return_pwm=80):
    timeout = time.time() + 5
    while time.time() < timeout:
        pos = comm.get_position()
        if abs(pos) <= 5:
            break
        comm.send("v", return_pwm if pos < 0 else -return_pwm)
        time.sleep(0.02)
    comm.send("v", 0)
    time.sleep(0.3)


def main():
    print("=" * 50)
    print("  DC Motor Connection Test")
    print("=" * 50)
    print(f"  Port: {PORT}")

    try:
        comm = Comm(PORT, BAUD)
    except Exception as e:
        print(f"\n  Failed to connect: {e}")
        return

    try:
        if not test_communication(comm):
            print("\n  Aborting - communication failed")
            return

        test_encoder(comm)

        input("\n  Press ENTER to start motor tests (ensure cart is free to move)...")
        test_forward(comm)
        test_reverse(comm)

        print("\n" + "=" * 50)
        print("  All tests complete!")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n  Interrupted")
    finally:
        comm.close()
        print("  Motor stopped.")


if __name__ == "__main__":
    main()
