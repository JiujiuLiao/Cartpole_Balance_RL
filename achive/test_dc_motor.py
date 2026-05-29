"""
Test script for DC Motor + IBT-2 + ESP32 Encoder setup.
Verifies motor control, encoder reading, and serial protocol compatibility.

Usage:
    python test_dc_motor.py [COM_PORT]

    Default port: /dev/ttyUSB0
"""

import serial
import time
import sys


# ---------------------------------------------------------------------------
#  SafeComm - identical to the one used by swing_up.py etc.
# ---------------------------------------------------------------------------
class SafeComm:
    def __init__(self, port, baud=115200):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # wait for ESP32 reboot after serial open
        self.ser.reset_input_buffer()
        print(f"Connected to {port} @ {baud}")

    def send(self, msg_type, value):
        """Send command, wait for echo. Returns True if echo received."""
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
        """Send command and return payload before the echo."""
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
        self.ser.close()


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------

def test_1_communication(comm):
    """Basic echo test."""
    print("\n" + "=" * 50)
    print("Test 1: Serial Communication")
    print("=" * 50)
    ok = comm.send("h", 0)
    if ok:
        print("  PASS - echo received")
    else:
        print("  FAIL - no echo. Check firmware & wiring.")
        return False
    return True


def test_2_encoder(comm):
    """Encoder reading and reset."""
    print("\n" + "=" * 50)
    print("Test 2: Encoder Reading")
    print("=" * 50)
    comm.send("h", 0)
    time.sleep(0.1)

    pos = comm.get_position()
    print(f"  After reset: position = {pos}  (expect 0)")

    print("  Manually rotate the motor shaft for 5 seconds...")
    start = time.time()
    while time.time() - start < 5:
        pos = comm.get_position()
        print(f"\r  Encoder count: {pos:>8}", end="", flush=True)
        time.sleep(0.1)
    print(f"\n  Final: {pos}")
    if pos != 0:
        print("  PASS - encoder is reading changes")
    else:
        print("  NOTE - count stayed 0. Either shaft was not turned,")
        print("         or encoder wiring needs checking.")
    return True


def test_3_forward(comm):
    """Motor forward at increasing PWM."""
    print("\n" + "=" * 50)
    print("Test 3: Motor Forward")
    print("=" * 50)
    comm.send("h", 0)
    time.sleep(0.2)

    for pwm in [50, 100, 150, 200, 255]:
        comm.send("v", pwm)
        time.sleep(0.8)
        pos = comm.get_position()
        print(f"  PWM = {pwm:>3}  ->  encoder = {pos:>8}")

    comm.send("v", 0)
    time.sleep(0.3)
    final = comm.get_position()
    print(f"  Stopped. Final position: {final}")
    if final > 0:
        print("  PASS - motor ran forward, encoder counted up")
    elif final < 0:
        print("  NOTE - encoder negative. Motor direction reversed vs convention.")
        print("         Swap M+/M- on IBT-2, or swap encoder A/B wires.")
    else:
        print("  FAIL - no movement detected")
    return True


def test_4_reverse(comm):
    """Motor reverse at increasing PWM."""
    print("\n" + "=" * 50)
    print("Test 4: Motor Reverse")
    print("=" * 50)
    comm.send("h", 0)
    time.sleep(0.2)

    for pwm in [-50, -100, -150, -200, -255]:
        comm.send("v", pwm)
        time.sleep(0.8)
        pos = comm.get_position()
        print(f"  PWM = {pwm:>4}  ->  encoder = {pos:>8}")

    comm.send("v", 0)
    time.sleep(0.3)
    final = comm.get_position()
    print(f"  Stopped. Final position: {final}")
    if final < 0:
        print("  PASS - motor ran reverse, encoder counted down")
    elif final > 0:
        print("  NOTE - encoder positive for reverse. See note in Test 3.")
    else:
        print("  FAIL - no movement detected")
    return True


def test_5_bidirectional(comm):
    """Run forward then reverse, check if encoder returns near zero."""
    print("\n" + "=" * 50)
    print("Test 5: Bidirectional Symmetry")
    print("=" * 50)
    comm.send("h", 0)
    time.sleep(0.2)

    print("  Forward PWM=150 for 2 seconds...")
    comm.send("v", 150)
    time.sleep(2)
    comm.send("v", 0)
    time.sleep(0.3)
    pos_fwd = comm.get_position()
    print(f"  After forward:  {pos_fwd:>8} counts")

    print("  Reverse PWM=-150 for 2 seconds...")
    comm.send("v", -150)
    time.sleep(2)
    comm.send("v", 0)
    time.sleep(0.3)
    pos_final = comm.get_position()
    print(f"  After reverse:  {pos_final:>8} counts")

    travel = abs(pos_fwd)
    error = abs(pos_final)
    pct = error / max(travel, 1) * 100
    print(f"  Return error: {error} counts ({pct:.1f}% of travel)")
    if pct < 20:
        print("  PASS - good symmetry")
    else:
        print("  NOTE - some asymmetry (normal for DC gearmotor friction/backlash)")
    return True


def test_6_speed_measurement(comm):
    """Measure encoder counts/sec at various PWM levels."""
    print("\n" + "=" * 50)
    print("Test 6: Speed Measurement")
    print("=" * 50)
    print("  (Assuming ~1320 counts/output-rev for 11PPR encoder + 30:1 gear)")
    print()

    for pwm in [80, 120, 180, 255]:
        comm.send("h", 0)
        time.sleep(0.3)

        comm.send("v", pwm)
        time.sleep(0.2)  # let motor stabilize
        pos1 = comm.get_position()
        t1 = time.time()

        time.sleep(1.0)

        pos2 = comm.get_position()
        t2 = time.time()
        comm.send("v", 0)

        dt = t2 - t1
        cps = (pos2 - pos1) / dt
        # 1320 CPR is a common value; your motor may differ
        rpm_est = abs(cps) / 1320 * 60
        print(f"  PWM={pwm:>3}:  {cps:>8.0f} counts/s  (~{rpm_est:.0f} RPM output)")

    time.sleep(0.3)
    print()
    print("  If RPM values look wrong, your encoder CPR might differ from 1320.")
    print("  Calibrate: rotate output shaft exactly 1 full turn, note the count.")
    return True


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"

    print("=" * 50)
    print("  DC Motor + IBT-2 + Encoder  Test Suite")
    print("=" * 50)
    print(f"  Port: {port}")

    try:
        comm = SafeComm(port)
    except Exception as e:
        print(f"\nFailed to connect: {e}")
        print("Usage:  python test_dc_motor.py [COM_PORT]")
        return

    try:
        # --- Test 1: must pass before anything else ---
        if not test_1_communication(comm):
            print("\nAborting - communication failed.")
            return

        # --- Test 2: encoder (no motor movement) ---
        test_2_encoder(comm)

        # --- Motor tests ---
        print()
        input("Press ENTER to start motor tests (cart must be free to move!) ...")

        test_3_forward(comm)
        test_4_reverse(comm)
        test_5_bidirectional(comm)
        test_6_speed_measurement(comm)

        print()
        print("=" * 50)
        print("  All tests complete!")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n\nInterrupted - stopping motor...")
        comm.send("v", 0)
    finally:
        comm.send("v", 0)
        time.sleep(0.1)
        comm.close()
        print("Motor stopped, port closed.")


if __name__ == "__main__":
    main()
