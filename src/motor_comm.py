"""
Shared serial communication with the ESP32 motor controller.

Replaces the duplicated `Comm` / `SerialComm` classes that used to live
in balance_pid.py, swing_up_dc.py, swing_up_energy_dc.py, measure_ppr.py,
test_pwm_range.py, test_accel.py, pc_end_to_end_latency.py, move_cart2_center.py.

Protocol (all messages framed as ``<type,value>``):
    <v,pwm>   set velocity, pwm in [-255, 255]
    <p,0>     query encoder position; firmware echoes ``<position> <p,0>``
    <h,0>     home / reset encoder (used by swing_up_energy_dc)

Two send paths are provided:
  - send_velocity(pwm)      fire-and-forget, fastest (control-loop use)
  - send(type, value)       wait for echo, returns bool (command-style use)
  - send_and_read(...)      wait for echo, returns the payload string

Usage:
    from motor_comm import MotorComm

    # Fast control loop (e.g. balance_pid):
    comm = MotorComm(port, baud, timeout=0.05, echo_timeout=0.08)
    comm.send_velocity(pwm)

    # Command-style (e.g. swing_up_dc, measure_ppr):
    comm = MotorComm(port, baud)            # defaults: timeout=1, echo_timeout=2
    comm.send("v", 100)
    pos = comm.get_position()
"""

import time
import serial

try:
    from config import (AUTO_RECENTER_ON_CLOSE,
                        RECENTER_PWM_MAX, RECENTER_PWM_FLOOR,
                        RECENTER_SLOW_ZONE, RECENTER_TOLERANCE,
                        RECENTER_TIMEOUT_S)
except ImportError:
    # Fallback defaults if config is not on sys.path
    AUTO_RECENTER_ON_CLOSE = False
    RECENTER_PWM_MAX = 120
    RECENTER_PWM_FLOOR = 80
    RECENTER_SLOW_ZONE = 300
    RECENTER_TOLERANCE = 10
    RECENTER_TIMEOUT_S = 10.0


class MotorComm:
    PWM_MIN = -255
    PWM_MAX = 255

    def __init__(self, port, baud=115200, timeout=1.0,
                 echo_timeout=2.0, boot_wait=2.0, verbose=False,
                 auto_recenter_on_close=None):
        """
        port                    serial device, e.g. "/dev/ttyUSB0" or "COM5"
        timeout                 pyserial read timeout (small for control loops)
        echo_timeout            how long send()/get_position() waits for the echo
        boot_wait               seconds to sleep after open (ESP32 reboots on serial connect)
        verbose                 print a connection banner
        auto_recenter_on_close  if True, close() drives cart back to encoder 0
                                before disconnecting. None = read from
                                config.AUTO_RECENTER_ON_CLOSE.
                                Pass False explicitly from move_cart2_center.py
                                or any other positioning script (avoids recursion).
        """
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(boot_wait)
        self.ser.reset_input_buffer()
        self.echo_timeout = echo_timeout
        self._auto_recenter = (AUTO_RECENTER_ON_CLOSE if auto_recenter_on_close is None
                               else auto_recenter_on_close)
        if verbose:
            print(f"  Connected: {port} @ {baud}")

    # ── fast path ──────────────────────────────────────────────
    def send_velocity(self, pwm):
        """Fire-and-forget velocity command. No echo wait. Clamps to ±255."""
        pwm = max(self.PWM_MIN, min(self.PWM_MAX, int(pwm)))
        self.ser.reset_input_buffer()
        self.ser.write(f"<v,{pwm}>".encode())

    # ── reliable path ──────────────────────────────────────────
    def send(self, msg_type, value, echo_timeout=None):
        """Send <type,value> and wait until the firmware echoes it. Returns bool."""
        return self._send_wait(msg_type, value, echo_timeout) is not False

    def send_and_read(self, msg_type, value, echo_timeout=None):
        """
        Send <type,value> and return the payload that precedes the echo.
        Returns None on timeout or empty payload.
        """
        result = self._send_wait(msg_type, value, echo_timeout, return_payload=True)
        return result if result else None

    def _send_wait(self, msg_type, value, echo_timeout, return_payload=False):
        msg = f"<{msg_type},{value}>".encode()
        self.ser.reset_input_buffer()
        self.ser.write(msg)
        deadline = time.time() + (echo_timeout if echo_timeout is not None
                                  else self.echo_timeout)
        buf = b""
        while time.time() < deadline:
            if self.ser.in_waiting:
                buf += self.ser.read(self.ser.in_waiting)
                if msg in buf:
                    if return_payload:
                        return buf[:buf.find(msg)].decode(errors="ignore").strip()
                    return True
            time.sleep(0.001)
        return None if return_payload else False

    # ── encoder query ──────────────────────────────────────────
    def get_position(self, echo_timeout=None, default=0):
        """
        Query encoder position. Returns int (pulses), or `default` on failure.

        Use a small echo_timeout (e.g. 0.04-0.08) inside fast control loops.
        Pass `default=None` when the caller must distinguish a read failure
        from a true zero position (e.g. positioning scripts where the wrong
        position would drive the motor in the wrong direction).
        """
        payload = self.send_and_read("p", 0, echo_timeout=echo_timeout)
        if not payload:
            return default
        try:
            # Firmware sometimes prefixes whitespace or extra tokens; take the last token.
            return int(payload.split()[-1])
        except (ValueError, IndexError):
            return default

    # ── lifecycle ──────────────────────────────────────────────
    def stop(self):
        """Send PWM=0 without waiting for echo."""
        self.send_velocity(0)

    def _recenter(self, timeout=None):
        """
        Drive cart toward encoder 0 with proportional PWM. Used by close()
        when auto_recenter_on_close=True. Silently no-ops if encoder reads fail.
        """
        if timeout is None:
            timeout = RECENTER_TIMEOUT_S
        pos = self.get_position(default=None)
        if pos is None or abs(pos) <= RECENTER_TOLERANCE:
            return

        deadline = time.time() + timeout
        while time.time() < deadline:
            pos = self.get_position(default=None)
            if pos is None:
                time.sleep(0.05)
                continue
            if abs(pos) <= RECENTER_TOLERANCE:
                break
            dist = abs(pos)
            if dist >= RECENTER_SLOW_ZONE:
                pwm = RECENTER_PWM_MAX
            else:
                ratio = dist / RECENTER_SLOW_ZONE
                pwm = int(RECENTER_PWM_FLOOR + ratio * (RECENTER_PWM_MAX - RECENTER_PWM_FLOOR))
                pwm = max(RECENTER_PWM_FLOOR, pwm)
            # pos > 0 → cart is right of center → drive left (negative PWM)
            # pos < 0 → cart is left  of center → drive right (positive PWM)
            self.send_velocity(-pwm if pos > 0 else pwm)
            time.sleep(0.02)
        self.stop()
        time.sleep(0.3)

    def close(self):
        if self._auto_recenter:
            self._recenter()
        self.stop()
        time.sleep(0.1)
        self.ser.close()

    # context-manager support: `with MotorComm(...) as comm:`
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
