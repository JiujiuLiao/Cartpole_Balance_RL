"""
Shared ZMQ subscriber for the camera angle stream.

Replaces the duplicated context/socket boilerplate in:
  balance_pid.py, swing_up_energy_dc.py, log_angle.py,
  hardware_measurement/{acquire_pendulum_dynamics.py, pc_end_to_end_latency.py}.

The publisher is camerapreview_detect_color2.py on the Raspberry Pi.

Wire format (CSV string from the Pi):
  parts[0..4]  five most recent angles (rad); parts[4] is the newest
  parts[5]     average angular velocity (rad/s)
  parts[6]     pole_up flag, "1"/"0" or "True"/"False"
  parts[7]     Pi loop start timestamp     (added 2026-04 for latency tests)
  parts[8]     Pi angle-computed timestamp (added 2026-04 for latency tests)
"""

import zmq


class CameraSubscriber:
    """ZMQ SUB socket for the Pi camera angle stream."""

    def __init__(self, addr, recv_timeout_ms=200, conflate=False):
        """
        addr             e.g. "tcp://192.168.0.109:9999"
        recv_timeout_ms  blocking-recv timeout in milliseconds
        conflate         if True, keep only the latest message in the buffer
                         (use for real-time control and latency measurement;
                          do NOT use for data-logging scripts that need every frame)
        """
        self.ctx  = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        if conflate:
            self.sock.setsockopt(zmq.CONFLATE, 1)  # must be set before connect()
        self.sock.connect(addr)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)

    def drain(self):
        """Discard any buffered messages. Returns count drained."""
        n = 0
        while True:
            try:
                self.sock.recv_string(zmq.NOBLOCK)
                n += 1
            except zmq.Again:
                return n

    def recv(self):
        """
        Blocking receive (respects the RCVTIMEO set in __init__).
        Returns the raw CSV string, or None if no message arrived in time.
        """
        try:
            return self.sock.recv_string()
        except zmq.Again:
            return None

    def latest(self):
        """
        Drain the buffer and return ONLY the most recent message string.
        Returns None if the buffer was empty.

        Use this in real-time control loops so you always act on the
        freshest angle, never on a backlog of stale frames.
        """
        msg = None
        while True:
            try:
                msg = self.sock.recv_string(zmq.NOBLOCK)
            except zmq.Again:
                return msg

    def latest_angle(self):
        """
        Drain + parse the most recent message.
        Returns (theta_rad, omega_rad_s, pole_up) or None.
        """
        msg = self.latest()
        return parse_angle(msg) if msg is not None else None

    def latest_angle_ts(self):
        """
        Drain + parse the most recent message, INCLUDING the capture timestamp.
        Returns (theta_rad, omega_rad_s, pole_up, sensor_ts_ns) or None.

        sensor_ts_ns is parts[8] = SensorTimestamp (ns, CLOCK_BOOTTIME): the
        photon-capture time stamped by libcamera. Compare it against
        time.clock_gettime_ns(time.CLOCK_BOOTTIME) to get the true per-frame age
        (including the ~20ms camera capture latency) for dynamic delay
        compensation. None if the publisher does not send it.
        """
        msg = self.latest()
        return parse_angle_ts(msg) if msg is not None else None

    def close(self):
        self.sock.close()
        self.ctx.destroy()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def parse_angle(msg):
    """
    Parse a raw CSV message into (theta_rad, omega_rad_s, pole_up).
    Returns None on parse failure or if the message has too few fields.

    theta_rad   parts[4]   most recent angle, rad (0=upright, ±π=hanging)
    omega       parts[5]   average angular velocity, rad/s
    pole_up     parts[6]   bool, True if pendulum is currently near upright
    """
    try:
        # Both comma- and space-separated formats are accepted (older Pi code used spaces).
        parts = msg.split(",") if "," in msg else msg.split()
        if len(parts) < 7:
            return None
        theta   = float(parts[4])
        omega   = float(parts[5])
        pole_up = parts[6].strip() in ("1", "True", "true")
        return theta, omega, pole_up
    except (ValueError, IndexError):
        return None


def parse_angle_ts(msg):
    """
    Like parse_angle(), but also returns the hardware capture timestamp.
    Returns (theta_rad, omega_rad_s, pole_up, sensor_ts_ns) or None.

    sensor_ts_ns  parts[8]  SensorTimestamp in ns on CLOCK_BOOTTIME (photon-capture
                            time stamped by libcamera). None if the field is absent
                            or 0 (older publisher / metadata unavailable). Compare
                            against time.clock_gettime_ns(time.CLOCK_BOOTTIME).
    """
    try:
        parts = msg.split(",") if "," in msg else msg.split()
        if len(parts) < 7:
            return None
        theta   = float(parts[4])
        omega   = float(parts[5])
        pole_up = parts[6].strip() in ("1", "True", "true")
        sensor_ts_ns = None
        if len(parts) > 8:
            v = int(parts[8])
            sensor_ts_ns = v if v > 0 else None      # 0 == "unavailable" sentinel
        return theta, omega, pole_up, sensor_ts_ns
    except (ValueError, IndexError):
        return None
