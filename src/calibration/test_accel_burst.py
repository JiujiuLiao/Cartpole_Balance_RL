"""
test_accel_burst.py
===================
Accurate cart max-acceleration test using the ESP32-side burst command <a,PWM>.

Why this is more accurate than test_accel.py:
  test_accel.py samples the encoder by polling <p,0> from the Pi inside the
  control loop, so every sample carries a variable serial round-trip latency
  and the Pi-side timestamp does not line up with when the ESP32 actually read
  the encoder. That jitter corrupts the velocity, and the acceleration (a
  second derivative) magnifies it.

  Here the ESP32 stamps micros() and reads its OWN encoder on a true 1 ms grid
  into RAM (firmware command <a,PWM>), then dumps the whole buffer afterwards.
  No serial sits in the timing loop, so Pi<->ESP32 latency is outside the
  measurement. We fit POSITION to x=1/2*a*t^2 (no differentiation), which is
  far less noise-sensitive than fitting velocity.

Requires the updated dc_motor_controller.ino (adds the <a,PWM> command).

Usage:
    python test_accel_burst.py                       # default port, PWM 200, both dirs, 5 runs
    python test_accel_burst.py COM5 --pwm 200 255 --runs 5 --dir both --plot
    python test_accel_burst.py --csv burst_200_a_1.csv   # offline re-analysis
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motor_comm import MotorComm
from config import DEFAULT_SERIAL_PORT, SERIAL_BAUD, M_PER_PULSE

G = 9.81  # m/s^2

# Cart-positioning params (match test_accel.py conventions)
RETURN_PWM     = 120     # PWM to drive back to center (above deadband)
STANDSTILL_THR = 3       # pulses/s below which the cart is "stopped"
STANDSTILL_S   = 0.4     # must stay still this long before a trial


# ── Cart positioning ──────────────────────────────────────────────
def return_to_origin(comm, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        pos = comm.get_position()
        if abs(pos) <= 5:
            break
        comm.send("v", RETURN_PWM if pos < 0 else -RETURN_PWM)
        time.sleep(0.02)
    comm.stop()
    time.sleep(0.3)


def wait_for_standstill(comm, timeout=3.0):
    still_since, prev = None, comm.get_position()
    prev_t, deadline = time.time(), time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.05)
        now = time.time()
        pos = comm.get_position()
        dt = now - prev_t
        vel = abs(pos - prev) / dt if dt > 0 else 0
        if vel < STANDSTILL_THR:
            if still_since is None:
                still_since = now
            elif now - still_since >= STANDSTILL_S:
                return True
        else:
            still_since = None
        prev, prev_t = pos, now
    return False


# ── Burst capture ─────────────────────────────────────────────────
def run_burst(comm, pwm, read_timeout=6.0):
    """Send <a,PWM>, capture the firmware CSV dump. Returns (t_s, x_m)."""
    comm.send("h", 0)
    comm.stop()
    wait_for_standstill(comm)

    comm.ser.reset_input_buffer()
    comm.ser.write(f"<a,{pwm}>".encode())

    lines, recording, deadline = [], False, time.time() + read_timeout
    while time.time() < deadline:
        raw = comm.ser.readline().decode(errors="replace").strip()
        if not raw:
            continue
        if raw.startswith("ACCEL_BEGIN"):
            recording, lines = True, []
            continue
        if raw.startswith("ACCEL_END"):
            break
        if recording and not raw.startswith("t_us"):
            lines.append(raw)
    comm.stop()

    t_us, pos = [], []
    for ln in lines:
        parts = ln.split(",")
        if len(parts) != 2:
            continue
        try:
            t_us.append(float(parts[0]))
            pos.append(float(parts[1]))
        except ValueError:
            continue
    t = np.array(t_us) / 1e6                  # s
    x = np.array(pos) * M_PER_PULSE           # m
    return t, x


# ── Analysis ──────────────────────────────────────────────────────
def analyze(t, x):
    """Fit x = 1/2 a t^2 over the early (still-accelerating) region."""
    x = x - x[0]
    if x[-1] < 0:
        x = -x                                # make travel positive

    v = np.gradient(x, t)
    if len(v) >= 5:
        v = np.convolve(v, np.ones(5) / 5.0, mode="same")
    v_term = float(np.median(v[int(0.7 * len(v)):])) if len(v) > 5 else float(v[-1])

    # early region = the contiguous initial accelerating part, ending at the
    # FIRST time velocity reaches 60% of terminal. Using the first crossing
    # (not the last index below threshold) makes this robust to the smoothing
    # edge-artifact at the END of the trace: np.convolve(mode="same") zero-pads
    # the final samples and drags their velocity down, which previously could
    # land below the threshold, push `cut` to the full length, and collapse the
    # fitted acceleration to ~0 (the spurious 0.13 m/s^2 "weak" runs).
    if v_term > 1e-6:
        above = np.where(v >= 0.6 * v_term)[0]
        cut = int(above[0]) if len(above) else len(t)
    else:
        cut = len(t)
    cut = int(np.clip(cut, 5, len(t)))
    te, xe = t[:cut], x[:cut]

    # quadratic fit of POSITION -> a = 2 * c2 (no differentiation noise)
    c = np.polyfit(te, xe, 2)
    a_pos = 2.0 * c[0]
    a_vel = float(np.polyfit(te, v[:cut], 1)[0]) if cut >= 3 else float("nan")

    x_fit = np.polyval(c, te)
    return {"t": t, "x": x, "v": v, "te": te, "x_fit": x_fit, "cut": cut,
            "a_max": a_pos, "a_vel": a_vel, "v_term": v_term}


def feasibility(a_max):
    print("\n  ===== FEASIBILITY (a_max = %.3f m/s^2) =====" % a_max)
    for deg in (5, 8, 10):
        need = G * np.tan(np.radians(deg))
        verdict = "OK" if a_max > need else "TOO WEAK"
        print("    +/-%2d deg needs %.3f m/s^2  ->  %-8s (x%.2f)"
              % (deg, need, verdict, a_max / need))
    print("    want ~1.5-2x margin over the +/-5 deg need (0.86 m/s^2)\n")


def plot(res, title, png, show):
    try:
        import matplotlib
    except ModuleNotFoundError:
        print("  [plot skipped] matplotlib not installed "
              "(pip install matplotlib, or re-analyze the saved CSV offline)")
        return
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    a1.plot(res["t"], res["x"] * 1000, ".", ms=3, label="measured position")
    a1.plot(res["te"], res["x_fit"] * 1000, "r-", lw=2,
            label="fit 0.5*a*t^2 (a=%.2f m/s^2)" % res["a_max"])
    a1.axvline(res["t"][res["cut"] - 1], color="gray", ls="--", lw=1,
               label="end of fit window")
    a1.set_ylabel("cart position (mm)"); a1.set_title(title)
    a1.legend(loc="upper left"); a1.grid(True, alpha=0.3)
    a2.plot(res["t"], res["v"], ".", ms=3, label="velocity")
    a2.axhline(res["v_term"], color="g", ls="--", lw=1,
               label="terminal v = %.3f m/s" % res["v_term"])
    a2.set_xlabel("time (s)"); a2.set_ylabel("cart velocity (m/s)")
    a2.legend(loc="lower right"); a2.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(png, dpi=120)
    print("  [saved] %s" % png)
    if show:
        plt.show()


# ── Main ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default=DEFAULT_SERIAL_PORT)
    ap.add_argument("--pwm", type=int, nargs="+", default=[255])
    ap.add_argument("--dir", choices=["a", "b", "both"], default="both",
                    help="a=forward, b=reverse, both=measure each and take min")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--plot", action="store_true", help="show plot (needs display)")
    ap.add_argument("--csv", help="offline: analyze a saved t_us,pos CSV")
    args = ap.parse_args()

    if args.csv:
        arr = []
        for ln in Path(args.csv).read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln[0].isalpha():
                continue                         # skip blanks / "# t_us,pos" header
            parts = ln.split(",")
            if len(parts) >= 2:
                try:
                    arr.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue
        arr = np.array(arr)
        res = analyze(arr[:, 0] / 1e6, arr[:, 1] * M_PER_PULSE)
        print("  a_max=%.3f  v_term=%.3f" % (res["a_max"], res["v_term"]))
        plot(res, "Cart accel (%s)" % args.csv, "burst_result.png", args.plot)
        feasibility(res["a_max"])
        return

    dirs = ["a", "b"] if args.dir == "both" else [args.dir]
    sign = {"a": +1, "b": -1}

    with MotorComm(args.port, SERIAL_BAUD, verbose=True) as comm:
        input("  Press ENTER with cart near CENTER and free to move...")
        dir_results = {}
        last_res = None
        for pwm in args.pwm:
            for d in dirs:
                spwm = sign[d] * pwm
                accels = []
                print("\n  == PWM=%d  dir=%s ==" % (pwm, d))
                for r in range(args.runs):
                    return_to_origin(comm)
                    t, x = run_burst(comm, spwm)
                    if len(t) < 5:
                        print("    run %d: too few samples, skip" % (r + 1))
                        continue
                    res = analyze(t, x)
                    last_res = res
                    accels.append(res["a_max"])
                    # save raw for offline re-analysis
                    fn = "burst_%d_%s_%d.csv" % (pwm, d, r + 1)
                    np.savetxt(fn, np.column_stack([t * 1e6, x / M_PER_PULSE]),
                               fmt="%.1f", delimiter=",", header="t_us,pos_pulses")
                    print("    run %d: a_max=%.3f  a(vel)=%.3f  v_term=%.3f  -> %s"
                          % (r + 1, res["a_max"], res["a_vel"], res["v_term"], fn))
                if accels:
                    med = float(np.median(accels))
                    dir_results[(pwm, d)] = med
                    print("    median a_max = %.3f m/s^2" % med)

        print("\n  ============ SUMMARY ============")
        for (pwm, d), med in dir_results.items():
            print("    PWM=%d dir=%s : a_max=%.3f m/s^2" % (pwm, d, med))

        if dir_results:
            a_worst = min(dir_results.values())   # conservative: weakest direction
            print("\n  Conservative a_max (weakest direction) = %.3f m/s^2" % a_worst)
            feasibility(a_worst)
        if last_res is not None:
            plot(last_res, "Cart accel (last run)", "burst_result.png", args.plot)


if __name__ == "__main__":
    main()
