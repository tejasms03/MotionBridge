import math
import threading
import time
from collections import deque

try:
    import serial
    _SERIAL_OK = True
except ImportError:
    _SERIAL_OK = False

_N_FIELDS        = 13
_STEADINESS_WIN  = 20


class ArduinoReader:
    def __init__(self, cfg: dict):
        port           = str  (cfg.get("port",           "COM5"))
        baud           = int  (cfg.get("baud",           115200))
        self._thresh   = float(cfg.get("emg_threshold",  600))
        self._cooldown = float(cfg.get("burst_cooldown", 8.0))

        self._emg        = 0.0
        self._wrist      = [0.0] * 6
        self._elbow      = [0.0] * 6
        self._last_burst = 0.0
        self._lock       = threading.Lock()
        self._running    = False

        self._mag_buf: deque = deque(maxlen=_STEADINESS_WIN)

        if not _SERIAL_OK:
            print("[Arduino] pyserial not installed — EMG/IMU disabled.")
            return

        try:
            self._ser     = serial.Serial(port, baud, timeout=0.01)
            self._running = True
            threading.Thread(target=self._loop, daemon=True).start()
            print(f"[Arduino] Connected on {port} @ {baud} baud.")
        except Exception as e:
            print(f"[Arduino] Could not open {port}: {e} — EMG/IMU disabled.")

    def _loop(self):
        buf = ""
        while self._running:
            try:
                buf += self._ser.read(128).decode("ascii", errors="ignore")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    self._parse(line.strip())
            except Exception:
                pass

    def _parse(self, line: str):
        if not line or line.startswith("#"):
            return
        parts = line.split(",")
        if len(parts) != _N_FIELDS:
            return
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            return
        w_mag = math.sqrt(vals[1]**2 + vals[2]**2 + vals[3]**2)
        e_mag = math.sqrt(vals[7]**2 + vals[8]**2 + vals[9]**2)
        with self._lock:
            self._emg   = vals[0]
            self._wrist = vals[1:7]
            self._elbow = vals[7:13]
            self._mag_buf.append((w_mag + e_mag) / 2.0)

    def check_burst(self) -> bool:
        if not self._running:
            return False
        now = time.time()
        with self._lock:
            val  = self._emg
            last = self._last_burst
        if val >= self._thresh and (now - last) >= self._cooldown:
            with self._lock:
                self._last_burst = now
            return True
        return False

    def burst_ready(self) -> tuple[bool, float]:
        """Returns (is_ready, seconds_remaining_in_cooldown)."""
        if not self._running:
            return False, 0.0
        now = time.time()
        with self._lock:
            last = self._last_burst
        elapsed   = now - last
        remaining = max(0.0, self._cooldown - elapsed)
        return remaining == 0.0, remaining

    def emg_value(self) -> float:
        with self._lock:
            return self._emg

    def imu_wrist(self) -> dict:
        """Returns {'ax','ay','az','gx','gy','gz'} or {} if disconnected."""
        if not self._running:
            return {}
        with self._lock:
            w = self._wrist[:]
        return {"ax": w[0], "ay": w[1], "az": w[2],
                "gx": w[3], "gy": w[4], "gz": w[5]}

    def imu_elbow(self) -> dict:
        """Returns {'ax','ay','az','gx','gy','gz'} or {} if disconnected."""
        if not self._running:
            return {}
        with self._lock:
            e = self._elbow[:]
        return {"ax": e[0], "ay": e[1], "az": e[2],
                "gx": e[3], "gy": e[4], "gz": e[5]}

    def steadiness(self) -> float:
        if not self._running:
            return 0.0
        with self._lock:
            buf = list(self._mag_buf)
        if len(buf) < 2:
            return 0.0
        mean = sum(buf) / len(buf)
        var  = sum((x - mean) ** 2 for x in buf) / len(buf)
        return math.sqrt(var)

    def close(self):
        self._running = False
        try:
            self._ser.close()
        except Exception:
            pass
