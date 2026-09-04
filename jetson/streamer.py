import json
import socket
import time


class UDPStreamer:
    def __init__(self, cfg):
        self._addr = (cfg["host"], cfg["port"])
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, joints: dict):
        payload = {
            "shoulder":  self._vec3(joints.get("right_shoulder")),
            "elbow":     self._vec3(joints.get("right_elbow")),
            "wrist":     self._vec3(joints.get("right_wrist")),
            "timestamp": time.time(),
        }
        try:
            self._sock.sendto(json.dumps(payload).encode(), self._addr)
        except OSError:
            pass

    def close(self):
        self._sock.close()

    @staticmethod
    def _vec3(v) -> dict:
        if v is None:
            return {"x": 0.0, "y": 0.0, "z": 0.0}
        arr = v.tolist() if hasattr(v, "tolist") else list(v)
        return {"x": float(arr[0]), "y": float(arr[1]), "z": float(arr[2])}
