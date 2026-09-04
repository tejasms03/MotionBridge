import json
import time
from enum import Enum, auto
from pathlib import Path

import numpy as np
import cv2


class Phase(Enum):
    NEUTRAL   = auto()
    ARM_UP    = auto()
    ARM_DOWN  = auto()
    ARM_LEFT  = auto()
    ARM_RIGHT = auto()
    DONE      = auto()


_INSTRUCTIONS = {
    Phase.NEUTRAL:   "Rest your arm on the desk in a relaxed position. Hold.",
    Phase.ARM_UP:    "Lift your arm OFF the desk as HIGH as you can. Hold.",
    Phase.ARM_DOWN:  "Return arm to desk — press it down flat. Hold.",
    Phase.ARM_LEFT:  "Slide / reach your arm as FAR LEFT along the desk as you can. Hold.",
    Phase.ARM_RIGHT: "Reach your arm as FAR RIGHT / FORWARD on the desk as you can. Hold.",
}

_PHASE_ORDER = [
    Phase.NEUTRAL,
    Phase.ARM_UP,
    Phase.ARM_DOWN,
    Phase.ARM_LEFT,
    Phase.ARM_RIGHT,
]


class CalibrationManager:
    def __init__(self, hold_duration: float, save_path: str):
        self.hold_duration = hold_duration
        self.save_path = Path(save_path)
        self.data: dict = {}
        self.active = False
        self._phase_idx = 0
        self._phase_start = 0.0
        self._buffer: list = []

    def start(self):
        self.active = True
        self._phase_idx = 0
        self._phase_start = time.time()
        self._buffer = []
        self.data = {}

    def update(self, joints) -> bool:
        """Returns True when all phases complete. Skips frames where joints is None."""
        if not self.active or self._current_phase() == Phase.DONE:
            return False
        if joints is None:
            return False

        self._buffer.append({k: v.copy() for k, v in joints.items()})

        if time.time() - self._phase_start >= self.hold_duration:
            self._commit()
            self._phase_idx += 1
            self._phase_start = time.time()
            self._buffer = []

            if self._current_phase() == Phase.DONE:
                self._save()
                self.active = False
                return True

        return False

    def load(self) -> bool:
        if not self.save_path.exists():
            return False
        with open(self.save_path) as f:
            raw = json.load(f)
        self.data = {
            phase: {joint: np.array(coords, dtype=np.float32)
                    for joint, coords in joints.items()}
            for phase, joints in raw.items()
        }
        return True

    def is_ready(self) -> bool:
        return bool(self.data) and "neutral" in self.data

    def draw_overlay(self, frame):
        if not self.active:
            return
        phase = self._current_phase()
        if phase == Phase.DONE:
            return

        h, w = frame.shape[:2]
        elapsed = time.time() - self._phase_start
        progress = min(elapsed / self.hold_duration, 1.0)
        remaining = max(0.0, self.hold_duration - elapsed)
        step = self._phase_idx + 1
        total = len(_PHASE_ORDER)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 115), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"CALIBRATION  ({step}/{total})",
                    (20, h - 88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, _INSTRUCTIONS[phase],
                    (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 180), 1, cv2.LINE_AA)

        bx, by, bw, bh = 20, h - 35, w - 40, 16
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
        fill_color = (0, 220, 100) if progress < 0.99 else (0, 255, 0)
        cv2.rectangle(frame, (bx, by), (bx + int(bw * progress), by + bh), fill_color, -1)
        cv2.putText(frame, f"{remaining:.1f}s",
                    (bx + bw + 8, by + bh - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    def _current_phase(self) -> Phase:
        if self._phase_idx >= len(_PHASE_ORDER):
            return Phase.DONE
        return _PHASE_ORDER[self._phase_idx]

    def _commit(self):
        if not self._buffer:
            return
        key = self._current_phase().name.lower()
        joint_names = list(self._buffer[0].keys())
        self.data[key] = {
            jn: np.mean([f[jn] for f in self._buffer], axis=0).astype(np.float32)
            for jn in joint_names
        }

    def _save(self):
        serialisable = {
            phase: {joint: coords.tolist() for joint, coords in joints.items()}
            for phase, joints in self.data.items()
        }
        with open(self.save_path, "w") as f:
            json.dump(serialisable, f, indent=2)
