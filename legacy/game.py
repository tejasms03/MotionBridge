import math
import random
import time

import cv2
import numpy as np

from tracker import _capsule

_BALLOON_COLORS = [
    ( 50,  50, 220),  # red
    ( 30, 210, 230),  # yellow
    ( 50, 190,  50),  # green
    (220,  90,  50),  # blue
    ( 40, 140, 220),  # orange
    (190,  50, 200),  # magenta
]

_ARM_CONNECTIONS = [
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("right_wrist",    "right_index"),
]
_RADII_FRAC = {
    "right_shoulder": 0.026,
    "right_elbow":    0.019,
    "right_wrist":    0.013,
    "right_index":    0.008,
}


def _draw_arm_canvas(canvas, px_joints, color=(210, 90, 20)):
    h = canvas.shape[0]
    radii = {k: int(v * h) for k, v in _RADII_FRAC.items()}
    outline = (20, 20, 20)
    for a, b in _ARM_CONNECTIONS:
        if a not in px_joints or b not in px_joints:
            continue
        _capsule(canvas, px_joints[a], px_joints[b],
                 radii.get(a, 8) + 3, radii.get(b, 6) + 3, outline)
    for a, b in _ARM_CONNECTIONS:
        if a not in px_joints or b not in px_joints:
            continue
        _capsule(canvas, px_joints[a], px_joints[b],
                 radii.get(a, 8), radii.get(b, 6), color)


class _Balloon:
    def __init__(self, w, h, min_x):
        self.r = random.randint(28, 50)
        self.x = float(random.randint(min_x + self.r + 10, w - self.r - 10))
        self.y = float(random.randint(self.r + 20, h - self.r - 55))
        self.color = random.choice(_BALLOON_COLORS)
        self.alive = True
        self._pop_t = None

        speed = random.uniform(1.2, 2.8)
        angle = random.uniform(0, 2 * math.pi)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

    def move(self, w, h):
        if not self.alive:
            return
        self.x += self.vx
        self.y += self.vy
        # Bounce off walls
        if self.x - self.r < 0:
            self.x = self.r
            self.vx = abs(self.vx)
        elif self.x + self.r > w:
            self.x = w - self.r
            self.vx = -abs(self.vx)
        if self.y - self.r < 0:
            self.y = self.r
            self.vy = abs(self.vy)
        elif self.y + self.r > h - 50:
            self.y = h - 50 - self.r
            self.vy = -abs(self.vy)

    def try_pop(self, hx, hy, reach=22) -> bool:
        if not self.alive:
            return False
        if math.hypot(self.x - hx, self.y - hy) < self.r + reach:
            self.alive = False
            self._pop_t = time.time()
            return True
        return False

    def is_done(self) -> bool:
        return not self.alive and (
            self._pop_t is None or time.time() - self._pop_t > 0.45
        )

    def draw(self, img):
        cx, cy = int(self.x), int(self.y)
        if self.alive:
            cv2.circle(img, (cx + 4, cy + 5), self.r, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), self.r, self.color, -1, cv2.LINE_AA)
            cv2.circle(img,
                       (cx - self.r // 4, cy - self.r // 4),
                       self.r // 4,
                       tuple(min(255, c + 90) for c in self.color),
                       -1, cv2.LINE_AA)
            # String trails behind movement direction
            tail_x = cx - int(self.vx * 8)
            tail_y = cy - int(self.vy * 8)
            cv2.line(img, (cx, cy + self.r), (tail_x, tail_y + self.r + 30),
                     (160, 160, 160), 2, cv2.LINE_AA)
            cv2.circle(img, (cx, cy + self.r), 3, self.color, -1, cv2.LINE_AA)

        elif self._pop_t:
            t = time.time() - self._pop_t
            if t < 0.45:
                alpha = 1.0 - t / 0.45
                for scale, thick in ((1.0, 3), (0.55, 2)):
                    pr = max(1, int((self.r + t * 200) * scale))
                    c = tuple(int(v * alpha) for v in self.color)
                    cv2.circle(img, (cx, cy), pr, c, thick, cv2.LINE_AA)
                for angle in range(0, 360, 45):
                    rad = math.radians(angle)
                    length = int((self.r + t * 130) * alpha)
                    ex = cx + int(math.cos(rad) * length)
                    ey = cy + int(math.sin(rad) * length)
                    c = tuple(int(v * alpha * 0.8) for v in self.color)
                    cv2.line(img, (cx, cy), (ex, ey), c, 2, cv2.LINE_AA)


class BalloonGame:
    def __init__(self, cfg):
        self.w = cfg.get("width",  960)
        self.h = cfg.get("height", 720)
        self._max      = cfg.get("max_balloons",   6)
        self._interval = cfg.get("spawn_interval", 2.0)

        # Arm rests on desk → shoulder is left anchor, arm extends rightward.
        # Anchor at far left, scale large enough so wrist reaches balloon zone.
        self.anchor = (int(self.w * 0.03), int(self.h * 0.55))
        self._scale = self.h * 1.4

        # Balloons spawn to the right where the hand can reach
        self._min_x = int(self.w * 0.30)

        self.score = 0
        self._balloons: list[_Balloon] = []
        self._last_spawn = time.time()

        for _ in range(self._max):
            self._balloons.append(_Balloon(self.w, self.h, self._min_x))

    # ------------------------------------------------------------------

    def joints_to_canvas(self, joints: dict) -> dict:
        """
        Map completed normalised joints → canvas pixel coords.
        Arm rests on desk in front of camera: shoulder is left anchor,
        arm extends rightward (positive x-delta → rightward on canvas).
        Lifting off desk (negative y-delta) → upward on canvas.
        """
        if "right_shoulder" not in joints:
            return {}
        s = joints["right_shoulder"]
        ax, ay = self.anchor
        out = {}
        for name, pos in joints.items():
            dx = pos[0] - s[0]
            dy = pos[1] - s[1]
            out[name] = (
                int(ax + dx * self._scale),
                int(ay + dy * self._scale),
            )
        return out

    def update(self, canvas_joints: dict) -> int:
        """Move balloons, check collisions. Returns pop count this frame."""
        for b in self._balloons:
            b.move(self.w, self.h)

        pops = 0
        for b in self._balloons:
            for name in ("right_wrist", "right_index"):
                if name in canvas_joints and b.try_pop(*canvas_joints[name]):
                    self.score += 1
                    pops += 1
                    break

        self._balloons = [b for b in self._balloons if not b.is_done()]
        self._maybe_spawn()
        return pops

    def render(self, canvas_joints: dict) -> np.ndarray:
        canvas = np.zeros((self.h, self.w, 3), np.uint8)

        for row in range(self.h):
            t = row / self.h
            canvas[row] = (
                int(28 + t * 22),
                int(10 + t * 12),
                int(52 + t * 30),
            )

        for b in self._balloons:
            b.draw(canvas)

        if canvas_joints:
            _draw_arm_canvas(canvas, canvas_joints)

        cv2.putText(canvas, f"Score: {self.score}",
                    (self.w - 175, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

        return canvas

    # ------------------------------------------------------------------

    def _maybe_spawn(self):
        now = time.time()
        live = sum(1 for b in self._balloons if b.alive)
        if live < self._max and now - self._last_spawn >= self._interval:
            self._balloons.append(_Balloon(self.w, self.h, self._min_x))
            self._last_spawn = now
