import math
import random
import time

import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE

from hud_utils import draw_status_overlay

_SKIN      = (0.87, 0.71, 0.53)
_SKIN_DARK = (0.72, 0.58, 0.42)

_SEGMENTS = [
    ("right_shoulder", "right_elbow", 0.115, 0.085),
    ("right_elbow",    "right_wrist",  0.085, 0.058),
    ("right_wrist",    "right_index",  0.058, 0.038),
]
_JOINT_RADII = {
    "right_shoulder": 0.125,
    "right_elbow":    0.092,
    "right_wrist":    0.065,
}

_TARGETS = [
    {"pos": np.array([0.70,  1.10, 0.0], np.float32), "label": "UP",        "color": (0.2, 0.9, 0.3)},
    {"pos": np.array([1.40,  0.85, 0.0], np.float32), "label": "UP-RIGHT",  "color": (0.9, 0.8, 0.1)},
    {"pos": np.array([1.80,  0.25, 0.0], np.float32), "label": "FAR RIGHT", "color": (0.9, 0.4, 0.1)},
    {"pos": np.array([1.40, -0.25, 0.0], np.float32), "label": "LOW-RIGHT", "color": (0.8, 0.1, 0.5)},
    {"pos": np.array([0.70, -0.30, 0.0], np.float32), "label": "DOWN",      "color": (0.2, 0.5, 0.9)},
    {"pos": np.array([0.35,  0.60, 0.0], np.float32), "label": "CLOSE-UP",  "color": (0.7, 0.2, 0.9)},
]

_TARGET_R    = 0.20
_HIT_R       = 0.28
_HOLD_SECS   = 0.80


def _cyl(quad, p1, p2, r1, r2, slices=18):
    v = p2 - p1
    L = float(np.linalg.norm(v))
    if L < 1e-5:
        return
    vn  = v / L
    z   = np.array([0.0, 0.0, 1.0])
    cr  = np.cross(z, vn)
    crl = float(np.linalg.norm(cr))
    dot = float(np.dot(z, vn))
    glPushMatrix()
    glTranslatef(*p1)
    if crl > 1e-6:
        glRotatef(math.degrees(math.acos(np.clip(dot, -1.0, 1.0))), *cr)
    elif dot < 0:
        glRotatef(180.0, 1.0, 0.0, 0.0)
    gluCylinder(quad, r1, r2, L, slices, 1)
    glPopMatrix()


def _sph(quad, pos, r, slices=18):
    glPushMatrix()
    glTranslatef(*pos)
    gluSphere(quad, r, slices, slices)
    glPopMatrix()


def _arc(cx, cy, cz, radius, progress, segs=48):
    count = max(1, int(progress * segs))
    glBegin(GL_LINE_STRIP)
    for i in range(count + 1):
        a = -math.pi / 2 + (i / segs) * 2 * math.pi
        glVertex3f(cx + radius * math.cos(a),
                   cy + radius * math.sin(a), cz)
    glEnd()


class ReachTargets:
    def __init__(self, cfg: dict):
        w = cfg.get("width",  800)
        h = cfg.get("height", 600)
        self._w, self._h = w, h
        self._scale      = float(cfg.get("arm_scale", 3.2))

        import os
        os.environ["SDL_VIDEO_WINDOW_POS"] = "490,30"

        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Reach & Hold — Hit every target!")
        pygame.display.set_mode((w, h), DOUBLEBUF | OPENGL)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)

        glLightfv(GL_LIGHT0, GL_POSITION, [1.5, 3.5, 5.0, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.00, 0.95, 0.90, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.28, 0.25, 0.22, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.90, 0.90, 0.90, 1.0])

        glMatrixMode(GL_PROJECTION)
        gluPerspective(52.0, w / h, 0.05, 50.0)
        glMatrixMode(GL_MODELVIEW)

        self._quad      = gluNewQuadric()
        gluQuadricNormals(self._quad, GLU_SMOOTH)
        self._font      = pygame.font.SysFont("Arial", 28, bold=True)
        self._score_tex = glGenTextures(1)

        self.score      = 0
        self._sequence  = list(range(len(_TARGETS)))
        random.shuffle(self._sequence)
        self._seq_idx   = 0
        self._active    = self._sequence[0]
        self._hold_start: float | None = None
        self._progress  = 0.0
        self._flash_t: float | None = None
        self._burst_flash_t: float | None = None

    def mp_to_world(self, joints: dict) -> dict:
        if "right_shoulder" not in joints:
            return {}
        s = joints["right_shoulder"]
        out = {}
        for name, pos in joints.items():
            out[name] = np.array([
                 (pos[0] - s[0]) * self._scale,
                -(pos[1] - s[1]) * self._scale,
                 (pos[2] - s[2]) * self._scale * 0.35,
            ], np.float32)
        return out

    def update(self, world_joints: dict) -> int:
        hand = next(
            (world_joints[k] for k in ("right_index", "right_wrist")
             if k in world_joints), None)

        target_pos = _TARGETS[self._active]["pos"]
        scored = 0

        if hand is not None:
            dist = float(np.linalg.norm(hand[:2] - target_pos[:2]))
            in_zone = dist < _HIT_R + _TARGET_R

            if in_zone:
                if self._hold_start is None:
                    self._hold_start = time.time()
                elapsed = time.time() - self._hold_start
                self._progress = min(elapsed / _HOLD_SECS, 1.0)

                if elapsed >= _HOLD_SECS:
                    self.score += 1
                    scored += 1
                    self._flash_t = time.time()
                    self._hold_start = None
                    self._progress   = 0.0
                    self._advance_target()
            else:
                self._hold_start = None
                self._progress   = 0.0
        else:
            self._hold_start = None
            self._progress   = 0.0

        return scored

    def render(self, world_joints: dict, steadiness: float = 0.0,
               burst_ready: bool = False, burst_rem: float = 0.0):
        glClearColor(0.04, 0.06, 0.16, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        gluLookAt(
            0.8,  0.5,  5.2,
            1.1,  0.4,  0.0,
            0.0,  1.0,  0.0,
        )

        self._draw_targets()

        if world_joints:
            self._draw_arm(world_joints)

        self._draw_hud()
        draw_status_overlay(self._w, self._h, self._font,
                            steadiness, burst_ready, burst_rem, self._score_tex)
        self._draw_burst_flash()
        pygame.display.flip()

    def pump_events(self) -> tuple[bool, set]:
        keys_down: set = set()
        for e in pygame.event.get():
            if e.type == QUIT:
                return False, keys_down
            if e.type == KEYDOWN:
                if e.key == K_ESCAPE:
                    return False, keys_down
                keys_down.add(e.key)
        return True, keys_down

    def emg_burst(self):
        self.score += 1
        self._flash_t    = time.time()
        self._hold_start = None
        self._progress   = 0.0
        self._advance_target()
        self._burst_flash_t = time.time()

    def close(self):
        gluDeleteQuadric(self._quad)
        pygame.quit()

    def _advance_target(self):
        self._seq_idx += 1
        if self._seq_idx >= len(self._sequence):
            # reshuffle but avoid repeating the last target
            last = self._active
            random.shuffle(self._sequence)
            if self._sequence[0] == last and len(self._sequence) > 1:
                self._sequence[0], self._sequence[1] = (
                    self._sequence[1], self._sequence[0])
            self._seq_idx = 0
        self._active = self._sequence[self._seq_idx]

    def _draw_targets(self):
        now = time.time()

        for i, tgt in enumerate(_TARGETS):
            pos   = tgt["pos"]
            col   = tgt["color"]
            is_active = (i == self._active)

            if is_active:
                pulse = 0.72 + 0.28 * math.sin(now * 4.5)
                r_draw = _TARGET_R * (0.95 + 0.10 * pulse)
                glColor3f(col[0] * pulse, col[1] * pulse, col[2] * pulse)
                glMaterialfv(GL_FRONT, GL_SPECULAR, [0.9, 0.9, 0.9, 1.0])
                glMaterialf (GL_FRONT, GL_SHININESS,  80.0)
            else:
                glColor3f(col[0]*0.22, col[1]*0.22, col[2]*0.22)
                glMaterialfv(GL_FRONT, GL_SPECULAR, [0.1, 0.1, 0.1, 1.0])
                glMaterialf (GL_FRONT, GL_SHININESS,  10.0)
                r_draw = _TARGET_R * 0.60

            _sph(self._quad, pos, r_draw)

            if is_active and self._progress > 0:
                glDisable(GL_LIGHTING)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glLineWidth(4.0)
                glColor4f(0.2, 0.2, 0.2, 0.5)
                _arc(pos[0], pos[1], pos[2], _TARGET_R * 1.55, 1.0)
                glColor4f(*col, 0.95)
                _arc(pos[0], pos[1], pos[2], _TARGET_R * 1.55, self._progress)
                glLineWidth(1.0)
                glDisable(GL_BLEND)
                glEnable(GL_LIGHTING)

            if self._flash_t is not None and i == self._active:
                t = now - self._flash_t
                if t < 0.4:
                    glDisable(GL_LIGHTING)
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                    alpha = (1.0 - t / 0.4) * 0.9
                    glColor4f(1.0, 1.0, 0.8, alpha)
                    glLineWidth(3.0)
                    _arc(pos[0], pos[1], pos[2], _TARGET_R * 2.2, 1.0)
                    glLineWidth(1.0)
                    glDisable(GL_BLEND)
                    glEnable(GL_LIGHTING)

    def _draw_arm(self, wj: dict):
        glMaterialfv(GL_FRONT, GL_SPECULAR,  [0.25, 0.20, 0.15, 1.0])
        glMaterialf (GL_FRONT, GL_SHININESS,  20.0)
        glColor3f(*_SKIN)
        for a, b, r1, r2 in _SEGMENTS:
            if a in wj and b in wj:
                _cyl(self._quad, wj[a], wj[b], r1, r2)
        glColor3f(*_SKIN_DARK)
        for name, r in _JOINT_RADII.items():
            if name in wj:
                _sph(self._quad, wj[name], r)

    def _draw_hud(self):
        tgt  = _TARGETS[self._active]
        pct  = int(self._progress * 100)
        line = (f"Score: {self.score}   Target: {tgt['label']}  "
                f"({'HOLD! ' + str(pct) + '%' if pct > 0 else 'reach it'})   ESC = menu")
        surf = self._font.render(line, True, (255, 255, 255))
        tw, th = surf.get_size()
        raw    = pygame.image.tostring(surf, "RGBA", True)

        glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D); glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindTexture(GL_TEXTURE_2D, self._score_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, th, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, raw)

        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        gluOrtho2D(0, self._w, 0, self._h)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()

        x, y = self._w - tw - 16, self._h - th - 16
        glBegin(GL_QUADS)
        glTexCoord2f(0,0); glVertex2f(x,      y)
        glTexCoord2f(1,0); glVertex2f(x + tw, y)
        glTexCoord2f(1,1); glVertex2f(x + tw, y + th)
        glTexCoord2f(0,1); glVertex2f(x,      y + th)
        glEnd()

        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glDisable(GL_TEXTURE_2D); glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)

    def _draw_burst_flash(self):
        if self._burst_flash_t is None:
            return
        t = time.time() - self._burst_flash_t
        if t > 0.45:
            self._burst_flash_t = None
            return
        alpha = (1.0 - t / 0.45) * 0.65
        glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        gluOrtho2D(0, self._w, 0, self._h)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glColor4f(1.0, 0.88, 0.20, alpha)
        glBegin(GL_QUADS)
        glVertex2f(0, 0); glVertex2f(self._w, 0)
        glVertex2f(self._w, self._h); glVertex2f(0, self._h)
        glEnd()
        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glDisable(GL_BLEND); glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
