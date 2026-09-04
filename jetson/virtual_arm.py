import math
import random
import time

import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE

from hud_utils import draw_status_overlay

_MONSTER_COLORS = [
    (0.18, 0.55, 0.12),
    (0.62, 0.08, 0.62),
    (0.80, 0.10, 0.10),
    (0.82, 0.42, 0.04),
    (0.20, 0.30, 0.62),
    (0.55, 0.50, 0.06),
]

_SKIN      = (0.87, 0.71, 0.53)
_SKIN_DARK = (0.72, 0.58, 0.42)

_DEFEND_PT   = np.array([0.0, 0.0, 0.0], np.float32)
_KILLS_PER_WAVE = 8

_SPAWN_RADIUS_MIN = 1.1
_SPAWN_RADIUS_MAX = 1.6

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


def _ring(cx, cy, cz, radius, segs=40):
    glBegin(GL_LINE_LOOP)
    for i in range(segs):
        a = 2 * math.pi * i / segs
        glVertex3f(cx + radius * math.cos(a), cy + radius * math.sin(a), cz)
    glEnd()


class _Monster:
    def __init__(self, speed_mult: float = 1.0):
        # Spawn from the arm side 75% of the time to encourage varied movements
        if random.random() < 0.75:
            angle = random.uniform(-1.1, 1.1)
        else:
            angle = random.uniform(-math.pi, math.pi)

        r  = random.uniform(_SPAWN_RADIUS_MIN, _SPAWN_RADIUS_MAX)
        sx = float(_DEFEND_PT[0] + r * math.cos(angle))
        sy = float(_DEFEND_PT[1] + r * math.sin(angle))
        sx = max(-0.6, min(2.4, sx))
        sy = max(-0.8, min(1.8, sy))

        self.pos   = np.array([sx, sy, 0.0], np.float32)
        self.r     = random.uniform(0.10, 0.16)
        self.color = random.choice(_MONSTER_COLORS)
        self.alive = True
        self._pop_t: float | None = None

        d = _DEFEND_PT[:2] - self.pos[:2]
        dn = float(np.linalg.norm(d))
        d  = (d / dn) if dn > 1e-5 else np.array([1.0, 0.0])
        spd = random.uniform(0.007, 0.013) * speed_mult
        self.vel = np.array([d[0] * spd, d[1] * spd, 0.0], np.float32)

    def move(self):
        if self.alive:
            self.pos += self.vel

    def reached_base(self) -> bool:
        if not self.alive:
            return False
        return float(np.linalg.norm(self.pos[:2] - _DEFEND_PT[:2])) < self.r + 0.18

    def try_pop(self, hand: np.ndarray) -> bool:
        if not self.alive:
            return False
        if float(np.linalg.norm(self.pos[:2] - hand[:2])) < self.r + 0.20:
            self.alive  = False
            self._pop_t = time.time()
            return True
        return False

    def is_done(self) -> bool:
        if self.alive:
            return False
        return self._pop_t is None or time.time() - self._pop_t > 0.55

    def draw(self, quad):
        if self.alive:
            self._draw_body(quad)
        elif self._pop_t is not None:
            t = time.time() - self._pop_t
            if t < 0.55:
                glDisable(GL_LIGHTING)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                for sc in (1.0, 1.7, 2.6):
                    alpha = max(0.0, (1.0 - t / 0.55) / sc)
                    glColor4f(*self.color, alpha)
                    _sph(quad, self.pos, (self.r + t * 0.9) * sc, slices=10)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                glDisable(GL_BLEND)
                glEnable(GL_LIGHTING)

    def _draw_body(self, quad):
        r = self.r
        p = self.pos

        glColor3f(*self.color)
        glMaterialfv(GL_FRONT, GL_SPECULAR,  [0.45, 0.45, 0.45, 1.0])
        glMaterialf (GL_FRONT, GL_SHININESS,  38.0)
        _sph(quad, p, r)

        glColor3f(0.96, 0.96, 0.96)
        glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glMaterialf (GL_FRONT, GL_SHININESS, 90.0)
        eye_r   = r * 0.24
        eye_fwd = r * 0.82
        eye_up  = r * 0.22
        eye_sep = r * 0.36
        lp = p + np.array([-eye_sep,  eye_up, eye_fwd], np.float32)
        rp = p + np.array([ eye_sep,  eye_up, eye_fwd], np.float32)
        _sph(quad, lp, eye_r, slices=10)
        _sph(quad, rp, eye_r, slices=10)

        glColor3f(0.92, 0.04, 0.04)
        pupil_r   = eye_r * 0.52
        pupil_fwd = eye_r * 0.55
        _sph(quad, lp + np.array([0, 0, pupil_fwd], np.float32), pupil_r, slices=8)
        _sph(quad, rp + np.array([0, 0, pupil_fwd], np.float32), pupil_r, slices=8)

        glColor3f(0.14, 0.10, 0.08)
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.05, 0.05, 0.05, 1.0])
        glMaterialf (GL_FRONT, GL_SHININESS,  8.0)
        for side in (-1, 1):
            base = p + np.array([side * r * 0.32,  r * 0.76, 0.0], np.float32)
            tip  = p + np.array([side * r * 0.54,  r * 1.48, 0.0], np.float32)
            _cyl(quad, base, tip, r * 0.09, 0.0, slices=7)

        glColor3f(0.94, 0.94, 0.90)
        fang_y = float(p[1]) - r * 0.52
        fang_z = float(p[2]) + r * 0.80
        for side in (-1, 1):
            fb = np.array([float(p[0]) + side * r * 0.21, fang_y,          fang_z], np.float32)
            ft = np.array([float(p[0]) + side * r * 0.21, fang_y - r*0.38, fang_z], np.float32)
            _cyl(quad, fb, ft, r * 0.07, 0.0, slices=6)


class VirtualArm:
    def __init__(self, cfg: dict):
        w = cfg.get("width",  800)
        h = cfg.get("height", 600)
        self._w, self._h = w, h
        self._scale          = float(cfg.get("arm_scale",      3.2))
        self._max_monsters   = int  (cfg.get("max_balloons",   3))
        self._spawn_interval = float(cfg.get("spawn_interval", 1.8))

        import os
        os.environ["SDL_VIDEO_WINDOW_POS"] = "490,30"

        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Arm Defense — Stop the Monsters!")
        self._screen = pygame.display.set_mode((w, h), DOUBLEBUF | OPENGL)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)

        glLightfv(GL_LIGHT0, GL_POSITION, [2.0, 3.5, 5.0, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.00, 0.95, 0.88, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.28, 0.25, 0.22, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.90, 0.90, 0.90, 1.0])

        glMatrixMode(GL_PROJECTION)
        gluPerspective(48.0, w / h, 0.05, 50.0)
        glMatrixMode(GL_MODELVIEW)

        self._quad = gluNewQuadric()
        gluQuadricNormals(self._quad, GLU_SMOOTH)

        self._font      = pygame.font.SysFont("Arial", 28, bold=True)
        self._score_tex = glGenTextures(1)

        self.score             = 0
        self._wave             = 1
        self._kills_this_wave  = 0
        self._monsters: list[_Monster] = []
        self._last_spawn       = time.time()
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
        for m in self._monsters:
            m.move()

        # monsters that reach the base just vanish — no lives system
        for m in self._monsters:
            if m.reached_base():
                m.alive  = False
                m._pop_t = None

        pops = 0
        hand = next(
            (world_joints[k] for k in ("right_index", "right_wrist") if k in world_joints),
            None,
        )
        if hand is not None:
            for m in self._monsters:
                if m.try_pop(hand):
                    self.score += 1
                    self._kills_this_wave += 1
                    pops += 1

        if self._kills_this_wave >= _KILLS_PER_WAVE:
            self._wave += 1
            self._kills_this_wave = 0

        self._monsters = [m for m in self._monsters if not m.is_done()]
        self._maybe_spawn()
        return pops

    def render(self, world_joints: dict, steadiness: float = 0.0,
               burst_ready: bool = False, burst_rem: float = 0.0):
        glClearColor(0.06, 0.04, 0.12, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        gluLookAt(
            0.8,  1.2,  5.2,
            0.7,  0.2,  0.0,
            0.0,  1.0,  0.0,
        )

        self._draw_base()

        for m in self._monsters:
            m.draw(self._quad)

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
        for m in self._monsters:
            if m.alive:
                m.alive  = False
                m._pop_t = time.time()
                self.score += 1
                self._kills_this_wave += 1
        if self._kills_this_wave >= _KILLS_PER_WAVE:
            self._wave += 1
            self._kills_this_wave = 0
        self._burst_flash_t = time.time()

    def close(self):
        gluDeleteQuadric(self._quad)
        pygame.quit()

    def _speed_mult(self) -> float:
        return 1.0 + (self._wave - 1) * 0.18

    def _maybe_spawn(self):
        now  = time.time()
        live = sum(1 for m in self._monsters if m.alive)
        if live < self._max_monsters and now - self._last_spawn >= self._spawn_interval:
            self._monsters.append(_Monster(self._speed_mult()))
            self._last_spawn = now

    def _draw_base(self):
        t     = time.time()
        pulse = 0.88 + 0.12 * math.sin(t * 3.5)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(2.0)
        dp = _DEFEND_PT
        for rad in (0.18, 0.30, 0.44):
            alpha = pulse * (0.85 - rad * 0.7)
            glColor4f(0.2, 0.9, 0.9, alpha)
            _ring(dp[0], dp[1], dp[2], rad * pulse)
        glDisable(GL_BLEND)
        glLineWidth(1.0)
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
        text = f"Wave {self._wave}   Score: {self.score}"
        surf = self._font.render(text, True, (255, 255, 255))
        tw, th = surf.get_size()
        raw    = pygame.image.tostring(surf, "RGBA", True)

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindTexture(GL_TEXTURE_2D, self._score_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, th, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, raw)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self._w, 0, self._h)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        x = self._w - tw - 16
        y = self._h - th - 16
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x,      y)
        glTexCoord2f(1, 0); glVertex2f(x + tw, y)
        glTexCoord2f(1, 1); glVertex2f(x + tw, y + th)
        glTexCoord2f(0, 1); glVertex2f(x,      y + th)
        glEnd()

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

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
