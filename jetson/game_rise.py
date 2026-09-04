import math
import random
import time

import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE

from hud_utils import draw_status_overlay

_COLORS = [
    (0.90, 0.18, 0.18),
    (0.95, 0.80, 0.08),
    (0.10, 0.80, 0.22),
    (0.15, 0.45, 0.95),
    (0.92, 0.48, 0.08),
    (0.70, 0.12, 0.92),
]
_SKIN      = (0.87, 0.71, 0.53)
_SKIN_DARK = (0.72, 0.58, 0.42)

_SPAWN_Y = -0.90
_EXIT_Y  =  2.20
_X_LO, _X_HI = 0.30, 1.90

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


class _Balloon:
    def __init__(self, speed_mult=1.0):
        x = random.uniform(_X_LO, _X_HI)
        self.pos   = np.array([x, _SPAWN_Y, 0.0], np.float32)
        self.r     = random.uniform(0.10, 0.17)
        self.color = random.choice(_COLORS)
        self.alive = True
        self._pop_t: float | None = None

        self.vel = np.array([
            random.uniform(-0.004, 0.004),
            random.uniform(0.008, 0.016) * speed_mult,
            0.0,
        ], np.float32)

    def move(self):
        if not self.alive:
            return
        self.pos += self.vel
        if self.pos[0] - self.r < _X_LO:
            self.pos[0] = _X_LO + self.r
            self.vel[0] = abs(self.vel[0])
        elif self.pos[0] + self.r > _X_HI:
            self.pos[0] = _X_HI - self.r
            self.vel[0] = -abs(self.vel[0])

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
            return self.pos[1] - self.r > _EXIT_Y
        return self._pop_t is None or time.time() - self._pop_t > 0.55

    def draw(self, quad):
        if self.alive:
            glColor3f(*self.color)
            glMaterialfv(GL_FRONT, GL_SPECULAR, [0.8, 0.8, 0.8, 1.0])
            glMaterialf (GL_FRONT, GL_SHININESS,  60.0)
            _sph(quad, self.pos, self.r)

            glDisable(GL_LIGHTING)
            glColor3f(0.6, 0.6, 0.6)
            glBegin(GL_LINES)
            glVertex3f(self.pos[0], self.pos[1] - self.r,        0.0)
            glVertex3f(self.pos[0], self.pos[1] - self.r - 0.20, 0.0)
            glEnd()
            glEnable(GL_LIGHTING)

        elif self._pop_t is not None:
            t = time.time() - self._pop_t
            if t < 0.55:
                glDisable(GL_LIGHTING)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                for sc in (1.0, 1.8, 2.6):
                    alpha = max(0.0, (1.0 - t / 0.55) / sc)
                    glColor4f(*self.color, alpha)
                    _sph(quad, self.pos, (self.r + t * 0.9) * sc, slices=10)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                glDisable(GL_BLEND)
                glEnable(GL_LIGHTING)


class BalloonRise:
    def __init__(self, cfg: dict):
        w = cfg.get("width",  800)
        h = cfg.get("height", 600)
        self._w, self._h   = w, h
        self._scale        = float(cfg.get("arm_scale",      3.2))
        self._max_bal      = int  (cfg.get("max_balloons",   3))
        self._spawn_ivl    = float(cfg.get("spawn_interval", 1.6))

        import os
        os.environ["SDL_VIDEO_WINDOW_POS"] = "490,30"

        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Balloon Rise — Raise your arm!")
        pygame.display.set_mode((w, h), DOUBLEBUF | OPENGL)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)

        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 4.0, 5.0, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.00, 0.95, 0.90, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.28, 0.25, 0.22, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.90, 0.90, 0.90, 1.0])

        glMatrixMode(GL_PROJECTION)
        gluPerspective(50.0, w / h, 0.05, 50.0)
        glMatrixMode(GL_MODELVIEW)

        self._quad      = gluNewQuadric()
        gluQuadricNormals(self._quad, GLU_SMOOTH)
        self._font      = pygame.font.SysFont("Arial", 28, bold=True)
        self._score_tex = glGenTextures(1)

        self.score  = 0
        self.missed = 0
        self._wave  = 1
        self._kills = 0
        self._balloons: list[_Balloon] = []
        self._last_spawn = time.time()
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
        for b in self._balloons:
            b.move()

        for b in self._balloons:
            if b.alive and b.pos[1] - b.r > _EXIT_Y:
                self.missed += 1

        hand = next(
            (world_joints[k] for k in ("right_index", "right_wrist")
             if k in world_joints), None)
        pops = 0
        if hand is not None:
            for b in self._balloons:
                if b.try_pop(hand):
                    self.score += 1
                    self._kills += 1
                    pops += 1

        if self._kills >= 10:
            self._wave += 1
            self._kills = 0

        self._balloons = [b for b in self._balloons if not b.is_done()]
        self._maybe_spawn()
        return pops

    def render(self, world_joints: dict, steadiness: float = 0.0,
               burst_ready: bool = False, burst_rem: float = 0.0):
        glClearColor(0.05, 0.08, 0.18, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        gluLookAt(
            0.8,  0.6,  5.0,
            1.1,  0.7,  0.0,
            0.0,  1.0,  0.0,
        )

        for b in self._balloons:
            b.draw(self._quad)

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
        for b in self._balloons:
            if b.alive:
                b.alive  = False
                b._pop_t = time.time()
                self.score += 1
                self._kills += 1
        if self._kills >= 10:
            self._wave  += 1
            self._kills  = 0
        self._burst_flash_t = time.time()

    def close(self):
        gluDeleteQuadric(self._quad)
        pygame.quit()

    def _maybe_spawn(self):
        now  = time.time()
        live = sum(1 for b in self._balloons if b.alive)
        if live < self._max_bal and now - self._last_spawn >= self._spawn_ivl:
            # stagger x so balloons don't stack on top of each other
            used_x = [b.pos[0] for b in self._balloons if b.alive]
            for _ in range(20):
                cand = random.uniform(_X_LO, _X_HI)
                if all(abs(cand - x) > 0.40 for x in used_x):
                    break
            spd = 1.0 + (self._wave - 1) * 0.15
            b = _Balloon(spd)
            b.pos[0] = cand
            self._balloons.append(b)
            self._last_spawn = now

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
        text = f"Wave {self._wave}   Score: {self.score}   Missed: {self.missed}   ESC = menu"
        surf = self._font.render(text, True, (255, 255, 255))
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
