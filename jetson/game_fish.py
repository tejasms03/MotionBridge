import math
import random
import time

import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, KEYDOWN, K_ESCAPE

from hud_utils import draw_status_overlay

_Y_TOP   =  1.70
_Y_BOT   = -0.70
_Y_MID   = (_Y_TOP + _Y_BOT) / 2

_BUBBLE_SPAWN_X =  3.2
_BUBBLE_EXIT_X  = -3.2
_BUBBLE_SCROLL  =  0.013

_MAX_VY  = 0.022
_DAMPING = 0.80


def _sph(quad, pos, r, slices=16):
    glPushMatrix()
    glTranslatef(float(pos[0]), float(pos[1]), float(pos[2]))
    gluSphere(quad, r, slices, slices // 2)
    glPopMatrix()


def _cyl(quad, p1, p2, r1, r2, slices=12):
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


def _draw_fish(quad, pos, angle_rad, t):
    wag = math.sin(t * 7.0) * 0.18

    glPushMatrix()
    glTranslatef(float(pos[0]), float(pos[1]), float(pos[2]))
    glRotatef(math.degrees(angle_rad), 0, 0, 1)

    glColor3f(0.95, 0.52, 0.08)
    glMaterialfv(GL_FRONT, GL_SPECULAR,  [0.7, 0.55, 0.2, 1.0])
    glMaterialf (GL_FRONT, GL_SHININESS,  55.0)
    glPushMatrix()
    glScalef(2.0, 1.0, 0.85)
    gluSphere(quad, 0.14, 18, 12)
    glPopMatrix()

    glColor3f(0.98, 0.92, 0.70)
    glPushMatrix()
    glScalef(1.4, 0.55, 0.90)
    gluSphere(quad, 0.13, 14, 10)
    glPopMatrix()

    glColor3f(0.15, 0.10, 0.05)
    glPushMatrix()
    glScalef(0.28, 1.20, 0.96)
    gluSphere(quad, 0.135, 10, 10)
    glPopMatrix()

    glDisable(GL_LIGHTING)

    glColor3f(0.95, 0.75, 0.10)
    tx = -0.30
    glBegin(GL_TRIANGLES)
    glVertex3f(tx,        wag * 0.5,  0.0)
    glVertex3f(tx - 0.20, 0.22 + wag, 0.0)
    glVertex3f(tx - 0.10, wag * 0.5,  0.0)
    glVertex3f(tx,        wag * 0.5,  0.0)
    glVertex3f(tx - 0.20, -0.22 + wag, 0.0)
    glVertex3f(tx - 0.10, wag * 0.5,  0.0)
    glEnd()

    glColor3f(0.90, 0.65, 0.08)
    glBegin(GL_TRIANGLES)
    glVertex3f( 0.06,  0.14, 0.0)
    glVertex3f( 0.14,  0.26, 0.0)
    glVertex3f( 0.24,  0.14, 0.0)
    glEnd()

    glBegin(GL_TRIANGLES)
    glVertex3f( 0.06, -0.05,  0.10)
    glVertex3f( 0.10, -0.20,  0.04)
    glVertex3f(-0.06, -0.05,  0.04)
    glEnd()

    glEnable(GL_LIGHTING)

    glColor3f(1.0, 1.0, 1.0)
    glMaterialfv(GL_FRONT, GL_SPECULAR,  [1.0, 1.0, 1.0, 1.0])
    glMaterialf (GL_FRONT, GL_SHININESS,  90.0)
    _sph(quad, np.array([0.21, 0.05, 0.11]), 0.042)

    glColor3f(0.05, 0.04, 0.04)
    _sph(quad, np.array([0.23, 0.05, 0.13]), 0.024)

    glPopMatrix()


class _Bubble:
    def __init__(self, speed_mult=1.0):
        self.pos   = np.array([
            _BUBBLE_SPAWN_X + random.uniform(0, 2.0),
            random.uniform(_Y_BOT + 0.25, _Y_TOP - 0.25),
            0.0,
        ], np.float32)
        self.r          = random.uniform(0.07, 0.14)
        self._speed     = _BUBBLE_SCROLL * speed_mult
        self.collected  = False
        self._collect_t: float | None = None

    def move(self):
        if not self.collected:
            self.pos[0] -= self._speed

    def try_collect(self, fish_pos: np.ndarray) -> bool:
        if self.collected:
            return False
        if float(np.linalg.norm(self.pos[:2] - fish_pos[:2])) < self.r + 0.20:
            self.collected  = True
            self._collect_t = time.time()
            return True
        return False

    def is_done(self) -> bool:
        if not self.collected:
            return self.pos[0] < _BUBBLE_EXIT_X
        return self._collect_t is not None and time.time() - self._collect_t > 0.45

    def draw(self, quad):
        if not self.collected:
            glow = 0.70 + 0.30 * math.sin(time.time() * 3.5 + self.pos[1] * 4)
            glColor3f(0.55 * glow, 0.82 * glow, glow)
            glMaterialfv(GL_FRONT, GL_SPECULAR,  [1.0, 1.0, 1.0, 1.0])
            glMaterialf (GL_FRONT, GL_SHININESS,  85.0)
            _sph(quad, self.pos, self.r)
        elif self._collect_t is not None:
            t = time.time() - self._collect_t
            if t < 0.45:
                glDisable(GL_LIGHTING)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                alpha = (1.0 - t / 0.45) * 0.9
                glColor4f(0.5, 0.85, 1.0, alpha)
                _sph(quad, self.pos, self.r + t * 0.6, slices=10)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                glDisable(GL_BLEND)
                glEnable(GL_LIGHTING)


class _BgParticle:
    def __init__(self):
        self._reset()

    def _reset(self):
        self.x = random.uniform(_BUBBLE_EXIT_X, _BUBBLE_SPAWN_X)
        self.y = random.uniform(_Y_BOT, _Y_TOP)
        self.r = random.uniform(0.012, 0.040)
        self.vy = random.uniform(0.003, 0.008)
        self.alpha = random.uniform(0.10, 0.28)

    def update(self):
        self.y += self.vy
        if self.y - self.r > _Y_TOP:
            self._reset()
            self.y = _Y_BOT

    def draw(self, quad):
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glColor4f(0.6, 0.85, 1.0, self.alpha)
        _sph(quad, np.array([self.x, self.y, -1.5]), self.r, slices=8)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)


class FishSwim:
    def __init__(self, cfg: dict):
        w = cfg.get("width",  800)
        h = cfg.get("height", 600)
        self._w, self._h = w, h
        self._scale      = float(cfg.get("arm_scale", 3.2))

        import os
        os.environ["SDL_VIDEO_WINDOW_POS"] = "490,30"

        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Fish Swim — tilt your arm to swim!")
        pygame.display.set_mode((w, h), DOUBLEBUF | OPENGL)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)

        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 4.0, 6.0, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.85, 0.92, 1.00, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.20, 0.25, 0.35, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.00, 1.00, 1.00, 1.0])

        glMatrixMode(GL_PROJECTION)
        gluPerspective(50.0, w / h, 0.05, 50.0)
        glMatrixMode(GL_MODELVIEW)

        self._quad      = gluNewQuadric()
        gluQuadricNormals(self._quad, GLU_SMOOTH)
        self._font      = pygame.font.SysFont("Arial", 28, bold=True)
        self._score_tex = glGenTextures(1)

        self._fish_y      = _Y_MID
        self._fish_vy     = 0.0
        self._fish_angle  = 0.0
        self._t           = 0.0

        self.score        = 0
        self.missed       = 0
        self._wave        = 1
        self._kills       = 0
        self._bubbles: list[_Bubble]      = []
        self._bg:      list[_BgParticle]  = [_BgParticle() for _ in range(30)]
        self._last_spawn  = time.time()
        self._spawn_ivl   = 1.4
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
        self._t += 1.0 / 60.0

        tilt = 0.0
        if ("right_elbow" in world_joints and "right_wrist" in world_joints):
            elbow = world_joints["right_elbow"]
            wrist = world_joints["right_wrist"]
            fore  = wrist - elbow
            arm2d = float(np.linalg.norm(fore[:2]))
            if arm2d > 0.05:
                tilt = float(fore[1]) / arm2d
                self._fish_angle = math.atan2(float(fore[1]), float(fore[0]))

        target_vy = tilt * _MAX_VY
        self._fish_vy = (self._fish_vy * _DAMPING
                         + target_vy * (1.0 - _DAMPING))
        self._fish_y  = float(np.clip(self._fish_y + self._fish_vy, _Y_BOT, _Y_TOP))

        if self._fish_y <= _Y_BOT + 0.05:
            self._fish_angle = max(self._fish_angle, 0.0)
        if self._fish_y >= _Y_TOP - 0.05:
            self._fish_angle = min(self._fish_angle, 0.0)

        fish_pos = np.array([0.0, self._fish_y, 0.0], np.float32)

        for p in self._bg:
            p.update()

        for b in self._bubbles:
            b.move()
            if not b.collected and b.pos[0] < _BUBBLE_EXIT_X:
                self.missed += 1
            if b.try_collect(fish_pos):
                self.score += 1
                self._kills += 1

        if self._kills >= 12:
            self._wave  += 1
            self._kills  = 0

        self._bubbles = [b for b in self._bubbles if not b.is_done()]
        self._maybe_spawn()
        return 0

    def render(self, world_joints: dict, steadiness: float = 0.0,
               burst_ready: bool = False, burst_rem: float = 0.0):
        glClearColor(0.02, 0.07, 0.22, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        cam_y = self._fish_y
        gluLookAt(
            0.0, cam_y, 5.5,
            0.0, cam_y, 0.0,
            0.0, 1.0,   0.0,
        )

        for p in self._bg:
            p.draw(self._quad)

        self._draw_surface(cam_y)

        for b in self._bubbles:
            b.draw(self._quad)

        fish_pos = np.array([0.0, self._fish_y, 0.0], np.float32)
        _draw_fish(self._quad, fish_pos, self._fish_angle, self._t)

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
        for b in self._bubbles:
            if not b.collected:
                b.collected  = True
                b._collect_t = time.time()
                self.score  += 1
                self._kills += 1
        if self._kills >= 12:
            self._wave  += 1
            self._kills  = 0
        self._burst_flash_t = time.time()

    def close(self):
        gluDeleteQuadric(self._quad)
        pygame.quit()

    def _speed_mult(self) -> float:
        return 1.0 + (self._wave - 1) * 0.15

    def _maybe_spawn(self):
        now  = time.time()
        live = sum(1 for b in self._bubbles if not b.collected)
        ivl  = self._spawn_ivl / self._speed_mult()
        if live < 5 and now - self._last_spawn >= ivl:
            # spread y so bubbles force arm movement across the full range
            used_y = [b.pos[1] for b in self._bubbles if not b.collected]
            for _ in range(20):
                cand = random.uniform(_Y_BOT + 0.2, _Y_TOP - 0.2)
                if all(abs(cand - y) > 0.35 for y in used_y):
                    break
            b = _Bubble(self._speed_mult())
            b.pos[1] = cand
            self._bubbles.append(b)
            self._last_spawn = now

    def _draw_surface(self, cam_y: float):
        if abs(_Y_TOP - cam_y) > 3.0:
            return
        t = self._t
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.5)
        glBegin(GL_LINE_STRIP)
        for i in range(60):
            x = _BUBBLE_EXIT_X + i * ((_BUBBLE_SPAWN_X - _BUBBLE_EXIT_X) / 59)
            y = _Y_TOP + 0.05 * math.sin(x * 4.0 + t * 2.5)
            glColor4f(0.5, 0.8, 1.0, 0.25)
            glVertex3f(x, y, 0.0)
        glEnd()
        glLineWidth(1.0)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    def _draw_hud(self):
        angle_deg = math.degrees(self._fish_angle)
        if angle_deg > 5:
            direction = f"↑ {angle_deg:.0f}°"
        elif angle_deg < -5:
            direction = f"↓ {abs(angle_deg):.0f}°"
        else:
            direction = "→ level"

        text = (f"Score: {self.score}   Wave {self._wave}   "
                f"Missed: {self.missed}   Arm: {direction}   ESC = menu")
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
        glTexCoord2f(0, 0); glVertex2f(x,      y)
        glTexCoord2f(1, 0); glVertex2f(x + tw, y)
        glTexCoord2f(1, 1); glVertex2f(x + tw, y + th)
        glTexCoord2f(0, 1); glVertex2f(x,      y + th)
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
