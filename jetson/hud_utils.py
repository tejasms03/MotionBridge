import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

_MAX_SHAKE = 0.4


def draw_status_overlay(w: int, h: int, font: pygame.font.Font,
                        steadiness: float, burst_ready: bool,
                        burst_rem: float, tex_id: int):
    if burst_ready:
        emg_txt   = "EMG: READY"
        emg_color = (80, 255, 120)
    else:
        emg_txt   = f"EMG: {burst_rem:.1f}s"
        emg_color = (180, 180, 180)

    surf = font.render(emg_txt, True, emg_color)
    tw, th = surf.get_size()
    raw    = pygame.image.tostring(surf, "RGBA", True)

    glDisable(GL_LIGHTING); glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    gluOrtho2D(0, w, 0, h)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()

    bx, by, bw, bh = 16, 16, 180, 14
    stability = max(0.0, 1.0 - steadiness / _MAX_SHAKE)
    filled    = int(bw * stability)
    r, g      = 1.0 - stability, stability

    glDisable(GL_TEXTURE_2D)

    glColor4f(0.12, 0.12, 0.12, 0.75)
    glBegin(GL_QUADS)
    glVertex2f(bx,      by);      glVertex2f(bx + bw, by)
    glVertex2f(bx + bw, by + bh); glVertex2f(bx,      by + bh)
    glEnd()

    if filled > 0:
        glColor4f(r, g, 0.0, 0.9)
        glBegin(GL_QUADS)
        glVertex2f(bx,          by);      glVertex2f(bx + filled, by)
        glVertex2f(bx + filled, by + bh); glVertex2f(bx,          by + bh)
        glEnd()

    glColor4f(0.65, 0.65, 0.65, 0.9)
    glBegin(GL_LINE_LOOP)
    glVertex2f(bx,      by);      glVertex2f(bx + bw, by)
    glVertex2f(bx + bw, by + bh); glVertex2f(bx,      by + bh)
    glEnd()

    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, th, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, raw)

    tx, ty = 16, by + bh + 6
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(tx,      ty)
    glTexCoord2f(1, 0); glVertex2f(tx + tw, ty)
    glTexCoord2f(1, 1); glVertex2f(tx + tw, ty + th)
    glTexCoord2f(0, 1); glVertex2f(tx,      ty + th)
    glEnd()

    glDisable(GL_TEXTURE_2D)
    lbl_surf = font.render("Stability", True, (220, 220, 220))
    lbl_surf = pygame.transform.scale(lbl_surf,
                   (lbl_surf.get_width() // 2, lbl_surf.get_height() // 2))
    lw, lh = lbl_surf.get_size()
    lraw   = pygame.image.tostring(lbl_surf, "RGBA", True)
    glEnable(GL_TEXTURE_2D)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, lw, lh, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, lraw)
    lx, ly = bx + 4, by + (bh - lh) // 2
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(lx,      ly)
    glTexCoord2f(1, 0); glVertex2f(lx + lw, ly)
    glTexCoord2f(1, 1); glVertex2f(lx + lw, ly + lh)
    glTexCoord2f(0, 1); glVertex2f(lx,      ly + lh)
    glEnd()

    glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glDisable(GL_TEXTURE_2D); glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING)
