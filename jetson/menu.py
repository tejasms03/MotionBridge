import pygame

GAMES = [
    {
        "key":   "1",
        "name":  "Monster Defense",
        "module": "virtual_arm",
        "class": "VirtualArm",
        "desc":  ["Monsters march toward your", "shoulder. Swipe them away!"],
        "trains": "All directions",
        "icon_color": (200, 50,  50),
    },
    {
        "key":   "2",
        "name":  "Balloon Rise",
        "module": "game_rise",
        "class": "BalloonRise",
        "desc":  ["Balloons float upward.", "Raise your arm to pop them!"],
        "trains": "Upward lift",
        "icon_color": (50, 130, 220),
    },
    {
        "key":   "3",
        "name":  "Reach & Hold",
        "module": "game_targets",
        "class": "ReachTargets",
        "desc":  ["Targets light up around you.", "Hold your hand on each one."],
        "trains": "Precise reach + sustained hold",
        "icon_color": (50, 190,  90),
    },
    {
        "key":   "4",
        "name":  "Fish Swim",
        "module": "game_fish",
        "class": "FishSwim",
        "desc":  ["Your arm IS the fish.", "Tilt up to rise, tilt down to dive!"],
        "trains": "Forearm rotation + shoulder angle",
        "icon_color": (30, 190, 200),
    },
]


def run_menu(cfg) -> dict | None:
    pygame.init()
    pygame.font.init()

    W, H = 1100, 480
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Arm Rehab — Pick a Game")

    f_title = pygame.font.SysFont("Arial", 38, bold=True)
    f_name  = pygame.font.SysFont("Arial", 22, bold=True)
    f_body  = pygame.font.SysFont("Arial", 17)
    f_small = pygame.font.SysFont("Arial", 14)

    CARD_W, CARD_H = 230, 270
    CARD_Y = 140
    GAP    = 36
    total  = len(GAMES) * CARD_W + (len(GAMES) - 1) * GAP
    start_x = (W - total) // 2

    cards = [
        pygame.Rect(start_x + i * (CARD_W + GAP), CARD_Y, CARD_W, CARD_H)
        for i in range(len(GAMES))
    ]

    clock   = pygame.time.Clock()
    hovered = None

    while True:
        mx, my = pygame.mouse.get_pos()
        hovered = next((i for i, r in enumerate(cards) if r.collidepoint(mx, my)), None)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return None
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return None
                for g in GAMES:
                    if e.key == getattr(pygame, f"K_{g['key']}"):
                        pygame.quit()
                        return g
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if hovered is not None:
                    pygame.quit()
                    return GAMES[hovered]

        screen.fill((14, 11, 26))

        t = f_title.render("Arm Rehab  —  Choose a Game", True, (210, 210, 255))
        screen.blit(t, ((W - t.get_width()) // 2, 44))

        for i, (g, rect) in enumerate(zip(GAMES, cards)):
            hot = (i == hovered)
            col = g["icon_color"]

            sh = pygame.Rect(rect.x + 5, rect.y + 5, rect.w, rect.h)
            pygame.draw.rect(screen, (0, 0, 0, 160), sh, border_radius=14)

            bg = (48, 42, 72) if hot else (30, 26, 50)
            pygame.draw.rect(screen, bg, rect, border_radius=14)

            pygame.draw.rect(screen, col,
                             pygame.Rect(rect.x, rect.y, rect.w, 7),
                             border_radius=14)

            pygame.draw.rect(screen, col if hot else (55, 50, 85),
                             rect, 2, border_radius=14)

            badge = pygame.Rect(rect.right - 36, rect.y + 14, 26, 26)
            pygame.draw.rect(screen, (18, 14, 38), badge, border_radius=6)
            pygame.draw.rect(screen, col, badge, 1, border_radius=6)
            ks = f_name.render(g["key"], True, col)
            screen.blit(ks, (badge.x + (badge.w - ks.get_width()) // 2,
                              badge.y + (badge.h - ks.get_height()) // 2))

            ns = f_name.render(g["name"], True, (228, 225, 255))
            screen.blit(ns, (rect.x + 16, rect.y + 24))

            yo = rect.y + 64
            for line in g["desc"]:
                ls = f_body.render(line, True, (175, 168, 210))
                screen.blit(ls, (rect.x + 16, yo))
                yo += 24

            yo += 12
            screen.blit(f_small.render("Best for:", True, (110, 105, 155)),
                        (rect.x + 16, yo))
            yo += 18
            screen.blit(f_body.render(g["trains"], True, col),
                        (rect.x + 16, yo))

            if hot:
                h_txt = f_small.render(f"Click or press  {g['key']}", True, (200, 195, 245))
                screen.blit(h_txt,
                            (rect.x + (rect.w - h_txt.get_width()) // 2,
                             rect.bottom - 26))

        ft = f_small.render(
            "Press  1 / 2 / 3 / 4  or click a card to start    ESC = quit",
            True, (90, 85, 130))
        screen.blit(ft, ((W - ft.get_width()) // 2, H - 36))

        pygame.display.flip()
        clock.tick(60)
