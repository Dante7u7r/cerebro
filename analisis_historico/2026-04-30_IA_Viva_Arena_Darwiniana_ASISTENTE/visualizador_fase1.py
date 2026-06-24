#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VISUALIZADOR FASE 1 - Pygame para ver la evolucion en tiempo real.

Muestra:
  - Organismos: poligonos coloreados por linaje, con segmentos conectados
  - Comida: puntos verdes (normal), dorados (rich), rojos (toxic)
  - Obstaculos: lineas grises
  - Depredadores: triangulos negros que persiguen
  - Panel lateral: top 3 fitness, morfoma del mejor, generacion, era
"""

from __future__ import annotations
import math
from typing import Iterable

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

from mundo_fisico import (FOOD_NORMAL, FOOD_RICH, FOOD_TOXIC, FOOD_COLOR,
                            OBS_COLOR, MundoFisico)


# ---------- Layout ----------
PANEL_W = 320
WORLD_DISPLAY_PAD = 10
MAX_WORLD_DISPLAY = 800   # F1.5: limite para que arenas 1200x1200 quepan


class Visualizador:
    def __init__(self, world: MundoFisico, organismos: Iterable, depredadores: Iterable):
        if not HAS_PYGAME:
            raise RuntimeError("pygame no esta instalado. pip install pygame")

        self.world = world
        pygame.init()
        pygame.display.set_caption("IA Viva - Arena Darwiniana (Fase 1.5)")

        # F1.5: scale para que el mundo quepa en pantalla
        self.scale = min(1.0, MAX_WORLD_DISPLAY / max(world.WORLD_W, world.WORLD_H))
        self.disp_w = int(world.WORLD_W * self.scale)
        self.disp_h = int(world.WORLD_H * self.scale)

        self.win_w = self.disp_w + PANEL_W
        self.win_h = self.disp_h
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 13)
        self.font_small = pygame.font.SysFont("consolas", 11)
        self.font_title = pygame.font.SysFont("consolas", 16, bold=True)

        # Trails para visualizar exploracion
        self._trails: dict[int, list[tuple[float, float]]] = {}

        # F1.5-B: surface para feromonas (se redibuja cada frame)
        self._phero_surface = pygame.Surface((self.disp_w, self.disp_h),
                                              pygame.SRCALPHA)

    # ------------------------------------------------------------------ helpers
    def _world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        return int(x * self.scale), int(y * self.scale)

    def _r_to_screen(self, r: float) -> int:
        return max(1, int(r * self.scale))

    def _draw_grid(self):
        # cuadrantes (C4)
        col = (50, 50, 60)
        mx = self.disp_w // 2
        my = self.disp_h // 2
        pygame.draw.line(self.screen, col, (mx, 0), (mx, self.disp_h), 1)
        pygame.draw.line(self.screen, col, (0, my), (self.disp_w, my), 1)

    def _draw_world_bg(self):
        self.screen.fill((22, 24, 30))
        pygame.draw.rect(self.screen, (32, 34, 42),
                         (0, 0, self.disp_w, self.disp_h))
        self._draw_friction_zones()
        self._draw_pheromones()
        self._draw_grid()

    def _draw_friction_zones(self):
        """F1.5-D: zonas de friccion como circulos tenues."""
        zones = getattr(self.world, "friction_zones", [])
        for cx, cy, r in zones:
            sx, sy = self._world_to_screen(cx, cy)
            sr = self._r_to_screen(r)
            # Circulo relleno semi-transparente
            surf = pygame.Surface((sr * 2, sr * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (90, 70, 110, 60), (sr, sr), sr)
            self.screen.blit(surf, (sx - sr, sy - sr))

    def _draw_pheromones(self):
        """F1.5-B: heat map suave de feromonas."""
        phero = getattr(self.world, "_phero", None)
        if phero is None:
            return
        # solo dibujar si hay senal significativa
        if float(phero.max()) < 0.1:
            return
        self._phero_surface.fill((0, 0, 0, 0))
        gN = self.world.PHERO_GRID
        cell_w = self.disp_w / gN
        cell_h = self.disp_h / gN
        max_v = max(0.5, float(phero.max()))
        for gy in range(gN):
            for gx in range(gN):
                v = phero[gy, gx]
                if v < 0.05:
                    continue
                alpha = int(min(110, 180 * v / max_v))
                pygame.draw.rect(self._phero_surface,
                                  (240, 180, 80, alpha),
                                  (int(gx * cell_w), int(gy * cell_h),
                                   int(cell_w + 1), int(cell_h + 1)))
        self.screen.blit(self._phero_surface, (0, 0))

    def _draw_obstacles(self):
        """E5: dibujar segun tipo. Plasticos llevan marcadores en extremos
        moldeables; dinamicos llevan flecha de direccion."""
        for ob in self.world.obstaculos:
            p1 = self._world_to_screen(*ob.p1)
            p2 = self._world_to_screen(*ob.p2)
            kind = getattr(ob, "kind", "rigid")
            color = OBS_COLOR.get(kind, (130, 130, 140))
            thick_px = max(2, int(ob.thickness * self.scale))
            pygame.draw.line(self.screen, color, p1, p2, thick_px)

            # Marcadores en extremos moldeables (plastico/vivo)
            if getattr(ob, "is_plastic", False):
                # Indicar deformacion acumulada con un anillo cuyo color
                # se intensifica si esta cerca del maximo.
                max_def = getattr(self.world, "MAX_DEFORM", 60.0)
                for ep, deform in ((p1, ob.p1_deform), (p2, ob.p2_deform)):
                    mag = (deform[0] ** 2 + deform[1] ** 2) ** 0.5
                    frac = min(1.0, mag / max_def)
                    # color de marcador: del marron base al rojo si muy deformado
                    mc = (int(180 + 60 * frac), int(120 - 80 * frac),
                          int(70 - 50 * frac))
                    pygame.draw.circle(self.screen, mc, ep, 4)
                    pygame.draw.circle(self.screen, (40, 30, 20), ep, 4, 1)

            # Flecha de direccion para dinamicos / vivos
            if getattr(ob, "is_dynamic", False):
                vx = getattr(ob, "vx", 0.0)
                vy = getattr(ob, "vy", 0.0)
                speed = (vx * vx + vy * vy) ** 0.5
                if speed > 0.01:
                    cx = (p1[0] + p2[0]) // 2
                    cy = (p1[1] + p2[1]) // 2
                    nx = vx / speed
                    ny = vy / speed
                    tip = (int(cx + nx * 8), int(cy + ny * 8))
                    pygame.draw.line(self.screen, (240, 240, 200),
                                     (cx, cy), tip, 1)
                    pygame.draw.circle(self.screen, (240, 240, 200), tip, 2)

    def _draw_food(self):
        for f in self.world.food_items:
            if not f.alive:
                continue
            color = FOOD_COLOR.get(f.kind, (200, 200, 200))
            x, y = self._world_to_screen(f.x, f.y)
            r_px = self._r_to_screen(f.radius)
            pygame.draw.circle(self.screen, color, (x, y), r_px)
            if f.kind == FOOD_TOXIC:
                pygame.draw.circle(self.screen, (255, 200, 200), (x, y),
                                    r_px, 1)

    def _draw_organisms(self, organismos):
        for o in organismos:
            if o.cuerpo is None or not o.cuerpo.bodies:
                continue
            color = o.lineage_color
            # Dibujar segmentos
            for shape in o.cuerpo.shapes:
                bx, by = shape.body.position
                sx, sy = self._world_to_screen(bx, by)
                if isinstance(shape, type(shape)) and hasattr(shape, "radius"):
                    r_world = float(getattr(shape, "radius", 6))
                    r_px = self._r_to_screen(r_world)
                    pygame.draw.circle(self.screen, color, (sx, sy), r_px)
                    ang = shape.body.angle
                    dx = int(math.cos(ang) * r_px)
                    dy = int(math.sin(ang) * r_px)
                    pygame.draw.circle(self.screen, (255, 255, 255),
                                       (sx + dx, sy + dy), 2)
            # Dibujar conexiones de joints
            for j in o.morphome.joints:
                if 0 <= j.seg_a < len(o.cuerpo.bodies) and 0 <= j.seg_b < len(o.cuerpo.bodies):
                    pa = o.cuerpo.bodies[j.seg_a].position
                    pb = o.cuerpo.bodies[j.seg_b].position
                    pygame.draw.line(self.screen, (200, 200, 220),
                                     self._world_to_screen(pa.x, pa.y),
                                     self._world_to_screen(pb.x, pb.y), 1)
            # Trail (ultimas posiciones)
            oid = id(o)
            trail = self._trails.setdefault(oid, [])
            trail.append((o.cuerpo.x, o.cuerpo.y))
            if len(trail) > 40:
                trail.pop(0)
            for i in range(len(trail) - 1):
                a = self._world_to_screen(*trail[i])
                b = self._world_to_screen(*trail[i + 1])
                alpha = int(80 * (i / len(trail)))
                tcol = (color[0] // 3, color[1] // 3, color[2] // 3)
                pygame.draw.line(self.screen, tcol, a, b, 1)

    def _draw_predators(self, depredadores):
        for pred in depredadores:
            x, y = self._world_to_screen(pred.x, pred.y)
            r = self._r_to_screen(14)
            pygame.draw.polygon(self.screen, (20, 20, 20), [
                (x, y - r), (x - r, y + r), (x + r, y + r),
            ])
            pygame.draw.polygon(self.screen, (200, 30, 30), [
                (x, y - r), (x - r, y + r), (x + r, y + r),
            ], 2)

    def _draw_panel(self, gen, step, organismos, depredadores):
        x0 = self.disp_w + WORLD_DISPLAY_PAD
        y = 8

        # Header
        title = self.font_title.render("IA VIVA - FASE 1.5", True, (240, 240, 240))
        self.screen.blit(title, (x0, y)); y += 20
        info = self.font.render(f"Gen {gen} Step {step}", True, (200, 200, 220))
        self.screen.blit(info, (x0, y)); y += 16
        info = self.font.render(
            f"Era {self.world.era}  Food {sum(1 for f in self.world.food_items if f.alive)}",
            True, (200, 200, 220))
        self.screen.blit(info, (x0, y)); y += 16
        info = self.font.render(
            f"Predators {len(depredadores)}  Obstacles {len(self.world.obstaculos)}",
            True, (200, 200, 220))
        self.screen.blit(info, (x0, y)); y += 22

        # Top 3 fitness
        top = sorted(organismos, key=lambda o: o.fitness(), reverse=True)[:5]
        head = self.font.render("Top 5 fitness:", True, (180, 220, 180))
        self.screen.blit(head, (x0, y)); y += 18
        for o in top:
            line = (f"{o.fitness():6.2f}  {o.nombre[:14]:<14} "
                    f"f={o.food_eaten:2d} q={o.quadrant}")
            t = self.font_small.render(line, True, o.lineage_color)
            self.screen.blit(t, (x0, y)); y += 13

        y += 10
        # Mejor: morfoma details
        if top:
            best = top[0]
            head = self.font.render(f"Mejor: {best.nombre}", True, best.lineage_color)
            self.screen.blit(head, (x0, y)); y += 16
            txt = best.morphome.describe()
            t = self.font_small.render(txt, True, (200, 200, 220))
            self.screen.blit(t, (x0, y)); y += 13
            t = self.font_small.render(f"parent: {best.parent}", True, (160, 160, 180))
            self.screen.blit(t, (x0, y)); y += 13
            if best.cuerpo:
                t = self.font_small.render(
                    f"toxic: {best.cuerpo._toxic_count}  "
                    f"hits: {best.cuerpo._predator_hits}",
                    True, (180, 180, 200))
                self.screen.blit(t, (x0, y)); y += 13

        y += 10
        # Distribucion por linaje
        head = self.font.render("Linajes activos:", True, (220, 180, 220))
        self.screen.blit(head, (x0, y)); y += 16
        from collections import defaultdict
        lin_count = defaultdict(int)
        for o in organismos:
            base = o.nombre.split("-")[0].rstrip("x")
            lin_count[base] += 1
        for base, count in sorted(lin_count.items(), key=lambda kv: -kv[1]):
            from arena_fase1 import _color_para_linaje
            col = _color_para_linaje(base)
            t = self.font_small.render(f"{base:<10} x {count}", True, col)
            self.screen.blit(t, (x0, y)); y += 13

        # Footer hint
        y = self.win_h - 30
        hint = self.font_small.render("ESC para cerrar", True, (120, 120, 140))
        self.screen.blit(hint, (x0, y))

    # ------------------------------------------------------------------ public
    def draw(self, gen: int, step: int, organismos, depredadores) -> bool:
        """Dibuja un frame. Devuelve False si el usuario cierra la ventana."""
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                return False

        self._draw_world_bg()
        self._draw_obstacles()
        self._draw_food()
        self._draw_organisms(organismos)
        self._draw_predators(depredadores)
        self._draw_panel(gen, step, organismos, depredadores)
        pygame.display.flip()
        # No clock.tick para no limitar el ritmo de la sim
        return True

    def close(self):
        try:
            pygame.quit()
        except Exception:
            pass
