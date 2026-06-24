#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPREDADOR - Entidad simple que persigue al organismo mas cercano.

C1: Crea presion de seleccion para velocidad y morfologias de escape.
Los organismos lentos (mucha masa sin motores) son castigados.
"""

from __future__ import annotations
import math
import pymunk

from cuerpo_evolvable import CuerpoEvolvable


class Depredador:
    """Cuerpo pymunk simple que persigue al organismo mas cercano."""

    def __init__(self, world, spawn_xy=(50.0, 50.0), speed: float = 45.0,
                 hit_radius: float = 16.0, hit_cooldown: int = 240):
        self.world = world
        self.space = world.space
        self.speed = speed
        self.hit_radius = hit_radius
        self.hit_cooldown = hit_cooldown   # frames entre golpes consecutivos al mismo objetivo
        self._last_hit_step: dict[int, int] = {}
        self._step = 0

        mass = 3.0
        radius = 14.0
        moment = pymunk.moment_for_circle(mass, 0, radius)
        self.body = pymunk.Body(mass, moment)
        self.body.position = spawn_xy
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.friction = 0.5
        self.shape.elasticity = 0.2
        self.shape.color_rgb = (40, 40, 40)
        self.shape.density = 0.05
        # Filtro de colision: choca con cuerpos pero no con comida (la comida es virtual)
        self.shape.filter = pymunk.ShapeFilter(categories=0b1000)
        self.space.add(self.body, self.shape)

    @property
    def x(self) -> float:
        return self.body.position.x
    @property
    def y(self) -> float:
        return self.body.position.y

    def step(self):
        """Avanza un paso: persigue al organismo mas cercano. Devuelve la lista
        de organismos golpeados en este step."""
        self._step += 1
        target = self._find_target()
        if target is None:
            return []

        # Direccion hacia el target
        dx = target.x - self.x
        dy = target.y - self.y
        dist = math.hypot(dx, dy)
        if dist > 0.01:
            ux = dx / dist
            uy = dy / dist
            # Aplicar fuerza proporcional al dt (asumiendo 1/60s)
            self.body.velocity = (ux * self.speed, uy * self.speed)
        else:
            self.body.velocity = (0.0, 0.0)

        # Detectar golpes
        hits = []
        for org in list(self.world.organisms):
            d = math.hypot(org.x - self.x, org.y - self.y)
            if d < self.hit_radius:
                org_id = id(org)
                last = self._last_hit_step.get(org_id, -9999)
                if self._step - last >= self.hit_cooldown:
                    self._last_hit_step[org_id] = self._step
                    hits.append(org)
        return hits

    def _find_target(self) -> CuerpoEvolvable | None:
        if not self.world.organisms:
            return None
        nearest = None
        nearest_d = float("inf")
        for org in self.world.organisms:
            if org.bodies is None or len(org.bodies) == 0:
                continue
            d = math.hypot(org.x - self.x, org.y - self.y)
            if d < nearest_d:
                nearest_d = d
                nearest = org
        return nearest

    def destroy(self):
        try:
            if self.shape in self.space.shapes:
                self.space.remove(self.shape)
            if self.body in self.space.bodies:
                self.space.remove(self.body)
        except Exception:
            pass
