#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MUNDO FISICO - Espacio pymunk con comida, paredes, organismos.

Reemplaza SharedWorld. La presion de seleccion la genera el entorno:
  - Comida limitada y dispersa
  - Paredes que detienen el movimiento
  - Sin script que diga "ve hacia la comida": solo fisica + dopamina

Modulo A (mundo rico):
  A1 - Food clusters: 65% comida en zonas calientes, 35% disperso
  A2 - Obstaculos estaticos en pymunk
  A3 - Tipos de comida: normal / rich / toxic
  A4 - Sistema de Eras (escalado de dificultad)

Fase 1.5 (mundo social):
  F1.5-A - Arena 1200x1200 (4x el area), 6 clusters (vs 4)
  F1.5-B - Feromonas: cuadricula 60x60 con depositos al comer y decay temporal
  F1.5-D - Zonas de friccion: 3 areas circulares que ralentizan a los organismos

Fase 1.6 (cadena alimenticia + obstaculos vivos):
  C2 - Depredacion biologicamente realista (ratio 4x, contacto sostenido,
       cooldown saciedad, recompensa proporcional)
  E1 - Obstaculos dinamicos: derivan lento, rebotan en paredes
  E2 - Obstaculos plasticos: sus extremos se moldean cuando un organismo
       suficientemente masivo los empuja
  E3 - Obstaculos vivos: combinan E1 + E2 (mueven Y se moldean)
"""

from __future__ import annotations
import math
import random
import pymunk


# ---------- Constantes de tipos de comida (A3) ----------
FOOD_NORMAL = "normal"
FOOD_RICH   = "rich"
FOOD_TOXIC  = "toxic"

FOOD_REWARD = {
    FOOD_NORMAL: 1.0,
    FOOD_RICH:   3.0,
    FOOD_TOXIC: -0.5,
}

FOOD_COLOR = {
    FOOD_NORMAL: (90, 200, 90),     # verde
    FOOD_RICH:   (240, 200, 60),    # dorado
    FOOD_TOXIC:  (220, 60, 60),     # rojo
}


class FoodItem:
    __slots__ = ("x", "y", "radius", "alive", "kind")

    def __init__(self, x: float, y: float, radius: float = 6.0, kind: str = FOOD_NORMAL):
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)
        self.alive = True
        self.kind = kind


# ---------- Tipos de obstaculos (Modulo E) ----------
OBS_RIGID = "rigid"          # No se mueve, no se moldea — ancla del mundo
OBS_DYNAMIC = "dynamic"      # Deriva, rebota en paredes
OBS_PLASTIC = "plastic"      # Extremos moldeables por masa suficiente
OBS_LIVING = "living"        # Deriva Y se moldea (combinacion mas dificil)

OBS_COLOR = {
    OBS_RIGID:   (110, 110, 120),   # gris oscuro
    OBS_DYNAMIC: (110, 130, 170),   # gris azulado
    OBS_PLASTIC: (170, 130,  90),   # marron calido
    OBS_LIVING:  (210, 150,  60),   # ambar/naranja
}


class Obstaculo:
    """Wrapper para un Segment estatico (A2 + Modulo E).

    Los extremos finales (p1, p2) se calculan como:
        p_final = p_orig + (drift_x, drift_y) + p_deform

    - drift = traslacion solida (obstaculos dinamicos / vivos)
    - deform = desplazamiento independiente por extremo (plasticos / vivos)
    """
    __slots__ = ("segment", "thickness", "kind",
                 "p1_orig", "p2_orig",
                 "drift_x", "drift_y", "vx", "vy",
                 "p1_deform", "p2_deform")

    def __init__(self, segment, p1, p2, thickness, kind: str = OBS_RIGID,
                 vx: float = 0.0, vy: float = 0.0):
        self.segment = segment
        self.thickness = thickness
        self.kind = kind
        self.p1_orig = (float(p1[0]), float(p1[1]))
        self.p2_orig = (float(p2[0]), float(p2[1]))
        self.drift_x = 0.0
        self.drift_y = 0.0
        self.vx = float(vx)
        self.vy = float(vy)
        self.p1_deform = (0.0, 0.0)
        self.p2_deform = (0.0, 0.0)

    @property
    def p1(self) -> tuple[float, float]:
        return (self.p1_orig[0] + self.drift_x + self.p1_deform[0],
                self.p1_orig[1] + self.drift_y + self.p1_deform[1])

    @property
    def p2(self) -> tuple[float, float]:
        return (self.p2_orig[0] + self.drift_x + self.p2_deform[0],
                self.p2_orig[1] + self.drift_y + self.p2_deform[1])

    @property
    def is_dynamic(self) -> bool:
        return self.kind in (OBS_DYNAMIC, OBS_LIVING)

    @property
    def is_plastic(self) -> bool:
        return self.kind in (OBS_PLASTIC, OBS_LIVING)


class MundoFisico:
    """Espacio pymunk 2D con comida, obstaculos y organismos."""

    # F1.5-A: Arena cuatro veces mas grande (de 600 a 1200)
    WORLD_W = 1200.0
    WORLD_H = 1200.0

    # F1.5-B: cuadricula de feromonas
    PHERO_GRID = 60          # 60x60 celdas (cada celda cubre 20x20 px)
    PHERO_DECAY = 0.998      # decay por step
    PHERO_DEPOSIT = 2.0      # cantidad depositada al comer

    # F1.5-D: zonas de friccion
    FRICTION_ZONE_FACTOR = 0.40   # multiplicador de fuerza dentro de zona

    def __init__(self, n_food: int = 120, n_obstacles: int = 6,
                 seed: int | None = None):
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)        # top-down
        self.space.damping = 0.6                # friccion al ambiente
        self.organisms: list = []                # lista de CuerpoEvolvable
        self.food_items: list[FoodItem] = []
        self.obstaculos: list[Obstaculo] = []
        self.n_food_target = n_food
        self.n_obstacles_target = n_obstacles
        self.rng = random.Random(seed)

        # A1: centros de clusters (se mantienen entre repoblamientos)
        self._cluster_centers: list[tuple[float, float]] = []
        # F1.5-A: 6 clusters en lugar de 4 (arena mas grande)
        self._n_clusters_default = 6
        # Proporciones por tipo de comida (A3) - ajustables por era
        self.food_kind_ratios = {
            FOOD_NORMAL: 0.75,
            FOOD_RICH:   0.15,
            FOOD_TOXIC:  0.10,
        }

        # F1.5-B: cuadricula de feromonas (numpy array)
        import numpy as _np
        self._phero = _np.zeros((self.PHERO_GRID, self.PHERO_GRID), dtype=_np.float32)

        # F1.5-D: zonas de friccion (lista de (cx, cy, radius))
        self.friction_zones: list[tuple[float, float, float]] = []

        # Compatibilidad legacy con SharedWorld (semilla_cerebro.py espera esto)
        self.WORLD_SIZE = self.WORLD_W   # asume mundo cuadrado

        # F2.0: Ecosistema retroalimentado (reemplaza eras)
        # `era` se mantiene como "nivel de estres ecologico" para compat con
        # semilla_cerebro (era >= 2 desactiva inmunidad social). Se incrementa
        # solo cuando el mundo entra en escasez sostenida.
        self.era = 0
        self.n_food_baseline = float(n_food)   # baseline dinamico
        self._consumed_last_gen = 0            # comida consumida en la gen anterior
        self._spawned_last_gen = max(1, n_food)  # comida disponible en la gen anterior
        self._gen_count = 0                    # contador de generaciones para ciclo estacional
        self._stress_streak = 0                # gens consecutivas en escasez (controla `era`)
        self.SEASON_PERIOD = 30                # gens por ciclo estacional completo
        self.SEASON_AMPLITUDE = 0.25           # ±25% de oscilacion estacional
        self.FOOD_BASELINE_MIN = 40.0
        self.FOOD_BASELINE_MAX = 110.0

        self.tool_x = self.WORLD_W * 0.5
        self.tool_y = self.WORLD_H * 0.5
        self.food_solo: FoodItem | None = None
        self.food_social: FoodItem | None = None

        self._build_walls()
        self._build_obstacles(n_obstacles)
        self._build_friction_zones(n=3)
        self._spawn_initial_food()
        self._refresh_anchored_food()

    # ---------------- compatibilidad SharedWorld ----------------
    @property
    def bodies(self):
        return self.organisms

    def register_body(self, body):
        if body not in self.organisms:
            self.organisms.append(body)

    @property
    def food_solo_x(self):
        return self.food_solo.x if (self.food_solo and self.food_solo.alive) else self.WORLD_W * 0.5
    @property
    def food_solo_y(self):
        return self.food_solo.y if (self.food_solo and self.food_solo.alive) else self.WORLD_H * 0.5
    @property
    def food_social_x(self):
        return self.food_social.x if (self.food_social and self.food_social.alive) else self.WORLD_W * 0.5
    @property
    def food_social_y(self):
        return self.food_social.y if (self.food_social and self.food_social.alive) else self.WORLD_H * 0.5

    def check_food_collision(self, x: float, y: float):
        """Compatibilidad con SharedWorld. Devuelve el FoodItem o None."""
        for f in self.food_items:
            if f.alive and math.hypot(f.x - x, f.y - y) < f.radius + 6.0:
                f.alive = False
                if f is self.food_solo or f is self.food_social:
                    self._refresh_anchored_food()
                return True
        return None

    def _refresh_anchored_food(self):
        alive = [f for f in self.food_items if f.alive]
        if len(alive) >= 2:
            self.food_solo = alive[0]
            self.food_social = alive[1]
        elif len(alive) == 1:
            self.food_solo = alive[0]
            self.food_social = alive[0]

    # ------------------------------------------------------------------ walls
    def _build_walls(self):
        static = self.space.static_body
        thick = 5.0
        W, H = self.WORLD_W, self.WORLD_H
        walls = [
            pymunk.Segment(static, (0, 0), (W, 0), thick),
            pymunk.Segment(static, (W, 0), (W, H), thick),
            pymunk.Segment(static, (W, H), (0, H), thick),
            pymunk.Segment(static, (0, H), (0, 0), thick),
        ]
        for w in walls:
            w.elasticity = 0.4
            w.friction = 0.5
            w.collision_type = 3
        self.space.add(*walls)

    # ------------------------------------------------------------------ obstaculos (A2 + Modulo E)
    # Distribucion objetivo de tipos (Modulo E):
    #   ~33% rigido, ~22% dinamico, ~33% plastico, ~12% vivo
    _OBS_TYPE_TARGETS = {
        OBS_RIGID:   2.0 / 6.0,
        OBS_DYNAMIC: 1.0 / 6.0,
        OBS_PLASTIC: 2.0 / 6.0,
        OBS_LIVING:  1.0 / 6.0,
    }

    def _pick_obstacle_kind(self) -> str:
        """Muestreo aleatorio segun ratios objetivo (para refresh)."""
        r = self.rng.random()
        acc = 0.0
        for kind, w in self._OBS_TYPE_TARGETS.items():
            acc += w
            if r < acc:
                return kind
        return OBS_RIGID

    def _build_obstacles(self, n: int):
        """Construye n obstaculos garantizando que todos los tipos esten presentes."""
        # Distribucion deterministica por deficit
        counts = {k: 0 for k in self._OBS_TYPE_TARGETS}
        kinds: list[str] = []
        for i in range(n):
            best_kind = OBS_RIGID
            best_deficit = -1.0
            for k, target in self._OBS_TYPE_TARGETS.items():
                deficit = target * (i + 1) - counts[k]
                if deficit > best_deficit:
                    best_deficit = deficit
                    best_kind = k
            counts[best_kind] += 1
            kinds.append(best_kind)
        self.rng.shuffle(kinds)
        for k in kinds:
            self._add_one_obstacle(kind=k)

    def _add_one_obstacle(self, kind: str = OBS_RIGID):
        """Anade un segmento estatico de un tipo concreto."""
        margin = 80.0
        cx = self.rng.uniform(margin, self.WORLD_W - margin)
        cy = self.rng.uniform(margin, self.WORLD_H - margin)
        length = self.rng.uniform(40.0, 110.0)
        angle = self.rng.uniform(0, 2 * math.pi)
        dx = math.cos(angle) * length / 2
        dy = math.sin(angle) * length / 2
        p1 = (cx - dx, cy - dy)
        p2 = (cx + dx, cy + dy)
        thick = self.rng.uniform(4.0, 8.0)
        seg = pymunk.Segment(self.space.static_body, p1, p2, thick)
        seg.elasticity = 0.3
        seg.friction = 0.6
        seg.collision_type = 4  # categoria obstaculo
        self.space.add(seg)

        # Velocidad inicial para tipos dinamicos (deriva 0.3-0.6 px/step)
        if kind in (OBS_DYNAMIC, OBS_LIVING):
            speed = self.rng.uniform(0.3, 0.6)
            ang_v = self.rng.uniform(0, 2 * math.pi)
            vx = math.cos(ang_v) * speed
            vy = math.sin(ang_v) * speed
        else:
            vx = vy = 0.0

        self.obstaculos.append(Obstaculo(seg, p1, p2, thick,
                                           kind=kind, vx=vx, vy=vy))

    def add_obstacle(self, kind: str | None = None):
        """Anade un nuevo obstaculo (usado por el sistema de eras)."""
        if kind is None:
            kind = self._pick_obstacle_kind()
        self._add_one_obstacle(kind=kind)
        self.n_obstacles_target = len(self.obstaculos)

    # ------------------------------------------------------------------ food
    def _generate_clusters(self, n_clusters: int | None = None):
        if n_clusters is None:
            n_clusters = self._n_clusters_default
        margin = 100.0
        self._cluster_centers = []
        for _ in range(n_clusters):
            cx = self.rng.uniform(margin, self.WORLD_W - margin)
            cy = self.rng.uniform(margin, self.WORLD_H - margin)
            self._cluster_centers.append((cx, cy))

    def _pick_food_kind(self) -> str:
        r = self.rng.random()
        acc = 0.0
        for kind, prob in self.food_kind_ratios.items():
            acc += prob
            if r < acc:
                return kind
        return FOOD_NORMAL

    def _spawn_initial_food(self):
        """A1: 65% en clusters, 35% disperso. Tipos sampleados por _pick_food_kind."""
        if not self._cluster_centers:
            self._generate_clusters()
        n_cluster = int(self.n_food_target * 0.65)
        margin = 30.0
        for _ in range(n_cluster):
            cx, cy = self.rng.choice(self._cluster_centers)
            x = cx + self.rng.gauss(0, 38)
            y = cy + self.rng.gauss(0, 38)
            x = max(margin, min(self.WORLD_W - margin, x))
            y = max(margin, min(self.WORLD_H - margin, y))
            self.food_items.append(FoodItem(x, y, radius=6.0, kind=self._pick_food_kind()))
        for _ in range(self.n_food_target - n_cluster):
            self._spawn_one_food()

    def _spawn_one_food(self):
        margin = 30.0
        # 70% probabilidad de respawn dentro de un cluster, 30% disperso (mantiene el patron)
        if self._cluster_centers and self.rng.random() < 0.7:
            cx, cy = self.rng.choice(self._cluster_centers)
            x = cx + self.rng.gauss(0, 38)
            y = cy + self.rng.gauss(0, 38)
        else:
            x = self.rng.uniform(margin, self.WORLD_W - margin)
            y = self.rng.uniform(margin, self.WORLD_H - margin)
        x = max(margin, min(self.WORLD_W - margin, x))
        y = max(margin, min(self.WORLD_H - margin, y))
        self.food_items.append(FoodItem(x, y, radius=6.0, kind=self._pick_food_kind()))

    def replenish_food(self):
        # mantener n_food_target items vivos
        self.food_items = [f for f in self.food_items if f.alive]
        while len(self.food_items) < self.n_food_target:
            self._spawn_one_food()

    # ------------------------------------------------------------------ M1: Cluster drift
    def drift_clusters(self, sigma: float = 12.0):
        """Migra ligeramente los centros de clusters cada generacion.
        La memoria espacial de los organismos se vuelve gradualmente obsoleta."""
        margin = 100.0
        new_centers = []
        for cx, cy in self._cluster_centers:
            ncx = cx + self.rng.gauss(0, sigma)
            ncy = cy + self.rng.gauss(0, sigma)
            ncx = max(margin, min(self.WORLD_W - margin, ncx))
            ncy = max(margin, min(self.WORLD_H - margin, ncy))
            new_centers.append((ncx, ncy))
        self._cluster_centers = new_centers

    # ------------------------------------------------------------------ M2: Food burst (eventos transitorios)
    def maybe_spawn_food_burst(self, prob: float = 0.015):
        """Probabilidad por step de generar un parche temporal de comida rica.
        Crea oportunidades transitorias que recompensan la atencion a novedad."""
        if self.rng.random() >= prob:
            return False
        margin = 80.0
        cx = self.rng.uniform(margin, self.WORLD_W - margin)
        cy = self.rng.uniform(margin, self.WORLD_H - margin)
        n = self.rng.randint(8, 12)
        for _ in range(n):
            x = cx + self.rng.gauss(0, 22)
            y = cy + self.rng.gauss(0, 22)
            x = max(30.0, min(self.WORLD_W - 30.0, x))
            y = max(30.0, min(self.WORLD_H - 30.0, y))
            # Burst = 70% rich, 20% normal, 10% toxic (trampa ocasional)
            r = self.rng.random()
            kind = FOOD_RICH if r < 0.7 else (FOOD_NORMAL if r < 0.9 else FOOD_TOXIC)
            self.food_items.append(FoodItem(x, y, radius=7.0, kind=kind))
        return True

    # ------------------------------------------------------------------ M3: Obstacle refresh
    def refresh_obstacles(self, n_remove: int = 2, n_add: int = 3):
        """Elimina algunos obstaculos viejos y anade nuevos en posiciones distintas.
        El layout del mundo deja de ser memorizable a largo plazo."""
        n_remove = min(n_remove, len(self.obstaculos))
        for _ in range(n_remove):
            idx = self.rng.randrange(len(self.obstaculos))
            obs = self.obstaculos.pop(idx)
            try:
                self.space.remove(obs.segment)
            except Exception:
                pass
        for _ in range(n_add):
            self._add_one_obstacle(kind=self._pick_obstacle_kind())
        self.n_obstacles_target = len(self.obstaculos)

    # ------------------------------------------------------------------ M4: Jitter en ratios
    def jitter_food_ratios(self, jitter: float = 0.10):
        """Perturba los ratios de tipos de comida con ruido +-jitter cada generacion.
        Elimina la predictibilidad escalonada de las eras."""
        base = dict(self.food_kind_ratios)
        perturbed = {}
        for k, v in base.items():
            noise = self.rng.uniform(-jitter, jitter)
            perturbed[k] = max(0.02, v + noise)
        # renormalizar
        total = sum(perturbed.values())
        self.food_kind_ratios = {k: v / total for k, v in perturbed.items()}

    def reset_food(self):
        """Reposiciona toda la comida desde cero (entre generaciones)."""
        self.food_items.clear()
        self._spawn_initial_food()
        self._refresh_anchored_food()

    def nearest_food(self, x: float, y: float):
        best = None
        best_d = float("inf")
        for f in self.food_items:
            if not f.alive:
                continue
            d = math.hypot(f.x - x, f.y - y)
            if d < best_d:
                best_d = d
                best = f
        if best is None:
            return None, float("inf")
        return (best.x, best.y), best_d

    # ------------------------------------------------------------------ F2.0: Ecosistema
    def update_ecosystem(self, food_consumed_this_gen: int):
        """F2.0 — Reemplaza el sistema de eras. Se llama al final de cada generacion.

        Logica:
          1. Feedback de consumo: si la poblacion comio mucho respecto a lo
             disponible, el baseline baja (sobrepastoreo). Si comio poco,
             sube (recuperacion ecologica).
          2. Ciclo estacional: el target oscila +/- SEASON_AMPLITUDE alrededor
             del baseline en periodos de SEASON_PERIOD generaciones.
          3. Toxicidad dinamica: aumenta cuando el ecosistema esta estresado.
          4. `era`: actua como contador de estres acumulado. Se incrementa
             tras 5 gens consecutivas con baseline cerca del minimo.
        """
        spawned = max(1, self._spawned_last_gen)
        consume_ratio = food_consumed_this_gen / spawned

        # 1. Feedback de consumo sobre el baseline
        if consume_ratio > 0.60:
            self.n_food_baseline -= 3.0     # sobrepastoreo
        elif consume_ratio < 0.20:
            self.n_food_baseline += 3.0     # recuperacion
        # zona muerta entre 0.20 y 0.60: equilibrio, sin cambios
        self.n_food_baseline = float(max(self.FOOD_BASELINE_MIN,
                                          min(self.FOOD_BASELINE_MAX,
                                              self.n_food_baseline)))

        # 2. Ciclo estacional sobre el target
        self._gen_count += 1
        import math as _math
        phase = 2.0 * _math.pi * self._gen_count / self.SEASON_PERIOD
        season_factor = 1.0 + self.SEASON_AMPLITUDE * _math.sin(phase)
        self.n_food_target = int(self.n_food_baseline * season_factor)

        # 3. Toxicidad y riqueza dinamicas (basadas en el baseline, no en `era`)
        # Cuanto mas estresado el ecosistema (baseline bajo), mas toxica la comida
        stress = (self.FOOD_BASELINE_MAX - self.n_food_baseline) / \
                 (self.FOOD_BASELINE_MAX - self.FOOD_BASELINE_MIN)
        toxic_pct = 0.10 + 0.20 * stress           # 0.10 -> 0.30
        rich_pct  = 0.18 - 0.10 * stress           # 0.18 -> 0.08
        self.food_kind_ratios = {
            FOOD_TOXIC: toxic_pct,
            FOOD_RICH:  rich_pct,
            FOOD_NORMAL: 1.0 - toxic_pct - rich_pct,
        }

        # 4. `era` como contador de estres (compat con era>=2 de semilla_cerebro)
        in_stress = self.n_food_baseline < (self.FOOD_BASELINE_MIN + 15)
        if in_stress:
            self._stress_streak += 1
            if self._stress_streak >= 5:
                self.era += 1
                self._stress_streak = 0
        else:
            self._stress_streak = max(0, self._stress_streak - 1)

        # Reset trackers para la proxima generacion
        self._consumed_last_gen = food_consumed_this_gen
        self._spawned_last_gen = max(1, len(self.food_items))

    # Alias deprecated para compatibilidad con checkpoints viejos
    def avanzar_era(self):
        """DEPRECADO: usado solo al cargar checkpoints pre-F2.0. No-op moderno."""
        self.era += 1
        # No alteramos n_food_target ni nada; el nuevo sistema toma control
        # tras la primera llamada a update_ecosystem()

    # ============================================================
    # F1.5-B: Feromonas
    # ============================================================
    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = int(x / self.WORLD_W * self.PHERO_GRID)
        gy = int(y / self.WORLD_H * self.PHERO_GRID)
        gx = max(0, min(self.PHERO_GRID - 1, gx))
        gy = max(0, min(self.PHERO_GRID - 1, gy))
        return gx, gy

    def deposit_pheromone(self, x: float, y: float,
                          amount: float | None = None):
        """Deposita feromona en la celda correspondiente. Llamado al comer."""
        if amount is None:
            amount = self.PHERO_DEPOSIT
        gx, gy = self._world_to_grid(x, y)
        self._phero[gy, gx] = min(10.0, self._phero[gy, gx] + amount)

    def sense_pheromone(self, x: float, y: float) -> float:
        """Lectura local de feromona normalizada [0, 1]."""
        gx, gy = self._world_to_grid(x, y)
        # Promedio 3x3 para suavizar
        x0 = max(0, gx - 1)
        x1 = min(self.PHERO_GRID, gx + 2)
        y0 = max(0, gy - 1)
        y1 = min(self.PHERO_GRID, gy + 2)
        local = self._phero[y0:y1, x0:x1]
        if local.size == 0:
            return 0.0
        return float(min(1.0, local.mean() / 5.0))

    def step_pheromones(self):
        """Decay global. Se llama cada step."""
        self._phero *= self.PHERO_DECAY

    # ============================================================
    # F1.5-D: Zonas de friccion
    # ============================================================
    def _build_friction_zones(self, n: int = 3):
        margin = 150.0
        self.friction_zones = []
        for _ in range(n):
            cx = self.rng.uniform(margin, self.WORLD_W - margin)
            cy = self.rng.uniform(margin, self.WORLD_H - margin)
            r = self.rng.uniform(80.0, 140.0)
            self.friction_zones.append((cx, cy, r))

    def friction_factor(self, x: float, y: float) -> float:
        """Devuelve el multiplicador de fuerza para esa posicion.

        1.0 = fuera de zona (normal). FRICTION_ZONE_FACTOR (~0.4) si dentro.
        """
        for cx, cy, r in self.friction_zones:
            if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                return self.FRICTION_ZONE_FACTOR
        return 1.0

    # ============================================================
    # Modulo E: Obstaculos vivos
    # ============================================================
    # Parametros de moldeo (E2)
    MOLD_MASS_THRESHOLD = 8.0    # masa minima del organismo para moldear
    MOLD_RADIUS = 18.0           # proximidad al extremo para que cuente
    MOLD_RATE = 2.0              # px de deformacion por contacto
    MAX_DEFORM = 60.0            # deformacion maxima acumulada
    MIN_MOLD_SPEED = 8.0         # velocidad minima del organismo para empujar

    def step_obstacles(self):
        """E1: avanza los obstaculos dinamicos un step y los rebota en paredes.

        Solo afecta obstaculos con kind in {dynamic, living}. Reindexa el
        spatial hash de los modificados.
        """
        if not self.obstaculos:
            return
        margin = 30.0
        any_moved = False
        for obs in self.obstaculos:
            if not obs.is_dynamic:
                continue
            # Tentativa de movimiento
            new_dx = obs.drift_x + obs.vx
            new_dy = obs.drift_y + obs.vy
            new_p1 = (obs.p1_orig[0] + new_dx + obs.p1_deform[0],
                      obs.p1_orig[1] + new_dy + obs.p1_deform[1])
            new_p2 = (obs.p2_orig[0] + new_dx + obs.p2_deform[0],
                      obs.p2_orig[1] + new_dy + obs.p2_deform[1])
            min_x = min(new_p1[0], new_p2[0])
            max_x = max(new_p1[0], new_p2[0])
            min_y = min(new_p1[1], new_p2[1])
            max_y = max(new_p1[1], new_p2[1])
            if min_x < margin or max_x > self.WORLD_W - margin:
                obs.vx = -obs.vx
                continue  # no aplicar movimiento este step
            if min_y < margin or max_y > self.WORLD_H - margin:
                obs.vy = -obs.vy
                continue
            obs.drift_x = new_dx
            obs.drift_y = new_dy
            try:
                obs.segment.unsafe_set_endpoints(obs.p1, obs.p2)
                any_moved = True
            except Exception:
                pass
        if any_moved:
            try:
                self.space.reindex_static()
            except Exception:
                pass

    def try_mold_obstacles(self, organism):
        """E2: si `organism` tiene masa suficiente y se mueve, deforma los
        extremos de obstaculos plasticos cercanos en la direccion del empuje.

        Comportamiento emergente: depredadores grandes pueden curvar obstaculos
        para acorralar; presas grandes pueden cerrar refugios.
        """
        bodies = getattr(organism, "bodies", None)
        morph = getattr(organism, "morphome", None)
        if not bodies or not morph or not morph.segments:
            return
        total_mass = sum(s.mass for s in morph.segments)
        if total_mass < self.MOLD_MASS_THRESHOLD:
            return
        # Velocidad del segmento raiz como direccion de empuje
        v = bodies[0].velocity
        speed = math.hypot(v.x, v.y)
        if speed < self.MIN_MOLD_SPEED:
            return
        nx = v.x / speed
        ny = v.y / speed

        any_deformed = False
        radius_sq = self.MOLD_RADIUS * self.MOLD_RADIUS
        for obs in self.obstaculos:
            if not obs.is_plastic:
                continue
            for which in (0, 1):
                ep = obs.p1 if which == 0 else obs.p2
                # Algun segmento del organismo en rango del extremo?
                close = False
                for body in bodies:
                    bx, by = body.position.x, body.position.y
                    if (bx - ep[0]) ** 2 + (by - ep[1]) ** 2 < radius_sq:
                        close = True
                        break
                if not close:
                    continue
                # Acumular deformacion en direccion del empuje
                cur = obs.p1_deform if which == 0 else obs.p2_deform
                ndx = cur[0] + nx * self.MOLD_RATE
                ndy = cur[1] + ny * self.MOLD_RATE
                mag = math.hypot(ndx, ndy)
                if mag > self.MAX_DEFORM:
                    scale = self.MAX_DEFORM / mag
                    ndx *= scale
                    ndy *= scale
                if which == 0:
                    obs.p1_deform = (ndx, ndy)
                else:
                    obs.p2_deform = (ndx, ndy)
                any_deformed = True
        if any_deformed:
            for obs in self.obstaculos:
                if obs.is_plastic:
                    try:
                        obs.segment.unsafe_set_endpoints(obs.p1, obs.p2)
                    except Exception:
                        pass
            try:
                self.space.reindex_static()
            except Exception:
                pass

    # ============================================================
    # F1.5-C: Cadena alimenticia entre organismos
    # ============================================================
    def detect_predation(self, mass_ratio: float = 4.0,
                         eat_radius: float = 18.0,
                         contact_threshold: int = 3,
                         cooldown_steps: int = 100):
        """C2: Cadena alimenticia biologicamente realista.

        Cuatro reglas que coexisten para que la depredacion sea rara, costosa
        y haya espacio para la coexistencia con presas:

          1. Ratio de masa 4x : A solo puede cazar a B si masa_A >= 4 * masa_B
             En la naturaleza un depredador suele pesar 4-8 veces su presa.
          2. Contacto sostenido: 3 contactos consecutivos a <= eat_radius (18px)
             Si la presa escapa (sale del rango), el contador se resetea.
             Simula 'atrapar y sujetar', no proximidad casual.
          3. Cooldown de saciedad: tras una caza exitosa, 100 steps sin cazar.
             Como un leon que descansa tras comer.
          4. Recompensa proporcional: food_eaten_victima + masa_victima * 0.3
             Cazar algo grande y bien alimentado vale mas que cazar algo debil.

        Esta funcion suele invocarse cada N steps (no cada step) — el contador
        de contactos cuenta llamadas consecutivas, no steps fisicos absolutos.
        """
        victims: list = []
        n = len(self.organisms)
        if n < 2:
            return victims

        # Pre-calcular masa total de cada organismo
        masses = []
        for o in self.organisms:
            try:
                m = sum(s.mass for s in o.morphome.segments)
            except Exception:
                m = 1.0
            masses.append(m)

        eaten = set()
        eat_radius_sq = eat_radius * eat_radius
        for i in range(n):
            if i in eaten:
                continue
            o_i = self.organisms[i]
            # Regla 3: cooldown de saciedad
            if getattr(o_i, "_predation_cooldown", 0) > 0:
                continue
            if not getattr(o_i, "bodies", None):
                continue
            # Asegurar dict de seguimiento
            pending = getattr(o_i, "_pending_contact", None)
            if pending is None:
                pending = {}
                o_i._pending_contact = pending

            for j in range(n):
                if i == j or j in eaten:
                    continue
                o_j = self.organisms[j]
                if not getattr(o_j, "bodies", None):
                    continue
                # Regla 1: ratio de masa
                if masses[i] < masses[j] * mass_ratio:
                    continue
                d2 = (o_i.x - o_j.x) ** 2 + (o_i.y - o_j.y) ** 2
                j_id = id(o_j)
                if d2 < eat_radius_sq:
                    # Regla 2: contacto sostenido (incrementar contador)
                    cnt = pending.get(j_id, 0) + 1
                    pending[j_id] = cnt
                    if cnt >= contact_threshold:
                        # Caza exitosa
                        # Regla 4: recompensa proporcional
                        prey_food = float(getattr(o_j, "_food_count", 0))
                        prey_mass = masses[j]
                        reward = prey_food + prey_mass * 0.3
                        o_i._lifetime_reward += reward
                        o_i.reward = reward
                        o_i._eaten_others = getattr(o_i, "_eaten_others", 0) + 1
                        # Regla 3: activar cooldown y limpiar tracking
                        o_i._predation_cooldown = cooldown_steps
                        o_i._pending_contact = {}
                        o_j._was_eaten = True
                        eaten.add(j)
                        victims.append(o_j)
                        break  # i ya cazo en este ciclo
                else:
                    # Fuera de rango: resetear contador para esta presa
                    if j_id in pending:
                        del pending[j_id]
        return victims

    # ------------------------------------------------------------------ organisms
    def add_organism(self, organism):
        self.organisms.append(organism)

    def remove_organism(self, organism):
        if organism in self.organisms:
            self.organisms.remove(organism)
        organism.destroy()

    # ------------------------------------------------------------------ step
    def step_physics(self, dt: float, substeps: int = 2):
        sub_dt = dt / substeps
        for _ in range(substeps):
            self.space.step(sub_dt)
        # F1.5-B: decay global de feromonas en cada step
        self.step_pheromones()
        # E1: drift de obstaculos dinamicos / vivos
        self.step_obstacles()
        # E2: intentar moldear obstaculos plasticos por organismos
        # con masa y velocidad suficientes (cada step para suavidad)
        for o in self.organisms:
            try:
                self.try_mold_obstacles(o)
            except Exception:
                pass
        # C2: decrementar cooldowns de depredacion
        for o in self.organisms:
            cd = getattr(o, "_predation_cooldown", 0)
            if cd > 0:
                o._predation_cooldown = cd - 1

    def random_spawn(self, near: tuple[float, float] | None = None,
                     radius: float = 150.0):
        """Spawn aleatorio. Si se da `near`, se mantiene cerca (C4: niching)."""
        margin = 60.0
        if near is not None:
            cx, cy = near
            for _ in range(20):
                x = cx + self.rng.uniform(-radius, radius)
                y = cy + self.rng.uniform(-radius, radius)
                x = max(margin, min(self.WORLD_W - margin, x))
                y = max(margin, min(self.WORLD_H - margin, y))
                if x != cx or y != cy:  # posicion valida encontrada
                    return (x, y)
        return (self.rng.uniform(margin, self.WORLD_W - margin),
                self.rng.uniform(margin, self.WORLD_H - margin))

    def quadrant_spawn(self, quadrant: int):
        """C4: spawn dentro del cuadrante 0..3 (TL, TR, BL, BR)."""
        margin = 60.0
        half_w = self.WORLD_W / 2
        half_h = self.WORLD_H / 2
        if quadrant == 0:    # TL
            x = self.rng.uniform(margin, half_w)
            y = self.rng.uniform(margin, half_h)
        elif quadrant == 1:  # TR
            x = self.rng.uniform(half_w, self.WORLD_W - margin)
            y = self.rng.uniform(margin, half_h)
        elif quadrant == 2:  # BL
            x = self.rng.uniform(margin, half_w)
            y = self.rng.uniform(half_h, self.WORLD_H - margin)
        else:                # BR
            x = self.rng.uniform(half_w, self.WORLD_W - margin)
            y = self.rng.uniform(half_h, self.WORLD_H - margin)
        return (x, y)
