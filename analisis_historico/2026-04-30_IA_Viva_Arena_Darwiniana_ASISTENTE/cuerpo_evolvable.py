#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUERPO EVOLVABLE - Constructor de cuerpos pymunk a partir de un Morphome.

Reemplaza VirtualBody. Implementa la misma interfaz minima que el cerebro espera:
  - get_sensory_input() -> np.ndarray(14,)
  - step(dt, motor_outputs: dict) -> bool   # True si comio
  - atributos: x, y, signal_output, reward
"""

from __future__ import annotations
import math
import numpy as np
import pymunk

from morfoma import Morphome, BRAIN_OUT_CHANNELS, BRAIN_IN_CHANNELS
from mundo_fisico import FOOD_REWARD


def _flatten_motor_outputs(motor_outputs: dict) -> np.ndarray:
    """Convierte el dict del cerebro a vector fijo de 11 canales."""
    out = np.zeros(BRAIN_OUT_CHANNELS, dtype=np.float64)
    out[0] = float(motor_outputs.get("left", 0.0))
    out[1] = float(motor_outputs.get("right", 0.0))
    sigs = motor_outputs.get("signals", [0.0] * 8)
    for i in range(8):
        out[2 + i] = float(sigs[i]) if i < len(sigs) else 0.0
    out[10] = float(motor_outputs.get("metabolism", 0.0))
    return out


class CuerpoEvolvable:
    """Cuerpo fisico construido desde un Morphome. Vive en un mundo pymunk."""

    # Categorias de colision
    CAT_BODY = 0b0001
    CAT_FOOD = 0b0010
    CAT_WALL = 0b0100

    # F1.5-C: contador para asignar group_id unico por organismo. Segmentos
    # del mismo organismo (mismo group != 0) no colisionan entre si, pero
    # SI colisionan con segmentos de otros organismos.
    _next_group_id = 1

    def __init__(self, morphome: Morphome, world, spawn_xy=(400.0, 400.0)):
        self.morphome = morphome
        self.world = world
        self.space: pymunk.Space = world.space

        # F1.5-C: id de grupo unico para no colisionar con uno mismo
        self._group_id = CuerpoEvolvable._next_group_id
        CuerpoEvolvable._next_group_id += 1

        # Estado expuesto al cerebro
        self._x = float(spawn_xy[0])
        self._y = float(spawn_xy[1])
        self._angle = 0.0
        self.signal_output = np.zeros(8)
        self.reward = 0.0
        self._lifetime_reward = 0.0
        self._steps_alive = 0
        self._food_count = 0
        self._toxic_count = 0       # A3: contador de tóxicos comidos
        self._predator_hits = 0     # C1: golpes de depredador
        self._dist_prev = float("inf")
        self._approach_bonus = 0.0  # acumulado de bonus por acercarse a comida
        self._eaten_others = 0      # F1.5-C: cuantos otros organismos se comio
        self._was_eaten = False     # F1.5-C: marca si fue comido (no fitness)
        # C2: estado de cadena alimenticia refinada
        self._predation_cooldown = 0   # steps sin poder cazar (saciedad)
        self._pending_contact: dict[int, int] = {}  # id(presa) -> contactos consecutivos

        # F2: back-reference al cerebro (asignada por acoplar_cerebro_a_mundo)
        self.brain = None
        # F2: cache de neuron_id -> celula para acceso O(1) a firing_rate (lazy)
        self._neuron_cache: dict[int, object] | None = None
        self._neuron_cache_brain_step = -1

        # Cuerpos pymunk creados
        self.bodies: list[pymunk.Body] = []
        self.shapes: list[pymunk.Shape] = []
        self.constraints: list[pymunk.Constraint] = []
        self._motor_constraints: list[pymunk.SimpleMotor | None] = []  # uno por joint, None si no tiene motor
        self._joint_pivots: list[pymunk.PivotJoint] = []  # un pivote por joint para mantener union

        self._build_from_morphome(spawn_xy)
        self._register_collision()

    # ------------------------------------------------------------------ build
    def _build_from_morphome(self, spawn_xy):
        morph = self.morphome
        # Crear todos los segmentos
        seg_bodies: list[pymunk.Body] = []
        for i, seg in enumerate(morph.segments):
            mass = max(0.05, seg.mass)
            if seg.shape == "box":
                moment = pymunk.moment_for_box(mass, (seg.size * 2, seg.size * 2))
            else:
                moment = pymunk.moment_for_circle(mass, 0, seg.size)
            body = pymunk.Body(mass, moment)
            body.position = (spawn_xy[0] + i * 0.5, spawn_xy[1] + i * 0.5)
            if seg.shape == "box":
                shape = pymunk.Poly.create_box(body, (seg.size * 2, seg.size * 2))
            else:
                shape = pymunk.Circle(body, seg.size)
            shape.friction = seg.friction
            shape.elasticity = 0.1
            shape.density = max(0.01, mass / max(1.0, seg.size ** 2))
            # F1.5-C: group_id unico — segmentos del mismo organismo no chocan
            # entre si, pero SI chocan con segmentos de otros organismos.
            shape.filter = pymunk.ShapeFilter(
                group=self._group_id,
                categories=self.CAT_BODY,
                mask=pymunk.ShapeFilter.ALL_MASKS(),
            )
            shape.color_rgb = seg.color
            shape._owner = self  # backref para callbacks
            seg_bodies.append(body)
            self.space.add(body, shape)
            self.bodies.append(body)
            self.shapes.append(shape)

        # Crear articulaciones
        for ji, j in enumerate(morph.joints):
            try:
                body_a = seg_bodies[j.seg_a]
                body_b = seg_bodies[j.seg_b]
            except IndexError:
                self._motor_constraints.append(None)
                self._joint_pivots.append(None)
                continue
            # Posicionar body_b cerca del anchor para no nacer destruido
            world_anchor = body_a.local_to_world(j.anchor_a)
            body_b.position = (world_anchor[0] - j.anchor_b[0], world_anchor[1] - j.anchor_b[1])
            pivot = pymunk.PivotJoint(body_a, body_b, j.anchor_a, j.anchor_b)
            pivot.collide_bodies = False
            pivot.error_bias = (1.0 - 0.15) ** 60  # rigido
            self.space.add(pivot)
            self.constraints.append(pivot)
            self._joint_pivots.append(pivot)

            # B3: Si rest_angle != 0, anadir resorte rotacional. El motor luchara
            # contra el resorte → comportamiento orgánico (tension/relajacion).
            if abs(j.rest_angle) > 0.01:
                spring = pymunk.DampedRotarySpring(
                    body_a, body_b,
                    rest_angle=j.rest_angle,
                    stiffness=2000.0,
                    damping=200.0,
                )
                self.space.add(spring)
                self.constraints.append(spring)

            if j.has_motor:
                motor = pymunk.SimpleMotor(body_a, body_b, 0.0)
                motor.max_force = j.max_torque
                self.space.add(motor)
                self.constraints.append(motor)
                self._motor_constraints.append(motor)
            else:
                self._motor_constraints.append(None)

    def _register_collision(self):
        # Listener para colisiones cuerpo<->comida
        for shape in self.shapes:
            shape.collision_type = 1  # body
        # El handler global se registra en MundoFisico

    # ------------------------------------------------------------------ destroy
    def destroy(self):
        for c in self.constraints:
            if c in self.space.constraints:
                self.space.remove(c)
        for s in self.shapes:
            if s in self.space.shapes:
                self.space.remove(s)
        for b in self.bodies:
            if b in self.space.bodies:
                self.space.remove(b)
        self.bodies.clear()
        self.shapes.clear()
        self.constraints.clear()
        self._motor_constraints.clear()
        self._joint_pivots.clear()

    # ------------------------------------------------------------------ pose
    @property
    def pose(self):
        """Posicion y angulo del segmento raiz (segmento 0)."""
        if not self.bodies:
            return self._x, self._y, self._angle
        b = self.bodies[0]
        return b.position.x, b.position.y, b.angle

    # x/y/angle getters: leen del cuerpo raiz pymunk si existe
    @property
    def x(self):
        if self.bodies:
            return self.bodies[0].position.x
        return self._x

    @x.setter
    def x(self, val):
        val = float(val)
        if self.bodies:
            dx = val - self.bodies[0].position.x
            for b in self.bodies:
                b.position = (b.position.x + dx, b.position.y)
        self._x = val

    @property
    def y(self):
        if self.bodies:
            return self.bodies[0].position.y
        return self._y

    @y.setter
    def y(self, val):
        val = float(val)
        if self.bodies:
            dy = val - self.bodies[0].position.y
            for b in self.bodies:
                b.position = (b.position.x, b.position.y + dy)
        self._y = val

    @property
    def angle(self):
        if self.bodies:
            return self.bodies[0].angle
        return self._angle

    @angle.setter
    def angle(self, val):
        val = float(val)
        if self.bodies:
            self.bodies[0].angle = val
        self._angle = val

    # ------------------------------------------------------------------ sensory
    def get_sensory_input(self) -> np.ndarray:
        """14 canales: vision/oido externo + interoception + sensores fisicos via genome."""
        if not self.bodies:
            return np.zeros(BRAIN_IN_CHANNELS)

        root = self.bodies[0]
        self.x, self.y = root.position.x, root.position.y

        # ---- canales globales (siempre presentes, equivalente al body anterior) ----
        # Comida mas cercana
        food_pos, _ = self.world.nearest_food(self.x, self.y)
        if food_pos is not None:
            dx = food_pos[0] - self.x
            dy = food_pos[1] - self.y
            dist = math.hypot(dx, dy)
            ang_to = math.atan2(dy, dx)
            rel = (ang_to - root.angle + math.pi) % (2 * math.pi) - math.pi
            s1 = 1.0 / (1.0 + dist / 200.0)
            s2 = rel / math.pi
        else:
            s1 = 0.0
            s2 = 0.0

        # Otros organismos
        s3 = 0.0
        s6_9 = [0.0] * 4
        others = [b for b in self.world.organisms if b is not self]
        if others:
            ds = [math.hypot(self.x - o.x, self.y - o.y) for o in others]
            idx = int(np.argmin(ds))
            s3 = 1.0 / (1.0 + ds[idx] / 200.0)
            s6_9 = list(others[idx].signal_output[:4])

        # Pared mas cercana (sustituye al "tool" del mundo viejo)
        s4 = self._wall_proximity()
        s5 = 0.5  # firma neutra

        # Interoception (estado interno)
        state = getattr(self, "_internal_state", None)
        if state is not None:
            s10 = float(state.get("emotional", 0.5))
            s11 = float(state.get("ram", 0.0))
            s12 = float(state.get("cpu", 0.0))
            s13 = float(state.get("buffer", 0.0))
            s14 = float(state.get("network", 0.0))
        else:
            s10 = s11 = s12 = s13 = s14 = 0.0

        retina = np.array([s1, s2, s3, s4, s5,
                           s6_9[0], s6_9[1], s6_9[2], s6_9[3],
                           s10, s11, s12, s13, s14], dtype=np.float64)

        # ---- sensores fisicos via genome ----
        for sm in self.morphome.sensor_maps:
            try:
                val = self._read_physics_sensor(sm.sensor_type, sm.target_idx)
                if val is None:
                    continue
                ch = sm.brain_channel
                if 0 <= ch < BRAIN_IN_CHANNELS:
                    # Mezcla aditiva con clipping
                    retina[ch] = float(np.clip(retina[ch] + val * sm.gain, -1.5, 1.5))
            except Exception:
                continue

        return retina

    def _read_physics_sensor(self, sensor_type: str, target_idx: int):
        if sensor_type == "angle":
            if 0 <= target_idx < len(self._joint_pivots):
                pivot = self._joint_pivots[target_idx]
                if pivot is None:
                    return None
                # angulo relativo entre cuerpos
                a = pivot.a.angle
                b = pivot.b.angle
                return math.sin(b - a)  # normalizado [-1, 1]
        elif sensor_type == "angvel":
            if 0 <= target_idx < len(self._joint_pivots):
                pivot = self._joint_pivots[target_idx]
                if pivot is None:
                    return None
                w = pivot.b.angular_velocity - pivot.a.angular_velocity
                return float(np.tanh(w * 0.3))
        elif sensor_type == "velocity":
            if 0 <= target_idx < len(self.bodies):
                v = self.bodies[target_idx].velocity.length
                return float(np.tanh(v * 0.02))
        elif sensor_type == "contact":
            # B1: Sensor de tacto. Devuelve 1.0 si el segmento toca cualquier
            # otra forma (pared, comida, otro cuerpo). Usa shape_query con un
            # margen pequeño para detectar contactos inminentes.
            if 0 <= target_idx < len(self.shapes):
                shape = self.shapes[target_idx]
                try:
                    queries = self.space.shape_query(shape)
                    return 1.0 if len(queries) > 0 else 0.0
                except Exception:
                    return 0.0
        elif sensor_type == "pheromone":
            # F1.5-B: lectura local de feromona. target_idx ignorado (lectura
            # en el centroide del organismo). Devuelve [0, 1].
            try:
                return float(self.world.sense_pheromone(self.x, self.y))
            except AttributeError:
                return 0.0
        elif sensor_type == "obstacle_endpoint":
            # E4: distancia (target_idx=0) o angulo (target_idx=1) al extremo
            # plastico mas cercano. Permite que el genoma evolucione el
            # comportamiento de moldear obstaculos.
            obstaculos = getattr(self.world, "obstaculos", [])
            best_d2 = float("inf")
            best_ep = None
            for obs in obstaculos:
                if not getattr(obs, "is_plastic", False):
                    continue
                for ep in (obs.p1, obs.p2):
                    d2 = (ep[0] - self.x) ** 2 + (ep[1] - self.y) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        best_ep = ep
            if best_ep is None:
                return 0.0
            if target_idx == 0:
                d = math.sqrt(best_d2)
                return float(1.0 / (1.0 + d / 200.0))
            elif target_idx == 1:
                root = self.bodies[0]
                ang = math.atan2(best_ep[1] - self.y, best_ep[0] - self.x)
                rel = (ang - root.angle + math.pi) % (2 * math.pi) - math.pi
                return float(rel / math.pi)
            return 0.0
        elif sensor_type == "nearby_organism":
            # F1.5-C: distancia inversa al organismo mas cercano (excluyendo
            # self). target_idx 0 = distancia, 1 = angulo relativo, 2 = tamano.
            others = [o for o in self.world.organisms if o is not self]
            if not others:
                return 0.0
            dx_dy = [(o.x - self.x, o.y - self.y) for o in others]
            ds = [dx * dx + dy * dy for dx, dy in dx_dy]
            i = int(np.argmin(ds))
            d = math.sqrt(ds[i])
            if target_idx == 0:
                return float(1.0 / (1.0 + d / 200.0))
            elif target_idx == 1:
                root = self.bodies[0]
                ang = math.atan2(dx_dy[i][1], dx_dy[i][0])
                rel = (ang - root.angle + math.pi) % (2 * math.pi) - math.pi
                return float(rel / math.pi)
            elif target_idx == 2:
                # tamano = num segmentos / 12 (normalizado)
                other = others[i]
                if hasattr(other, "morphome") and other.morphome:
                    return float(min(1.0, len(other.morphome.segments) / 12.0))
                return 0.0
        return None

    def _wall_proximity(self) -> float:
        W = self.world.WORLD_W
        H = self.world.WORLD_H
        d = min(self.x, self.y, W - self.x, H - self.y)
        return 1.0 / (1.0 + d / 100.0)

    # ------------------------------------------------------------------ step
    def step(self, dt: float, motor_outputs: dict) -> bool:  # noqa: ARG002
        """Aplica salidas motoras al fisico, lee colisiones, devuelve True si comio."""
        self._steps_alive += 1
        self.reward = 0.0
        if not self.bodies:
            return False

        out_vec = _flatten_motor_outputs(motor_outputs)
        self.signal_output = out_vec[2:10].copy()  # canales 2..9 son signals[0..7]

        # F1.5-D: factor de friccion zonal — atenua motores si esta en zona pesada
        try:
            f_factor = self.world.friction_factor(self.x, self.y)
        except AttributeError:
            f_factor = 1.0

        # Aplicar motores via genome — acumulamos contribuciones por joint
        # para que reflexos y canales espontaneos puedan sumar al control normal
        # en vez de pisarlo.
        motor_signals: dict[int, float] = {}  # joint_idx -> rate acumulado

        for mm in self.morphome.motor_maps:
            if 0 <= mm.joint_idx < len(self._motor_constraints):
                motor = self._motor_constraints[mm.joint_idx]
                if motor is None:
                    continue
                ch = mm.brain_channel
                if 0 <= ch < BRAIN_OUT_CHANNELS:
                    val = out_vec[ch]
                    centered = (val - 0.3) * 3.0
                    motor_signals[mm.joint_idx] = motor_signals.get(mm.joint_idx, 0.0) \
                                                   + centered * mm.gain

        # F1: REFLEXOS — sensor->motor directo, sin cerebro de por medio.
        # Reaccion sub-frame: si la senal del sensor supera el umbral, aplicamos
        # torque proporcional. Util para evasion de depredadores, contacto, etc.
        for rx in getattr(self.morphome, "reflexes", []):
            if not (0 <= rx.motor_joint_idx < len(self._motor_constraints)):
                continue
            if self._motor_constraints[rx.motor_joint_idx] is None:
                continue
            try:
                sensor_val = self._read_physics_sensor(rx.sensor_type, rx.sensor_target)
            except Exception:
                sensor_val = None
            if sensor_val is None:
                continue
            mag = abs(sensor_val)
            if mag <= rx.threshold:
                continue
            excess = (mag - rx.threshold) * (1.0 if sensor_val >= 0 else -1.0)
            motor_signals[rx.motor_joint_idx] = motor_signals.get(rx.motor_joint_idx, 0.0) \
                                                  + excess * rx.gain

        # F2: CANALES MOTORES ESPONTANEOS — neuronas internas como motores.
        # Si la neurona referenciada por neuron_id existe en el cerebro, usamos
        # su firing_rate como senal motora extra.
        sp_motors = getattr(self.morphome, "spontaneous_motors", [])
        if sp_motors and self.brain is not None:
            cells = getattr(self.brain, "cells", None) or []
            # Refrescar cache de id->cell si cambio el cerebro
            brain_step = getattr(self.brain, "step_count", -1)
            if self._neuron_cache is None or brain_step != self._neuron_cache_brain_step:
                self._neuron_cache = {getattr(c, "id", -1): c for c in cells}
                self._neuron_cache_brain_step = brain_step
            for sp in sp_motors:
                if not (0 <= sp.joint_idx < len(self._motor_constraints)):
                    continue
                if self._motor_constraints[sp.joint_idx] is None:
                    continue
                cell = self._neuron_cache.get(sp.neuron_id)
                if cell is None:
                    continue   # neurona no existe en este cerebro
                fr = float(getattr(cell, "firing_rate", 0.0))
                # Normalizar firing_rate (~0..50Hz tipico) a [0,1] y centrar
                norm = min(1.0, fr / 30.0)
                centered = (norm - 0.3) * 3.0
                motor_signals[sp.joint_idx] = motor_signals.get(sp.joint_idx, 0.0) \
                                                + centered * sp.gain

        # Aplicar las senales acumuladas a los motores
        for joint_idx, rate in motor_signals.items():
            motor = self._motor_constraints[joint_idx]
            if motor is None:
                continue
            motor.rate = float(np.clip(rate * f_factor, -8.0, 8.0))

        # Empuje base si no tiene motores aun (para no morir totalmente quieto)
        # Forma "pre-evolucion": empuje proporcional a left+right, giro a la diferencia
        if not self.morphome.motor_maps:
            root = self.bodies[0]
            forward = (out_vec[0] + out_vec[1]) * 250.0 * f_factor
            turn    = (out_vec[1] - out_vec[0]) * 0.25 * f_factor
            root.apply_force_at_local_point((forward, 0), (0, 0))
            root.angular_velocity += turn
            # Friccion ambiental simulada
            root.velocity = (root.velocity.x * 0.95, root.velocity.y * 0.95)

        # La actualizacion del Space la hace el mundo
        # Bonus por acercarse a la comida mas cercana
        food_pos, _ = self.world.nearest_food(self.x, self.y)
        if food_pos is not None:
            dist_now = math.hypot(food_pos[0] - self.x, food_pos[1] - self.y)
            if dist_now < self._dist_prev:
                self._approach_bonus += 0.001
            self._dist_prev = dist_now
        else:
            self._dist_prev = float("inf")

        # Detectar colision con comida (consulta directa)
        kind_eaten = self._consume_nearby_food()
        if kind_eaten is not None:
            reward = FOOD_REWARD.get(kind_eaten, 1.0)
            self.reward = reward
            if reward > 0:
                self._food_count += 1
                # F1.5-B: depositar feromona positiva al comer comida buena
                try:
                    self.world.deposit_pheromone(self.x, self.y)
                except AttributeError:
                    pass
            else:
                self._toxic_count += 1
            self._lifetime_reward += reward
            self._dist_prev = float("inf")  # resetear al comer
            return True
        return False

    def _consume_nearby_food(self):
        """Devuelve el `kind` (str) de la comida ingerida, o None.

        A3: cada tipo (normal/rich/toxic) da reward distinto.
        Busca solo comida viva cerca del centroide del cuerpo para reducir comparaciones.
        """
        cx, cy = self.x, self.y
        eat_range = 80.0  # radio de busqueda inicial para filtrar candidatos
        candidates = [f for f in self.world.food_items
                      if f.alive and abs(f.x - cx) < eat_range and abs(f.y - cy) < eat_range]
        for shape in self.shapes:
            seg_radius = getattr(shape, "radius", 6.0)
            sx, sy = shape.body.position.x, shape.body.position.y
            for food in candidates:
                dx = sx - food.x
                dy = sy - food.y
                if (dx * dx + dy * dy) < (food.radius + seg_radius + 4.0) ** 2:
                    food.alive = False
                    return food.kind
        return None

    # ------------------------------------------------------------------ stats
    def fitness(self) -> float:
        """Score de seleccion.

        B2: costo energético proporcional a masa total y tiempo de vida.
        A3: bonus por comida rica, malus por toxicos.
        C1: malus por contactos con depredador.

        Diseño: el organismo "default" (1 segmento, no come) debe quedar cerca
        de 0, no muy negativo. La selección distingue los que comieron de los
        que no, sin enterrar el rango con costos fijos.
        """
        score = self._lifetime_reward

        # B2: costo energético — solo penaliza el EXCESO de masa (delta sobre baseline 1.5).
        # Cuerpos pequeños no pagan; cuerpos grandes pagan proporcional a tiempo.
        if self.morphome and self.morphome.segments:
            total_mass = sum(s.mass for s in self.morphome.segments)
            extra_mass = max(0.0, total_mass - 1.5)
            energy_cost = extra_mass * 0.00015 * self._steps_alive
        else:
            energy_cost = 0.0

        # C1: malus por depredador — penalizacion fuerte por hit pero acotada
        # al 50% del score + 5. Antes era 0.5/hit, irrelevante para organismos
        # exitosos. Ahora 2.0/hit crea presion selectiva real para velocidad
        # y morfologias de escape.
        predator_penalty = min(self._predator_hits * 2.0, score * 0.5 + 5.0)

        # Bonus por acercarse a comida — premia movimiento direccional.
        # Maximo ~3.0 si siempre se acerca (3000 steps * 0.001).
        # Crea presion de seleccion para motor maps que mejoren la navegacion.
        approach_bonus = min(self._approach_bonus, 3.0)

        return float(score - energy_cost - predator_penalty + approach_bonus)
