#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MORFOMA - Genoma morfologico evolvable.

Define el cuerpo como un grafo de segmentos + articulaciones + mapeo cerebro-cuerpo.
NADA del cuerpo esta predefinido: todo emerge de mutaciones acumuladas.

El cerebro (semilla_cerebro.py) no se toca. Solo cambia como se interpretan sus
14 canales sensoriales (entrada) y 11 canales motores (salida = left/right/signals[8]/metabolism).
"""

from __future__ import annotations
import copy
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

# 11 canales de salida del cerebro: [left, right, sig0..sig7, metabolism]
BRAIN_OUT_CHANNELS = 11
# 14 canales de entrada del cerebro
BRAIN_IN_CHANNELS = 14


@dataclass
class SegmentGene:
    """Una pieza fisica del cuerpo."""
    shape: str = "circle"      # 'circle' o 'box'
    size: float = 8.0          # radio (circle) o semi-lado (box)
    mass: float = 1.0
    friction: float = 0.6
    color: Tuple[int, int, int] = (180, 180, 220)


@dataclass
class JointGene:
    """Una articulacion entre dos segmentos."""
    seg_a: int                 # indice en segments[]
    seg_b: int                 # indice en segments[]
    anchor_a: Tuple[float, float] = (0.0, 0.0)   # punto de anclaje en seg_a (local)
    anchor_b: Tuple[float, float] = (0.0, 0.0)   # punto de anclaje en seg_b (local)
    has_motor: bool = False
    max_torque: float = 8000.0
    rest_angle: float = 0.0


@dataclass
class MotorMap:
    """Conecta un canal de salida del cerebro a una articulacion."""
    brain_channel: int         # 0..BRAIN_OUT_CHANNELS-1
    joint_idx: int
    gain: float = 1.0          # signo y magnitud


@dataclass
class SensorMap:
    """Conecta una senal del cuerpo a un canal sensorial del cerebro.

    Tipos:
      'angle', 'angvel'         -> joint_idx
      'contact', 'velocity'     -> segment_idx
      'pheromone'               -> target ignorado (sensa en centroide)
      'nearby_organism'         -> target_idx en {0=dist, 1=angulo, 2=tamano}
    """
    sensor_type: str
    target_idx: int
    brain_channel: int         # 0..BRAIN_IN_CHANNELS-1
    gain: float = 1.0


@dataclass
class ReflexGene:
    """F1: Arco reflejo directo sensor->motor sin pasar por el cerebro.

    Si la senal del sensor supera el umbral, aplica torque proporcional al
    motor de la articulacion indicada. Permite reacciones sub-100ms que el
    cerebro (que tarda varios steps en integrar) no puede generar.
    """
    sensor_type: str           # 'contact', 'angle', 'angvel', 'velocity',
                                # 'pheromone', 'nearby_organism', 'obstacle_endpoint'
    sensor_target: int         # mismo significado que SensorMap.target_idx
    motor_joint_idx: int       # articulacion a controlar (debe tener motor)
    threshold: float = 0.4     # nivel minimo de senal para activar
    gain: float = 1.0          # signo y magnitud del torque (-2..+2)


@dataclass
class SpontaneousMotorChannel:
    """F2: Canal motor extra mapeado a una neurona interna por ID.

    A diferencia de MotorMap (que usa los 11 canales de salida estandar del
    cerebro), este lee el firing_rate de una neurona individual identificada
    por su ID interno y lo aplica como senal motora. Permite que neuronas
    'internas' se especialicen como motores de facto si la evolucion lo favorece.
    """
    neuron_id: int             # ID de la neurona en cerebro.cells
    joint_idx: int             # articulacion motorizada destino
    gain: float = 1.0          # signo y magnitud (-2..+2)


@dataclass
class Morphome:
    """Genoma morfologico completo."""
    segments: List[SegmentGene] = field(default_factory=list)
    joints: List[JointGene] = field(default_factory=list)
    motor_maps: List[MotorMap] = field(default_factory=list)
    sensor_maps: List[SensorMap] = field(default_factory=list)
    reflexes: List[ReflexGene] = field(default_factory=list)
    spontaneous_motors: List[SpontaneousMotorChannel] = field(default_factory=list)
    generation: int = 0
    lineage: str = "raiz"

    # ---------------------------------------------------------------- factories
    @staticmethod
    def semilla_minima(lineage: str = "raiz") -> "Morphome":
        """Organismo minimo: 1 torso, sin articulaciones, sin motores ni sensores fisicos.

        El cerebro recibira solo las senales 'globales' (vision, oido, interoception).
        Las mutaciones agregaran segmentos y motores progresivamente.
        """
        return Morphome(
            segments=[SegmentGene(shape="circle", size=10.0, mass=2.0)],
            joints=[],
            motor_maps=[],
            sensor_maps=[],
            generation=0,
            lineage=lineage,
        )

    # ---------------------------------------------------------------- copy
    def copy(self) -> "Morphome":
        return copy.deepcopy(self)

    def __post_init__(self):
        # Retrocompat: morfomas viejos pueden no tener estos campos al deserializar
        if not hasattr(self, "reflexes") or self.reflexes is None:
            self.reflexes = []
        if not hasattr(self, "spontaneous_motors") or self.spontaneous_motors is None:
            self.spontaneous_motors = []

    # ---------------------------------------------------------------- mutate
    def mutate(self, rate: float = 1.0) -> "Morphome":
        """Devuelve una copia mutada. rate escala la probabilidad global."""
        m = self.copy()
        m.generation += 1

        # Catalogo de mutaciones (probabilidades base)
        ops = [
            ("add_segment",    0.18),
            ("remove_segment", 0.06),
            ("tweak_segment",  0.20),
            ("tweak_joint",    0.15),
            ("add_motor_map",  0.20),
            ("remove_motor_map", 0.05),
            ("tweak_motor_map",  0.15),
            ("add_sensor_map", 0.10),
            ("tweak_sensor_map", 0.08),
            # F1: reflexos corporales
            ("add_reflex",     0.10),
            ("remove_reflex",  0.04),
            ("tweak_reflex",   0.08),
            # F2: canales motores espontaneos
            ("add_spontaneous_motor",    0.08),
            ("remove_spontaneous_motor", 0.04),
            ("tweak_spontaneous_motor",  0.06),
            ("nada", 0.50),
        ]
        # Aplicar 2-5 mutaciones por reproduccion (mas variabilidad por generacion)
        n_muts = random.choices([2, 3, 4, 5], weights=[0.4, 0.3, 0.2, 0.1])[0]
        for _ in range(n_muts):
            op = random.choices([o for o, _ in ops], weights=[w * rate for _, w in ops])[0]
            try:
                getattr(m, f"_op_{op}")()
            except Exception:
                pass  # mutacion fallida = sin efecto

        return m

    # ---------------------------------------------------------------- ops
    def _op_nada(self):
        return

    def _op_add_segment(self):
        if len(self.segments) >= 12:
            return  # limite suave
        parent_idx = random.randrange(len(self.segments))
        parent = self.segments[parent_idx]
        new_seg = SegmentGene(
            shape=random.choice(["circle", "box"]),
            size=random.uniform(3.0, 9.0),
            mass=random.uniform(0.3, 2.0),
            friction=random.uniform(0.3, 0.9),
            color=(random.randint(100, 240), random.randint(100, 240), random.randint(100, 240)),
        )
        self.segments.append(new_seg)
        new_idx = len(self.segments) - 1
        # articulacion al padre
        d = parent.size + new_seg.size
        ang = random.uniform(0, 2 * np.pi)
        anchor_a = (np.cos(ang) * parent.size, np.sin(ang) * parent.size)
        anchor_b = (-np.cos(ang) * new_seg.size, -np.sin(ang) * new_seg.size)
        self.joints.append(JointGene(
            seg_a=parent_idx, seg_b=new_idx,
            anchor_a=anchor_a, anchor_b=anchor_b,
            has_motor=random.random() < 0.5,
            max_torque=random.uniform(4000.0, 14000.0),
            rest_angle=0.0,
        ))

    def _op_remove_segment(self):
        if len(self.segments) <= 1:
            return
        # solo se puede borrar un segmento "hoja" (sin hijos)
        children = {j.seg_a for j in self.joints} | set()  # padres usados
        # buscar hojas: un segmento que es seg_b en una sola articulacion y no es seg_a en ninguna
        appears_as_a = {j.seg_a for j in self.joints}
        leaves = [i for i in range(len(self.segments))
                  if i not in appears_as_a and any(j.seg_b == i for j in self.joints)]
        if not leaves:
            return
        victim = random.choice(leaves)
        # Quitar articulaciones que la usen
        kept_joints = []
        removed_joint_indices = set()
        for ji, j in enumerate(self.joints):
            if j.seg_a == victim or j.seg_b == victim:
                removed_joint_indices.add(ji)
            else:
                kept_joints.append(j)
        # Quitar mapas que apunten a articulaciones eliminadas
        new_joint_idx = {}
        for new_i, old_i in enumerate([i for i in range(len(self.joints)) if i not in removed_joint_indices]):
            new_joint_idx[old_i] = new_i
        self.joints = kept_joints
        self.motor_maps = [MotorMap(m.brain_channel, new_joint_idx[m.joint_idx], m.gain)
                           for m in self.motor_maps if m.joint_idx in new_joint_idx]
        # Reindex segmentos
        new_seg_idx = {}
        new_segments = []
        for i, s in enumerate(self.segments):
            if i == victim:
                continue
            new_seg_idx[i] = len(new_segments)
            new_segments.append(s)
        self.segments = new_segments
        for j in self.joints:
            j.seg_a = new_seg_idx[j.seg_a]
            j.seg_b = new_seg_idx[j.seg_b]
        # Sensor maps que apuntaban al segmento eliminado
        valid_sm = []
        for sm in self.sensor_maps:
            if sm.sensor_type in ("angle", "angvel"):
                if sm.target_idx in new_joint_idx:
                    valid_sm.append(SensorMap(sm.sensor_type, new_joint_idx[sm.target_idx],
                                              sm.brain_channel, sm.gain))
            elif sm.sensor_type in ("pheromone", "nearby_organism",
                                     "obstacle_endpoint"):
                # No dependen de indices de segments/joints
                valid_sm.append(SensorMap(sm.sensor_type, sm.target_idx,
                                          sm.brain_channel, sm.gain))
            else:
                if sm.target_idx in new_seg_idx:
                    valid_sm.append(SensorMap(sm.sensor_type, new_seg_idx[sm.target_idx],
                                              sm.brain_channel, sm.gain))
        self.sensor_maps = valid_sm

        # F1: reflexos — re-mapear o descartar segun joints/segments eliminados
        valid_rx = []
        for r in self.reflexes:
            if r.motor_joint_idx not in new_joint_idx:
                continue  # joint motor desaparecio
            new_motor = new_joint_idx[r.motor_joint_idx]
            if r.sensor_type in ("angle", "angvel"):
                if r.sensor_target in new_joint_idx:
                    valid_rx.append(ReflexGene(r.sensor_type, new_joint_idx[r.sensor_target],
                                                new_motor, r.threshold, r.gain))
            elif r.sensor_type in ("pheromone", "nearby_organism", "obstacle_endpoint"):
                valid_rx.append(ReflexGene(r.sensor_type, r.sensor_target,
                                            new_motor, r.threshold, r.gain))
            else:  # contact, velocity -> segmentos
                if r.sensor_target in new_seg_idx:
                    valid_rx.append(ReflexGene(r.sensor_type, new_seg_idx[r.sensor_target],
                                                new_motor, r.threshold, r.gain))
        self.reflexes = valid_rx

        # F2: canales espontaneos — solo re-mapear joint_idx
        valid_sp = []
        for sm in self.spontaneous_motors:
            if sm.joint_idx in new_joint_idx:
                valid_sp.append(SpontaneousMotorChannel(
                    neuron_id=sm.neuron_id,
                    joint_idx=new_joint_idx[sm.joint_idx],
                    gain=sm.gain,
                ))
        self.spontaneous_motors = valid_sp

    def _op_tweak_segment(self):
        if not self.segments:
            return
        s = random.choice(self.segments)
        s.size = float(np.clip(s.size + random.gauss(0, 1.5), 2.5, 14.0))
        s.mass = float(np.clip(s.mass + random.gauss(0, 0.3), 0.2, 4.0))

    def _op_tweak_joint(self):
        if not self.joints:
            return
        j = random.choice(self.joints)
        if random.random() < 0.3:
            j.has_motor = not j.has_motor
        j.max_torque = float(np.clip(j.max_torque + random.gauss(0, 1500.0), 1000.0, 20000.0))
        # B3: rest_angle puede mutar para crear resortes con posicion natural
        if random.random() < 0.4:
            j.rest_angle = float(np.clip(j.rest_angle + random.gauss(0, 0.4),
                                          -np.pi, np.pi))

    def _op_add_motor_map(self):
        motor_joints = [i for i, j in enumerate(self.joints) if j.has_motor]
        if not motor_joints:
            return
        # evitar duplicados exactos
        existing = {(m.brain_channel, m.joint_idx) for m in self.motor_maps}
        for _ in range(8):
            ch = random.randrange(BRAIN_OUT_CHANNELS)
            j = random.choice(motor_joints)
            if (ch, j) not in existing:
                self.motor_maps.append(MotorMap(
                    brain_channel=ch, joint_idx=j,
                    gain=random.choice([-1.0, 1.0]) * random.uniform(0.5, 1.5),
                ))
                return

    def _op_remove_motor_map(self):
        if self.motor_maps:
            self.motor_maps.pop(random.randrange(len(self.motor_maps)))

    def _op_tweak_motor_map(self):
        if not self.motor_maps:
            return
        m = random.choice(self.motor_maps)
        m.gain = float(np.clip(m.gain + random.gauss(0, 0.3), -2.0, 2.0))

    def _op_add_sensor_map(self):
        if not self.joints and not self.segments:
            return
        # B1: 'contact' = tacto. F1.5-B: 'pheromone' = comunicacion quimica.
        # F1.5-C: 'nearby_organism' = sentir a otros organismos (3 sub-canales).
        # E4: 'obstacle_endpoint' = sentir extremos de obstaculos plasticos
        #      para evolucionar comportamiento de moldeo (2 sub-canales).
        sensor_type = random.choices(
            ["contact", "angle", "angvel", "velocity",
             "pheromone", "nearby_organism", "obstacle_endpoint"],
            weights=[0.25, 0.16, 0.12, 0.09, 0.13, 0.13, 0.12]
        )[0]
        if sensor_type in ("angle", "angvel"):
            if not self.joints:
                return
            target = random.randrange(len(self.joints))
        elif sensor_type == "pheromone":
            target = 0  # ignorado, lectura en centroide
        elif sensor_type == "nearby_organism":
            target = random.randrange(3)  # 0=dist, 1=angulo, 2=tamano
        elif sensor_type == "obstacle_endpoint":
            target = random.randrange(2)  # 0=dist, 1=angulo
        else:  # velocity y contact apuntan a segmentos
            target = random.randrange(len(self.segments))
        ch = random.randrange(BRAIN_IN_CHANNELS)
        self.sensor_maps.append(SensorMap(
            sensor_type=sensor_type, target_idx=target,
            brain_channel=ch, gain=random.uniform(0.3, 1.2),
        ))

    def _op_tweak_sensor_map(self):
        if not self.sensor_maps:
            return
        s = random.choice(self.sensor_maps)
        s.gain = float(np.clip(s.gain + random.gauss(0, 0.2), 0.1, 2.0))

    # ---------------------------------------------------------------- F1: reflexos
    def _op_add_reflex(self):
        """Anade un arco reflejo sensor->motor. Requiere al menos un joint motorizado."""
        motor_joints = [i for i, j in enumerate(self.joints) if j.has_motor]
        if not motor_joints:
            return
        sensor_type = random.choices(
            ["contact", "angle", "angvel", "velocity",
             "pheromone", "nearby_organism", "obstacle_endpoint"],
            weights=[0.30, 0.14, 0.10, 0.10, 0.12, 0.14, 0.10]
        )[0]
        if sensor_type in ("angle", "angvel"):
            if not self.joints:
                return
            target = random.randrange(len(self.joints))
        elif sensor_type == "pheromone":
            target = 0
        elif sensor_type == "nearby_organism":
            target = random.randrange(3)
        elif sensor_type == "obstacle_endpoint":
            target = random.randrange(2)
        else:  # contact, velocity -> segmentos
            if not self.segments:
                return
            target = random.randrange(len(self.segments))
        self.reflexes.append(ReflexGene(
            sensor_type=sensor_type,
            sensor_target=target,
            motor_joint_idx=random.choice(motor_joints),
            threshold=random.uniform(0.2, 0.7),
            gain=random.choice([-1.0, 1.0]) * random.uniform(0.5, 1.8),
        ))

    def _op_remove_reflex(self):
        if self.reflexes:
            self.reflexes.pop(random.randrange(len(self.reflexes)))

    def _op_tweak_reflex(self):
        if not self.reflexes:
            return
        r = random.choice(self.reflexes)
        if random.random() < 0.5:
            r.threshold = float(np.clip(r.threshold + random.gauss(0, 0.1), 0.05, 0.95))
        else:
            r.gain = float(np.clip(r.gain + random.gauss(0, 0.25), -2.0, 2.0))

    # ---------------------------------------------------------------- F2: canales espontaneos
    def _op_add_spontaneous_motor(self):
        """Anade un canal motor que lee una neurona interna por ID.

        El neuron_id es un valor 'aspiracional' — si la neurona no existe en el
        cerebro al ejecutar, el canal simplemente no aporta. La presion selectiva
        favorecera IDs que sí existen Y cuya activacion correlaciona con fitness.
        """
        motor_joints = [i for i, j in enumerate(self.joints) if j.has_motor]
        if not motor_joints:
            return
        # Rango realista para IDs de neuronas (los cerebros llegan a ~3000 neurons IDs)
        neuron_id = random.randint(0, 2500)
        self.spontaneous_motors.append(SpontaneousMotorChannel(
            neuron_id=neuron_id,
            joint_idx=random.choice(motor_joints),
            gain=random.choice([-1.0, 1.0]) * random.uniform(0.4, 1.2),
        ))

    def _op_remove_spontaneous_motor(self):
        if self.spontaneous_motors:
            self.spontaneous_motors.pop(random.randrange(len(self.spontaneous_motors)))

    def _op_tweak_spontaneous_motor(self):
        if not self.spontaneous_motors:
            return
        sm = random.choice(self.spontaneous_motors)
        # Pequenos saltos en neuron_id permiten "explorar" neuronas vecinas
        if random.random() < 0.3:
            sm.neuron_id = max(0, sm.neuron_id + random.randint(-50, 50))
        else:
            sm.gain = float(np.clip(sm.gain + random.gauss(0, 0.25), -2.0, 2.0))

    # ---------------------------------------------------------------- crossover (C2)
    @staticmethod
    def crossover(padre_a: "Morphome", padre_b: "Morphome") -> "Morphome":
        """Reproduccion sexual: combina genes de dos padres.

        Estrategia: el hijo hereda el cuerpo (segments + joints) del padre A
        y el sistema de mapeo neuronal (motor_maps + sensor_maps) del padre B
        (o viceversa). Asi se mezclan dos 'inventos' independientes.

        Solo es valido si los indices de joint_idx y target_idx del B son
        compatibles con la estructura del A. Filtramos los que no encajen.
        """
        hijo = padre_a.copy()
        hijo.generation = max(padre_a.generation, padre_b.generation) + 1

        # 50%: trasplantar motor_maps de B
        if padre_b.motor_maps and random.random() < 0.5:
            n_joints_a = len(hijo.joints)
            valid = [m for m in padre_b.motor_maps
                     if 0 <= m.joint_idx < n_joints_a
                     and hijo.joints[m.joint_idx].has_motor]
            if valid:
                hijo.motor_maps = copy.deepcopy(valid)

        # 50%: trasplantar sensor_maps de B
        if padre_b.sensor_maps and random.random() < 0.5:
            valid = []
            for s in padre_b.sensor_maps:
                if s.sensor_type in ("angle", "angvel"):
                    if 0 <= s.target_idx < len(hijo.joints):
                        valid.append(s)
                elif s.sensor_type in ("pheromone", "nearby_organism",
                                        "obstacle_endpoint"):
                    valid.append(s)  # globales, siempre validos
                else:
                    if 0 <= s.target_idx < len(hijo.segments):
                        valid.append(s)
            if valid:
                hijo.sensor_maps = copy.deepcopy(valid)

        # F1: trasplantar reflexos de B (50%) — validar contra estructura de A
        if padre_b.reflexes and random.random() < 0.5:
            n_j = len(hijo.joints)
            n_s = len(hijo.segments)
            valid_rx = []
            for r in padre_b.reflexes:
                if not (0 <= r.motor_joint_idx < n_j and hijo.joints[r.motor_joint_idx].has_motor):
                    continue
                if r.sensor_type in ("angle", "angvel"):
                    if 0 <= r.sensor_target < n_j:
                        valid_rx.append(r)
                elif r.sensor_type in ("pheromone", "nearby_organism", "obstacle_endpoint"):
                    valid_rx.append(r)
                else:
                    if 0 <= r.sensor_target < n_s:
                        valid_rx.append(r)
            if valid_rx:
                hijo.reflexes = copy.deepcopy(valid_rx)

        # F2: trasplantar canales espontaneos de B (50%)
        if padre_b.spontaneous_motors and random.random() < 0.5:
            n_j = len(hijo.joints)
            valid_sp = [sm for sm in padre_b.spontaneous_motors
                        if 0 <= sm.joint_idx < n_j and hijo.joints[sm.joint_idx].has_motor]
            if valid_sp:
                hijo.spontaneous_motors = copy.deepcopy(valid_sp)

        # Adicionalmente trasplantar 1-2 segmentos de B con sus joints
        # si el hijo tiene espacio
        if (len(padre_b.segments) >= 2 and len(hijo.segments) < 10
                and random.random() < 0.3):
            # Tomar el ultimo segmento de B (probablemente sea hoja)
            new_seg = copy.deepcopy(padre_b.segments[-1])
            hijo.segments.append(new_seg)
            new_idx = len(hijo.segments) - 1
            # Conectarlo a un segmento del padre A elegido al azar
            parent_idx = random.randrange(len(hijo.segments) - 1)
            parent = hijo.segments[parent_idx]
            ang = random.uniform(0, 2 * np.pi)
            anchor_a = (np.cos(ang) * parent.size, np.sin(ang) * parent.size)
            anchor_b = (-np.cos(ang) * new_seg.size, -np.sin(ang) * new_seg.size)
            hijo.joints.append(JointGene(
                seg_a=parent_idx, seg_b=new_idx,
                anchor_a=anchor_a, anchor_b=anchor_b,
                has_motor=random.random() < 0.5,
                max_torque=random.uniform(4000.0, 14000.0),
            ))

        # Ligera mutacion adicional
        return hijo.mutate(rate=0.5)

    # ---------------------------------------------------------------- info
    def describe(self) -> str:
        n_motors = sum(1 for j in self.joints if j.has_motor)
        return (f"gen={self.generation:03d} {self.lineage:>8} "
                f"segs={len(self.segments):2d} joints={len(self.joints):2d} "
                f"motors={n_motors:2d} mmap={len(self.motor_maps):2d} "
                f"smap={len(self.sensor_maps):2d} "
                f"rx={len(self.reflexes):2d} spm={len(self.spontaneous_motors):2d}")
