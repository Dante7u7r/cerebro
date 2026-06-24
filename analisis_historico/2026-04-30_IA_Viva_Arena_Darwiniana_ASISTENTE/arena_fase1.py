#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARENA FASE 1 - Evolucion de cuerpos morfologicamente abiertos.

Cada organismo = Morphome (cuerpo evolvable) + BrainSimulation (cerebro de semilla_cerebro.py).
La seleccion natural opera sobre el organismo completo: cuerpo + cerebro.

No hay 'patas predefinidas'. No hay 'ojo correcto'. Solo: comio o no comio.
Si comio, sus genes (cuerpo) y su arquitectura neural (cerebro) se heredan con mutaciones.

Modulo C (presion de seleccion):
  C1 - Depredador simple
  C2 - Reproduccion sexual (crossover)
  C3 - Seleccion por torneo
  C4 - Niching geografico (4 cuadrantes)
"""

from __future__ import annotations
import os
import time
import pickle
import random
import argparse
import traceback
from dataclasses import dataclass, field

# Cargar el motor neural inyectando sus clases en __main__ para que unpickle funcione
def _bootstrap_semilla_cerebro():
    """Ejecuta el contenido de semilla_cerebro.py en globals() para que unpickle funcione."""
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(here, "semilla_cerebro.py")
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()
    src = src.replace('if __name__ == "__main__":', 'if False:')
    src = src.replace("if __name__ == '__main__':", "if False:")
    g = globals()
    exec(compile(src, src_path, "exec"), g)

_bootstrap_semilla_cerebro()
BrainSimulation = globals()["BrainSimulation"]
CellType = globals()["CellType"]

from morfoma import Morphome
from cuerpo_evolvable import CuerpoEvolvable
from mundo_fisico import MundoFisico
from depredador import Depredador


# ============================================================================
# Organismo = cerebro + cuerpo
# ============================================================================

@dataclass
class Organismo:
    nombre: str
    cerebro: BrainSimulation
    morphome: Morphome
    cuerpo: CuerpoEvolvable | None = None
    food_eaten: int = 0
    steps_alive: int = 0
    born_step: int = 0
    parent: str = ""
    quadrant: int = 0  # C4: cuadrante de origen (0..3)
    lineage_color: tuple[int, int, int] = field(default=(180, 180, 180))

    def fitness(self) -> float:
        if self.cuerpo is None:
            return 0.0
        return self.cuerpo.fitness()


# Colores por linaje (D: visualizador)
LINAGE_COLORS = {
    "alfa":  (220, 80, 80),    # rojo
    "beta":  (80, 130, 230),   # azul
    "gamma": (80, 200, 100),   # verde
    "delta": (240, 200, 80),   # amarillo
    "raiz":  (170, 170, 170),  # gris
}


def _color_para_linaje(nombre_base: str) -> tuple[int, int, int]:
    for prefix, color in LINAGE_COLORS.items():
        if nombre_base.lower().startswith(prefix):
            return color
    return LINAGE_COLORS["raiz"]


# ============================================================================
# Construccion / herencia
# ============================================================================

def crear_cerebro_desde_pkl(pkl_path: str, nombre: str,
                              limpieza: bool = True,
                              plasticidad: float = 0.2) -> BrainSimulation:
    """Carga un BrainSimulation guardado, sin su body anterior.

    Si `limpieza=True` (default), aplica un "reseteo de adaptacion" porque los
    cerebros pkl fueron entrenados con un cuerpo virtual diferente y sus pesos,
    memoria y mapas espaciales son ruido en el cuerpo fisico nuevo:
      - Memoria episodica: vacia (recuerdos de mundo inexistente).
      - Curiosity grid: zeros (mapa espacial obsoleto).
      - Neuron utility: decaida a 0.5 (neutro, sin matar arquitectura).
      - Pesos sinapticos: x `plasticidad` (default 0.2) para conservar la
        topologia pero permitir que STDP recalibre desde casi-blanco.
      - Frustration/resilience/courage: reseteados a defaults.
    Lo que se conserva: numero de neuronas y conexiones (la "arquitectura").
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    brain = BrainSimulation()
    brain.cells = data.get("cells", brain.cells)
    brain.time = 0.0 if limpieza else data.get("time", 0.0)
    brain.step_count = 0 if limpieza else data.get("step_count", 0)

    if limpieza:
        # 1. Memoria episodica vacia
        from collections import deque
        brain._episodic_buffer = deque(maxlen=20000)
        # 2. Curiosity grid en cero
        import numpy as np
        brain._curiosity_grid = np.zeros((20, 20), dtype=np.float32)
        # 3. Neuron utility neutralizada
        old_util = data.get("_neuron_utility", {})
        brain._neuron_utility = {k: 0.5 for k in old_util.keys()}
        # 4. Pesos sinapticos reducidos para ganar plasticidad
        # cells puede ser lista (formato interno) o dict (formato legacy)
        cells_iter = (brain.cells.values() if isinstance(brain.cells, dict)
                      else brain.cells)
        for cell in cells_iter:
            for syn_list_name in ("synapses_in", "synapses_out"):
                for syn in getattr(cell, syn_list_name, []):
                    if hasattr(syn, "weight"):
                        syn.weight = float(syn.weight) * plasticidad
        # 5. Estados emocionales en defaults
        brain.frustration = 0.0
        brain.resilience = 0.5
        brain.courage = 0.5
    else:
        if "frustration" in data:
            brain.frustration = data["frustration"]
        if "resilience" in data:
            brain.resilience = data["resilience"]
        if "courage" in data:
            brain.courage = data["courage"]
        if "_episodic_buffer" in data and data["_episodic_buffer"] is not None:
            from collections import deque
            items = list(data["_episodic_buffer"])[-5000:]
            brain._episodic_buffer = deque(items, maxlen=20000)
        if "_neuron_utility" in data:
            brain._neuron_utility = dict(data["_neuron_utility"])
        if "_curiosity_grid" in data and data["_curiosity_grid"] is not None:
            brain._curiosity_grid = data["_curiosity_grid"]

    brain.name = nombre
    return brain


def cerebro_semilla_minimo(nombre: str) -> BrainSimulation:
    """Cerebro nuevo, sin pkl. Para hijos de cero."""
    brain = BrainSimulation()
    brain.name = nombre
    return brain


def acoplar_cerebro_a_mundo(brain: BrainSimulation, morphome: Morphome,
                             world: MundoFisico, nombre: str,
                             spawn_xy=None) -> CuerpoEvolvable:
    """Construye CuerpoEvolvable desde el genoma y lo conecta al cerebro."""
    if spawn_xy is None:
        spawn_xy = world.random_spawn()
    cuerpo = CuerpoEvolvable(morphome, world, spawn_xy=spawn_xy)
    brain.body = cuerpo
    cuerpo.brain = brain  # F2: back-reference para canales motores espontaneos
    if not hasattr(brain, "history_food_dist"):
        brain.history_food_dist = []
        brain.consecutive_errors = 0
        brain.last_dist = float("inf")
    world.add_organism(cuerpo)
    return cuerpo


def reproducir(padre: Organismo, world: MundoFisico, gen_idx: int,
               otro_padre: Organismo | None = None,
               near_xy: tuple[float, float] | None = None) -> Organismo:
    """Crea un hijo con cuerpo mutado y cerebro heredado.

    C2: si otro_padre se proporciona, hace crossover (reproducción sexual).
    C4: si near_xy se proporciona, spawnea cerca de esa posición (niching).
    """
    # Cuerpo: mutación o crossover
    if otro_padre is not None and random.random() < 1.0:
        nuevo_morphome = Morphome.crossover(padre.morphome, otro_padre.morphome)
        sex_marker = "x"
    else:
        nuevo_morphome = padre.morphome.mutate()
        sex_marker = ""

    # Cerebro: herencia del padre principal
    import copy
    nuevo_brain = BrainSimulation()
    nuevo_brain.cells = copy.deepcopy(padre.cerebro.cells)
    nuevo_brain.time = 0.0
    nuevo_brain.step_count = 0
    nuevo_brain.next_id = padre.cerebro.next_id
    nuevo_brain._neuron_utility = copy.deepcopy(getattr(padre.cerebro, "_neuron_utility", {}))
    if hasattr(padre.cerebro, "_curiosity_grid"):
        nuevo_brain._curiosity_grid = padre.cerebro._curiosity_grid.copy()
    from collections import deque
    nuevo_brain._episodic_buffer = deque(maxlen=20000)
    nuevo_brain.frustration = 0.0
    nuevo_brain.resilience = 0.5
    nuevo_brain.courage = 0.5

    nombre = f"{padre.nombre[:1]}{sex_marker}-g{nuevo_morphome.generation:02d}-{gen_idx:03d}"
    nuevo_brain.name = nombre

    # Spawn (C4: cerca del padre si niching activo)
    spawn_xy = world.random_spawn(near=near_xy) if near_xy else world.random_spawn()

    cuerpo = acoplar_cerebro_a_mundo(nuevo_brain, nuevo_morphome, world, nombre, spawn_xy=spawn_xy)
    return Organismo(
        nombre=nombre,
        cerebro=nuevo_brain,
        morphome=nuevo_morphome,
        cuerpo=cuerpo,
        born_step=0,
        parent=padre.nombre,
        quadrant=padre.quadrant,
        lineage_color=padre.lineage_color,
    )


# ============================================================================
# Generacion 0 con niching geografico (C4)
# ============================================================================

PKL_FUENTES = [
    "semillas/arena_save_omega.pkl",
    "semillas/arena_save_sigma.pkl",
    "semillas/arena_save_hibrido.pkl",
    "semillas/arena_save_omega2.pkl",
    "semillas/arena_save_sigma2.pkl",
    "semillas/originales/arena_save_alfa.pkl",
    "semillas/originales/arena_save_beta.pkl",
    "semillas/originales/arena_save_gamma.pkl",
    "semillas/originales/arena_save_delta.pkl",
]


def construir_generacion_0(world: MundoFisico, n_total: int = 12,
                            niching: bool = True) -> list[Organismo]:
    """Mezcla cerebros heredados de los pkls + cerebros vacios.

    C4: cada cerebro madre nace en su propio cuadrante para iniciar especiacion.
    """
    organismos: list[Organismo] = []
    pkls_disponibles = [p for p in PKL_FUENTES if os.path.exists(p)]

    # Cuadrantes para los 4 linajes principales
    QUADRANTS = {"alfa": 0, "beta": 1, "gamma": 2, "delta": 3}

    for i, pkl in enumerate(pkls_disponibles):
        nombre_base = os.path.basename(pkl).replace("arena_save_", "").replace(".pkl", "")
        try:
            brain = crear_cerebro_desde_pkl(pkl, nombre_base)
        except Exception as e:
            print(f"[!] No pude cargar {pkl}: {e}")
            continue
        morphome = Morphome.semilla_minima(lineage=nombre_base)
        for _ in range(random.randint(1, 2)):
            morphome = morphome.mutate()
        quadrant = QUADRANTS.get(nombre_base, i % 4)
        spawn_xy = world.quadrant_spawn(quadrant) if niching else world.random_spawn()
        cuerpo = acoplar_cerebro_a_mundo(brain, morphome, world, nombre_base, spawn_xy=spawn_xy)
        organismos.append(Organismo(
            nombre=nombre_base, cerebro=brain, morphome=morphome,
            cuerpo=cuerpo, born_step=0, parent="(semilla)",
            quadrant=quadrant, lineage_color=_color_para_linaje(nombre_base),
        ))

    # Resto: organismos raiz, distribuidos en cuadrantes ciclicamente
    for i in range(len(organismos), n_total):
        nombre = f"raiz-{i:02d}"
        brain = cerebro_semilla_minimo(nombre)
        morphome = Morphome.semilla_minima(lineage=nombre)
        for _ in range(random.randint(0, 2)):
            morphome = morphome.mutate()
        quadrant = i % 4
        spawn_xy = world.quadrant_spawn(quadrant) if niching else world.random_spawn()
        cuerpo = acoplar_cerebro_a_mundo(brain, morphome, world, nombre, spawn_xy=spawn_xy)
        organismos.append(Organismo(
            nombre=nombre, cerebro=brain, morphome=morphome,
            cuerpo=cuerpo, born_step=0, parent="(raiz)",
            quadrant=quadrant, lineage_color=_color_para_linaje(nombre),
        ))

    return organismos


# ============================================================================
# C3: Seleccion por torneo
# ============================================================================

def seleccion_por_torneo(organismos: list[Organismo], n_survivors: int,
                          tournament_size: int = 3) -> list[Organismo]:
    """Hace n_survivors torneos. Devuelve los ganadores (sin repetir).

    Si los torneos generan duplicados, completa con los mejores no usados.
    """
    pool = list(organismos)
    sobrevivientes = []
    used = set()
    while len(sobrevivientes) < n_survivors and len(used) < len(pool):
        candidatos = random.sample(pool, min(tournament_size, len(pool)))
        ganador = max(candidatos, key=lambda o: o.fitness())
        if id(ganador) not in used:
            sobrevivientes.append(ganador)
            used.add(id(ganador))

    # Completar con los mejores no seleccionados si faltan
    if len(sobrevivientes) < n_survivors:
        sobrantes = [o for o in pool if id(o) not in used]
        sobrantes.sort(key=lambda o: o.fitness(), reverse=True)
        for o in sobrantes:
            if len(sobrevivientes) >= n_survivors:
                break
            sobrevivientes.append(o)

    return sobrevivientes


# ============================================================================
# Loop principal
# ============================================================================

def loop_evolutivo(steps_per_generation: int = 8000,
                   n_generations: int = 50,
                   pop_size: int = 12,
                   n_food: int = 50,
                   n_obstacles: int = 6,
                   n_predators: int = 1,
                   eras_every: int = 8,
                   crossover_prob: float = 0.20,
                   tournament_size: int = 3,
                   niching: bool = True,
                   visual: bool = False,
                   verbose: bool = True,
                   save_every: int = 5,
                   load_from: str | None = None,
                   seed: int | None = None,
                   save_dir: str | None = None):
    global CHECKPOINT_DIR
    if save_dir is not None:
        CHECKPOINT_DIR = save_dir

    print("=" * 70)
    print(" ARENA FASE 1 - Evolucion de cuerpos en mundo rico")
    print("=" * 70)
    print(f" pop={pop_size} food={n_food} obstacles={n_obstacles} "
          f"predators={n_predators} eras_every={eras_every}")
    print(f" crossover_prob={crossover_prob} tournament_size={tournament_size} "
          f"niching={niching} visual={visual} save_dir={CHECKPOINT_DIR}")
    print()

    world = MundoFisico(n_food=n_food, n_obstacles=n_obstacles, seed=seed)

    historial = []
    gen_offset = 0
    if load_from and os.path.exists(load_from):
        organismos, historial, gen_offset = cargar_checkpoint(load_from, world,
                                                                niching=niching)
        # Si la población cargada es menor que pop_size, completar con hijos
        while len(organismos) < pop_size:
            padre = random.choice(organismos)
            hijo = reproducir(padre, world, gen_idx=len(organismos),
                              near_xy=(padre.cuerpo.x, padre.cuerpo.y) if niching else None)
            organismos.append(hijo)
    else:
        # Inicio fresco: archivar checkpoints viejos para no contaminar la nueva corrida
        _archivar_checkpoints_viejos()
        organismos = construir_generacion_0(world, n_total=pop_size, niching=niching)

    # C1: depredadores
    depredadores: list[Depredador] = []
    for i in range(n_predators):
        margin = 60.0
        sx = world.WORLD_W * (0.2 + 0.6 * (i / max(1, n_predators)))
        sy = world.WORLD_H * 0.85
        depredadores.append(Depredador(world, spawn_xy=(sx, sy)))

    # Si cargamos desde un checkpoint con era > 0, anadir los depredadores
    # que se hubieran sumado en cada era (1 por era). Asi la presion selectiva
    # escalada es consistente entre corridas frescas y continuaciones.
    if world.era > 0:
        for _ in range(world.era):
            margin = 60.0
            sx = world.rng.uniform(margin, world.WORLD_W - margin)
            sy = world.rng.uniform(margin, world.WORLD_H - margin)
            depredadores.append(Depredador(world, spawn_xy=(sx, sy)))
        print(f"[+] Restaurando depredadores: +{world.era} por era previa "
              f"(total: {len(depredadores)})")

    print(f"[*] Generacion 0: {len(organismos)} organismos")
    for o in organismos:
        print(f"    - {o.nombre:<12} q={o.quadrant} | {o.morphome.describe()}")
    print()

    # Visualizador (D)
    visualizer = None
    if visual:
        try:
            from visualizador_fase1 import Visualizador
            visualizer = Visualizador(world, organismos, depredadores)
        except Exception as e:
            print(f"[!] Visualizador no disponible: {e}")
            visualizer = None

    try:
        for gen_local in range(n_generations):
            gen = gen_offset + gen_local
            t0 = time.time()
            # ---- vida ----
            for step in range(steps_per_generation):
                for o in organismos:
                    if o.cuerpo is None:
                        continue
                    try:
                        o.cerebro.step()
                        o.steps_alive += 1
                        if o.cuerpo.reward > 0:
                            o.food_eaten += 1
                    except Exception as e:
                        if verbose:
                            print(f"[!] {o.nombre} murio en step {step}: {e}")
                        if o.cuerpo:
                            world.remove_organism(o.cuerpo)
                            o.cuerpo = None

                # C1: depredadores se mueven y golpean
                for pred in depredadores:
                    hits = pred.step()
                    for cuerpo_hit in hits:
                        cuerpo_hit._predator_hits += 1

                # Avanzar fisica
                world.step_physics(dt=1 / 60.0, substeps=1)

                # F1.5-C: cadena alimenticia entre organismos (cada 8 steps)
                if step % 8 == 0:
                    try:
                        victimas = world.detect_predation()
                        for victima in victimas:
                            for o in organismos:
                                if o.cuerpo is victima:
                                    if verbose:
                                        print(f"   [!] {o.nombre} fue devorado")
                                    world.remove_organism(o.cuerpo)
                                    o.cuerpo = None
                                    break
                    except Exception:
                        pass

                if step % 200 == 0:
                    world.replenish_food()
                    world._refresh_anchored_food()

                # M2: food burst (eventos transitorios de comida rica)
                if step % 30 == 0:
                    world.maybe_spawn_food_burst(prob=0.45)

                # Visualizacion
                if visualizer is not None and step % 2 == 0:
                    keep_running = visualizer.draw(gen=gen, step=step,
                                                    organismos=organismos,
                                                    depredadores=depredadores)
                    if not keep_running:
                        print("[!] Visualizador cerrado por usuario.")
                        return historial, organismos

            # ---- evaluacion ----
            organismos.sort(key=lambda o: o.fitness(), reverse=True)
            elapsed = time.time() - t0
            comida_viva = sum(1 for f in world.food_items if f.alive)
            print(f"[Gen {gen:02d}] {elapsed:.1f}s era={world.era} food={comida_viva} "
                  f"obstacles={len(world.obstaculos)}")
            for o in organismos[:min(8, len(organismos))]:
                tox = o.cuerpo._toxic_count if o.cuerpo else 0
                hits = o.cuerpo._predator_hits if o.cuerpo else 0
                print(f"   {o.fitness():7.2f}  {o.nombre:<14} q={o.quadrant} "
                      f"food={o.food_eaten:3d} tox={tox} hits={hits} "
                      f"| {o.morphome.describe()}")

            # F2.0: feedback ecosistemico (reemplaza eras)
            food_consumed_this_gen = sum(getattr(o, "_food_count", 0) for o in organismos) \
                                   + sum(getattr(o, "_toxic_count", 0) for o in organismos)

            historial.append({
                "gen": gen,
                "best": {"nombre": organismos[0].nombre, "fitness": organismos[0].fitness()},
                "fitness_distribution": [o.fitness() for o in organismos],
                "era": world.era,
                "food_consumed": food_consumed_this_gen,
                "n_food_baseline": world.n_food_baseline,
                "n_food_target": world.n_food_target,
            })

            # ---- F2.0: actualizar ecosistema (sustituye avanzar_era) ----
            prev_era = world.era
            world.update_ecosystem(food_consumed_this_gen)
            print(f"   [eco] consumed={food_consumed_this_gen} "
                  f"baseline={world.n_food_baseline:.1f} "
                  f"target={world.n_food_target} stress_era={world.era}")
            # Si el estres ecosistemico subio, anadir depredador (presion escalada natural)
            if world.era > prev_era:
                margin = 60.0
                spawn_x = world.rng.uniform(margin, world.WORLD_W - margin)
                spawn_y = world.rng.uniform(margin, world.WORLD_H - margin)
                nuevo_pred = Depredador(world, spawn_xy=(spawn_x, spawn_y))
                depredadores.append(nuevo_pred)
                print(f"   [+] Estres ecologico subio. Depredador anadido. Total: {len(depredadores)}")

            # ---- seleccion (C3: torneo) ----
            n_survivors = max(2, pop_size // 2)
            sobrevivientes = seleccion_por_torneo(organismos, n_survivors,
                                                    tournament_size=tournament_size)
            survivor_ids = {id(o) for o in sobrevivientes}
            muertos = [o for o in organismos if id(o) not in survivor_ids]
            for o in muertos:
                if o.cuerpo:
                    world.remove_organism(o.cuerpo)
                    o.cuerpo = None

            # ---- reproduccion (C2: crossover ocasional, C4: cerca del padre) ----
            nuevos = []
            idx = 0
            while len(sobrevivientes) + len(nuevos) < pop_size:
                padre = random.choice(sobrevivientes)
                otro = None
                if random.random() < crossover_prob and len(sobrevivientes) >= 2:
                    candidatos = [o for o in sobrevivientes if o is not padre]
                    if candidatos:
                        otro = random.choice(candidatos)
                # Spawn cerca del padre si niching activo
                near = (padre.cuerpo.x, padre.cuerpo.y) if (niching and padre.cuerpo) else None
                try:
                    hijo = reproducir(padre, world, gen_idx=idx,
                                      otro_padre=otro, near_xy=near)
                    nuevos.append(hijo)
                except Exception as e:
                    if verbose:
                        print(f"   [!] Error reproduciendo de {padre.nombre}: {e}")
                        traceback.print_exc()
                idx += 1
                if idx > pop_size * 4:
                    break

            # Resetear contadores y reposicionar sobrevivientes
            for o in sobrevivientes:
                o.food_eaten = 0
                o.steps_alive = 0
                if o.cuerpo:
                    o.cuerpo._food_count = 0
                    o.cuerpo._toxic_count = 0
                    o.cuerpo._predator_hits = 0
                    o.cuerpo._lifetime_reward = 0.0
                    o.cuerpo._steps_alive = 0
                    o.cuerpo._approach_bonus = 0.0
                    o.cuerpo._dist_prev = float("inf")
                    if niching:
                        spawn = world.quadrant_spawn(o.quadrant)
                    else:
                        spawn = world.random_spawn()
                    o.cuerpo.x = spawn[0]
                    o.cuerpo.y = spawn[1]

            organismos = sobrevivientes + nuevos

            # M1: clusters migran ligeramente cada generacion
            world.drift_clusters(sigma=12.0)
            # M4: jitter en ratios de tipos de comida
            world.jitter_food_ratios(jitter=0.08)

            # Repoblar comida para nueva generacion
            for f in world.food_items:
                f.alive = True
            world.replenish_food()
            world._refresh_anchored_food()

            if (gen + 1) % save_every == 0:
                guardar_estado(historial, organismos, gen, world)
    finally:
        # Limpieza de visualizador
        if visualizer is not None:
            try:
                visualizer.close()
            except Exception:
                pass

    # Guardar checkpoint final
    if organismos:
        try:
            guardar_estado(historial, organismos,
                           gen_offset + n_generations - 1, world)
        except Exception as e:
            print(f"[!] No pude guardar checkpoint final: {e}")

    print("\n" + "=" * 70)
    print(" Fase 1 completada")
    print("=" * 70)
    return historial, organismos


CHECKPOINT_DIR = "checkpoints"  # se sobreescribe desde CLI con --save_dir


def _archivar_checkpoints_viejos():
    """Si existen checkpoints de una corrida previa, los mueve a una carpeta
    de archivo con timestamp para no contaminar la nueva corrida."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return
    contenido = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pkl")]
    if not contenido:
        return
    import shutil
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo = f"{CHECKPOINT_DIR}_archive_{stamp}"
    shutil.move(CHECKPOINT_DIR, archivo)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"[+] Checkpoints viejos archivados en: {archivo}/")
    print(f"    ({len(contenido)} archivos preservados, nueva carpeta limpia)")


def _serializar_cerebro(brain: BrainSimulation) -> dict:
    """Extrae los componentes serializables de un cerebro."""
    return {
        "cells": brain.cells,
        "time": brain.time,
        "step_count": brain.step_count,
        "next_id": getattr(brain, "next_id", 0),
        "frustration": getattr(brain, "frustration", 0.0),
        "resilience": getattr(brain, "resilience", 0.5),
        "courage": getattr(brain, "courage", 0.5),
        "_neuron_utility": dict(getattr(brain, "_neuron_utility", {})),
        "_curiosity_grid": getattr(brain, "_curiosity_grid", None),
        "_episodic_buffer": list(getattr(brain, "_episodic_buffer", []))[-5000:],
        "name": getattr(brain, "name", "anon"),
    }


def _restaurar_cerebro(data: dict) -> BrainSimulation:
    """Reconstruye un BrainSimulation desde un dict serializado."""
    brain = BrainSimulation()
    brain.cells = data.get("cells", brain.cells)
    brain.time = data.get("time", 0.0)
    brain.step_count = data.get("step_count", 0)
    brain.next_id = data.get("next_id", getattr(brain, "next_id", 0))
    brain.frustration = data.get("frustration", 0.0)
    brain.resilience = data.get("resilience", 0.5)
    brain.courage = data.get("courage", 0.5)
    brain._neuron_utility = dict(data.get("_neuron_utility", {}))
    if data.get("_curiosity_grid") is not None:
        brain._curiosity_grid = data["_curiosity_grid"]
    from collections import deque
    items = data.get("_episodic_buffer", []) or []
    brain._episodic_buffer = deque(items, maxlen=20000)
    brain.name = data.get("name", "anon")
    return brain


def guardar_estado(historial, organismos, gen, world: MundoFisico):
    """Guarda checkpoint completo: cerebros + morfomas + estado del mundo."""
    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        snap = {
            "gen": gen,
            "historial": historial,
            "world_era": world.era,
            "world_n_food": world.n_food_target,
            "world_n_obstacles": len(world.obstaculos),
            "organismos": [
                {
                    "nombre": o.nombre,
                    "morphome": o.morphome,
                    "cerebro": _serializar_cerebro(o.cerebro),
                    "fitness": o.fitness(),
                    "food_eaten": o.food_eaten,
                    "parent": o.parent,
                    "quadrant": o.quadrant,
                    "lineage_color": o.lineage_color,
                }
                for o in organismos
            ],
        }
        # Guardar versión numerada y "latest" para continuar fácil
        path_num = os.path.join(CHECKPOINT_DIR, f"checkpoint_gen{gen:04d}.pkl")
        path_latest = os.path.join(CHECKPOINT_DIR, "latest.pkl")
        with open(path_num, "wb") as f:
            pickle.dump(snap, f)
        with open(path_latest, "wb") as f:
            pickle.dump(snap, f)
        print(f"   [+] Checkpoint guardado: {path_num}")
    except Exception as e:
        print(f"   [!] No pude guardar checkpoint: {e}")
        traceback.print_exc()


def cargar_checkpoint(path: str, world: MundoFisico,
                       niching: bool = True) -> tuple[list[Organismo], list, int]:
    """Carga organismos desde un checkpoint y los acopla al mundo.

    Devuelve (organismos, historial, gen_inicial).
    """
    with open(path, "rb") as f:
        snap = pickle.load(f)

    # Restaurar estado del mundo
    if "world_era" in snap:
        for _ in range(snap["world_era"]):
            world.avanzar_era()

    organismos: list[Organismo] = []
    for od in snap["organismos"]:
        brain = _restaurar_cerebro(od["cerebro"])
        morphome = od["morphome"]
        spawn = (world.quadrant_spawn(od.get("quadrant", 0)) if niching
                 else world.random_spawn())
        cuerpo = acoplar_cerebro_a_mundo(brain, morphome, world,
                                           od["nombre"], spawn_xy=spawn)
        organismos.append(Organismo(
            nombre=od["nombre"], cerebro=brain, morphome=morphome,
            cuerpo=cuerpo, born_step=0, parent=od.get("parent", "(checkpoint)"),
            quadrant=od.get("quadrant", 0),
            lineage_color=od.get("lineage_color", _color_para_linaje(od["nombre"])),
        ))

    historial = snap.get("historial", [])
    gen_inicial = snap.get("gen", 0) + 1
    print(f"[+] Checkpoint cargado: gen={gen_inicial - 1} "
          f"organismos={len(organismos)} era={snap.get('world_era', 0)}")
    return organismos, historial, gen_inicial


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=20)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--pop", type=int, default=10)
    parser.add_argument("--food", type=int, default=60)
    parser.add_argument("--obstacles", type=int, default=6)
    parser.add_argument("--predators", type=int, default=1)
    parser.add_argument("--eras_every", type=int, default=8)
    parser.add_argument("--crossover_prob", type=float, default=0.20)
    parser.add_argument("--tournament", type=int, default=3)
    parser.add_argument("--no-niching", action="store_true")
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test rapido (1 gen, 200 steps, 4 organismos)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--load", type=str, default=None,
                        help="Cargar checkpoint para continuar (ej: checkpoints/latest.pkl)")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Guardar checkpoint cada N generaciones")
    parser.add_argument("--continue", dest="cont", action="store_true",
                        help="Atajo: cargar checkpoints/latest.pkl si existe")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Carpeta de checkpoints (default: checkpoints). Usar para ramas paralelas.")
    args = parser.parse_args()

    load_from = args.load
    # --save_dir sobreescribe el directorio de checkpoints para ramas paralelas
    effective_save_dir = args.save_dir  # None = usar default "checkpoints"

    if args.cont and load_from is None:
        # Buscar en save_dir si se especificó, si no en el directorio default
        search_dir = effective_save_dir if effective_save_dir else CHECKPOINT_DIR
        candidate = os.path.join(search_dir, "latest.pkl")
        if os.path.exists(candidate):
            load_from = candidate
            print(f"[+] --continue: cargando {candidate}")
        else:
            print(f"[!] --continue: no existe {candidate}, empezando de cero")

    if args.smoke:
        loop_evolutivo(steps_per_generation=200, n_generations=1, pop_size=4,
                       n_food=20, n_obstacles=4, n_predators=1,
                       eras_every=999, niching=True,
                       visual=args.visual, verbose=True, save_every=999,
                       load_from=load_from, seed=args.seed,
                       save_dir=effective_save_dir)
    else:
        loop_evolutivo(steps_per_generation=args.steps, n_generations=args.gens,
                       pop_size=args.pop, n_food=args.food,
                       n_obstacles=args.obstacles, n_predators=args.predators,
                       eras_every=args.eras_every,
                       crossover_prob=args.crossover_prob,
                       tournament_size=args.tournament,
                       niching=not args.no_niching,
                       visual=args.visual, verbose=True,
                       save_every=args.save_every,
                       load_from=load_from, seed=args.seed,
                       save_dir=effective_save_dir)
