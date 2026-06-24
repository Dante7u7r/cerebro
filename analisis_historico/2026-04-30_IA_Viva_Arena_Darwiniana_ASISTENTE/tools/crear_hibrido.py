#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crear_hibrido.py — Fusiona las topologias de alfa + beta en un cerebro hibrido.

Resultado: arena_save_hibrido.pkl
  - ~728 neuronas (483 alfa + 245 beta, conteo real al momento del merge)
  - IDs de beta remapeados (offset = max_alfa_id + 1) para no colisionar
  - Pesos escalados a 0.05 (casi cero) — STDP aprende desde cero sobre la
    topologia heredada
  - Memoria episodica y curiosity grid vacios (sin recuerdos del mundo viejo)
  - Utilidad neuronal neutra (0.5) para todas las neuronas

Uso:
    python crear_hibrido.py

Luego la proxima corrida en arena_fase1.py detectara arena_save_hibrido.pkl
y lo usara como un linaje mas en la generacion 0.
"""

import os
import copy
import pickle
import numpy as np

# Cargar clases de semilla_cerebro en globals para que unpickle funcione
def _bootstrap():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raiz del proyecto
    src = os.path.join(here, "semilla_cerebro.py")
    with open(src, "r", encoding="utf-8") as f:
        code = f.read()
    code = code.replace('if __name__ == "__main__":', 'if False:')
    code = code.replace("if __name__ == '__main__':", "if False:")
    exec(compile(code, src, "exec"), globals())

_bootstrap()
BrainSimulation = globals()["BrainSimulation"]


def cargar_brain_data(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def limpiar_pesos(cells: list, factor: float = 0.05):
    """Escala todos los pesos sinapticos por `factor` (casi cero)."""
    for cell in cells:
        for syn in getattr(cell, "synapses_in", []):
            if hasattr(syn, "weight"):
                syn.weight = float(syn.weight) * factor
        for syn in getattr(cell, "synapses_out", []):
            if hasattr(syn, "weight"):
                syn.weight = float(syn.weight) * factor


def remap_ids_list(cells: list, offset: int) -> list:
    """
    Crea una copia de la lista de cells con todos los IDs (cell.id, syn.pre_id,
    syn.post_id) desplazados por `offset`.
    """
    new_cells = []
    for cell in cells:
        cell_copy = copy.deepcopy(cell)
        cell_copy.id = cell.id + offset

        for syn in getattr(cell_copy, "synapses_in", []):
            syn.pre_id = syn.pre_id + offset
            syn.post_id = syn.post_id + offset

        for syn in getattr(cell_copy, "synapses_out", []):
            syn.pre_id = syn.pre_id + offset
            syn.post_id = syn.post_id + offset

        new_cells.append(cell_copy)

    return new_cells


def fusionar(alfa_path: str, beta_path: str, output_path: str,
             peso_factor: float = 0.05):
    print(f"[*] Cargando {alfa_path} ...")
    alfa_data = cargar_brain_data(alfa_path)
    print(f"[*] Cargando {beta_path} ...")
    beta_data = cargar_brain_data(beta_path)

    # cells es una lista de IntelligentCell, cada una con .id
    alfa_cells: list = alfa_data.get("cells", [])
    beta_cells: list = beta_data.get("cells", [])

    n_alfa = len(alfa_cells)
    n_beta = len(beta_cells)
    print(f"    alfa: {n_alfa} neuronas | beta: {n_beta} neuronas")

    # Offset: siguiente ID libre despues del ID maximo de alfa
    if alfa_cells:
        max_alfa_id = max(c.id for c in alfa_cells)
    else:
        max_alfa_id = 0
    offset = max_alfa_id + 1

    # Remap beta para que sus IDs no colisionen con alfa
    beta_remapped = remap_ids_list(copy.deepcopy(beta_cells), offset)

    # Fusionar: copia profunda de alfa + beta remapeado
    merged_cells = copy.deepcopy(alfa_cells) + beta_remapped

    print(f"    hibrido: {len(merged_cells)} neuronas totales")

    # Limpiar pesos en todo el hibrido (STDP recalibrara desde casi-cero)
    limpiar_pesos(merged_cells, factor=peso_factor)

    # Nuevo next_id
    new_next_id = max(c.id for c in merged_cells) + 1

    # _neuron_utility indexado por id de neurona
    utility = {c.id: 0.5 for c in merged_cells}

    # Construir data dict en el mismo formato que usa crear_cerebro_desde_pkl
    from collections import deque
    hibrido_data = {
        "cells": merged_cells,
        "time": 0.0,
        "step_count": 0,
        "next_id": new_next_id,
        "frustration": 0.0,
        "resilience": 0.5,
        "courage": 0.5,
        "_neuron_utility": utility,
        "_curiosity_grid": np.zeros((20, 20), dtype=np.float32),
        "_episodic_buffer": [],
        "name": "hibrido",
    }

    with open(output_path, "wb") as f:
        pickle.dump(hibrido_data, f)

    print(f"[+] Cerebro hibrido guardado en: {output_path}")
    print(f"    Neuronas: {len(merged_cells)} | next_id: {new_next_id}")
    print(f"    Pesos escalados a x{peso_factor} (STDP recalibrara desde casi-cero)")
    print(f"    Memoria episodica: vacia | Curiosity grid: cero | Utilidad: 0.5 neutral")


if __name__ == "__main__":
    alfa_path = "semillas/originales/arena_save_alfa.pkl"
    beta_path = "semillas/originales/arena_save_beta.pkl"
    output_path = "semillas/arena_save_hibrido.pkl"

    missing = [p for p in (alfa_path, beta_path) if not os.path.exists(p)]
    if missing:
        print(f"[!] No se encontraron: {missing}")
        print("    Ejecuta desde la raiz del proyecto: python tools/crear_hibrido.py")
        raise SystemExit(1)

    fusionar(alfa_path, beta_path, output_path, peso_factor=0.05)
    print("\nListo. arena_fase1.py detectara el hibrido en semillas/.")
