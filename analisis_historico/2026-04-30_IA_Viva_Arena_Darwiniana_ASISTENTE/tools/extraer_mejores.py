#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extraer_mejores.py — Extrae los mejores organismos evolucionados de
checkpoints/ y checkpoints_fresh/ y los guarda como nuevas semillas pkl.

A diferencia de los pkl originales (alfa/beta/gamma/delta) que solo guardan
cerebros entrenados en mundo virtual, estas semillas EVOLUCIONADAS contienen:
  - cerebro (compatible con crear_cerebro_desde_pkl)
  - morphome evolucionado (12 segmentos, ~13-15 motor maps)
  - metadata: rama, generacion, era, linaje

Resultado:
  arena_save_omega.pkl   — mejor de Rama A (12 segs, 13 mmap, 13 smap)
  arena_save_omega2.pkl  — segundo de Rama A
  arena_save_sigma.pkl   — mejor de Rama B (12 segs, 15 mmap, 10 smap)
  arena_save_sigma2.pkl  — segundo de Rama B

Estos representan organismos forjados en fisica real bajo presion selectiva
extrema (era 10 en A, era 19 en B). Son MUCHO mejores semillas que los
originales para fases futuras.
"""

import os
import pickle


def _bootstrap():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raiz del proyecto
    src = os.path.join(here, "semilla_cerebro.py")
    with open(src, "r", encoding="utf-8") as f:
        code = f.read()
    code = code.replace('if __name__ == "__main__":', 'if False:')
    code = code.replace("if __name__ == '__main__':", "if False:")
    exec(compile(code, src, "exec"), globals())

_bootstrap()


def score_morfologico(org_dict):
    """Suma de complejidad morfologica como proxy de aptitud evolutiva."""
    m = org_dict["morphome"]
    return len(m.segments) + len(m.motor_maps) + len(m.sensor_maps)


def extraer_top(checkpoint_path: str, n_top: int = 2):
    with open(checkpoint_path, "rb") as f:
        snap = pickle.load(f)
    orgs = snap["organismos"]
    orgs_sorted = sorted(orgs, key=score_morfologico, reverse=True)
    return orgs_sorted[:n_top], snap


def guardar_semilla_evolucionada(org_dict, snap, output_path: str,
                                   nombre_nuevo: str, rama: str):
    """Guarda un organismo como nuevo pkl semilla."""
    cerebro = org_dict["cerebro"]
    morphome = org_dict["morphome"]

    # Datos del cerebro (formato compatible con crear_cerebro_desde_pkl)
    data = {
        "cells": cerebro["cells"],
        "time": cerebro.get("time", 0.0),
        "step_count": cerebro.get("step_count", 0),
        "next_id": cerebro.get("next_id", 0),
        "frustration": cerebro.get("frustration", 0.0),
        "resilience": cerebro.get("resilience", 0.5),
        "courage": cerebro.get("courage", 0.5),
        "_neuron_utility": cerebro.get("_neuron_utility", {}),
        "_curiosity_grid": cerebro.get("_curiosity_grid"),
        "_episodic_buffer": cerebro.get("_episodic_buffer", []),
        "name": nombre_nuevo,

        # Extension: morfoma evolucionado + metadata
        "morphome": morphome,
        "metadata": {
            "rama": rama,
            "gen_extraido": snap["gen"],
            "era": snap["world_era"],
            "nombre_original": org_dict["nombre"],
            "parent": org_dict.get("parent", "?"),
            "morfologia": (
                f"segs={len(morphome.segments)} "
                f"mmap={len(morphome.motor_maps)} "
                f"smap={len(morphome.sensor_maps)}"
            ),
            "n_neuronas": len(cerebro["cells"]),
        },
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    meta = data["metadata"]
    print(f"  [+] {output_path}")
    print(f"      Rama {meta['rama']} | gen {meta['gen_extraido']} | era {meta['era']}")
    print(f"      Original: {meta['nombre_original']} (parent: {meta['parent']})")
    print(f"      Cuerpo: {meta['morfologia']}")
    print(f"      Cerebro: {meta['n_neuronas']} neuronas")


if __name__ == "__main__":
    print("=" * 70)
    print(" EXTRACCION DE SEMILLAS EVOLUCIONADAS")
    print("=" * 70)

    # Rama A — checkpoints/
    print("\n[*] Rama A (checkpoints/latest.pkl)")
    top_a, snap_a = extraer_top("checkpoints/latest.pkl", n_top=2)
    guardar_semilla_evolucionada(top_a[0], snap_a,
                                   "semillas/arena_save_omega.pkl", "omega", "A")
    guardar_semilla_evolucionada(top_a[1], snap_a,
                                   "semillas/arena_save_omega2.pkl", "omega2", "A")

    # Rama B — checkpoints_fresh/
    print("\n[*] Rama B (checkpoints_fresh/latest.pkl)")
    top_b, snap_b = extraer_top("checkpoints_fresh/latest.pkl", n_top=2)
    guardar_semilla_evolucionada(top_b[0], snap_b,
                                   "semillas/arena_save_sigma.pkl", "sigma", "B")
    guardar_semilla_evolucionada(top_b[1], snap_b,
                                   "semillas/arena_save_sigma2.pkl", "sigma2", "B")

    print("\n" + "=" * 70)
    print(" Listo. 4 semillas evolucionadas guardadas:")
    print("   omega, omega2  — descendientes de Rama A (era 10)")
    print("   sigma, sigma2  — descendientes de Rama B (era 19)")
    print("=" * 70)
