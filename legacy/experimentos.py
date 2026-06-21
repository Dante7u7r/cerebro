#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IA VIVA - EXPERIMENTOS DE FASE 2

Experimento 1: Respuesta de dopamina a novedad
Experimento 2: Frustracion y poda neuronal
Experimento 3: Consolidacion en sueno REM

Ejecutar: python experimentos.py
"""

import sys
import os
import numpy as np
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))
from asistente_cerebro import (
    BrainSimulation, BrainObserver, StimulusController, EventLogger,
    CellType, NeuronSubtype, BrainState, SLEEP_CYCLE_PERIOD
)


def separador(nombre):
    print(f"\n{'#'*60}")
    print(f"  {nombre}")
    print(f"{'#'*60}")


def registrar_estado(observer, label=""):
    s = observer.snapshot()
    if label:
        print(f"\n  --- {label} ---")
    print(f"  t={s['time']:.1f}ms  paso={s['step']}  estado={s['brain_state']}")
    print(f"  Neuronas: {s['total_neurons']}  Sinapsis: {s['total_synapses']}")
    print(f"  DA={s['dopamine']:.4f}  5HT={s['serotonin']:.4f}  ACh={s['acetylcholine']:.4f}")
    print(f"  Fru={s['frustration']:.4f}  Res={s['resilience']:.4f}  Cou={s['courage']:.4f}")
    print(f"  w_prom={s['avg_weight']:.4f}  w_max={s['max_weight']:.4f}")
    print(f"  Energia: {s['energy_avg']:.1f}")
    return s


# ============================================================================
# EXPERIMENTO 1: Respuesta de dopamina a novedad
# ============================================================================

def experimento_1():
    separador("EXPERIMENTO 1: Respuesta de dopamina a novedad")
    print("Objetivo: Observar como la dopamina responde a estmulos nuevos")
    print("          y como afecta la plasticidad STDP.\n")

    cerebro = BrainSimulation(name="Exp1")
    observer = BrainObserver(cerebro)
    stim = StimulusController(cerebro)
    logger = EventLogger(cerebro, directorio="logs/exp1")
    cerebro._logger = logger

    historial_da = []
    historial_peso = []
    historial_fru = []

    print("Fase 1: Basal (200 pasos sin estmulos)")
    stim.silencio(200)
    s = registrar_estado(observer, "Estado basal")
    historial_da.append(('basal', s['dopamine']))
    historial_peso.append(('basal', s['avg_weight']))
    historial_fru.append(('basal', s['frustration']))

    intensidades = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for intensidad in intensidades:
        print(f"\nFase: Novedad intensidad={intensidad}")
        stim.inyectar_novedad(intensidad)
        stim.silencio(100)
        s = registrar_estado(observer, f"Despues de novedad={intensidad}")
        historial_da.append((f'novedad_{intensidad}', s['dopamine']))
        historial_peso.append((f'novedad_{intensidad}', s['avg_weight']))
        historial_fru.append((f'novedad_{intensidad}', s['frustration']))

    print("\nFase: Recompensa (para ver consolidacion)")
    stim.inyectar_recompensa()
    stim.silencio(200)
    s = registrar_estado(observer, "Despues de recompensa")
    historial_da.append(('recompensa', s['dopamine']))
    historial_peso.append(('recompensa', s['avg_weight']))
    historial_fru.append(('recompensa', s['frustration']))

    print("\nFase: Recuperacion (500 pasos)")
    stim.silencio(500)
    s = registrar_estado(observer, "Recuperacion final")
    historial_da.append(('recuperacion', s['dopamine']))
    historial_peso.append(('recuperacion', s['avg_weight']))
    historial_fru.append(('recuperacion', s['frustration']))

    print("\n" + "="*50)
    print("  RESUMEN EXPERIMENTO 1")
    print("="*50)
    print(f"  {'Fase':<20s} {'DA':>8s} {'Peso Prom':>10s} {'Frustracion':>12s}")
    print(f"  {'-'*50}")
    for i, (fase, da) in enumerate(historial_da):
        peso = historial_peso[i][1]
        fru = historial_fru[i][1]
        print(f"  {fase:<20s} {da:>8.4f} {peso:>10.4f} {fru:>12.4f}")

    logger.guardar("exp1_novedad")
    cerebro.save_brain("logs/exp1/cerebro_exp1.pkl")
    print("\nExperimento 1 completado.")


# ============================================================================
# EXPERIMENTO 2: Frustracion y poda neuronal
# ============================================================================

def experimento_2():
    separador("EXPERIMENTO 2: Frustracion y poda neuronal")
    print("Objetivo: Observar como la frustracion crece con errores")
    print("          repetidos y como la microglia poda sinapsis.\n")

    cerebro = BrainSimulation(name="Exp2")
    observer = BrainObserver(cerebro)
    stim = StimulusController(cerebro)
    logger = EventLogger(cerebro, directorio="logs/exp2")
    cerebro._logger = logger

    historial = []

    print("Fase 1: Basal (300 pasos)")
    stim.silencio(300)
    s = registrar_estado(observer, "Estado basal")
    historial.append({
        'fase': 'basal',
        'neuronas': s['total_neurons'],
        'sinapsis': s['total_synapses'],
        'da': s['dopamine'],
        'fru': s['frustration'],
        'res': s['resilience'],
        'peso_prom': s['avg_weight'],
    })

    rondas_errores = 10
    for ronda in range(1, rondas_errores + 1):
        print(f"\nRonda de error {ronda}/{rondas_errores}")
        stim.inyectar_error(0.8)
        stim.silencio(200)
        s = registrar_estado(observer, f"Error ronda {ronda}")
        historial.append({
            'fase': f'error_{ronda}',
            'neuronas': s['total_neurons'],
            'sinapsis': s['total_synapses'],
            'da': s['dopamine'],
            'fru': s['frustration'],
            'res': s['resilience'],
            'peso_prom': s['avg_weight'],
        })

    print("\nFase: Recompensa (para ver si se recupera)")
    stim.inyectar_recompensa()
    stim.silencio(300)
    s = registrar_estado(observer, "Despues de recompensa")
    historial.append({
        'fase': 'recompensa',
        'neuronas': s['total_neurons'],
        'sinapsis': s['total_synapses'],
        'da': s['dopamine'],
        'fru': s['frustration'],
        'res': s['resilience'],
        'peso_prom': s['avg_weight'],
    })

    print("\nFase: Recuperacion (500 pasos)")
    stim.silencio(500)
    s = registrar_estado(observer, "Recuperacion final")
    historial.append({
        'fase': 'recuperacion',
        'neuronas': s['total_neurons'],
        'sinapsis': s['total_synapses'],
        'da': s['dopamine'],
        'fru': s['frustration'],
        'res': s['resilience'],
        'peso_prom': s['avg_weight'],
    })

    print("\n" + "="*70)
    print("  RESUMEN EXPERIMENTO 2")
    print("="*70)
    print(f"  {'Fase':<15s} {'Neur':>5s} {'Syn':>5s} {'DA':>7s} {'Fru':>7s} {'Res':>7s} {'w_prom':>8s}")
    print(f"  {'-'*70}")
    for h in historial:
        print(f"  {h['fase']:<15s} {h['neuronas']:>5d} {h['sinapsis']:>5d} "
              f"{h['da']:>7.4f} {h['fru']:>7.4f} {h['res']:>7.4f} {h['peso_prom']:>8.4f}")

    neuronas_inicial = historial[0]['neuronas']
    neuronas_final = historial[-1]['neuronas']
    sinapsis_inicial = historial[0]['sinapsis']
    sinapsis_final = historial[-1]['sinapsis']
    print(f"\n  Cambio neuronas: {neuronas_inicial} -> {neuronas_final} "
          f"({neuronas_final - neuronas_inicial:+d})")
    print(f"  Cambio sinapsis: {sinapsis_inicial} -> {sinapsis_final} "
          f"({sinapsis_final - sinapsis_inicial:+d})")

    logger.guardar("exp2_frustracion")
    cerebro.save_brain("logs/exp2/cerebro_exp2.pkl")
    print("\nExperimento 2 completado.")


# ============================================================================
# EXPERIMENTO 3: Consolidacion en sueno REM
# ============================================================================

def experimento_3():
    separador("EXPERIMENTO 3: Consolidacion en sueno REM")
    print("Objetivo: Inyectar un patron temporal, dejar que el cerebro")
    print("          duerma, y observar si consolida memoria.\n")

    cerebro = BrainSimulation(name="Exp3")
    observer = BrainObserver(cerebro)
    stim = StimulusController(cerebro)
    logger = EventLogger(cerebro, directorio="logs/exp3")
    cerebro._logger = logger

    historial = []

    print("Fase 1: Basal (200 pasos)")
    stim.silencio(200)
    s = registrar_estado(observer, "Estado basal")
    historial.append({
        'fase': 'basal',
        'da': s['dopamine'],
        'buffer_size': len(cerebro._episodic_buffer),
        'brain_state': s['brain_state'],
    })

    print("\nFase 2: Inyectar patron temporal (10 Hz, 5 repeticiones)")
    patron_frecuencia = 10.0
    for rep in range(5):
        print(f"  Repeticion {rep+1}/5")
        stim.inyectar_senal_temporal(patron_frecuencia, duracion_pasos=100, amplitud=500.0)
        stim.silencio(50)
    s = registrar_estado(observer, "Despues de 5 repeticiones de patron")
    historial.append({
        'fase': 'post_patron',
        'da': s['dopamine'],
        'buffer_size': len(cerebro._episodic_buffer),
        'brain_state': s['brain_state'],
    })

    print(f"\n  Buffer episodico: {len(cerebro._episodic_buffer)} entradas")

    print("\nFase 3: Esperar a que entre en sueno lento...")
    while cerebro.brain_state != BrainState.SLOW_WAVE_SLEEP:
        stim.silencio(100)
        if cerebro.step_count > 100000:
            break
    stim.silencio(500)
    s = registrar_estado(observer, "En sueno lento")
    historial.append({
        'fase': 'sueno_lento',
        'da': s['dopamine'],
        'buffer_size': len(cerebro._episodic_buffer),
        'brain_state': s['brain_state'],
    })

    print("\nFase 4: Esperar a REM...")
    while cerebro.brain_state != BrainState.REM:
        stim.silencio(100)
        if cerebro.step_count > 100000:
            break
    stim.silencio(200)
    s = registrar_estado(observer, "En REM")
    historial.append({
        'fase': 'rem',
        'da': s['dopamine'],
        'buffer_size': len(cerebro._episodic_buffer),
        'brain_state': s['brain_state'],
    })

    print("\nFase 5: Despertar y verificar consolidacion")
    while cerebro.brain_state != BrainState.AWAKE:
        stim.silencio(100)
        if cerebro.step_count > 100000:
            break
    s = registrar_estado(observer, "Despierto de nuevo")
    historial.append({
        'fase': 'post_sueno',
        'da': s['dopamine'],
        'buffer_size': len(cerebro._episodic_buffer),
        'brain_state': s['brain_state'],
    })

    print(f"\n  Buffer episodico final: {len(cerebro._episodic_buffer)} entradas")

    print("\nFase 6: Inyectar el mismo patron otra vez para comparar")
    stim.inyectar_senal_temporal(patron_frecuencia, duracion_pasos=100, amplitud=500.0)
    s = registrar_estado(observer, "Segunda exposicion al patron")
    historial.append({
        'fase': 'segunda_exposicion',
        'da': s['dopamine'],
        'buffer_size': len(cerebro._episodic_buffer),
        'brain_state': s['brain_state'],
    })

    print("\n" + "="*60)
    print("  RESUMEN EXPERIMENTO 3")
    print("="*60)
    print(f"  {'Fase':<25s} {'DA':>7s} {'Buffer':>7s} {'Estado':>15s}")
    print(f"  {'-'*60}")
    for h in historial:
        print(f"  {h['fase']:<25s} {h['da']:>7.4f} {h['buffer_size']:>7d} {h['brain_state']:>15s}")

    logger.guardar("exp3_rem")
    cerebro.save_brain("logs/exp3/cerebro_exp3.pkl")
    print("\nExperimento 3 completado.")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  IA VIVA - EXPERIMENTOS DE FASE 2")
    print("=" * 60)
    print("  1. Respuesta de dopamina a novedad")
    print("  2. Frustracion y poda neuronal")
    print("  3. Consolidacion en sueno REM")
    print("=" * 60)

    experimento_1()
    print("\n")
    experimento_2()
    print("\n")
    experimento_3()

    print("\n" + "=" * 60)
    print("  TODOS LOS EXPERIMENTOS COMPLETADOS")
    print("=" * 60)
    print("  Revisa la carpeta logs/ para los CSV.")
    print("  Cada experimento tiene su propia subcarpeta.")
    print("=" * 60)
