#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from brian2 import *

# Asegurar importación de cerebro_unico
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from cerebro_unico import BrainUnico

def run_test():
    state_file = os.path.join(SCRIPT_DIR, "logs", "cerebro_unico_state.pkl")
    if not os.path.exists(state_file):
        print(f"[ERROR] No se encontró el estado pre-entrenado en: {state_file}")
        sys.exit(1)
        
    print("=== [TEST] CARGANDO CEREBRO EN ESTADO BASE (SIN REGULACIÓN) ===")
    brain = BrainUnico(state_path=state_file)
    
    # Asegurar que por defecto vienen inactivos
    brain.scaler.active = False
    brain.gain_control.active = False
    
    # 1. Medir pesos entrantes de excitación antes del escalado
    w = brain.synapses.w[:]
    j_indices = brain.synapses.j[:]
    is_exc = brain.synapses.is_excitatory[:] == 1.0
    is_active = brain.synapses.is_active[:] == 1.0
    valid_exc_mask = is_exc & is_active
    
    max_incoming_w = 0.0
    for neuron_id in range(len(brain.neurons)):
        syn_indices = np.where((j_indices == neuron_id) & valid_exc_mask)[0]
        if len(syn_indices) > 0:
            sum_w = np.sum(w[syn_indices])
            if sum_w > max_incoming_w:
                max_incoming_w = sum_w
                
    print(f"[*] Excitación entrante máxima inicial: {max_incoming_w:.4f}")
    print(f"[*] Estado cerebral inicial: {brain.brain_state}")
    
    # Forzar estado AWAKE para probar control de ganancia bajo estímulo
    # Modificamos el tiempo de la simulación para estar en fase AWAKE (fase < 14000 ms)
    # SLEEP_CYCLE_PERIOD = 20000.0 (14s AWAKE)
    brain.time = 0.0  # Resetear tiempo local para forzar AWAKE
    
    print("\n=== [TEST] ACTIVANDO CONTROLES DE REGULACIÓN BIOLÓGICA ===")
    brain.scaler.active = True
    brain.gain_control.active = True
    brain.scaler.target_sum_w = 6.0
    brain.gain_control.target_rate = 8.0
    brain.gain_control.alpha_gain = 0.05
    
    # Correr 30 pasos (15 segundos biológicos de simulación)
    print("Corriendo 30 pasos de simulación con regulación activa...")
    for step_idx in range(30):
        brain.step()
        
        motor_firing = float(np.mean(brain.neurons.firing_rate[brain.layer_indices == 2] / Hz))
        print(f"Paso {brain.step_count:02d} | Firing Motor: {motor_firing:5.2f} Hz | V_offset: {brain.gain_control.v_offset:6.3f} mV | Predicción: {brain.history[-1]['prediction']:5.3f} | Target: {brain.history[-1]['target']:5.3f}")

    # Forzar una última escala sináptica manual para evaluar la acción directa del scaler
    brain.scaler.scale(brain)

    # 2. Medir pesos entrantes tras escalado L1
    w_final = brain.synapses.w[:]
    max_incoming_w_final = 0.0
    for neuron_id in range(len(brain.neurons)):
        syn_indices = np.where((j_indices == neuron_id) & valid_exc_mask)[0]
        if len(syn_indices) > 0:
            sum_w = np.sum(w_final[syn_indices])
            if sum_w > max_incoming_w_final:
                max_incoming_w_final = sum_w
                
    print("\n=== [TEST] RESULTADOS DE VALIDACIÓN ===")
    print(f"[*] Excitación entrante máxima post-regulada (forzada): {max_incoming_w_final:.4f} (Objetivo: <= {brain.scaler.target_sum_w})")
    print(f"[*] Offset final del umbral motor: {brain.gain_control.v_offset:.4f} mV")
    
    # Validaciones asertivas
    assert max_incoming_w_final <= brain.scaler.target_sum_w + 1e-5, "¡Error! El escalado L1 no limitó la excitación entrante."
    print("[OK] Verificación de SynapticScaler exitosa (Excitación acotada).")
    
    # Verificar que el offset de umbral se aplicó a v_thresh_base de Brian2
    motor_indices = np.where(brain.layer_indices == 2)[0]
    expected_v_thresh_base = brain.v_thresh_arr[motor_indices] + brain.gain_control.v_offset
    actual_v_thresh_base = brain.neurons.v_thresh_base[motor_indices]
    assert np.allclose(actual_v_thresh_base, expected_v_thresh_base), "¡Error! El offset de umbral no se aplicó al NeuronGroup de Brian2."
    print("[OK] Verificación de GainController en Brian2 exitosa (Offset de umbral aplicado).")
    
    print("\n[ÉXITO] Todas las validaciones dinámicas y estructurales pasaron con éxito.")

if __name__ == '__main__':
    run_test()
