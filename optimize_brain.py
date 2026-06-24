#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OPTIMIZE BRAIN — Optimizador de Hiperparámetros Homeostáticos para Cerebro Único v3.0
===================================================================================
Aplica un algoritmo de búsqueda por Ascenso de Colina (Hill Climbing) para encontrar
los rangos de parámetros que garanticen la estabilidad de disparo (evitar coma/epilepsia),
el equilibrio de energía y la plasticidad de pesos sinápticos en Brian2.
"""

import os
import sys
import numpy as np
import brian2 as b2

# Silenciar advertencias de compilación de Brian2 para no ensuciar la salida
b2.prefs.codegen.target = 'numpy'  # Usar numpy en optimización para ejecuciones ultra-rápidas
b2.BrianLogger.log_level = 'error'

# Asegurar que podemos importar cerebro_unico
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cerebro_unico import BrainUnico, N_TOTAL

# Límites operativos de los parámetros
BOUNDS = {
    'noise_base': (1.2, 3.8),
    'cpg_amplitude': (0.5, 4.5),
    'decay_factor': (0.970, 0.997),
    'target_rate': (4.0, 14.0)
}

def evaluar_fitness(params):
    """Ejecuta una simulación corta y calcula la salud biológica de la red (fitness)."""
    noise_base = params['noise_base']
    cpg_amplitude = params['cpg_amplitude']
    decay_factor = params['decay_factor']
    target_rate = params['target_rate']
    
    try:
        # Instanciar el cerebro en modo silencioso (sin cargar pickle anterior para evaluar desde cero)
        brain = BrainUnico(state_path=None)
        
        # Inyectar parámetros candidatos
        brain.neurons.noise_base = noise_base
        brain.neurons.cpg_amplitude = cpg_amplitude
        brain.decay_factor = decay_factor
        brain.gain_control.target_rate = target_rate
        
        # Correr simulación corta: 30 pasos (15 segundos biológicos)
        # En modo numpy esto toma menos de 0.2 segundos de CPU
        for _ in range(30):
            brain.step()
            
        # Extraer métricas finales (últimos 10 pasos de simulación para estabilidad)
        historico_reciente = brain.history[-10:]
        
        # 1. Tasa de disparo promedio de la red (punto óptimo en 6.0 Hz)
        avg_firing_rates = [np.mean(brain.neurons.firing_rate[:] / b2.Hz) for _ in range(10)]
        mean_firing = float(np.mean(avg_firing_rates))
        
        # Penalización cuadrática muy dura si la red entra en coma (<2 Hz) o epilepsia (>15 Hz)
        loss_fr = 0.0
        if mean_firing < 3.0:
            loss_fr = (3.0 - mean_firing)**2 * 35.0
        elif mean_firing > 11.0:
            loss_fr = (mean_firing - 11.0)**2 * 35.0
        else:
            loss_fr = (mean_firing - 6.5)**2 * 2.0  # Penalización suave cerca del centro
            
        # 2. Energía metabólica media (punto óptimo > 80%)
        energies = [h['energy_mean'] for h in historico_reciente]
        mean_energy = float(np.mean(energies))
        loss_energy = 0.0
        if mean_energy < 0.75:
            loss_energy = (0.75 - mean_energy)**2 * 150.0
            
        # 3. Pesos sinápticos (queremos evitar que colapsen a 0 o se saturen a 2.0)
        w_all = brain.synapses.w[:]
        w_active = w_all[w_all > 0.02]
        loss_w = 0.0
        if len(w_active) == 0:
            loss_w = 500.0  # Penalización por muerte sináptica total
        else:
            w_mean = float(np.mean(w_active))
            if w_mean < 0.35:
                loss_w = (0.35 - w_mean)**2 * 200.0
            elif w_mean > 0.95:
                loss_w = (w_mean - 0.95)**2 * 200.0
                
        # 4. Densidad de sinapsis activas (punto óptimo en 550 conexiones activas)
        n_active = len(w_active)
        loss_syn = ((n_active - 550) / 100.0)**2 * 5.0
        
        # 5. Desvío de predicción (capacidad de aprendizaje de la onda)
        errors = [abs(h['target'] - h['prediction']) for h in historico_reciente]
        mean_error = float(np.mean(errors))
        loss_pred = mean_error * 15.0
        
        # Fitness total (mientras más cercano a 0, mejor)
        fitness = -(loss_fr + loss_energy + loss_w + loss_syn + loss_pred)
        
        # Debugging de métricas internas
        info = {
            'fr': mean_firing,
            'energy': mean_energy,
            'w_mean': w_mean if len(w_active) > 0 else 0.0,
            'syns': n_active,
            'pred_err': mean_error
        }
        
        return fitness, info
    except Exception as e:
        # En caso de error de cálculo, penalizar con fitness muy bajo
        return -9999.0, {'error': str(e)}

def perturbar_parametro(valor, bounds, scale):
    """Aplica una perturbación aleatoria gaussiana y mantiene el valor dentro de los límites."""
    min_b, max_b = bounds
    rango = max_b - min_b
    nuevo_val = valor + np.random.normal(0, scale * rango)
    return float(np.clip(nuevo_val, min_b, max_b))

def main():
    print("=" * 80)
    print("  INICIANDO OPTIMIZADOR HOMEOSTÁTICO DE CEREBRO ÚNICO v3.0 (Hill Climbing)")
    print("=" * 80)
    
    # Parámetros de arranque (los actuales en producción)
    actual_params = {
        'noise_base': 2.2,
        'cpg_amplitude': 2.5,
        'decay_factor': 0.985,
        'target_rate': 8.0
    }
    
    print("[i] Evaluando configuración actual de control...")
    actual_fit, actual_info = evaluar_fitness(actual_params)
    print(f"    Fitness Inicial : {actual_fit:.2f}")
    print(f"    Métricas        : FR={actual_info['fr']:.2f} Hz | Energía={actual_info['energy']:.1%} | "
          f"W Mean={actual_info['w_mean']:.3f} | Sinapsis={actual_info['syns']} | Pred Error={actual_info['pred_err']:.3f}")
    
    best_params = actual_params.copy()
    best_fit = actual_fit
    best_info = actual_info
    
    # Ejecutaremos 35 iteraciones de optimización
    max_iterations = 35
    print(f"\n[i] Iniciando optimización por 35 ciclos...")
    
    # Escala de perturbación (comienza grande y se enfría lentamente)
    perturb_scale = 0.12
    
    mejoras = 0
    for i in range(1, max_iterations + 1):
        # Enfriamiento de la escala de perturbación
        current_scale = perturb_scale * (1.0 - (i / max_iterations) * 0.5)
        
        # Crear candidato perturbando un parámetro a la vez de forma aleatoria
        candidato = best_params.copy()
        param_to_mutate = np.random.choice(list(BOUNDS.keys()))
        candidato[param_to_mutate] = perturbar_parametro(
            best_params[param_to_mutate], 
            BOUNDS[param_to_mutate], 
            current_scale
        )
        
        # Evaluar
        fit, info = evaluar_fitness(candidato)
        
        if fit > best_fit:
            best_fit = fit
            best_params = candidato.copy()
            best_info = info
            mejoras += 1
            print(f"  [Ciclo {i:02d}] *MEJORA* Fitness={fit:.2f} | Mutado: {param_to_mutate}={candidato[param_to_mutate]:.4f}")
            print(f"             FR={info['fr']:.2f} Hz | Energía={info['energy']:.1%} | W Mean={info['w_mean']:.3f} | Sinapsis={info['syns']}")
        else:
            if i % 5 == 0:
                print(f"  [Ciclo {i:02d}] Evaluando... Fitness actual={best_fit:.2f} (candidato fallido={fit:.2f})")
                
    print("\n" + "=" * 80)
    print("  OPTIMIZACIÓN COMPLETADA CON ÉXITO")
    print("=" * 80)
    print(f"  Total mejoras encontradas: {mejoras}")
    print(f"  Fitness final alcanzado  : {best_fit:.2f} (Mejora: {best_fit - actual_fit:+.2f})\n")
    
    print("  PARÁMETROS RECOMENDADOS (ÓPTIMOS):")
    print(f"    * noise_base    = {best_params['noise_base']:.4f}  (Antes: {actual_params['noise_base']})")
    print(f"    * cpg_amplitude = {best_params['cpg_amplitude']:.4f}  (Antes: {actual_params['cpg_amplitude']})")
    print(f"    * decay_factor  = {best_params['decay_factor']:.4f}  (Antes: {actual_params['decay_factor']})")
    print(f"    * target_rate   = {best_params['target_rate']:.4f}  (Antes: {actual_params['target_rate']})\n")
    
    print("  MÉTRICAS RESULTANTES DE SALUD NEURONAL:")
    print(f"    * Firing rate medio  : {best_info['fr']:.2f} Hz (Rango sano: 3-10 Hz)")
    print(f"    * Energía media final: {best_info['energy']:.2%}")
    print(f"    * Peso sináptico medio: {best_info['w_mean']:.4f}")
    print(f"    * Sinapsis activas    : {best_info['syns']} conexiones")
    print(f"    * Error de predicción : {best_info['pred_err']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
