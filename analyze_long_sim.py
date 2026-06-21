import pickle
import numpy as np
import os
import sys

LOGS_DIR = r"C:\Users\maruc\Proyectos\trabajando\cerebro-main\logs"
pkl_path = os.path.join(LOGS_DIR, "cerebro_unico_sim_results.pkl")

if not os.path.exists(pkl_path):
    print(f"Error: {pkl_path} no existe.")
    sys.exit(1)

with open(pkl_path, 'rb') as f:
    history = pickle.load(f)

print(f"Historial cargado con {len(history)} registros.")
if len(history) == 0:
    print("El historial está vacío.")
    sys.exit(0)

# 1. Metricas Generales
first = history[0]
last = history[-1]
print(f"Rango de Tiempo Simulado: {first['time']/1000:.1f}s - {last['time']/1000:.1f}s (Duración: {last['time']/1000 - first['time']/1000:.1f}s) | Pasos en historial: {len(history)}")
print(f"Sinapsis activas finales: {last['synapses']}")
print(f"Pesos sinápticos finales: promedio={last['w_mean']:.4f}, max={last['w_max']:.4f}")
print(f"Total spikes: {last['spikes']}")
print(f"Poda total: {last['pruned']} | Creadas total: {last['created']}")

# 2. Análisis de Aprendizaje por Bloques (MSE de predicción durante AWAKE)
print("\n--- Rendimiento de Aprendizaje por Bloques (MSE en AWAKE) ---")
block_size = 300
n_blocks = len(history) // block_size
for b in range(n_blocks):
    start_idx = b * block_size
    end_idx = (b + 1) * block_size
    block_data = history[start_idx:end_idx]
    
    # Calcular MSE de predicción durante estado AWAKE
    awake_block = [h for h in block_data if h['state'] == 'AWAKE']
    if len(awake_block) > 0:
        errors = [h['target'] - h['prediction'] for h in awake_block]
        mse = np.mean([e**2 for e in errors])
        mean_da = np.mean([h['da'] for h in awake_block])
        mean_w = np.mean([h['w_mean'] for h in awake_block])
        print(f"Bloque {b+1} ({block_data[0]['time']/1000:.1f}s - {block_data[-1]['time']/1000:.1f}s): MSE={mse:.4f} | DA Promedio={mean_da:.3f} | W_Mean Promedio={mean_w:.4f}")
    else:
        print(f"Bloque {b+1}: Sin periodos AWAKE")

# 3. Análisis de Retención / Olvido en Sueño
# Buscamos transiciones de AWAKE -> SWS -> REM -> AWAKE
cycles = []
current_cycle = {}
for i in range(1, len(history)):
    prev_h = history[i-1]
    curr_h = history[i]
    
    # Transición a SWS (comienzo del sueño)
    if prev_h['state'] == 'AWAKE' and curr_h['state'] == 'SLOW_WAVE_SLEEP':
        current_cycle['awake_end_w'] = prev_h['w_mean']
        current_cycle['awake_end_step'] = prev_h['step']
    
    # Transición de SWS a REM
    if prev_h['state'] == 'SLOW_WAVE_SLEEP' and curr_h['state'] == 'REM':
        current_cycle['sws_end_w'] = prev_h['w_mean']
        
    # Transición de REM a AWAKE (despertar)
    if prev_h['state'] == 'REM' and curr_h['state'] == 'AWAKE':
        current_cycle['rem_end_w'] = prev_h['w_mean']
        current_cycle['awake_start_step'] = curr_h['step']
        cycles.append(current_cycle)
        current_cycle = {}

print("\n--- Olvido y Consolidación durante los Ciclos de Sueño ---")
if len(cycles) > 0:
    print(f"{'Ciclo':<6} | {'Paso':<12} | {'W Antes Sueño':<14} | {'W Post-SWS':<12} | {'W Post-REM':<12} | {'Pérdida %':<10}")
    print("-" * 75)
    for idx, c in enumerate(cycles[:10]):  # Mostrar los primeros 10 ciclos
        if 'awake_end_w' in c and 'sws_end_w' in c and 'rem_end_w' in c:
            loss = (c['awake_end_w'] - c['rem_end_w']) / c['awake_end_w'] * 100 if c['awake_end_w'] > 0 else 0
            print(f"{idx+1:<6} | {c['awake_end_step']:>4}->{c['awake_start_step']:<6} | {c['awake_end_w']:<14.4f} | {c['sws_end_w']:<12.4f} | {c['rem_end_w']:<12.4f} | {loss:<9.1f}%")
    if len(cycles) > 10:
        print(f"... y {len(cycles) - 10} ciclos más.")
else:
    print("No se pudieron detectar ciclos de sueño completos en las transiciones de estado.")

# 4. Dinámicas de Energía y Frustración
mean_energy = np.mean([h['energy_mean'] for h in history])
min_energy = np.min([h['energy_mean'] for h in history])
print(f"\n--- Métricas de Estabilidad Homeostática ---")
print(f"Energía promedio global: {mean_energy:.2%}")
print(f"Energía mínima registrada: {min_energy:.2%}")
