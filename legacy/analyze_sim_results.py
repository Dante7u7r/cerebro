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
if len(history) > 0:
    print("\n--- Detalle de Pasos 15 a 45 ---")
    print(f"{'Paso':<6} | {'Estado':<10} | {'W_Mean':<7} | {'W_Max':<7} | {'Energy':<7} | {'Spikes':<8} | {'Dopamina':<8} | {'Frust':<7}")
    print("-" * 85)
    for idx in range(14, min(45, len(history))):
        h = history[idx]
        print(f"{h['step']:<6} | {h['state']:<10} | {h['w_mean']:<7.4f} | {h['w_max']:<7.4f} | {h['energy_mean']:<7.2%} | {h['spikes']:<8} | {h['da']:<8.4f} | {h['prediction']:<7.4f}")


