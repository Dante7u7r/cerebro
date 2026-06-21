import time as time_mod
import pickle
import os
import sys
import numpy as np

# Asegurar que podemos importar cerebro_unico
sys.path.insert(0, r"C:\Users\maruc\Proyectos\trabajando\cerebro-main")
from cerebro_unico import BrainUnico, LOGS_DIR

# 2 horas = 7200 segundos de tiempo real
DURATION_SECONDS = 7200 
state_path = os.path.join(LOGS_DIR, "cerebro_unico_state.pkl")

print("=" * 80)
print(f"  INICIANDO EJECUCIÓN CRÍTICA TEMPORIZADA (Límite: {DURATION_SECONDS/3600:.1f} horas de tiempo real)")
print("=" * 80)

# Cargar cerebro con persistencia
cerebro = BrainUnico(state_path=state_path)

start_real_time = time_mod.time()
end_real_time = start_real_time + DURATION_SECONDS
step = 0

print(f"\n[INFO] Simulación en marcha. Se guardarán checkpoints cada 500 pasos.")
print(f"       Presiona Ctrl+C en la terminal de la tarea si deseas detenerla antes de tiempo.")
print("-" * 80)

try:
    while time_mod.time() < end_real_time:
        step += 1
        cerebro.step()
        
        # Reportar cada 100 pasos
        if step % 100 == 0:
            elapsed = time_mod.time() - start_real_time
            remaining = end_real_time - time_mod.time()
            h = cerebro.history[-1]
            print(f"[Paso {h['step']}] | Biol: {h['time']/1000:.1f}s | Real Transcurrido: {elapsed/60:.1f} min | "
                  f"Restante: {remaining/60:.1f} min | Sinapsis: {h['synapses']} | "
                  f"W_Mean: {h['w_mean']:.4f} | DA: {h['da']:.2f} | Frust: {cerebro.frustration:.2f} | "
                  f"Energy: {h['energy_mean']:.2%}")
            
        # Checkpoint cada 500 pasos
        if step % 500 == 0:
            print(f"[CHECKPOINT] Guardando estado en paso {cerebro.step_count}...")
            cerebro.save_state(state_path)
            
except KeyboardInterrupt:
    print("\n[WARN] Simulación interrumpida por el usuario.")
finally:
    # Guardar estado final
    print("\n[INFO] Finalizando simulación. Guardando estado y resultados finales...")
    cerebro.save_state(state_path)
    
    # Guardar historial
    log_path = os.path.join(LOGS_DIR, "cerebro_unico_sim_results.pkl")
    try:
        with open(log_path, 'wb') as f:
            pickle.dump(cerebro.history, f)
        print(f"[SAVE] Historial de simulación guardado en: {log_path}")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar el historial: {e}")
        
    print("=" * 80)
    print("  SIMULACIÓN TIMED COMPLETADA")
    print("=" * 80)
