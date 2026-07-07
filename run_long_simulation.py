import sys
import os
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from cerebro_unico import BrainUnico

cerebro = BrainUnico()
print("Starting long simulation: 1500 steps (750 seconds of biological time)...")

try:
    for step in range(1, 1501):
        cerebro.step()
        if step % 50 == 0:
            h = cerebro.history[-1]
            print(f"[Step {step}/1500] t={h['time']/1000:.1f}s | State: {h['state']} | "
                  f"Active Synapses: {h['synapses']} | W_Mean: {h['w_mean']:.4f} | "
                  f"DA: {h['da']:.2f} | Energy: {h['energy_mean']:.2%}")
except KeyboardInterrupt:
    print("\nSimulation interrupted.")

# Save long history
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
log_path = os.path.join(LOGS_DIR, "cerebro_unico_long_sim_results.pkl")
with open(log_path, 'wb') as f:
    pickle.dump(cerebro.history, f)
print(f"Results saved to {log_path}")
