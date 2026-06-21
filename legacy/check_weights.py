import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from cerebro_unico import BrainUnico

def main():
    state_file = os.path.join(SCRIPT_DIR, "logs", "cerebro_unico_state.pkl")
    brain = BrainUnico(state_path=state_file)
    brain.scaler.active = True
    brain.gain_control.active = True
    
    print("=== DIAGNÓSTICO PASO A PASO ===")
    w_init = brain.synapses.w[:]
    active_syns = np.sum(brain.synapses.is_active[:] == 1.0)
    w_mean = np.mean(w_init[brain.synapses.is_active[:] == 1.0])
    print(f"Inicio | Sinapsis Activas: {active_syns} | W_Mean: {w_mean:.4f} | Max W: {np.max(w_init):.4f}")
    
    for i in range(1, 101):
        brain.step()
        w = brain.synapses.w[:]
        active_syns = np.sum(brain.synapses.is_active[:] == 1.0)
        w_mean = np.mean(w[brain.synapses.is_active[:] == 1.0]) if active_syns > 0 else 0.0
        motor_firing = float(np.mean(brain.neurons.firing_rate[brain.layer_indices == 2] / 1.0))
        print(f"Paso {i:02d} | Estado: {brain.brain_state} | Sinapsis Activas: {active_syns} | W_Mean: {w_mean:.4f} | Motor Firing: {motor_firing:.2f} Hz | V_offset: {brain.gain_control.v_offset:.3f}")
        if active_syns == 0:
            print("[ALERTA] ¡Colapso completo de sinapsis en el paso", i)
            break

if __name__ == '__main__':
    main()
