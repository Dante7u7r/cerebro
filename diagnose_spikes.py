#!/usr/bin/env python3
"""Diagnóstico de conductancias: ¿cuánta corriente sináptica llegan a las ocultas?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brian2 import *
import numpy as np
prefs.codegen.target = 'numpy'

from cerebro_unico import BrainUnico, N_TOTAL, N_SENSORY, BATCH_MS

cerebro = BrainUnico()

# Inyectar corriente sensorial
cerebro.neurons.I_ext[:N_SENSORY] = 35.0

# Correr 1 batch
cerebro.network.run(BATCH_MS * ms, report=None)

# Verificar conductancias en las ocultas (10-24)
g_ampa_soma_hidden = cerebro.neurons.g_ampa_soma[10:25]
g_gaba_soma_hidden = cerebro.neurons.g_gaba_soma[10:25]
g_ampa_dend_hidden = cerebro.neurons.g_ampa_dend[10:25]
v_hidden = cerebro.neurons.v[10:25]

print("=== Conductancias DESPUÉS de 500ms de estimulación sensorial ===")
print(f"g_ampa_soma (ocultas):  {np.array2string(np.array(g_ampa_soma_hidden), precision=6)}")
print(f"g_ampa_dend (ocultas):  {np.array2string(np.array(g_ampa_dend_hidden), precision=6)}")
print(f"g_gaba_soma (ocultas):  {np.array2string(np.array(g_gaba_soma_hidden), precision=6)}")
print(f"v (ocultas):            {np.array2string(np.array(v_hidden), precision=2)}")
print()

# Verificar pesos sinápticos sensorial -> oculta
i_idx = cerebro.synapses.i[:]
j_idx = cerebro.synapses.j[:]
w = cerebro.synapses.w[:]
is_active = cerebro.synapses.is_active[:]

# Sinapsis activas de sensorial (0-9) a oculta (10-24)
mask_s2h = (i_idx < 10) & (j_idx >= 10) & (j_idx < 25) & (is_active == 1.0)
w_s2h = w[mask_s2h]
print(f"Sinapsis Sensorial -> Oculta activas: {np.sum(mask_s2h)}")
print(f"  Pesos: {np.array2string(w_s2h, precision=3)}")
print(f"  w medio: {np.mean(w_s2h):.3f}" if len(w_s2h) > 0 else "  SIN SINAPSIS")
print()

# Sinapsis activas de oculta (10-24) a motor (25-39)
mask_h2m = (i_idx >= 10) & (i_idx < 25) & (j_idx >= 25) & (j_idx < 40) & (is_active == 1.0)
w_h2m = w[mask_h2m]
print(f"Sinapsis Oculta -> Motor activas: {np.sum(mask_h2m)}")
print(f"  Pesos: {np.array2string(w_h2m, precision=3)}")
print(f"  w medio: {np.mean(w_h2m):.3f}" if len(w_h2m) > 0 else "  SIN SINAPSIS")
print()

# Calcular la corriente sináptica teórica máxima que una sinapsis puede inyectar
# I_syn = g_ampa * (E_ampa - v) = w * release * (0 - (-65)) = w * release * 65
# Con STP: release = u * x, donde u típico ~ 0.2, x ~ 1.0 al inicio
# Entonces I_syn_max_teórica = 0.6 * 0.2 * 1.0 * 65 = 7.8
# Pero necesitamos al menos (v_thresh - v_rest) / (E_ampa - v_rest) * tau_ampa/tau_m para disparar
# = (-55 - (-65)) / (0 - (-65)) * 5/15 = 10/65 * 0.33 = 0.051
# Necesitamos g_ampa_peak >= 0.051 para disparar UNA neurona sin ruido
print("=== Cálculo teórico de umbral de conductancia ===")
print(f"  Para disparar una oculta (v_rest=-65, v_thresh=-55, E_ampa=0):")
print(f"  g_ampa_peak necesaria ≈ (v_thresh - v_rest) * tau_ampa / ((E_ampa - v_rest) * tau_m)")
print(f"  = (-55-(-65)) * 5ms / ((0-(-65)) * 15ms) = 10*5 / (65*15) = {50/(65*15):.4f}")
print(f"  Un spike pre aporta: Δg_ampa = w * u * x * (1+0.6*myel)")
print(f"  Con w=0.6, u=0.2, x=1.0, myel=0.2: Δg = 0.6*0.2*1.0*1.12 = {0.6*0.2*1.0*1.12:.4f}")
print(f"  Necesitamos ~{0.0513/0.1344:.0f} spikes convergentes simultáneos para disparar sin ruido")
