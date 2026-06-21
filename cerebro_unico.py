#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CEREBRO ÚNICO v2.0 — Simulador de Spiking Neurobiológico Unificado en Brian2
==========================================================================
Integra:
1. LIF Multicompartimental (Soma-Dendrita) con conductancias AMPA/GABA.
2. STP (Tsodyks-Markram) + Metaplasticidad STDP de 3 neuromoduladores (DA, 5-HT, ACh).
3. Ciclo Sueño-Vigilia (AWAKE / SWS / REM) con Replay REM de ráfagas Poisson.
4. Umbral Adaptativo dinámico (Plasticidad Intrínseca).
5. Resonancia Estocástica Dinámica (Ruido Adaptativo según la Frustración) y Ritmo Schumann CPG.
6. Plasticidad Estructural (Poda de w<0.01 y sinaptogénesis activa).
7. Homeostasis Vectorizada en NumPy.
8. Servidor HTTP integrado que sirve un Dashboard 3D interactivo en Three.js.
"""

from brian2 import *
import numpy as np
import os, sys, pickle, json, threading
import time as time_mod
from collections import deque
import http.server
import socketserver

# Forzar backend de numpy en Windows para evadir bloqueos de compilador/AppLocker
if sys.platform.startswith('win'):
    prefs.codegen.target = 'numpy'
else:
    prefs.codegen.target = 'auto'

# Resolver rutas absolutas relativas al archivo del script para robustez
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(SCRIPT_DIR, "web_visualizer")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")

# ============================================================================
# CONFIGURACIÓN DE PARÁMETROS
# ============================================================================
DT = 1.0 * ms
defaultclock.dt = DT
BATCH_MS = 500.0  # ms por paso de simulación
SLEEP_CYCLE_PERIOD = 20000.0  # 20s de simulación por ciclo (14s AWAKE, 4s SWS, 2s REM)
SIGNAL_PERIOD = 127.7  # Periodo de oscilación de Schumann (ms)

# Topología 3D
N_SENSORY = 10
N_HIDDEN = 15
N_MOTOR = 15
N_PFC = 10
N_TOTAL = N_SENSORY + N_HIDDEN + N_MOTOR + N_PFC

np.random.seed(42)

# ============================================================================
# CLASE DEL SERVIDOR HTTP INTEGRADO
# ============================================================================
class BrainHTTPServer:
    """Servidor web multihilo para servir el visualizador web 3D interactivo."""
    def __init__(self, directory, port=8000):
        self.directory = os.path.abspath(directory)
        self.port = port
        
        # Clase interna para asegurar que sirve la carpeta correcta
        class CustomHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)
                
            def end_headers(self):
                # Deshabilitar cache para sim_state.json
                self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
                super().end_headers()
                
            def log_message(self, format, *args):
                # Suprimir logs HTTP redundantes en la consola
                pass
                
        socketserver.TCPServer.allow_reuse_address = True
        self.httpd = socketserver.TCPServer(("", self.port), CustomHandler)
        self.thread = threading.Thread(target=self.httpd.serve_forever)
        self.thread.daemon = True
        
    def start(self):
        self.thread.start()
        print(f"[HTTP] Servidor en ejecución en http://localhost:{self.port} (Directorio: {self.directory})")
        
    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        print("[HTTP] Servidor detenido.")

# ============================================================================
# CLASE DEL SISTEMA NEUROMODULADOR
# ============================================================================
class NeuromodulatorSystem:
    """Modulación continua de Dopamina, Serotonina y Acetilcolina."""
    def __init__(self):
        self.dopamine = 0.5
        self.serotonin = 0.5
        self.acetylcholine = 0.5
        self.tau_da = 2.0  # s
        self.tau_5ht = 8.0 # s
        self.tau_ach = 1.0 # s
        
    def update(self, dt_sec, brain_state, prediction_error):
        # Dopamina responde al error de predicción (novedad/sorpresa)
        da_target = float(np.clip(0.5 + 0.5 * np.tanh(abs(prediction_error) * 2.0), 0.0, 1.0))
        if brain_state == 'SLOW_WAVE_SLEEP':
            da_target = 0.2
        elif brain_state == 'REM':
            da_target = 0.6
            
        # Serotonina y Acetilcolina regulan estado de vigilia y sueño
        ser_target = 0.7 if brain_state == 'AWAKE' else (0.4 if brain_state == 'SLOW_WAVE_SLEEP' else 0.1)
        ach_target = 0.8 if brain_state == 'AWAKE' else (0.2 if brain_state == 'SLOW_WAVE_SLEEP' else 0.7)
        
        # Integrar decaimiento exponencial simple
        self.dopamine += (da_target - self.dopamine) * dt_sec / self.tau_da
        self.serotonin += (ser_target - self.serotonin) * dt_sec / self.tau_5ht
        self.acetylcholine += (ach_target - self.acetylcholine) * dt_sec / self.tau_ach
        
        self.dopamine = np.clip(self.dopamine, 0.0, 1.0)
        self.serotonin = np.clip(self.serotonin, 0.0, 1.0)
        self.acetylcholine = np.clip(self.acetylcholine, 0.0, 1.0)

# ============================================================================
# REGULADORES HOMEOSTÁTICOS EVOLUTIVOS (BIOLÓGICOS)
# ============================================================================
class SynapticScaler:
    """Modulador astrocítico lento. Normaliza de forma descentralizada (neurona por neurona)
    la suma de pesos excitatorios entrantes para evitar la saturación de excitación.
    """
    def __init__(self, target_sum_w=6.0, active=False):
        self.target_sum_w = target_sum_w
        self.active = active
        
    def scale(self, brain):
        """Suma los pesos de las sinapsis excitatorias entrantes activas de cada neurona
        y las escala si superan el umbral target_sum_w.
        """
        w = brain.synapses.w[:]
        j_indices = brain.synapses.j[:]
        is_exc = brain.synapses.is_excitatory[:] == 1.0
        is_active = brain.synapses.is_active[:] == 1.0
        
        valid_syn_mask = is_exc & is_active
        if not np.any(valid_syn_mask):
            return
            
        w_new = w.copy()
        for neuron_id in range(N_TOTAL):
            syn_indices = np.where((j_indices == neuron_id) & valid_syn_mask)[0]
            if len(syn_indices) == 0:
                continue
            sum_w = np.sum(w[syn_indices])
            if sum_w > self.target_sum_w:
                factor = self.target_sum_w / sum_w
                w_new[syn_indices] *= factor
                
        brain.synapses.w = np.clip(w_new, 0.0, 2.0)


class GainController:
    """Modulador tálamo-cortical lento. Adapta el umbral de disparo (plasticidad intrínseca)
    de la capa motora basándose en el desvío respecto a una tasa de disparo objetivo.
    """
    def __init__(self, target_rate=8.0, alpha_gain=0.05, active=False):
        self.target_rate = target_rate
        self.alpha_gain = alpha_gain
        self.active = active
        self.v_offset = 0.0
        
    def adapt(self, brain, motor_firing):
        """Ajusta el offset del umbral base de la capa motora para mantener la tasa de disparo
        dentro del rango operativo.
        """
        delta_offset = self.alpha_gain * (motor_firing - self.target_rate)
        self.v_offset = np.clip(self.v_offset + delta_offset, -5.0, 10.0)
        
        motor_mask = brain.layer_indices == 2
        new_v_thresh_base = brain.v_thresh_arr.copy()
        new_v_thresh_base[motor_mask] += self.v_offset
        brain.neurons.v_thresh_base = new_v_thresh_base


# ============================================================================
# CLASE PRINCIPAL: CEREBRO ÚNICO
# ============================================================================
class BrainUnico:
    def __init__(self, state_path=None):
        self.time = 0.0
        self.step_count = 0
        self.brain_state = 'AWAKE'
        self.pruned_synapses = 0
        self.created_synapses = 0
        
        # Asignar capas, subtipos y coordenadas 3D de las neuronas
        self.layer_indices = np.zeros(N_TOTAL, dtype=int)
        self.layer_indices[N_SENSORY:N_SENSORY+N_HIDDEN] = 1 # Oculta
        self.layer_indices[N_SENSORY+N_HIDDEN:N_SENSORY+N_HIDDEN+N_MOTOR] = 2 # Motor/Readout
        self.layer_indices[N_SENSORY+N_HIDDEN+N_MOTOR:] = 3 # PFC
        
        self.neuron_types = np.ones(N_TOTAL, dtype=int) # 1 = Excitatoria, 4 = Inhibitoria
        
        # 20% de las ocultas y de PFC son inhibitorias
        h_idx = np.where(self.layer_indices == 1)[0]
        p_idx = np.where(self.layer_indices == 3)[0]
        
        self.neuron_types[h_idx[-3:]] = 4
        self.neuron_types[p_idx[-2:]] = 4
        
        # Asignar coordenadas 3D para visualización y conectividad
        self.x = np.random.uniform(-40, 40, N_TOTAL)
        self.y = np.random.uniform(-40, 40, N_TOTAL)
        self.z = np.zeros(N_TOTAL)
        self.z[self.layer_indices == 0] = 5.0   # Sensorial
        self.z[self.layer_indices == 1] = 25.0  # Oculta
        self.z[self.layer_indices == 2] = 45.0  # Motor
        self.z[self.layer_indices == 3] = 65.0  # PFC
        
        # Parámetros LIF individuales para umbral base
        self.v_thresh_arr = np.zeros(N_TOTAL)
        self.v_thresh_arr[self.neuron_types == 1] = -55.0
        self.v_thresh_arr[self.neuron_types == 4] = -57.0
        
        self.tau_m_arr = np.zeros(N_TOTAL)
        self.tau_m_arr[self.neuron_types == 1] = 15.0 # ms
        self.tau_m_arr[self.neuron_types == 4] = 10.0 # ms
        self.tau_m_arr = self.tau_m_arr * ms
        
        # ==========================================
        # ECUACIONES DE BRIAN2 EN HILO INTEGRADO
        # ==========================================
        
        # Neurona LIF Multicompartimental con Umbral Adaptativo, Ruido Dinámico y Ritmo CPG
        eqs_neurons = '''
        dv/dt = (-(v - v_rest) + I_syn_soma + I_coupling + I_ext + I_cpg) / tau_m + noise_base * (1.0 + 1.2 * frustration) * (0.8 + 0.4 * ach) * (0.2 + 0.8 * energy) * (tau_m**-0.5) * xi : 1 (unless refractory)
        dv_dend/dt = (-(v_dend - v_rest) + I_syn_dend) / tau_dend : 1
        
        I_coupling = g_coupling * (v_dend - v) : 1
        I_syn_soma = g_ampa_soma * (E_ampa - v) + g_gaba_soma * (E_gaba - v) : 1
        I_syn_dend = g_ampa_dend * (E_ampa - v_dend) + g_gaba_dend * (E_gaba - v_dend) : 1
        
        dg_ampa_soma/dt = -g_ampa_soma / tau_ampa : 1
        dg_gaba_soma/dt = -g_gaba_soma / tau_gaba : 1
        dg_ampa_dend/dt = -g_ampa_dend / tau_ampa : 1
        dg_gaba_dend/dt = -g_gaba_dend / tau_gaba : 1
        
        # Umbral Adaptativo (Plasticidad Intrínseca acoplada metabólicamente)
        dv_thresh/dt = (v_thresh_base_effective - v_thresh) / tau_thresh : 1
        v_thresh_base_effective = v_thresh_base + 15.0 * (1.0 - energy)**2 : 1
        v_thresh_base : 1
        tau_thresh : second
        delta_v_thresh : 1
        
        # Ruido Dinámico Adaptativo base
        noise_base : 1
        
        # Ritmo Sincrónico CPG (Portadora Sincronizante)
        I_cpg = cpg_amplitude * sin(2 * pi * t / signal_period) * is_sensory : 1
        is_sensory : 1
        cpg_amplitude : 1
        signal_period : second
        
        # Neuromodulación continua (copiada por el script en cada paso)
        da : 1
        ser : 1
        ach : 1
        
        v_rest : 1
        tau_m : second
        tau_dend : second
        g_coupling : 1
        E_ampa : 1
        E_gaba : 1
        
        I_ext : 1
        
        energy : 1
        firing_rate : Hz
        last_spike : second
        frustration : 1
        resilience : 1
        '''
        
        self.neurons = NeuronGroup(
            N_TOTAL, eqs_neurons,
            threshold='v > v_thresh',
            reset='v = v_rest; last_spike = t; v_thresh = clip(v_thresh + delta_v_thresh, -60.0, -35.0)',
            refractory=2 * ms,
            method='euler',
            namespace={
                'tau_ampa': 5.0 * ms,
                'tau_gaba': 10.0 * ms,
                'tau_da': 2.0 * second,
                'tau_5ht': 8.0 * second,
                'tau_ach': 1.0 * second
            },
            name='neurons'
        )
        
        self.neurons.v_rest = -65.0
        self.neurons.v = -65.0 + np.random.randn(N_TOTAL) * 3.0
        self.neurons.v_dend = -65.0 + np.random.randn(N_TOTAL) * 3.0
        self.neurons.tau_m = self.tau_m_arr
        self.neurons.tau_dend = 30.0 * ms
        self.neurons.g_coupling = 0.15
        self.neurons.E_ampa = 0.0
        self.neurons.E_gaba = -75.0
        self.neurons.da = 0.5
        self.neurons.ser = 0.5
        self.neurons.ach = 0.5
        self.neurons.I_ext = 0.0
        self.neurons.energy = 1.0
        self.neurons.frustration = 0.0
        self.neurons.resilience = 0.2
        self.neurons.firing_rate = 0.0 * Hz
        self.neurons.last_spike = -1000 * ms
        
        # Inicialización del umbral adaptativo
        self.neurons.v_thresh_base = self.v_thresh_arr
        self.neurons.v_thresh = self.v_thresh_arr
        
        tau_t_arr = np.zeros(N_TOTAL)
        tau_t_arr[self.neuron_types == 1] = 80.0 # ms
        tau_t_arr[self.neuron_types == 4] = 40.0 # ms
        self.neurons.tau_thresh = tau_t_arr * ms
        
        delta_t_arr = np.zeros(N_TOTAL)
        delta_t_arr[self.neuron_types == 1] = 1.8
        delta_t_arr[self.neuron_types == 4] = 1.0
        self.neurons.delta_v_thresh = delta_t_arr
        
        # Inicialización del ruido dinámico
        self.neurons.noise_base = 2.2
        
        # Parámetros del Ritmo Sincrónico CPG
        self.neurons.is_sensory = (self.layer_indices == 0).astype(float)
        self.neurons.cpg_amplitude = 2.5
        self.neurons.signal_period = SIGNAL_PERIOD * ms
        
        # Conexiones Sinápticas (STP + STDP 3-Factores + Mielinización)
        eqs_synapses = '''
        w : 1
        myelination : 1
        is_excitatory : 1
        target_is_dendrite : 1
        is_active : 1
        
        # Traza STP
        dx_stp/dt = (1 - x_stp) / tau_d : 1 (event-driven)
        du_stp/dt = (U_stp - u_stp) / tau_f : 1 (event-driven)
        
        # Traza STDP
        dapre/dt = -apre / taupre : 1 (event-driven)
        dapost/dt = -apost / taupost : 1 (event-driven)
        
        U_stp : 1
        tau_d : second
        tau_f : second
        '''
        
        taupre = taupost = 20 * ms
        wmax = 2.0
        Apre = 0.015
        Apost = -Apre * 1.05
        
        self.synapses = Synapses(
            self.neurons, self.neurons,
            model=eqs_synapses,
            on_pre='''
            u_stp = u_stp + U_stp * (1.0 - u_stp)
            release = u_stp * x_stp
            x_stp = clip(x_stp - release, 0.0, 1.0)
            
            effective_weight = w * release * (1.0 + 0.6 * myelination)
            
            g_ampa_soma_post += effective_weight * is_excitatory * (1.0 - target_is_dendrite)
            g_ampa_dend_post += effective_weight * is_excitatory * target_is_dendrite
            g_gaba_soma_post += effective_weight * (1.0 - is_excitatory) * (1.0 - target_is_dendrite)
            g_gaba_dend_post += effective_weight * (1.0 - is_excitatory) * target_is_dendrite
            
            apre += Apre * is_excitatory
            w = clip(w + apost * is_excitatory * is_active * (1.5 - da_post + frustration_post) / (0.5 + 1.5 * ser_post), 0.0, wmax)
            ''',
            on_post='''
            apost += Apost * is_excitatory
            w = clip(w + apre * is_excitatory * is_active * da_post * (1.0 + 2.5 * ach_post) / (0.5 + 1.5 * ser_post), 0.0, wmax)
            ''',
            namespace={
                'taupre': taupre, 'taupost': taupost,
                'Apre': Apre, 'Apost': Apost, 'wmax': wmax
            },
            name='synapses'
        )
        
        # Conectamos todos los pares posibles para recableado flexible
        self.synapses.connect(p=1.0)
        
        # Inicialización de variables sinápticas
        self.synapses.w = 0.0
        self.synapses.myelination = 0.0
        pre_types = self.neuron_types[self.synapses.i[:]]
        self.synapses.is_excitatory = (pre_types == 1).astype(float)
        self.synapses.target_is_dendrite = (np.random.rand(len(self.synapses)) < 0.5).astype(float)
        
        # Parámetros STP
        self.synapses.U_stp = np.where(pre_types == 1, 0.2, 0.5)
        self.synapses.tau_d = np.where(pre_types == 1, 200.0, 100.0) * ms
        self.synapses.tau_f = np.where(pre_types == 1, 50.0, 150.0) * ms
        self.synapses.x_stp = 1.0
        self.synapses.u_stp = self.synapses.U_stp
        
        # Conectividad activa inicial
        i_idx = self.synapses.i[:]
        j_idx = self.synapses.j[:]
        pre_layer = self.layer_indices[i_idx]
        post_layer = self.layer_indices[j_idx]
        
        active_mask = (
            ((pre_layer == 0) & (post_layer == 1) & (np.random.rand(len(i_idx)) < 0.4)) |
            ((pre_layer == 1) & (post_layer == 2) & (np.random.rand(len(i_idx)) < 0.4)) |
            ((pre_layer == 2) & (post_layer == 3) & (np.random.rand(len(i_idx)) < 0.4)) |
            ((pre_layer == 3) & (post_layer == 2) & (np.random.rand(len(i_idx)) < 0.3)) |
            ((pre_layer == 3) & (post_layer == 1) & (np.random.rand(len(i_idx)) < 0.2)) |
            ((pre_layer == 1) & (post_layer == 1) & (np.random.rand(len(i_idx)) < 0.2)) |
            ((pre_layer == 3) & (post_layer == 3) & (np.random.rand(len(i_idx)) < 0.2))
        )
        
        # Pesos e indicación de existencia inicial
        self.synapses.is_active = active_mask.astype(float)
        
        w_init = np.zeros(len(self.synapses))
        w_init[active_mask] = np.random.uniform(0.35, 0.85, np.sum(active_mask))
        inh_mask = active_mask & (self.synapses.is_excitatory[:] == 0.0)
        w_init[inh_mask] = 0.95
        self.synapses.w = w_init
        
        # Mielinización inicial
        myel_init = np.zeros(len(self.synapses))
        myel_init[active_mask] = np.random.uniform(0.1, 0.3, np.sum(active_mask))
        self.synapses.myelination = myel_init
        
        # Configurar delays basados en distancia 3D
        pos_pre = np.column_stack((self.x[i_idx], self.y[i_idx], self.z[i_idx]))
        pos_post = np.column_stack((self.x[j_idx], self.y[j_idx], self.z[j_idx]))
        dist_3d = np.linalg.norm(pos_pre - pos_post, axis=1)
        self.base_delays = (1.5 + 2.5 * (dist_3d / np.max(dist_3d))) * ms
        self.synapses.delay = self.base_delays
        
        # Monitoreo
        self.spike_mon = SpikeMonitor(self.neurons, name='spike_monitor')
        self.network = Network(self.neurons, self.synapses, self.spike_mon)
        
        # Buffers e históricos
        self.neuromod = NeuromodulatorSystem()
        self._episodic_buffer = deque(maxlen=1000)
        self.history = []
        
        self.frustration = 0.0
        self.resilience = 0.2
        
        # Inicialización de módulos de regulación biológica
        self.scaler = SynapticScaler(target_sum_w=6.0, active=True)
        self.gain_control = GainController(target_rate=8.0, alpha_gain=0.05, active=True)
        
        if state_path is not None and os.path.exists(state_path):
            self.load_state(state_path)
        
    def _get_brain_state(self):
        phase = self.time % SLEEP_CYCLE_PERIOD
        if phase < SLEEP_CYCLE_PERIOD * 0.7:
            return 'AWAKE'
        elif phase < SLEEP_CYCLE_PERIOD * 0.9:
            return 'SLOW_WAVE_SLEEP'
        else:
            return 'REM'
            
    def _homeostasis(self):
        """Escalado homeostático multiplicativo de Excitación e Inhibición vectorizado en NumPy."""
        rates = self.neurons.firing_rate[:] / Hz
        w = self.synapses.w[:]
        j_indices = self.synapses.j[:]
        is_exc = self.synapses.is_excitatory[:] == 1.0
        
        # Factores para Excitadoras e Inhibidoras
        scale_exc = np.ones(N_TOTAL)
        scale_inh = np.ones(N_TOTAL)
        
        # Neuronas hiperactivas: deprimir entradas excitatorias, potenciar inhibitorias para frenar la red
        dn_mask = rates > 8.0
        scale_exc[dn_mask] = 0.95
        scale_inh[dn_mask] = 1.05
        
        # Neuronas hipoactivas: potenciar entradas excitatorias, deprimir inhibitorias para animar la red
        up_mask = rates < 1.0
        scale_exc[up_mask] = 1.05
        scale_inh[up_mask] = 0.95
        
        # Mapear factores a cada sinapsis
        syn_scales_exc = scale_exc[j_indices]
        syn_scales_inh = scale_inh[j_indices]
        
        w_new = w.copy()
        # Escalar excitatorias
        w_new[is_exc] *= syn_scales_exc[is_exc]
        # Escalar inhibitorias
        w_new[~is_exc] *= syn_scales_inh[~is_exc]
        
        # Conservar el límite máximo wmax = 2.0
        self.synapses.w = np.clip(w_new, 0.0, 2.0)
        
        # Escalado sináptico descentralizado adicional (capa de regulación glial)
        if hasattr(self, 'scaler') and self.scaler.active:
            self.scaler.scale(self)
            
    def _structural_plasticity(self):
        """Recableado dinámico disperso (poda w < 0.02 y sinaptogénesis lenta)."""
        w = self.synapses.w[:]
        is_active_arr = self.synapses.is_active[:]
        is_exc = self.synapses.is_excitatory[:] == 1.0
        
        # Poda de sinapsis excitatorias inactivas o extremadamente débiles
        prune_mask = is_exc & (is_active_arr == 1.0) & (w < 0.02)
        n_pruned = np.sum(prune_mask)
        w[prune_mask] = 0.0
        is_active_arr[prune_mask] = 0.0
        self.pruned_synapses += n_pruned
        
        # Sinaptogénesis lenta para no saturar la red (conectividad dispersa)
        inactive_mask = (is_active_arr == 0.0) & is_exc
        if np.any(inactive_mask):
            inactive_indices = np.where(inactive_mask)[0]
            i_idx = self.synapses.i[inactive_indices]
            j_idx = self.synapses.j[inactive_indices]
            
            pos_pre = np.column_stack((self.x[i_idx], self.y[i_idx], self.z[i_idx]))
            pos_post = np.column_stack((self.x[j_idx], self.y[j_idx], self.z[j_idx]))
            dists = np.linalg.norm(pos_pre - pos_post, axis=1)
            
            # Probabilidad de sinaptogénesis baja (0.005 base)
            p_conn = 0.005 * np.exp(-dists / 30.0)
            connect_mask = np.random.rand(len(inactive_indices)) < p_conn
            
            if np.any(connect_mask):
                chosen_syns = inactive_indices[connect_mask]
                # Peso inicial bajo para que la poda pueda limpiarlo si no se potencia por STDP
                w[chosen_syns] = np.random.uniform(0.015, 0.045, len(chosen_syns))
                is_active_arr[chosen_syns] = 1.0
                
                # Resetear parámetros de traza STP
                x_vals = self.synapses.x_stp[:]
                u_vals = self.synapses.u_stp[:]
                x_vals[chosen_syns] = 1.0
                u_vals[chosen_syns] = self.synapses.U_stp[chosen_syns]
                self.synapses.x_stp = x_vals
                self.synapses.u_stp = u_vals
                self.created_synapses += len(chosen_syns)
                
        self.synapses.w = w
        self.synapses.is_active = is_active_arr
        
    def _sleep_replay(self):
        """Simula Replay de Sueño REM en ráfagas Poisson rápidas (50 Hz)."""
        if len(self._episodic_buffer) > 0:
            memory = np.random.choice(list(self._episodic_buffer))
            sensory_pattern = memory['sensory']
            noise = np.random.rand(N_SENSORY) < 0.2
            burst_current = sensory_pattern * 45.0 * noise
            self.neurons.I_ext[:N_SENSORY] = burst_current
            self.neuromod.dopamine = min(1.0, self.neuromod.dopamine + 0.02)
            
    def step(self):
        self.time += BATCH_MS
        self.step_count += 1
        self.brain_state = self._get_brain_state()
        
        # Sincronizar neuromodulación a Brian2
        self.neurons.da = self.neuromod.dopamine
        self.neurons.ser = self.neuromod.serotonin
        self.neurons.ach = self.neuromod.acetylcholine
        self.neurons.frustration = self.frustration
        self.neurons.resilience = self.resilience
        
        # Limpiar corriente externa
        self.neurons.I_ext = 0.0
        
        target_signal = 0.0
        prediction = 0.0
        error = 0.0
        
        if self.brain_state == 'AWAKE':
            target_signal = np.sin(2 * np.pi * self.time / SIGNAL_PERIOD)
            
            # Estimulación en los primeros 5 canales
            sensory_pulse = np.zeros(N_SENSORY)
            sensory_pulse[0:5] = 45.0 * (target_signal > 0.0)
            sensory_pulse += np.random.uniform(0.0, 5.0, N_SENSORY)
            self.neurons.I_ext[:N_SENSORY] = sensory_pulse
            
            if self.neuromod.dopamine > 0.65:
                self._episodic_buffer.append({
                    'sensory': sensory_pulse.copy(),
                    'time': self.time
                })
                
        elif self.brain_state == 'REM':
            self._sleep_replay()
            
        # Ejecutar simulación Brian2 por 500 ms
        self.network.run(BATCH_MS * ms, report=None)
        
        # Obtener tasas de disparo medias en los últimos 200 ms
        t_sec = self.time * 0.001
        spike_t = np.asarray(self.spike_mon.t)
        spike_i = np.asarray(self.spike_mon.i)
        
        window_start = t_sec - 0.2
        recent_mask = (spike_t > window_start) & (spike_t <= t_sec)
        recent_spikes_i = spike_i[recent_mask]
        
        for i in range(N_TOTAL):
            count = np.sum(recent_spikes_i == i)
            self.neurons.firing_rate[i] = (count / 0.2) * Hz
            
        # Calcular error de predicción en base a la capa motora (Z: 45)
        motor_firing = float(np.mean(self.neurons.firing_rate[self.layer_indices == 2] / Hz))
        
        # Aplicar regulación intrínseca de ganancia en el cerebro
        if hasattr(self, 'gain_control') and self.gain_control.active:
            self.gain_control.adapt(self, motor_firing)
            
        prediction = motor_firing / 15.0
        if self.brain_state == 'AWAKE':
            error = target_signal - prediction
        else:
            error = 0.0
            
        # Actualizar frustración y resiliencia
        if self.brain_state == 'AWAKE':
            if abs(error) > 0.35:
                self.frustration = min(1.0, self.frustration + 0.06)
            else:
                self.frustration = max(0.0, self.frustration - 0.04)
                self.resilience = min(1.0, self.resilience + 0.01)
        else:
            self.frustration = max(0.0, self.frustration - 0.02)
            
        self.neuromod.update(0.5, self.brain_state, error)
        
        # Consumo y recuperación de energía
        has_spiked = np.zeros(N_TOTAL, dtype=bool)
        if len(recent_spikes_i) > 0:
            has_spiked[np.unique(recent_spikes_i)] = True
        self.neurons.energy[has_spiked] -= 0.05
        self.neurons.energy[~has_spiked] += 0.01
        self.neurons.energy[:] = np.clip(self.neurons.energy[:], 0.0, 1.0)
        
        # Homeostasis y plasticidad estructural
        if self.step_count % 10 == 0:
            self._homeostasis()
        if self.step_count % 20 == 0:
            self._structural_plasticity()
            
        # Mielinización dinámica
        if self.neuromod.dopamine > 0.6:
            active_syns = (self.synapses.w[:] > 0.1) & (self.synapses.is_excitatory[:] == 1.0)
            if np.any(active_syns):
                myel = self.synapses.myelination[:]
                myel[active_syns] = np.clip(myel[active_syns] + 0.03, 0.0, 1.0)
                self.synapses.myelination = myel
                self.synapses.delay = self.base_delays / (1.0 + 3.0 * myel)
            
        # Registrar historial
        w_all = self.synapses.w[:]
        is_active_arr = self.synapses.is_active[:]
        w_active = w_all[is_active_arr == 1.0]
        self.history.append({
            'time': self.time,
            'step': self.step_count,
            'state': self.brain_state,
            'da': self.neuromod.dopamine,
            'ser': self.neuromod.serotonin,
            'ach': self.neuromod.acetylcholine,
            'w_mean': float(np.mean(w_active)) if len(w_active) > 0 else 0.0,
            'w_max': float(np.max(w_active)) if len(w_active) > 0 else 0.0,
            'synapses': len(w_active),
            'spikes': self.spike_mon.num_spikes,
            'energy_mean': float(np.mean(self.neurons.energy[:])),
            'pruned': self.pruned_synapses,
            'created': self.created_synapses,
            'prediction': prediction,
            'target': target_signal
        })
        
        if len(self.history) > 1000:
            self.history.pop(0)

        # Escribir estado actual en JSON
        self.write_state_json()
        
    def write_state_json(self):
        w_syn = self.synapses.w[:]
        i_indices = self.synapses.i[:]
        j_indices = self.synapses.j[:]
        is_exc = self.synapses.is_excitatory[:] == 1.0
        
        # Filtro de sinapsis activas para el JSON (w > 0.01) para agilizar Three.js
        active_syn_mask = w_syn > 0.01
        syn_list = []
        if np.any(active_syn_mask):
            pre_ids = i_indices[active_syn_mask]
            post_ids = j_indices[active_syn_mask]
            weights = w_syn[active_syn_mask]
            excs = is_exc[active_syn_mask]
            for idx in range(len(pre_ids)):
                syn_list.append({
                    'pre': int(pre_ids[idx]),
                    'post': int(post_ids[idx]),
                    'w': float(weights[idx]),
                    'exc': bool(excs[idx])
                })
                
        neuron_list = []
        firing_rates = self.neurons.firing_rate[:] / Hz
        energies = self.neurons.energy[:]
        for idx in range(N_TOTAL):
            neuron_list.append({
                'layer': int(self.layer_indices[idx]),
                'x': float(self.x[idx]),
                'y': float(self.y[idx]),
                'z': float(self.z[idx]),
                'firing': float(firing_rates[idx]),
                'energy': float(energies[idx])
            })
            
        w_active = w_syn[w_syn > 0.0]
        state_data = {
            'time': float(self.time),
            'step': int(self.step_count),
            'state': str(self.brain_state),
            'da': float(self.neuromod.dopamine),
            'ser': float(self.neuromod.serotonin),
            'ach': float(self.neuromod.acetylcholine),
            'w_mean': float(np.mean(w_active)) if len(w_active) > 0 else 0.0,
            'synapses_active': int(len(w_active)),
            'synapses_pruned': int(self.pruned_synapses),
            'synapses_created': int(self.created_synapses),
            'energy_mean': float(np.mean(energies)),
            'prediction': float(self.history[-1]['prediction']) if len(self.history) > 0 else 0.0,
            'target': float(self.history[-1]['target']) if len(self.history) > 0 else 0.0,
            'neurons': neuron_list,
            'synapses': syn_list
        }
        
        # Escritura atómica simple en la ruta absoluta WEB_DIR
        target_path = os.path.join(WEB_DIR, "sim_state.json")
        temp_path = target_path + ".tmp"
        try:
            with open(temp_path, 'w') as f:
                json.dump(state_data, f)
            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename(temp_path, target_path)
        except Exception as e:
            pass

    def save_state(self, filepath):
        """Persiste el estado biológico y de simulación completo en un archivo pickle."""
        state_data = {
            'time': self.time,
            'step_count': self.step_count,
            'brain_state': self.brain_state,
            'pruned_synapses': self.pruned_synapses,
            'created_synapses': self.created_synapses,
            'frustration': self.frustration,
            'resilience': self.resilience,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'layer_indices': self.layer_indices,
            'neuron_types': self.neuron_types,
            'v_thresh_arr': self.v_thresh_arr,
            'tau_m_arr': self.tau_m_arr,
            'dopamine': self.neuromod.dopamine,
            'serotonin': self.neuromod.serotonin,
            'acetylcholine': self.neuromod.acetylcholine,
            'history': self.history,
            'episodic_buffer': list(self._episodic_buffer),
            'neurons_v': self.neurons.v[:],
            'neurons_v_dend': self.neurons.v_dend[:],
            'neurons_v_thresh': self.neurons.v_thresh[:],
            'neurons_energy': self.neurons.energy[:],
            'neurons_g_ampa_soma': self.neurons.g_ampa_soma[:],
            'neurons_g_gaba_soma': self.neurons.g_gaba_soma[:],
            'neurons_g_ampa_dend': self.neurons.g_ampa_dend[:],
            'neurons_g_gaba_dend': self.neurons.g_gaba_dend[:],
            'neurons_firing_rate': self.neurons.firing_rate[:],
            'neurons_last_spike': self.neurons.last_spike[:],
            'synapses_w': self.synapses.w[:],
            'synapses_myelination': self.synapses.myelination[:],
            'synapses_is_active': self.synapses.is_active[:],
            'synapses_delay': self.synapses.delay[:],
            'synapses_x_stp': self.synapses.x_stp[:],
            'synapses_u_stp': self.synapses.u_stp[:],
            'synapses_is_excitatory': self.synapses.is_excitatory[:],
            'synapses_target_is_dendrite': self.synapses.target_is_dendrite[:],
            'scaler_active': self.scaler.active if hasattr(self, 'scaler') else False,
            'scaler_target_sum_w': self.scaler.target_sum_w if hasattr(self, 'scaler') else 6.0,
            'gain_control_active': self.gain_control.active if hasattr(self, 'gain_control') else False,
            'gain_control_target_rate': self.gain_control.target_rate if hasattr(self, 'gain_control') else 8.0,
            'gain_control_alpha_gain': self.gain_control.alpha_gain if hasattr(self, 'gain_control') else 0.05,
            'gain_control_v_offset': self.gain_control.v_offset if hasattr(self, 'gain_control') else 0.0
        }
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(state_data, f)
            print(f"[SAVE] Estado completo guardado en: {filepath}")
        except Exception as e:
            print(f"[ERROR] No se pudo guardar el estado cerebral: {e}")

    def load_state(self, filepath):
        """Restaura el estado completo de simulación a partir de un archivo pickle."""
        if not os.path.exists(filepath):
            print(f"[LOAD] Archivo de estado no encontrado: {filepath}")
            return False
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.time = data['time']
            self.step_count = data['step_count']
            self.brain_state = data['brain_state']
            self.pruned_synapses = data['pruned_synapses']
            self.created_synapses = data['created_synapses']
            self.frustration = data['frustration']
            self.resilience = data['resilience']
            self.x = data['x']
            self.y = data['y']
            self.z = data['z']
            self.layer_indices = data['layer_indices']
            self.neuron_types = data['neuron_types']
            self.v_thresh_arr = data['v_thresh_arr']
            self.tau_m_arr = data['tau_m_arr']
            self.neuromod.dopamine = data['dopamine']
            self.neuromod.serotonin = data['serotonin']
            self.neuromod.acetylcholine = data['acetylcholine']
            self.history = data['history']
            self._episodic_buffer = deque(data['episodic_buffer'], maxlen=1000)
            self.neurons.v = data['neurons_v']
            self.neurons.v_dend = data['neurons_v_dend']
            self.neurons.v_thresh = data['neurons_v_thresh']
            self.neurons.energy = data['neurons_energy']
            self.neurons.g_ampa_soma = data['neurons_g_ampa_soma']
            self.neurons.g_gaba_soma = data['neurons_g_gaba_soma']
            self.neurons.g_ampa_dend = data['neurons_g_ampa_dend']
            self.neurons.g_gaba_dend = data['neurons_g_gaba_dend']
            self.neurons.firing_rate = data['neurons_firing_rate']
            self.neurons.last_spike = data['neurons_last_spike']
            self.synapses.w = data['synapses_w']
            self.synapses.myelination = data['synapses_myelination']
            self.synapses.is_active = data['synapses_is_active']
            self.synapses.delay = data['synapses_delay']
            self.synapses.x_stp = data['synapses_x_stp']
            self.synapses.u_stp = data['synapses_u_stp']
            self.synapses.is_excitatory = data['synapses_is_excitatory']
            self.synapses.target_is_dendrite = data['synapses_target_is_dendrite']
            self.neurons.is_sensory = (self.layer_indices == 0).astype(float)
            
            # Restaurar estado de los reguladores
            if 'scaler_active' in data:
                self.scaler.active = data['scaler_active']
                self.scaler.target_sum_w = data['scaler_target_sum_w']
            if 'gain_control_active' in data:
                self.gain_control.active = data['gain_control_active']
                self.gain_control.target_rate = data['gain_control_target_rate']
                self.gain_control.alpha_gain = data['gain_control_alpha_gain']
                self.gain_control.v_offset = data['gain_control_v_offset']
                
            new_v_thresh_base = self.v_thresh_arr.copy()
            if hasattr(self, 'gain_control') and self.gain_control.active:
                motor_mask = self.layer_indices == 2
                new_v_thresh_base[motor_mask] += self.gain_control.v_offset
            self.neurons.v_thresh_base = new_v_thresh_base
            
            i_idx = self.synapses.i[:]
            j_idx = self.synapses.j[:]
            pos_pre = np.column_stack((self.x[i_idx], self.y[i_idx], self.z[i_idx]))
            pos_post = np.column_stack((self.x[j_idx], self.y[j_idx], self.z[j_idx]))
            dist_3d = np.linalg.norm(pos_pre - pos_post, axis=1)
            self.base_delays = (1.5 + 2.5 * (dist_3d / np.max(dist_3d))) * ms
            print(f"[LOAD] Estado completo restaurado desde: {filepath}")
            return True
        except Exception as e:
            print(f"[ERROR] Fallo al cargar el estado cerebral: {e}")
            return False

# ============================================================================
# PROGRAMA PRINCIPAL Y BUCLE DE SIMULACIÓN
# ============================================================================
def simular_y_servir():
    print("=" * 80)
    print("  INICIALIZANDO SIMULACIÓN DE CEREBRO ÚNICO HEADLESS (Servidor Web Activo)")
    print(f"  Total Neuronas: {N_TOTAL} (Sensorial: {N_SENSORY}, Oculta: {N_HIDDEN}, Motor: {N_MOTOR}, PFC: {N_PFC})")
    print("=" * 80)
    
    # Asegurar la existencia de directorios absolutos
    os.makedirs(WEB_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Crear e inicializar el cerebro con carga de estado persistente
    state_path = os.path.join(LOGS_DIR, "cerebro_unico_state.pkl")
    cerebro = BrainUnico(state_path=state_path)
    
    # Iniciar servidor HTTP integrado
    server = BrainHTTPServer(WEB_DIR, port=8000)
    server.start()
    
    # Lanzar visualizador en modo App (ventana emergente nativa)
    def lanzar_app_window():
        import subprocess
        time_mod.sleep(1.0)
        print("[HTTP] Lanzando visualizador en ventana emergente nativa...")
        try:
            if sys.platform.startswith('win'):
                subprocess.Popen(['cmd', '/c', 'start', 'msedge', '--app=http://localhost:8000'],
                                 shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            pass
            
    threading.Thread(target=lanzar_app_window, daemon=True).start()
    
    print("\nSimulación iniciada. Presiona Ctrl+C para finalizar de forma segura.")
    print("Monitorea la simulación en: http://localhost:8000")
    
    try:
        # Corremos 300 pasos de simulación (150 segundos biológicos simulados)
        max_steps = 300
        for _ in range(max_steps):
            cerebro.step()
            h = cerebro.history[-1]
            if cerebro.step_count % 10 == 0:
                print(f"[Paso {h['step']}] t={h['time']/1000:.1f}s | Estado: {h['state']} | "
                      f"Sinapsis Activas: {h['synapses']} | W Promedio: {h['w_mean']:.3f} | "
                      f"Dopamina: {h['da']:.2f} | Frustración: {cerebro.frustration:.2f} | "
                      f"Energía Media: {h['energy_mean']:.2%}")
            
            # Pequeño sleep para control de CPU en ejecuciones de telemetría en vivo
            time_mod.sleep(0.05)
            
        print("\n[INFO] Simulación completada con éxito.")
        print("El servidor HTTP sigue activo en http://localhost:8000 para que puedas navegar el dashboard final.")
        print("Presiona Ctrl+C para finalizar el proceso del servidor de forma segura.")
        while True:
            time_mod.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nInterrupción detectada. Apagando servidor y simulación...")
    finally:
        # Guardar el estado biológico completo para la próxima ejecución
        cerebro.save_state(state_path)

        # Guardar resultados finales de la corrida (historial)
        log_path = os.path.join(LOGS_DIR, "cerebro_unico_sim_results.pkl")
        try:
            with open(log_path, 'wb') as f:
                pickle.dump(cerebro.history, f)
            print(f"[SAVE] Historial de simulación guardado en: {log_path}")
        except Exception as e:
            print(f"[ERROR] No se pudo guardar el historial: {e}")
            
        # Guardar una copia estática del último JSON en logs para registro histórico
        try:
            with open(os.path.join(LOGS_DIR, "cerebro_unico_state_final.json"), 'w') as f:
                json.dump(cerebro.history[-1], f)
        except:
            pass
            
        server.stop()
        print("Cerebro apagado de forma segura.")

if __name__ == "__main__":
    simular_y_servir()
