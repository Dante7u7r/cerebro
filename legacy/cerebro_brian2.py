#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IA VIVA - CEREBRO NEURONAL EN BRIAN2

Simulacion neuronal con:
- NeuronGroup LIF con ruido y corriente externa
- STDP nativo (on_pre / on_post)
- Neuromodulacion
- Sistema emocional y ciclo sueno-vigilia
- 6 experimentos

Ejecutar: python cerebro_brian2.py
"""

from brian2 import *
import numpy as np
import csv, os, pickle, time as time_mod
from collections import deque

# ============================================================================
# CONFIGURACION GLOBAL
# ============================================================================

# Mapeo de codegen target compatible con múltiples plataformas.
# Si Cython y un compilador C++ no están listos, Brian2 cae automáticamente a numpy.
prefs.codegen.target = 'cython'
DT = 0.1 * ms
SLEEP_CYCLE_PERIOD = 20000.0  # ms
SIGNAL_PERIOD = 127.7        # ms

N_SENSORIAL = 5
N_OCULTA = 3
N_MOTOR = 4
N_PFC = 3
N_TOTAL = N_SENSORIAL + N_OCULTA + N_MOTOR + N_PFC

np.random.seed(42)

# ============================================================================
# NEUROMODULADOR
# ============================================================================

class NeuromodulatorSystem:
    def __init__(self):
        self.dopamine = 0.5
        self.serotonin = 0.5
        self.acetylcholine = 0.5
        self.tau_da = 2000.0   # ~2s reuptake (realista)
        self.tau_5ht = 8000.0  # ~8s clearance (serotonina lenta)
        self.tau_ach = 1000.0  # ~1s hidrolisis (acetilcolina rapida)
        self.last_pe = 0.0
        self.da_target = 0.5
        self.ser_target = 0.5
        self.ach_target = 0.5

    def update(self, dt, brain_state, prediction_error):
        self.last_pe = prediction_error
        
        # Calculate targets based on current state and prediction error
        self.da_target = float(np.clip(0.5 + 0.5 * np.tanh(abs(prediction_error) * 2.0), 0.0, 1.0))
        if brain_state == 'SLOW_WAVE_SLEEP':
            self.da_target = 0.2
        elif brain_state == 'REM':
            self.da_target = 0.6
            
        self.ser_target = 0.7 if brain_state == 'AWAKE' else (0.4 if brain_state == 'SLOW_WAVE_SLEEP' else 0.1)
        self.ach_target = 0.8 if brain_state == 'AWAKE' else (0.2 if brain_state == 'SLOW_WAVE_SLEEP' else 0.7)

    def stdp_modulation(self):
        return 0.5 + self.dopamine

# ============================================================================
# CEREBRO BRIAN2
# ============================================================================

class BrainBrian2:
    def __init__(self, name="CerebroBrian2", n_sensory=25, n_hidden=20, n_motor=20, n_pfc=15,
                 plasticity=True, drive_amplitude=15.0, gated_plasticity=False, homeostatic_scaling=False):
        self.name = name
        self.time = 0.0
        self.step_count = 0
        self.brain_state = 'AWAKE'

        N_S = n_sensory; N_H = n_hidden; N_M = n_motor; N_P = n_pfc
        n_total = N_S + N_H + N_M + N_P

        layer = np.zeros(n_total, dtype=int)
        layer[N_S:N_S+N_H] = 1; layer[N_S+N_H:N_S+N_H+N_M] = 2; layer[N_S+N_H+N_M:] = 3

        # Configurar subtipos de neuronas
        # Capa sensorial (0) -> Subtipo 1 (Excitatorio sensorial/oculto)
        # Capa oculta (1) -> 80% Subtipo 1, 20% Subtipo 4 (Inhibitorio local)
        # Capa motor (2) -> Todo Subtipo 2 (Motor excitatorio)
        # Capa PFC (3) -> 80% Subtipo 3 (PFC excitatorio), 20% Subtipo 4 (Inhibitorio local)
        subtype = np.ones(n_total, dtype=int)

        s_idx = np.where(layer == 0)[0]
        h_idx = np.where(layer == 1)[0]
        m_idx = np.where(layer == 2)[0]
        p_idx = np.where(layer == 3)[0]

        # Asignar motor
        subtype[m_idx] = 2
        
        # Asignar PFC excitatorio por defecto
        subtype[p_idx] = 3

        # Determinar cuántas de la capa oculta y PFC serán subtipo 4 (inhibidoras)
        n_h_inh = int(np.round(0.2 * N_H))
        n_p_inh = int(np.round(0.2 * N_P))

        # Seleccionar los últimos índices de las capas para interneuronas inhibidoras
        h_inh_idx = h_idx[-n_h_inh:] if n_h_inh > 0 else []
        p_inh_idx = p_idx[-n_p_inh:] if n_p_inh > 0 else []

        subtype[h_inh_idx] = 4
        subtype[p_inh_idx] = 4

        v_rest_val = -65.0
        
        # Configurar v_thresh y tau_m por subtipo
        v_thresh_arr = np.zeros(n_total)
        v_thresh_arr[subtype == 1] = -55.0
        v_thresh_arr[subtype == 2] = -45.0
        v_thresh_arr[subtype == 3] = -52.0
        v_thresh_arr[subtype == 4] = -57.0

        tau_m_arr = np.zeros(n_total)
        tau_m_arr[subtype == 1] = 15.0
        tau_m_arr[subtype == 2] = 10.0
        tau_m_arr[subtype == 3] = 25.0
        tau_m_arr[subtype == 4] = 10.0
        tau_m_arr = tau_m_arr * ms

        eqs = '''
        dv/dt = (-(v - v_rest) + I_sensory * pacemaker_gate + I_ext + I_pacemaker) / tau_m + noise_sigma * (2/tau_m)**0.5 * xi : 1
        pacemaker_gate = (1.0 - is_sleep) + is_sleep * pacemaker_wave : 1
        I_pacemaker = is_sleep * osc_amp * pacemaker_wave : 1
        pacemaker_wave = (0.5 + 0.5 * sin(2 * pi * theta_freq * t)) * (0.5 + 0.5 * sin(2 * pi * gamma_freq * t)) : 1
        is_sleep : 1
        v_rest : 1
        tau_m : second
        v_thresh : 1
        layer : integer
        subtipo : integer
        I_sensory : 1
        I_ext : 1
        firing_rate : Hz
        last_spike : second
        energy : 1
        
        # Trazas neuromoduladoras continuas integradas en C++
        dda/dt = (da_target - da) / (tau_da * ms) : 1
        dser/dt = (ser_target - ser) / (tau_5ht * ms) : 1
        dach/dt = (ach_target - ach) / (tau_ach * ms) : 1
        
        da_target : 1
        ser_target : 1
        ach_target : 1
        
        noise_sigma : 1
        '''

        self.neurons = NeuronGroup(
            n_total, eqs,
            threshold='v > v_thresh',
            reset='v = v_rest; last_spike = t',
            refractory='2*ms',
            method='euler',
            namespace={
                'tau_da': 2000.0, 'tau_5ht': 8000.0, 'tau_ach': 1000.0,
                'theta_freq': 8.0 * Hz, 'gamma_freq': 40.0 * Hz,
                'osc_amp': 8.0
            },
            name='neurons'
        )

        self.neurons.v_rest = v_rest_val
        self.neurons.tau_m = tau_m_arr
        self.neurons.v_thresh = v_thresh_arr
        self.neurons.v = v_rest_val + np.random.randn(n_total) * 3.0
        self.neurons.layer = layer
        self.neurons.subtipo = subtype
        self.neurons.I_sensory = 0.0
        self.neurons.I_ext = 0.0
        self.neurons.firing_rate = 0.0 * Hz
        self.neurons.last_spike = -1000 * ms
        self.neurons.energy = 1.0
        
        # Inicializar variables neuromoduladoras continuas y de ruido
        self.neurons.da = 0.5
        self.neurons.ser = 0.5
        self.neurons.ach = 0.5
        self.neurons.da_target = 0.5
        self.neurons.ser_target = 0.5
        self.neurons.ach_target = 0.5
        self.neurons.noise_sigma = 0.0
        self.neurons.is_sleep = 0.0

        # STDP
        taupre = taupost = 20 * ms
        wmax = 1.0
        Apre = 0.05 if plasticity else 0.0
        Apost = (-Apre * taupre / taupost * 1.05) if plasticity else 0.0

        self.synapses = Synapses(
            self.neurons, self.neurons,
            '''w : 1
               synaptic_sign : 1
               is_excitatory : 1
               dapre/dt = -apre/taupre : 1 (clock-driven)
               dapost/dt = -apost/taupost : 1 (clock-driven)''',
            on_pre='''
               v_post += w * synaptic_sign
               apre += Apre * is_excitatory * {gate}
               w = clip(w + apost * is_excitatory * {gate}, 0, wmax)
            '''.format(gate='ach_post' if gated_plasticity else '1.0'),
            on_post='''
               apost += Apost * is_excitatory * {gate}
               w = clip(w + apre * is_excitatory * {gate}, 0, wmax)
            '''.format(gate='ach_post' if gated_plasticity else '1.0'),
            namespace={
                'taupre': taupre, 'taupost': taupost,
                'wmax': wmax, 'Apre': Apre, 'Apost': Apost
            },
            name='synapses'
        )

        # Separar índices de excitatorias e inhibidoras
        h_exc_idx = h_idx[subtype[h_idx] != 4]
        h_inh_idx = h_idx[subtype[h_idx] == 4]
        p_exc_idx = p_idx[subtype[p_idx] != 4]
        p_inh_idx = p_idx[subtype[p_idx] == 4]

        # Topología por capas con conexiones probabilísticas
        # Feedforward: Sensorial -> Oculta (toda la capa oculta)
        for pre in s_idx:
            for post in h_idx:
                if np.random.random() < 0.4:
                    self.synapses.connect(i=pre, j=post)
                    
        # Feedforward: Oculta Excitatoria -> Motor
        for pre in h_exc_idx:
            for post in m_idx:
                if np.random.random() < 0.4:
                    self.synapses.connect(i=pre, j=post)
                    
        # Feedforward: Motor -> PFC (toda la capa PFC)
        for pre in m_idx:
            for post in p_idx:
                if np.random.random() < 0.4:
                    self.synapses.connect(i=pre, j=post)
                    
        # Feedback: PFC Excitatoria -> Motor
        for pre in p_exc_idx:
            for post in m_idx:
                if np.random.random() < 0.4:
                    self.synapses.connect(i=pre, j=post)
                    
        # Feedback: PFC Excitatoria -> Oculta (toda la capa oculta)
        for pre in p_exc_idx:
            for post in h_idx:
                if np.random.random() < 0.2:
                    self.synapses.connect(i=pre, j=post)
                    
        # Feedback: Oculta Excitatoria -> Sensorial
        for pre in h_exc_idx:
            for post in s_idx:
                if np.random.random() < 0.2:
                    self.synapses.connect(i=pre, j=post)
                    
        # --- BUCLES DE INHIBICIÓN LATERAL (LOCAL FEEDBACK) ---
        # Capa oculta: Excitatoria -> Inhibitoria y viceversa
        for pre in h_exc_idx:
            for post in h_inh_idx:
                if np.random.random() < 0.5:
                    self.synapses.connect(i=pre, j=post)
        for pre in h_inh_idx:
            for post in h_exc_idx:
                if np.random.random() < 0.5:
                    self.synapses.connect(i=pre, j=post)
                    
        # Capa PFC: Excitatoria -> Inhibitoria y viceversa
        for pre in p_exc_idx:
            for post in p_inh_idx:
                if np.random.random() < 0.5:
                    self.synapses.connect(i=pre, j=post)
        for pre in p_inh_idx:
            for post in p_exc_idx:
                if np.random.random() < 0.5:
                    self.synapses.connect(i=pre, j=post)

        # Inicialización de variables sinápticas
        self.synapses.w = 'rand() * 0.1'
        self.synapses.synaptic_sign = 1.0 - 2.0 * (self.synapses.subtipo_pre == 4)
        self.synapses.is_excitatory = (self.synapses.subtipo_pre != 4).astype(float)
        self.synapses.w[self.synapses.subtipo_pre == 4] = 0.25
        self.synapses.apre = 0.0
        self.synapses.apost = 0.0
        self.synapses.delay = '(2 + rand() * 3) * ms'

        self.homeostatic_scaling = homeostatic_scaling
        self.initial_w_sum = np.zeros(n_total)
        if homeostatic_scaling:
            w_init = np.array(self.synapses.w[:])
            j_indices = np.array(self.synapses.j[:])
            is_exc = np.array(self.synapses.is_excitatory[:]) == 1.0
            for post_id in range(n_total):
                mask = (j_indices == post_id) & is_exc
                if np.any(mask):
                    self.initial_w_sum[post_id] = np.sum(w_init[mask])

        self.n_total = n_total
        self.N_S = N_S; self.N_H = N_H; self.N_M = N_M; self.N_P = N_P
        self.drive_amplitude = drive_amplitude
        self.drive_mode = 'oscillation'  # 'oscillation' o 'tonic'

        self.spike_mon = SpikeMonitor(self.neurons, name='spikemonitor')
        self.state_mon = None
        self.network = Network(self.neurons, self.synapses,
                                self.spike_mon)

        self.frustration = 0.0
        self.resilience = 0.2
        self.courage = 0.2
        self.neuromod = NeuromodulatorSystem()
        self._episodic_buffer = deque(maxlen=5000)
        self._best_da_threshold = 0.70
        self.history = []
        self._last_error = 0.0
        self._logger = None
        self.force_state = None  # para exp de privacion de sueno
        self.noise_sigma = 0.0
        self.consolidation_threshold = 0.28

    def _get_brain_state(self):
        if self.force_state is not None:
            return self.force_state
        phase = self.time % SLEEP_CYCLE_PERIOD
        if phase < SLEEP_CYCLE_PERIOD * 0.7: return 'AWAKE'
        elif phase < SLEEP_CYCLE_PERIOD * 0.9: return 'SLOW_WAVE_SLEEP'
        else: return 'REM'

    def step(self):
        self.time += 500.0
        self.step_count += 1
        self.brain_state = self._get_brain_state()

        if self.drive_mode == 'tonic':
            ext = self.drive_amplitude
        else:
            ext = self.drive_amplitude * np.sin(2 * np.pi * self.time / SIGNAL_PERIOD)
        if self.brain_state != 'AWAKE':
            ext = 0.0
        self.neurons.I_ext = ext

        # Sleep Replay: Reactivar patrones grabados durante el sueño (Priorizado por debilidad sináptica)
        if self.brain_state in ['SLOW_WAVE_SLEEP', 'REM'] and len(self._episodic_buffer) > 0:
            weights = self.synapses.w[:]
            pre_indices = self.synapses.i[:]
            channel_w = {}
            for ch in range(self.N_S):
                mask = (pre_indices == ch)
                channel_w[ch] = float(np.mean(weights[mask])) if np.any(mask) else 0.0
            
            episodes = list(self._episodic_buffer)
            priorities = []
            for ep in episodes:
                channels = ep.get('channels', [])
                if len(channels) > 0:
                    avg_w = np.mean([channel_w.get(ch, 0.5) for ch in channels])
                    # Solo reactivamos si la memoria no está consolidada (peso <= consolidation_threshold)
                    if avg_w > self.consolidation_threshold:
                        priorities.append(0.0)
                    else:
                        priorities.append(1.0 / (avg_w + 0.05))
                else:
                    priorities.append(0.0)
            
            sum_p = sum(priorities)
            if sum_p > 0:
                probs = [p / sum_p for p in priorities]
                idx = np.random.choice(len(episodes), p=probs)
                episode = episodes[idx]
                channels = episode.get('channels', [])
                if len(channels) > 0:
                    self.neurons.I_sensory[channels] = 45.0

        pred = float(np.mean(self.neurons.firing_rate[:] / 100.0))
        target = np.sin(2 * np.pi * self.time / SIGNAL_PERIOD)
        error = target - pred
        self._last_error = error
        
        # Calcular targets neuromoduladores
        self.neuromod.update(500.0, self.brain_state, error)
        
        # Alimentar targets y nivel de ruido a NeuronGroup
        self.neurons.da_target = self.neuromod.da_target
        self.neurons.ser_target = self.neuromod.ser_target
        self.neurons.ach_target = self.neuromod.ach_target
        self.neurons.noise_sigma = self.noise_sigma
        self.neurons.is_sleep = 1.0 if self.brain_state in ['SLOW_WAVE_SLEEP', 'REM'] else 0.0

        self.network.run(500 * ms, report=None)
        
        # Sincronizar niveles reales continuos calculados en C++ de vuelta a Python
        self.neuromod.dopamine = float(np.mean(self.neurons.da[:]))
        self.neuromod.serotonin = float(np.mean(self.neurons.ser[:]))
        self.neuromod.acetylcholine = float(np.mean(self.neurons.ach[:]))

        t_sec = self.time * 0.001
        spike_t = np.asarray(self.spike_mon.t)
        spike_i = np.asarray(self.spike_mon.i)
        
        # We only care about spikes in the last 100 ms (0.1 seconds)
        window_start = t_sec - 0.1
        window_end = t_sec
        
        # Find indices of spikes within this window
        in_window = (spike_t > window_start) & (spike_t <= window_end)
        recent_i = spike_i[in_window]
        
        for i in range(self.n_total):
            spikes_in_window = np.sum(recent_i == i)
            self.neurons.firing_rate[i] = spikes_in_window * 10.0 * Hz

        # Energia: se consume con spikes, se recupera lentamente
        spike_mask = np.zeros(self.n_total, dtype=bool)
        recent_mask = (spike_t > t_sec - 0.05) & (spike_t <= t_sec)
        if np.any(recent_mask):
            spike_mask[np.unique(spike_i[recent_mask])] = True
        self.neurons.energy[spike_mask] -= 0.05
        self.neurons.energy[~spike_mask] += 0.01
        self.neurons.energy[:] = np.clip(self.neurons.energy[:], 0.0, 1.0)


        # Poda: eliminar sinapsis debiles cuando energia baja
        if self.step_count % 10 == 0:
            energy_avg = float(np.mean(self.neurons.energy[:]))
            if energy_avg < 0.3:
                weak = self.synapses.w[:] < 0.03
                if np.any(weak):
                    self.synapses.w[weak] = 0.0

        # Memoria episodica: registrar cuando DA > umbral y hay actividad
        if self.neuromod.dopamine > self._best_da_threshold and self.brain_state == 'AWAKE':
            n_spikes = self.spike_mon.num_spikes
            if n_spikes > 0:
                sensory_vals = np.asarray(self.neurons.I_sensory[:])
                active_channels = list(np.where(sensory_vals > 0.1)[0])
                if len(active_channels) > 0:
                    self._episodic_buffer.append({
                        'time': self.time,
                        'da': self.neuromod.dopamine,
                        'spikes': n_spikes,
                        'w_mean': float(np.mean(self.synapses.w[:])),
                        'channels': active_channels
                    })

        self.neurons.I_sensory = 0.0

        if self.homeostatic_scaling:
            w = np.array(self.synapses.w[:])
            j_indices = np.array(self.synapses.j[:])
            is_exc = np.array(self.synapses.is_excitatory[:]) == 1.0
            for post_id in range(self.n_total):
                target_sum = self.initial_w_sum[post_id]
                if target_sum > 0.0:
                    mask = (j_indices == post_id) & is_exc
                    if np.any(mask):
                        curr_sum = np.sum(w[mask])
                        if curr_sum > 0.0:
                            factor = target_sum / curr_sum
                            gradual_factor = 1.0 + 0.2 * (factor - 1.0)
                            w[mask] = np.clip(w[mask] * gradual_factor, 0.0, 1.0)
            self.synapses.w = w

        weights = self.synapses.w[:]
        self.history.append({
            'time': self.time, 'neuron': self.n_total, 'synapse': len(self.synapses),
            'da': self.neuromod.dopamine, 'fru': self.frustration, 'res': self.resilience,
            'brain_state': self.brain_state,
            'w_mean': float(np.mean(weights)) if len(weights) > 0 else 0.0,
            'w_max': float(np.max(weights)) if len(weights) > 0 else 0.0,
            'spikes': self.spike_mon.num_spikes,
            'energy': float(np.mean(self.neurons.energy[:])),
            '5ht': self.neuromod.serotonin,
            'ach': self.neuromod.acetylcholine,
        })
        if len(self.history) > 5000: self.history.pop(0)

        # Auto-log si hay logger
        if self._logger is not None:
            self._logger.registrar_neuromod()

        return self.spike_mon.num_spikes > 0

    @property
    def t(self): return self.time * ms

    def save_brain(self, path=None):
        if path is None: path = f"cerebro_brian2_save_{self.name.lower()}.pkl"
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        data = {
            'weights': self.synapses.w[:], 'time': self.time,
            'step_count': self.step_count, 'frustration': self.frustration,
            'resilience': self.resilience, 'courage': self.courage,
            'dopamine': self.neuromod.dopamine, 'serotonin': self.neuromod.serotonin,
            'acetylcholine': self.neuromod.acetylcholine,
            'spikes': self.spike_mon.num_spikes,
        }
        with open(path, 'wb') as f: pickle.dump(data, f)
        print(f"[SAVE] Cerebro Brian2 guardado en {path}")

    def load_brain(self, path):
        if not os.path.exists(path): print(f"[LOAD] No se encontro {path}"); return
        with open(path, 'rb') as f: data = pickle.load(f)
        for k, v in data.items():
            if k == 'weights': self.synapses.w = v
            elif k == 'time': self.time = v
            elif k == 'step_count': self.step_count = v
            elif k == 'frustration': self.frustration = v
            elif k == 'resilience': self.resilience = v
            elif k == 'courage': self.courage = v
            elif k == 'dopamine': self.neuromod.dopamine = v
            elif k == 'serotonin': self.neuromod.serotonin = v
            elif k == 'acetylcholine': self.neuromod.acetylcholine = v
        print(f"[LOAD] Cerebro Brian2 cargado desde {path}")

# ============================================================================
# BRAIN OBSERVER
# ============================================================================

class BrainObserver:
    def __init__(self, brain): self.brain = brain

    def snapshot(self):
        b = self.brain; weights = b.synapses.w[:]; n_syn = len(weights)
        nd = {}
        for i in range(b.n_total):
            ln = ['sensorial', 'oculta', 'motor', 'pfc'][b.neurons.layer[i]]
            nd.setdefault(ln, {'total':0, 'active':0, 'rates':[]})
            nd[ln]['total'] += 1
            rate = float(b.neurons.firing_rate[i] / Hz)
            nd[ln]['rates'].append(rate)
            if rate > 0: nd[ln]['active'] += 1
        layers = {}
        for name, data in nd.items():
            r = data['rates']
            layers[name] = {'total': data['total'], 'active': data['active'],
                            'firing_rate': np.mean(r) if r else 0.0, 'avg_weight': 0.0}
        return {
            'time': b.time, 'step': b.step_count, 'brain_state': b.brain_state,
            'total_neurons': b.n_total, 'total_synapses': n_syn,
            'avg_weight': float(np.mean(weights)) if n_syn > 0 else 0.0,
            'max_weight': float(np.max(weights)) if n_syn > 0 else 0.0,
            'dopamine': b.neuromod.dopamine, 'serotonin': b.neuromod.serotonin,
            'acetylcholine': b.neuromod.acetylcholine,
            'frustration': b.frustration, 'resilience': b.resilience, 'courage': b.courage,
            'total_spikes': b.spike_mon.num_spikes,
            'energy_avg': float(np.mean(b.neurons.energy[:])),
            'episodic_buffer': len(b._episodic_buffer),
        }

    def print_snapshot(self):
        s = self.snapshot()
        print(f"\n{'='*60}")
        print(f"  t={s['time']:.0f}ms  paso={s['step']}  estado={s['brain_state']}")
        print(f"{'='*60}")
        print(f"  Neuronas: {s['total_neurons']}  Sinapsis: {s['total_synapses']}")
        print(f"  w_prom={s['avg_weight']:.4f}  w_max={s['max_weight']:.4f}")
        print(f"  DA={s['dopamine']:.3f}  5HT={s['serotonin']:.3f}  ACh={s['acetylcholine']:.3f}")
        print(f"  Fru={s['frustration']:.3f}  Res={s['resilience']:.3f}")
        print(f"  Spikes totales: {s['total_spikes']}")
        print(f"{'='*60}")

# ============================================================================
# STIMULUS CONTROLLER
# ============================================================================

class StimulusController:
    def __init__(self, brain): self.brain = brain

    def inyectar_novedad(self, intensidad=0.5):
        val = float(self.brain.neuromod.dopamine) + intensidad * 0.3
        self.brain.neuromod.dopamine = val if val < 1.0 else 1.0

    def inyectar_error(self, severidad=0.5):
        val = float(self.brain.frustration) + severidad * 0.2
        self.brain.frustration = val if val < 1.0 else 1.0

    def inyectar_recompensa(self):
        val = float(self.brain.neuromod.dopamine) + 0.4
        self.brain.neuromod.dopamine = val if val < 1.0 else 1.0
        self.brain.frustration *= 0.5
        val2 = float(self.brain.resilience) + 0.05
        self.brain.resilience = val2 if val2 < 1.0 else 1.0

    def inyectar_pulso(self, canal, voltaje, duracion_pasos=10):
        idx = canal % self.brain.N_S
        for _ in range(duracion_pasos):
            self.brain.neurons.I_sensory[idx] = voltaje
            self.brain.step()

    def inyectar_senal_temporal(self, frecuencia_hz, duracion_pasos=100, amplitud=4.0):
        for paso in range(duracion_pasos):
            t = self.brain.time
            senal = amplitud * np.sin(2 * np.pi * frecuencia_hz * t / 1000.0)
            for i in range(self.brain.N_S):
                self.brain.neurons.I_sensory[i] = senal
            self.brain.step()

    def inyectar_senal_a_capa(self, capa_nombre, voltaje, duracion_pasos=10):
        layer_map = {'sensorial': 0, 'oculta': 1, 'motor': 2, 'pfc': 3}
        layer_id = layer_map.get(capa_nombre, 0)
        indices = [i for i in range(self.brain.n_total)
                   if self.brain.neurons.layer[i] == layer_id]
        for _ in range(duracion_pasos):
            for i in indices: self.brain.neurons.I_sensory[i] = voltaje
            self.brain.step()
        for i in indices: self.brain.neurons.I_sensory[i] = 0.0

    def inyectar_ruido_gaussiano(self, amplitud=5.0, duracion_pasos=20):
        for _ in range(duracion_pasos):
            for i in range(self.brain.N_S):
                self.brain.neurons.I_sensory[i] = amplitud * np.random.randn()
            self.brain.step()

    def silencio(self, pasos=50):
        for _ in range(pasos): self.brain.step()

# ============================================================================
# EVENT LOGGER
# ============================================================================

class EventLogger:
    def __init__(self, brain, directorio="logs_brian2"):
        self.brain = brain; self.directorio = directorio
        self._spikes = []; self._neuromod = []; self._step_data = []
        os.makedirs(directorio, exist_ok=True)

    def registrar_neuromod(self):
        b = self.brain
        self._neuromod.append({
            'time': b.time, 'dopamine': b.neuromod.dopamine,
            'serotonin': b.neuromod.serotonin, 'acetylcholine': b.neuromod.acetylcholine,
            'frustration': b.frustration, 'resilience': b.resilience,
            'courage': b.courage, 'spike_rate': float(np.mean(b.neurons.firing_rate[:] / Hz)),
            'w_mean': float(np.mean(b.synapses.w[:])),
        })

    def registrar_spikes(self):
        b = self.brain
        for t_val, i_val in zip(b.spike_mon.t, b.spike_mon.i):
            self._spikes.append({
                'time': float(t_val / ms), 'neuron_id': int(i_val),
                'capa': ['sensorial','oculta','motor','pfc'][int(b.neurons.layer[i_val])],
            })

    def _guardar_csv(self, datos, nombre):
        if not datos: return
        path = os.path.join(self.directorio, nombre)
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=datos[0].keys())
            w.writeheader(); w.writerows(datos)

    def guardar(self, sufijo=""):
        ts = int(time_mod.time())
        base = f"{ts}_{sufijo}" if sufijo else str(ts)
        self._guardar_csv(self._spikes, f"{base}_spikes.csv")
        self._guardar_csv(self._neuromod, f"{base}_neuromod.csv")
        self._guardar_csv(self._step_data, f"{base}_steps.csv")
        self._guardar_csv(self.brain.history, f"{base}_history.csv")
        print(f"[LOG] {self.directorio}/ ({base}_*.csv)")
        print(f"      Spikes: {len(self._spikes)}  Neuromod: {len(self._neuromod)}  "
              f"History: {len(self.brain.history)}")

# ============================================================================
# EXPERIMENTOS
# ============================================================================

def separador(nombre):
    print(f"\n{'#'*60}"); print(f"  {nombre}"); print(f"{'#'*60}")

def registrar_estado(observer, label=""):
    s = observer.snapshot()
    if label: print(f"\n  --- {label} ---")
    print(f"  t={s['time']:.0f}ms  paso={s['step']}  estado={s['brain_state']}")
    print(f"  Neuronas: {s['total_neurons']}  Sinapsis: {s['total_synapses']}")
    print(f"  DA={s['dopamine']:.4f}  5HT={s['serotonin']:.4f}  ACh={s['acetylcholine']:.4f}")
    print(f"  Fru={s['frustration']:.4f}  Res={s['resilience']:.4f}  Energia={s['energy_avg']:.3f}")
    print(f"  w_prom={s['avg_weight']:.4f}  w_max={s['max_weight']:.4f}")
    print(f"  Spikes totales: {s['total_spikes']}")
    return s

# ============================================================================
# EXPERIMENTOS ELIMINADOS (1-18) PARA CONCENTRAR LA INVESTIGACIÓN EN EL ACPLAMIENTO DE FASE (EXP 19)
# ============================================================================

def experimento_19():
    separador("EXPERIMENTO 19: Sincronía Oscilatoria Theta/Gamma (Acoplamiento de Fase) en Replay de Sueño")
    print("Objetivo: Demostrar que el acoplamiento oscilatorio Theta/Gamma (pacemaker) durante el replay en el sueño consolida la memoria de A (+100 spikes) con HSS.\n")

    def entrenar_estimulo(cerebro, canal, rondas=30):
        cerebro.drive_amplitude = 10.0
        for _ in range(rondas):
            for _ in range(5):
                cerebro.neurons.I_sensory[canal] = 15.0
                cerebro.step()
            cerebro.neurons.I_sensory[canal] = 0.0
            for _ in range(3): cerebro.step()

    def medir(cerebro, canal, duracion=5):
        cerebro.drive_amplitude = 9.0
        cerebro.drive_mode = 'tonic'
        idx_antes = len(cerebro.spike_mon.i)
        for _ in range(duracion):
            cerebro.neurons.I_sensory[canal] = 15.0
            cerebro.step()
        cerebro.neurons.I_sensory[canal] = 0.0
        recent_i = np.asarray(cerebro.spike_mon.i[idx_antes:])
        layers = np.asarray(cerebro.neurons.layer[:])
        downstream_spikes = np.sum(layers[recent_i] > 0)
        return downstream_spikes

    def test_triple(cerebro, label=""):
        a = [medir(cerebro, 0) for _ in range(3)]
        for _ in range(3): cerebro.step()
        b = [medir(cerebro, 2) for _ in range(3)]
        for _ in range(3): cerebro.step()
        c = [medir(cerebro, 4) for _ in range(3)]
        for _ in range(3): cerebro.step()
        mu_a, mu_b, mu_c = np.mean(a), np.mean(b), np.mean(c)
        if label:
            print(f"  {label}:  A={mu_a:.0f}  B={mu_b:.0f}  C={mu_c:.0f}  "
                  f"d(A,B)={mu_a-mu_b:+.0f}  d(A,C)={mu_a-mu_c:+.0f}")
        return mu_a, mu_b, mu_c

    # ===== Inicializar cerebro con HSS y Pacemaker =====
    np.random.seed(102)
    cerebro = BrainBrian2(name="Beta_Replay_Oscilador", plasticity=True, drive_amplitude=0.0, homeostatic_scaling=True)
    cerebro.force_state = 'AWAKE'
    logger = EventLogger(cerebro, "logs_brian2/exp19")
    cerebro._logger = logger
    for _ in range(5): cerebro.step()

    s0_weights_pre = cerebro.synapses.w[:][cerebro.synapses.i[:] == 0]
    s2_weights_pre = cerebro.synapses.w[:][cerebro.synapses.i[:] == 2]
    print(f"  Topologia Beta: {len(s0_weights_pre)}/{len(s2_weights_pre)} conexiones S0/S2 -> oculta")

    # ===== FASE 1: Entrenar A =====
    print("\n  --- FASE 1: Entrenar A (canal 0, 50 rondas) ---")
    entrenar_estimulo(cerebro, canal=0, rondas=50)
    w_post_a = float(np.mean(cerebro.synapses.w[:]))
    s0_weights_post_a = cerebro.synapses.w[:][cerebro.synapses.i[:] == 0]
    print(f"  w S0 post-A: {float(np.mean(s0_weights_post_a)):.4f} (Entrenado)")

    # Test pre-B
    a_pre_b, b_pre_b, c_pre_b = test_triple(cerebro, "Pre-B ")

    # ===== FASE 2: Entrenar B =====
    print("\n  --- FASE 2: Entrenar B (canal 2, 30 rondas) ---")
    entrenar_estimulo(cerebro, canal=2, rondas=30)
    s0_weights_post_b = cerebro.synapses.w[:][cerebro.synapses.i[:] == 0]
    s2_weights_post_b = cerebro.synapses.w[:][cerebro.synapses.i[:] == 2]
    print(f"  w S0 post-B: {float(np.mean(s0_weights_post_b)):.4f}  w S2 post-B: {float(np.mean(s2_weights_post_b)):.4f}")

    # Test post-B (olvido catastrófico de A)
    a_post_b, b_post_b, c_post_b = test_triple(cerebro, "Post-B")

    # ===== FASE 3: Consolidación por Sueño con HSS y Acoplamiento Theta/Gamma =====
    print("\n  --- FASE 3: Fase de Sueño (350 pasos en SLOW_WAVE_SLEEP con Replay, HSS y Pacemaker activos) ---")
    cerebro.force_state = 'SLOW_WAVE_SLEEP'
    cerebro.consolidation_threshold = 0.38
    cerebro.noise_sigma = 1.2
    for paso in range(350):
        cerebro.step()
    cerebro.noise_sigma = 0.0
    
    s0_weights_sleep = cerebro.synapses.w[:][cerebro.synapses.i[:] == 0]
    s2_weights_sleep = cerebro.synapses.w[:][cerebro.synapses.i[:] == 2]
    print(f"  w S0 post-sueno: {float(np.mean(s0_weights_sleep)):.4f}  w S2 post-sueno: {float(np.mean(s2_weights_sleep)):.4f}")

    # Despertar
    cerebro.force_state = 'AWAKE'
    for _ in range(5): cerebro.step()

    # Test post-Sueño
    a_post_sleep, b_post_sleep, c_post_sleep = test_triple(cerebro, "Post-S ")

    # ===== ANALISIS =====
    print("\n" + "="*60)
    print("  RESULTADOS: Rescate de memoria con Replay, HSS y Oscilador Theta/Gamma")
    print("="*60)
    print(f"  {'Metrica':<25s} {'Pre-B':>10s} {'Post-B':>10s} {'Post-Sueño':>12s}")
    print(f"  {'-'*60}")
    print(f"  {'Respuesta a A':<25s} {a_pre_b:>10.0f} {a_post_b:>10.0f} {a_post_sleep:>12.0f}")
    print(f"  {'Respuesta a B':<25s} {b_pre_b:>10.0f} {b_post_b:>10.0f} {b_post_sleep:>12.0f}")
    print(f"  {'w S0 (canal A)':<25s} {float(np.mean(s0_weights_post_a)):>10.4f} {float(np.mean(s0_weights_post_b)):>10.4f} {float(np.mean(s0_weights_sleep)):>12.4f}")
    print(f"  {'w S2 (canal B)':<25s} {float(np.mean(s2_weights_pre)):>10.4f} {float(np.mean(s2_weights_post_b)):>10.4f} {float(np.mean(s2_weights_sleep)):>12.4f}")

    print("\n  --- CONCLUSION ---")
    delta_recuperacion = a_post_sleep - a_post_b
    if delta_recuperacion > 100:
        print(f"  ¡ÉXITO! El Replay acoplado a Oscilaciones Theta/Gamma y HSS RESCATÓ la memoria de A (+{delta_recuperacion:.0f} spikes).")
        print(f"  La sincronía de fase potenció la coincidencia de disparo (LTP), consolidando w S0 en {float(np.mean(s0_weights_sleep)):.4f}.")
    else:
        print(f"  El acoplamiento oscilatorio no fue suficiente para rescatar la memoria (+{delta_recuperacion:.0f} spikes).")

    logger.registrar_spikes()
    logger.guardar("exp19_sleep_replay_oscilador")
    cerebro.save_brain("cerebro_sleep_replay_oscilador_exp19.pkl")

if __name__ == "__main__":
    import sys
    
    experimentos = {
        19: experimento_19
    }
    
    if len(sys.argv) > 1:
        try:
            exp_num = int(sys.argv[1])
            if exp_num in experimentos:
                print(f"\n============================================================")
                print(f"  EJECUTANDO EXPERIMENTO {exp_num} EN PROCESO INDEPENDIENTE")
                print(f"============================================================\n")
                experimentos[exp_num]()
            else:
                print(f"Error: El experimento {exp_num} no existe. Rango valido: 19.")
                sys.exit(1)
        except ValueError:
            print("Error: El argumento debe ser un numero entero 19.")
            sys.exit(1)
    else:
        print("=" * 60)
        print("  IA VIVA - EXPERIMENTOS BRIAN2 (v2)")
        print("=" * 60)
        print("\n[INFO] Nota: En Windows se recomienda correr los experimentos en procesos")
        print("separados para evitar bloqueos de archivos DLL. Ejemplo:")
        print("PowerShell: for ($i=19; $i -le 19; $i++) { .\\.venv\\Scripts\\python.exe cerebro_brian2.py $i }\n")
        print("Ejecutando suite completa secuencial...")
        
        for num, func in experimentos.items():
            func()
            print("\n" + "="*60 + "\n")
            
        print("\n" + "=" * 60)
        print("  TODOS LOS EXPERIMENTOS COMPLETADO (19)")
        print("  Revisa logs_brian2/ para los CSV.")
    print("  Revisa logs_brian2/ para los CSV.")
    print("=" * 60)
