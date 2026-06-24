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

    def update(self, dt, brain_state, prediction_error):
        self.last_pe = prediction_error
        da_target = 0.5 + 0.5 * np.tanh(abs(prediction_error) * 2.0)
        if brain_state == 'SLOW_WAVE_SLEEP':
            da_target = 0.2
        elif brain_state == 'REM':
            da_target = 0.6
        self.dopamine += (da_target - self.dopamine) * dt / self.tau_da
        ht_target = 0.7 if brain_state == 'AWAKE' else (0.4 if brain_state == 'SLOW_WAVE_SLEEP' else 0.1)
        self.serotonin += (ht_target - self.serotonin) * dt / self.tau_5ht
        ach_target = 0.8 if brain_state == 'AWAKE' else (0.2 if brain_state == 'SLOW_WAVE_SLEEP' else 0.7)
        self.acetylcholine += (ach_target - self.acetylcholine) * dt / self.tau_ach
        self.dopamine = float(np.clip(self.dopamine, 0.0, 1.0))
        self.serotonin = float(np.clip(self.serotonin, 0.0, 1.0))
        self.acetylcholine = float(np.clip(self.acetylcholine, 0.0, 1.0))

    def stdp_modulation(self):
        return 0.5 + self.dopamine

# ============================================================================
# CEREBRO BRIAN2
# ============================================================================

class BrainBrian2:
    def __init__(self, name="CerebroBrian2", n_sensory=5, n_hidden=3, n_motor=4, n_pfc=3,
                 plasticity=True, drive_amplitude=15.0):
        self.name = name
        self.time = 0.0
        self.step_count = 0
        self.brain_state = 'AWAKE'

        N_S = n_sensory; N_H = n_hidden; N_M = n_motor; N_P = n_pfc
        n_total = N_S + N_H + N_M + N_P

        layer = np.zeros(n_total, dtype=int)
        layer[N_S:N_S+N_H] = 1; layer[N_S+N_H:N_S+N_H+N_M] = 2; layer[N_S+N_H+N_M:] = 3

        subtype = np.ones(n_total, dtype=int)
        subtype[N_S+N_H:N_S+N_H+N_M] = 2

        v_rest_val = -65.0
        v_thresh_arr = np.where(subtype == 2, -45.0, np.where(subtype == 3, -52.0, -55.0))
        tau_m_arr = np.where(subtype == 2, 10.0, np.where(subtype == 3, 25.0, 15.0)) * ms

        eqs = '''
        dv/dt = (-(v - v_rest) + I_sensory + I_ext) / tau_m : 1
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
        '''

        self.neurons = NeuronGroup(
            n_total, eqs,
            threshold='v > v_thresh',
            reset='v = v_rest; last_spike = t',
            refractory='2*ms',
            method='euler'
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

        # STDP
        taupre = taupost = 20 * ms
        wmax = 1.0
        Apre = 0.05 if plasticity else 0.0
        Apost = (-Apre * taupre / taupost * 1.05) if plasticity else 0.0

        self.synapses = Synapses(
            self.neurons, self.neurons,
            '''w : 1
               dapre/dt = -apre/taupre : 1 (clock-driven)
               dapost/dt = -apost/taupost : 1 (clock-driven)''',
            on_pre='''
               v_post += w
               apre += Apre
               w = clip(w + apost, 0, wmax)
            ''',
            on_post='''
               apost += Apost
               w = clip(w + apre, 0, wmax)
            ''',
            namespace={
                'taupre': taupre, 'taupost': taupost,
                'wmax': wmax, 'Apre': Apre, 'Apost': Apost
            }
        )
        # Topologia por capas (conexiones probabilisticas)
        s_idx = np.where(layer == 0)[0]
        h_idx = np.where(layer == 1)[0]
        m_idx = np.where(layer == 2)[0]
        p_idx = np.where(layer == 3)[0]
        for pre in s_idx:
            for post in h_idx:
                if np.random.random() < 0.4:
                    self.synapses.connect(i=pre, j=post)
        for pre in h_idx:
            for post in m_idx:
                if np.random.random() < 0.4:
                    self.synapses.connect(i=pre, j=post)
        for pre in m_idx:
            for post in p_idx:
                if np.random.random() < 0.4:
                    self.synapses.connect(i=pre, j=post)
        for pre in p_idx:
            for post in m_idx:
                if np.random.random() < 0.4:
                    self.synapses.connect(i=pre, j=post)
        for pre in p_idx:
            for post in h_idx:
                if np.random.random() < 0.2:
                    self.synapses.connect(i=pre, j=post)
        for pre in h_idx:
            for post in s_idx:
                if np.random.random() < 0.2:
                    self.synapses.connect(i=pre, j=post)

        self.synapses.w = 'rand() * 0.1'
        self.synapses.apre = 0.0
        self.synapses.apost = 0.0
        self.synapses.delay = '(2 + rand() * 3) * ms'

        self.n_total = n_total
        self.N_S = N_S; self.N_H = N_H; self.N_M = N_M; self.N_P = N_P
        self.drive_amplitude = drive_amplitude
        self.drive_mode = 'oscillation'  # 'oscillation' o 'tonic'

        self.spike_mon = SpikeMonitor(self.neurons)
        self.state_mon = StateMonitor(self.neurons,
            ['v', 'firing_rate', 'I_sensory', 'energy'],
            record=range(5 if n_total > 5 else n_total))
        self.network = Network(self.neurons, self.synapses,
                                self.spike_mon, self.state_mon)

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

        pred = float(np.mean(self.neurons.firing_rate[:] / 100.0))
        target = np.sin(2 * np.pi * self.time / SIGNAL_PERIOD)
        error = target - pred
        self._last_error = error
        self.neuromod.update(500.0, self.brain_state, error)

        self.network.run(500 * ms, report=None)

        for i in range(self.n_total):
            spikes_in_window = np.sum(
                (self.spike_mon.t[(self.spike_mon.i == i)] > self.t - 100*ms) &
                (self.spike_mon.t[(self.spike_mon.i == i)] <= self.t))
            self.neurons.firing_rate[i] = spikes_in_window * 10.0 * Hz

        # Energia: se consume con spikes, se recupera lentamente
        spike_mask = np.zeros(self.n_total, dtype=bool)
        recent = self.spike_mon.t > self.t - 50*ms
        if np.any(recent):
            spike_mask[np.unique(self.spike_mon.i[recent])] = True
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
                self._episodic_buffer.append({
                    'time': self.time,
                    'da': self.neuromod.dopamine,
                    'spikes': n_spikes,
                    'w_mean': float(np.mean(self.synapses.w[:])),
                })

        self.neurons.I_sensory = 0.0

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

def experimento_1():
    separador("EXPERIMENTO 1: Dopamina y novedad")
    print("Objetivo: DA responde a novedad con spikes reales.\n")

    cerebro = BrainBrian2(name="Exp1")
    observer = BrainObserver(cerebro)
    stim = StimulusController(cerebro)
    logger = EventLogger(cerebro, "logs_brian2/exp1")
    cerebro._logger = logger

    hist_da = []; hist_w = []

    print("Basal (30 pasos)")
    stim.silencio(30)
    s = registrar_estado(observer, "basal")
    hist_da.append(('basal', s['dopamine'])); hist_w.append(('basal', s['avg_weight']))

    for intensidad in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        print(f"Novedad={intensidad}")
        stim.inyectar_novedad(intensidad)
        stim.silencio(15)
        s = registrar_estado(observer, f"nov={intensidad}")
        hist_da.append((f'nov_{intensidad}', s['dopamine']))
        hist_w.append((f'nov_{intensidad}', s['avg_weight']))

    print("Recompensa")
    stim.inyectar_recompensa(); stim.silencio(20)
    s = registrar_estado(observer, "recompensa")
    hist_da.append(('recompensa', s['dopamine'])); hist_w.append(('recompensa', s['avg_weight']))

    print("Recuperacion (40 pasos)")
    stim.silencio(40)
    s = registrar_estado(observer, "recup")
    hist_da.append(('recup', s['dopamine'])); hist_w.append(('recup', s['avg_weight']))

    print("\n" + "="*50)
    print("  RESUMEN EXP 1")
    print("="*50)
    print(f"  {'Fase':<15s} {'DA':>8s} {'w_prom':>8s}")
    print(f"  {'-'*35}")
    for i, (f, da) in enumerate(hist_da):
        print(f"  {f:<15s} {da:>8.4f} {hist_w[i][1]:>8.4f}")

    logger.registrar_neuromod(); logger.registrar_spikes()
    logger.guardar("exp1_novedad")
    cerebro.save_brain("logs_brian2/exp1/cerebro_exp1.pkl")

def experimento_2():
    separador("EXPERIMENTO 2: Frustracion y poda")
    print("Objetivo: Frustracion crece con errores, recompensa la reduce.\n")

    cerebro = BrainBrian2(name="Exp2")
    observer = BrainObserver(cerebro)
    stim = StimulusController(cerebro)
    logger = EventLogger(cerebro, "logs_brian2/exp2")
    cerebro._logger = logger

    hist = []

    print("Basal (30 pasos)"); stim.silencio(30)
    s = registrar_estado(observer, "basal")
    hist.append({'fase':'basal','n':s['total_neurons'],'s':s['total_synapses'],
                 'da':s['dopamine'],'fru':s['frustration'],'w':s['avg_weight'],
                 'spk':s['total_spikes']})

    for ronda in range(1, 11):
        print(f"Error {ronda}/10")
        stim.inyectar_error(0.8); stim.silencio(15)
        s = registrar_estado(observer, f"err_{ronda}")
        hist.append({'fase':f'err_{ronda}','n':s['total_neurons'],'s':s['total_synapses'],
                     'da':s['dopamine'],'fru':s['frustration'],'w':s['avg_weight'],
                     'spk':s['total_spikes']})

    print("Recompensa"); stim.inyectar_recompensa(); stim.silencio(20)
    s = registrar_estado(observer, "recompensa")
    hist.append({'fase':'recompensa','n':s['total_neurons'],'s':s['total_synapses'],
                 'da':s['dopamine'],'fru':s['frustration'],'w':s['avg_weight'],
                 'spk':s['total_spikes']})
    print("Recup (40 pasos)"); stim.silencio(40)
    s = registrar_estado(observer, "recup")
    hist.append({'fase':'recup','n':s['total_neurons'],'s':s['total_synapses'],
                 'da':s['dopamine'],'fru':s['frustration'],'w':s['avg_weight'],
                 'spk':s['total_spikes']})

    print("\n" + "="*70)
    print("  RESUMEN EXP 2")
    print("="*70)
    print(f"  {'Fase':<12s} {'N':>4s} {'S':>5s} {'DA':>7s} {'Fru':>7s} {'w_prom':>8s} {'Spk':>5s}")
    print(f"  {'-'*70}")
    for h in hist:
        print(f"  {h['fase']:<12s} {h['n']:>4d} {h['s']:>5d} {h['da']:>7.4f} {h['fru']:>7.4f} {h['w']:>8.4f} {h['spk']:>5d}")

    logger.registrar_neuromod(); logger.registrar_spikes()
    logger.guardar("exp2_frustracion")
    cerebro.save_brain("logs_brian2/exp2/cerebro_exp2.pkl")

def experimento_3():
    separador("EXPERIMENTO 3: Sueno REM")
    print("Objetivo: Ciclo sueno-vigilia y memoria episodica.\n")

    cerebro = BrainBrian2(name="Exp3")
    observer = BrainObserver(cerebro)
    stim = StimulusController(cerebro)
    logger = EventLogger(cerebro, "logs_brian2/exp3")
    cerebro._logger = logger

    hist = []

    print("Basal (10 pasos)"); stim.silencio(10)
    s = registrar_estado(observer, "basal")
    hist.append({'fase':'basal','da':s['dopamine'],'buf':len(cerebro._episodic_buffer),
                 'state':s['brain_state'],'spk':s['total_spikes']})

    print("Patron 10Hz (3 reps)")
    for rep in range(3):
        print(f"  Rep {rep+1}/3")
        stim.inyectar_senal_temporal(10.0, 10, 8.0)
        stim.silencio(5)
    s = registrar_estado(observer, "post_patron")
    hist.append({'fase':'post_patron','da':s['dopamine'],'buf':len(cerebro._episodic_buffer),
                 'state':s['brain_state'],'spk':s['total_spikes']})

    print("Esperar SWS...")
    while cerebro.brain_state != 'SLOW_WAVE_SLEEP':
        stim.silencio(2)
        if cerebro.step_count > 500: break
    s = registrar_estado(observer, "SWS")
    hist.append({'fase':'SWS','da':s['dopamine'],'buf':len(cerebro._episodic_buffer),
                 'state':s['brain_state'],'spk':s['total_spikes']})

    print("Esperar REM...")
    while cerebro.brain_state != 'REM':
        stim.silencio(2)
        if cerebro.step_count > 500: break
    s = registrar_estado(observer, "REM")
    hist.append({'fase':'REM','da':s['dopamine'],'buf':len(cerebro._episodic_buffer),
                 'state':s['brain_state'],'spk':s['total_spikes']})

    print("Despertar...")
    while cerebro.brain_state != 'AWAKE':
        stim.silencio(2)
        if cerebro.step_count > 500: break
    s = registrar_estado(observer, "post_sueno")
    hist.append({'fase':'post_sueno','da':s['dopamine'],'buf':len(cerebro._episodic_buffer),
                 'state':s['brain_state'],'spk':s['total_spikes']})

    print("Segunda exposicion")
    stim.inyectar_senal_temporal(10.0, 10, 8.0)
    s = registrar_estado(observer, "segunda_exp")
    hist.append({'fase':'segunda_exp','da':s['dopamine'],'buf':len(cerebro._episodic_buffer),
                 'state':s['brain_state'],'spk':s['total_spikes']})

    print("\n" + "="*60)
    print("  RESUMEN EXP 3")
    print("="*60)
    print(f"  {'Fase':<15s} {'DA':>7s} {'Buf':>5s} {'Estado':>15s} {'Spk':>5s}")
    print(f"  {'-'*60}")
    for h in hist:
        print(f"  {h['fase']:<15s} {h['da']:>7.4f} {h['buf']:>5d} {h['state']:>15s} {h['spk']:>5d}")
    print(f"\n  Buffer episodico: {len(cerebro._episodic_buffer)} entradas")

    logger.registrar_neuromod(); logger.registrar_spikes()
    logger.guardar("exp3_rem")
    cerebro.save_brain("logs_brian2/exp3/cerebro_exp3.pkl")

def experimento_4():
    separador("EXPERIMENTO 4: Plasticidad STDP")
    print("Objetivo: Ver pesos sinapticos cambiar con estimulacion repetida.\n")

    cerebro = BrainBrian2(name="Exp4")
    observer = BrainObserver(cerebro)
    stim = StimulusController(cerebro)
    logger = EventLogger(cerebro, "logs_brian2/exp4")
    cerebro._logger = logger

    print("Fase 1: Pesos iniciales")
    s = observer.snapshot()
    w_inicial = s['avg_weight']
    print(f"  w_prom inicial: {w_inicial:.4f}")

    print("Fase 2: Estimular par sensorial -> oculta (20 rondas)")
    for ronda in range(20):
        if ronda % 5 == 0:
            print(f"  Ronda {ronda+1}/20")
        stim.inyectar_senal_temporal(10.0, 10, 10.0)
        stim.silencio(3)

    s = registrar_estado(observer, "post_estimulacion")
    w_post = s['avg_weight']
    print(f"\n  w_prom inicial: {w_inicial:.4f} -> w_prom final: {w_post:.4f}")
    print(f"  Cambio: {(w_post - w_inicial) / w_inicial * 100:.1f}%")

    logger.registrar_neuromod(); logger.registrar_spikes()
    logger.guardar("exp4_stdp")
    cerebro.save_brain("logs_brian2/exp4/cerebro_exp4.pkl")

def experimento_5():
    separador("EXPERIMENTO 5: Resiliencia al ruido")
    print("Objetivo: La red mantiene actividad frente a ruido gaussiano.\n")

    cerebro = BrainBrian2(name="Exp5")
    observer = BrainObserver(cerebro)
    stim = StimulusController(cerebro)
    logger = EventLogger(cerebro, "logs_brian2/exp5")
    cerebro._logger = logger

    print("Fase 1: Sin ruido (20 pasos)")
    stim.silencio(20)
    s1 = registrar_estado(observer, "sin_ruido")

    print("Fase 2: Con ruido sensorial (30 pasos)")
    for _ in range(30):
        stim.inyectar_ruido_gaussiano(8.0, 1)
    s2 = registrar_estado(observer, "con_ruido")

    print("Fase 3: Recuperacion (20 pasos)")
    stim.silencio(20)
    s3 = registrar_estado(observer, "recuperacion")

    print(f"\n  Spikes sin ruido: {s1['total_spikes']}")
    print(f"  Spikes con ruido: {s2['total_spikes']}")
    print(f"  Spikes recuperacion: {s3['total_spikes']}")

    logger.registrar_neuromod(); logger.registrar_spikes()
    logger.guardar("exp5_ruido")
    cerebro.save_brain("logs_brian2/exp5/cerebro_exp5.pkl")

def experimento_6():
    separador("EXPERIMENTO 6: Escalado de red")
    print("Objetivo: Red ampliada (15 -> 30 neuronas) con mas conexiones.\n")

    import time as tm
    t0 = tm.time()
    cerebro = BrainBrian2(name="Exp6", n_sensory=10, n_hidden=8, n_motor=8, n_pfc=4)
    observer = BrainObserver(cerebro)
    stim = StimulusController(cerebro)
    logger = EventLogger(cerebro, "logs_brian2/exp6")
    cerebro._logger = logger
    init_t = tm.time() - t0
    print(f"Inicializacion: {init_t:.2f}s")
    print(f"Total neuronas: {cerebro.n_total}")
    print(f"Conexiones sinapticas: {len(cerebro.synapses)}")

    print("Estimulacion (40 pasos)")
    t1 = tm.time()
    for i in range(10):
        stim.inyectar_senal_temporal(5.0 + i, 4, 10.0)
    s = registrar_estado(observer, "post_estimulacion")
    sim_t = tm.time() - t1
    print(f"\n  Tiempo de simulacion: {sim_t:.2f}s")
    print(f"  Spikes totales: {s['total_spikes']}")
    print(f"  Sinapsis: {s['total_synapses']}")

    logger.registrar_neuromod(); logger.registrar_spikes()
    logger.guardar("exp6_escalado")
    cerebro.save_brain("logs_brian2/exp6/cerebro_exp6.pkl")

# ============================================================================
# EXPERIMENTOS CREATIVOS (7-11)
# ============================================================================

def experimento_7():
    separador("EXPERIMENTO 7: Aprendizaje de secuencia temporal")
    print("Objetivo: La red aprende que A->B->C es distinto de C->A->B.\n")

    cerebro = BrainBrian2(name="Exp7")
    observer = BrainObserver(cerebro)
    stim = StimulusController(cerebro)
    logger = EventLogger(cerebro, "logs_brian2/exp7")
    cerebro._logger = logger

    print("Fase 1: Entrenar secuencia A->B->C (10 repeticiones)")
    for rep in range(10):
        if rep % 3 == 0: print(f"  Entrenamiento {rep+1}/10")
        for canal in [0, 1, 2]:
            stim.inyectar_pulso(canal, 15.0, 3)
            stim.silencio(1)
        stim.silencio(2)
    w_entrenado = float(np.mean(cerebro.synapses.w[:]))
    da_entrenado = cerebro.neuromod.dopamine
    print(f"  w_prom post-entrenamiento: {w_entrenado:.4f}  DA={da_entrenado:.4f}")

    print("\nFase 2: Medir respuesta a A->B->C (secuencia aprendida)")
    spikes_abc = 0
    for _ in range(3):
        spk_before = cerebro.spike_mon.num_spikes
        for canal in [0, 1, 2]:
            stim.inyectar_pulso(canal, 15.0, 3)
            stim.silencio(1)
        stim.silencio(2)
        spikes_abc += cerebro.spike_mon.num_spikes - spk_before
    print(f"  Spikes en A->B->C: {spikes_abc}")

    print("\nFase 3: Medir respuesta a C->A->B (secuencia invertida)")
    spikes_cab = 0
    for _ in range(3):
        spk_before = cerebro.spike_mon.num_spikes
        for canal in [2, 0, 1]:
            stim.inyectar_pulso(canal, 15.0, 3)
            stim.silencio(1)
        stim.silencio(2)
        spikes_cab += cerebro.spike_mon.num_spikes - spk_before
    print(f"  Spikes en C->A->B: {spikes_cab}")

    print(f"\n  Diferencia (aprendizaje de orden): {spikes_abc} vs {spikes_cab}")
    divisor = spikes_cab if spikes_cab > 1 else 1
    ratio = spikes_abc / divisor
    print(f"  Ratio sec. aprendida / invertida: {ratio:.2f}x")
    if ratio > 1.2:
        print("  >> La red discrimina el orden temporal!")
    elif ratio < 0.8:
        print("  >> La red prefiere la secuencia INVERSA! (inesperado)")
    else:
        print("  >> Sin preferencia significativa de orden")

    logger.registrar_neuromod(); logger.registrar_spikes()
    logger.guardar("exp7_secuencia")
    cerebro.save_brain("logs_brian2/exp7/cerebro_exp7.pkl")

def experimento_8():
    separador("EXPERIMENTO 8: Privacion de sueño")
    print("Objetivo: Comparar consolidacion con vs sin sueno.\n")

    print("=== CONDICION 1: Ciclo normal con sueno ===")
    cerebro1 = BrainBrian2(name="Exp8_normal")
    observer1 = BrainObserver(cerebro1)
    stim1 = StimulusController(cerebro1)
    logger1 = EventLogger(cerebro1, "logs_brian2/exp8_normal")
    cerebro1._logger = logger1

    print("Entrenamiento (5 patrones)")
    for _ in range(5):
        stim1.inyectar_senal_temporal(10.0, 8, 10.0)
        stim1.silencio(3)

    s_pre_sueno = observer1.snapshot()
    w_pre = s_pre_sueno['avg_weight']
    print(f"  w_prom pre-sueno: {w_pre:.4f}  DA={s_pre_sueno['dopamine']:.4f}")

    print("Dejar dormir (50 pasos, cubre ciclos completos)")
    stim1.silencio(50)
    s_post_sueno = observer1.snapshot()
    w_post = s_post_sueno['avg_weight']
    buf_normal = s_post_sueno['episodic_buffer']
    print(f"  w_prom post-sueno: {w_post:.4f}  Buffer={buf_normal}  Spikes={s_post_sueno['total_spikes']}")
    delta_normal = w_post - w_pre

    print("\n=== CONDICION 2: Privado de sueno (AWAKE forzado) ===")
    cerebro2 = BrainBrian2(name="Exp8_insomnio")
    cerebro2.force_state = 'AWAKE'
    observer2 = BrainObserver(cerebro2)
    stim2 = StimulusController(cerebro2)
    logger2 = EventLogger(cerebro2, "logs_brian2/exp8_insomnio")
    cerebro2._logger = logger2

    print("Entrenamiento (5 patrones identicos)")
    for _ in range(5):
        stim2.inyectar_senal_temporal(10.0, 8, 10.0)
        stim2.silencio(3)

    s2_pre = observer2.snapshot()
    w2_pre = s2_pre['avg_weight']
    print(f"  w_prom pre-forzado: {w2_pre:.4f}  DA={s2_pre['dopamine']:.4f}")

    print("Forzar AWAKE (50 pasos sin dormir)")
    stim2.silencio(50)
    s2_post = observer2.snapshot()
    w2_post = s2_post['avg_weight']
    buf_insomnio = s2_post['episodic_buffer']
    print(f"  w_prom post-forzado: {w2_post:.4f}  Buffer={buf_insomnio}  Spikes={s2_post['total_spikes']}")
    delta_insomnio = w2_post - w2_pre

    print("\n" + "="*60)
    print("  COMPARATIVA: Sueno vs Insomnio")
    print("="*60)
    print(f"  {'Metrica':<25s} {'Con sueno':>15s} {'Sin sueno':>15s}")
    print(f"  {'-'*55}")
    print(f"  {'Delta w_prom':<25s} {delta_normal:>15.4f} {delta_insomnio:>15.4f}")
    print(f"  {'Buffer episodico':<25s} {buf_normal:>15d} {buf_insomnio:>15d}")
    print(f"  {'DA final':<25s} {s_post_sueno['dopamine']:>15.4f} {s2_post['dopamine']:>15.4f}")
    print(f"  {'Energia final':<25s} {s_post_sueno['energy_avg']:>15.3f} {s2_post['energy_avg']:>15.3f}")

    if delta_normal > delta_insomnio:
        print("\n  >> El sueño MEJORA la consolidacion de pesos (+%.4f vs +%.4f)" % (delta_normal, delta_insomnio))
    else:
        print("\n  >> La privacion de sueño NO afecto negativamente (inesperado)")

    logger1.registrar_spikes(); logger1.guardar("exp8_normal")
    logger2.registrar_spikes(); logger2.guardar("exp8_insomnio")
    cerebro1.save_brain("logs_brian2/exp8/cerebro_exp8_normal.pkl")
    cerebro2.save_brain("logs_brian2/exp8/cerebro_exp8_insomnio.pkl")

def experimento_9():
    separador("EXPERIMENTO 9: Modulacion emocional de memoria")
    print("Objetivo: Frustracion vs recompensa afectan la consolidacion.\n")

    print("=== CONDICION 1: Aprendizaje CON frustracion ===")
    cerebro1 = BrainBrian2(name="Exp9_fru")
    observer1 = BrainObserver(cerebro1)
    stim1 = StimulusController(cerebro1)
    logger1 = EventLogger(cerebro1, "logs_brian2/exp9_fru")
    cerebro1._logger = logger1

    for rep in range(8):
        stim1.inyectar_senal_temporal(10.0, 5, 8.0)
        stim1.inyectar_error(0.5)
        stim1.silencio(3)
    s1_pre_sueno = observer1.snapshot()
    w1_pre = s1_pre_sueno['avg_weight']
    da1_pre = s1_pre_sueno['dopamine']
    fru1 = s1_pre_sueno['frustration']
    print(f"  Post-aprendizaje: w={w1_pre:.4f}  DA={da1_pre:.4f}  Fru={fru1:.4f}")

    print("  Dejar dormir (30 pasos)")
    stim1.silencio(30)
    s1_post = observer1.snapshot()
    delta1 = s1_post['avg_weight'] - w1_pre
    buf1 = s1_post['episodic_buffer']
    print(f"  Post-sueno: w={s1_post['avg_weight']:.4f}  Delta={delta1:+.4f}  Buffer={buf1}")

    print("\n=== CONDICION 2: Aprendizaje CON recompensa ===")
    cerebro2 = BrainBrian2(name="Exp9_rec")
    observer2 = BrainObserver(cerebro2)
    stim2 = StimulusController(cerebro2)
    logger2 = EventLogger(cerebro2, "logs_brian2/exp9_rec")
    cerebro2._logger = logger2

    for rep in range(8):
        stim2.inyectar_senal_temporal(10.0, 5, 8.0)
        stim2.inyectar_recompensa()
        stim2.silencio(3)
    s2_pre_sueno = observer2.snapshot()
    w2_pre = s2_pre_sueno['avg_weight']
    da2_pre = s2_pre_sueno['dopamine']
    fru2 = s2_pre_sueno['frustration']
    print(f"  Post-aprendizaje: w={w2_pre:.4f}  DA={da2_pre:.4f}  Fru={fru2:.4f}")

    print("  Dejar dormir (30 pasos)")
    stim2.silencio(30)
    s2_post = observer2.snapshot()
    delta2 = s2_post['avg_weight'] - w2_pre
    buf2 = s2_post['episodic_buffer']
    print(f"  Post-sueno: w={s2_post['avg_weight']:.4f}  Delta={delta2:+.4f}  Buffer={buf2}")

    print("\n" + "="*60)
    print("  COMPARATIVA: Frustracion vs Recompensa")
    print("="*60)
    print(f"  {'Metrica':<25s} {'Frustracion':>15s} {'Recompensa':>15s}")
    print(f"  {'-'*55}")
    print(f"  {'DA pre-sueno':<25s} {da1_pre:>15.4f} {da2_pre:>15.4f}")
    print(f"  {'Delta w_prom':<25s} {delta1:>+15.4f} {delta2:>+15.4f}")
    print(f"  {'Buffer episodico':<25s} {buf1:>15d} {buf2:>15d}")

    if abs(delta2) > abs(delta1):
        print("\n  >> La RECOMPENSA genera mayor consolidacion (emocion positiva -> mejor memoria)")
    elif abs(delta1) > abs(delta2):
        print("\n  >> La FRUSTRACION genera mayor consolidacion (memoria emocional intensa)")
    else:
        print("\n  >> Sin diferencia significativa entre condiciones emocionales")

    logger1.registrar_spikes(); logger1.guardar("exp9_frustracion")
    logger2.registrar_spikes(); logger2.guardar("exp9_recompensa")
    cerebro1.save_brain("logs_brian2/exp9/cerebro_exp9_frustracion.pkl")
    cerebro2.save_brain("logs_brian2/exp9/cerebro_exp9_recompensa.pkl")

def experimento_10():
    separador("EXPERIMENTO 10: Periodo critico de plasticidad")
    print("Objetivo: La plasticidad es mayor en etapas tempranas.\n")

    cerebro = BrainBrian2(name="Exp10")
    observer = BrainObserver(cerebro)
    stim = StimulusController(cerebro)
    logger = EventLogger(cerebro, "logs_brian2/exp10")
    cerebro._logger = logger

    # Fase 1: Plasticidad temprana (pasos 0-50)
    print("Fase 1: Estimulacion temprana (20 pasos)")
    s_inicial = observer.snapshot()
    w_inicial = s_inicial['avg_weight']

    cambios_tempranos = []
    for paso in range(20):
        w_pre = float(np.mean(cerebro.synapses.w[:]))
        stim.inyectar_senal_temporal(10.0, 2, 8.0)
        stim.silencio(1)
        w_post = float(np.mean(cerebro.synapses.w[:]))
        cambios_tempranos.append(w_post - w_pre)

    delta_temprano = float(np.mean(cerebro.synapses.w[:])) - w_inicial
    s_temprano = observer.snapshot()
    print(f"  w_inicial={w_inicial:.4f}  w_final_temprano={s_temprano['avg_weight']:.4f}")
    print(f"  Delta temprano: {delta_temprano:+.4f}")
    print(f"  Cambio promedio por paso: {np.mean(np.abs(cambios_tempranos)):.6f}")
    print(f"  Spikes fase temprana: {s_temprano['total_spikes']}")

    # Fase 2: Plasticidad tardia (pasos 200-250)
    print(f"\nFase 2: Avanzar a periodo tardio (hasta paso 200 con silencio)")
    while cerebro.step_count < 200:
        stim.silencio(5)
    w_tardio_inicial = float(np.mean(cerebro.synapses.w[:]))

    print("Fase 2: Estimulacion tardia (20 pasos)")
    cambios_tardios = []
    for paso in range(20):
        w_pre = float(np.mean(cerebro.synapses.w[:]))
        stim.inyectar_senal_temporal(10.0, 2, 8.0)
        stim.silencio(1)
        w_post = float(np.mean(cerebro.synapses.w[:]))
        cambios_tardios.append(w_post - w_pre)

    delta_tardio = float(np.mean(cerebro.synapses.w[:])) - w_tardio_inicial
    s_tardio = observer.snapshot()
    print(f"  w_inicial_tardio={w_tardio_inicial:.4f}  w_final_tardio={s_tardio['avg_weight']:.4f}")
    print(f"  Delta tardio: {delta_tardio:+.4f}")
    print(f"  Cambio promedio por paso: {np.mean(np.abs(cambios_tardios)):.6f}")
    print(f"  Spikes fase tardia: {s_tardio['total_spikes'] - s_temprano['total_spikes']}")

    print("\n" + "="*60)
    print("  COMPARATIVA: Periodo temprano vs tardio")
    print("="*60)
    print(f"  {'Metrica':<30s} {'Temprano':>12s} {'Tardio':>12s}")
    print(f"  {'-'*55}")
    print(f"  {'Delta total':<30s} {delta_temprano:>+12.4f} {delta_tardio:>+12.4f}")
    print(f"  {'Cambio medio/paso':<30s} {np.mean(np.abs(cambios_tempranos)):>12.6f} {np.mean(np.abs(cambios_tardios)):>12.6f}")

    if abs(delta_temprano) > abs(delta_tardio) * 1.3:
        print("\n  >> Existe PERIODO CRITICO: la plasticidad temprana es mayor!")
    elif abs(delta_tardio) > abs(delta_temprano) * 1.3:
        print("\n  >> Plasticidad TARDIA mayor (posible consolidacion acumulada)")
    else:
        div = abs(delta_tardio) if abs(delta_tardio) > 1e-6 else 1e-6
        print(f"\n  >> Plasticidad similar en ambas fases (razon={abs(delta_temprano)/div:.1f}x)")

    logger.registrar_neuromod(); logger.registrar_spikes()
    logger.guardar("exp10_critico")
    cerebro.save_brain("logs_brian2/exp10/cerebro_exp10.pkl")

def experimento_11():
    separador("EXPERIMENTO 11: Incertidumbre y prediccion")
    print("Objetivo: La DA refleja error de prediccion bajo ruido sensorial.")
    print("Hipotesis: A mayor ruido, mayor error de prediccion, mayor DA.\n")

    cerebro = BrainBrian2(name="Exp11")
    cerebro.force_state = 'AWAKE'  # Eliminar interferencia del ciclo sueño
    observer = BrainObserver(cerebro)
    stim = StimulusController(cerebro)
    logger = EventLogger(cerebro, "logs_brian2/exp11")
    cerebro._logger = logger

    print("Fase 1: Estabilizar DA basal (20 pasos AWAKE forzado)")
    stim.silencio(20)
    s_base = observer.snapshot()
    da_basal = s_base['dopamine']
    spk_basal = s_base['total_spikes']
    print(f"  DA basal estabilizada: {da_basal:.4f}  Spikes: {spk_basal}")

    niveles_ruido = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    resultados = []

    print("\nFase 2: Medir DA bajo niveles de ruido creciente")
    for nivel in niveles_ruido:
        # Estabilizar: 5 pasos con ruido constante + señal base
        spk_antes = cerebro.spike_mon.num_spikes
        rates_ronda = []
        for _ in range(8):
            # Inyectar señal de referencia (siempre igual)
            for i in range(cerebro.N_S):
                cerebro.neurons.I_sensory[i] = 5.0
            # Añadir ruido gaussiano adicional
            if nivel > 0:
                for i in range(cerebro.N_S):
                    cerebro.neurons.I_sensory[i] += nivel * np.random.randn()
            cerebro.step()
            rates_ronda.append(float(np.mean(cerebro.neurons.firing_rate[:] / Hz)))

        spikes_ronda = cerebro.spike_mon.num_spikes - spk_antes

        # Dejar estabilizar DA 5 pasos extra con el mismo nivel de ruido
        for _ in range(5):
            for i in range(cerebro.N_S):
                cerebro.neurons.I_sensory[i] = 5.0
                if nivel > 0:
                    cerebro.neurons.I_sensory[i] += nivel * np.random.randn()
            cerebro.step()

        s = observer.snapshot()
        da_respuesta = s['dopamine']
        rate_mean = np.mean(rates_ronda) if rates_ronda else 0
        rate_std = np.std(rates_ronda) if len(rates_ronda) > 1 else 0

        resultados.append({
            'ruido': nivel,
            'da': da_respuesta,
            'spikes': spikes_ronda,
            'energia': s['energy_avg'],
            'rate_mean': rate_mean,
            'rate_std': rate_std,
        })
        print(f"  Ruido={nivel:>4.0f}  DA={da_respuesta:.4f}  Spikes={spikes_ronda:>4d}  "
              f"Rate={rate_mean:.1f}Hz ±{rate_std:.1f}")

    # Volver a basal para comparar
    print("\nFase 3: Recuperacion post-ruido (10 pasos sin ruido)")
    stim.silencio(10)
    s_rec = observer.snapshot()
    da_rec = s_rec['dopamine']

    print("\n" + "="*65)
    print("  RESULTADOS: Dopamina como señal de incertidumbre")
    print("="*65)
    print(f"  {'Ruido':>6s} {'DA':>8s} {'Spikes':>7s} {'Rate μ':>7s} {'Rate σ':>7s} {'Energia':>8s}")
    print(f"  {'-'*55}")
    for r in resultados:
        print(f"  {r['ruido']:>6.1f} {r['da']:>8.4f} {r['spikes']:>7d} {r['rate_mean']:>7.1f} "
              f"{r['rate_std']:>7.1f} {r['energia']:>8.3f}")

    # Analisis estadistico
    ruidos_arr = np.array([r['ruido'] for r in resultados])
    das_arr = np.array([r['da'] for r in resultados])
    rates_std_arr = np.array([r['rate_std'] for r in resultados])
    spikes_arr = np.array([r['spikes'] for r in resultados])

    corr_da_ruido = np.corrcoef(ruidos_arr, das_arr)[0, 1] if len(resultados) > 2 else 0
    corr_da_var = np.corrcoef(rates_std_arr, das_arr)[0, 1] if len(resultados) > 2 else 0
    corr_spk_ruido = np.corrcoef(ruidos_arr, spikes_arr)[0, 1] if len(resultados) > 2 else 0

    print(f"\n  Correlaciones (Pearson):")
    print(f"    DA vs Ruido       : {corr_da_ruido:+.3f}")
    print(f"    DA vs Var. Rate   : {corr_da_var:+.3f}")
    print(f"    Spikes vs Ruido   : {corr_spk_ruido:+.3f}")
    print(f"\n  DA basal: {da_basal:.4f}  DA max: {das_arr.max():.4f}  "
          f"DA recuperacion: {da_rec:.4f}")

    # Conclusion
    print("\n  --- Interpretacion ---")
    if corr_da_ruido > 0.5:
        print(f"  DA CORRELACIONA FUERTEMENTE con ruido (r={corr_da_ruido:.2f}).")
        print("  La red usa dopamina como señal de incertidumbre: a mas ruido, mas DA.")
    elif corr_da_ruido > 0.2:
        print(f"  DA correlaciona DEBILMENTE con ruido (r={corr_da_ruido:.2f}).")
        print("  La DA responde parcialmente al ruido, pero otros factores influyen.")
    elif corr_da_ruido < -0.2:
        print(f"  DA correlaciona NEGATIVAMENTE con ruido (r={corr_da_ruido:.2f}).")
        print("  La red REDUCE DA bajo incertidumbre (posible efecto inhibitorio).")
    else:
        print(f"  DA NO correlaciona con ruido (r={corr_da_ruido:.2f}).")
        print("  El sistema de error de prediccion no es suficientemente sensible al ruido.")

    if corr_spk_ruido > 0.5:
        print(f"  Los spikes AUMENTAN con ruido (r={corr_spk_ruido:.2f}) — respuesta sensorial intacta.")

    logger.registrar_spikes()
    logger.guardar("exp11_incertidumbre")
    cerebro.save_brain("logs_brian2/exp11/cerebro_exp11.pkl")

def experimento_12():
    separador("EXPERIMENTO 12: Selectividad a estimulos")
    print("Pregunta: ¿STDP no supervisado genera preferencia por estimulo entrenado?")
    print("(Topologia por capas, drive=0 durante test, sin saturacion previa)\n")

    def medir_respuesta(cerebro, canal, duracion=5):
        spk_antes = cerebro.spike_mon.num_spikes
        for _ in range(duracion):
            cerebro.neurons.I_sensory[canal] = 15.0
            cerebro.step()
        cerebro.neurons.I_sensory[canal] = 0.0
        return cerebro.spike_mon.num_spikes - spk_antes

    def medir_3x(cerebro, canal, duracion=5):
        medidas = [medir_respuesta(cerebro, canal, duracion) for _ in range(3)]
        for _ in range(3): cerebro.step()
        return np.mean(medidas), np.std(medidas)

    seeds = [42, 123, 456]
    resultados = []

    for seed in seeds:
        np.random.seed(seed)
        print(f"\n=== Semilla {seed} ===")

        # --- STDP ---
        cerebro = BrainBrian2(name=f"Exp12_STDP_s{seed}", n_sensory=10, n_hidden=15,
                              n_motor=10, n_pfc=5, plasticity=True, drive_amplitude=0.0)
        cerebro.force_state = 'AWAKE'
        logger = EventLogger(cerebro, f"logs_brian2/exp12_STDP_s{seed}")
        cerebro._logger = logger

        print("  Reposo (5 pasos, I_ext=0)")
        for _ in range(5): cerebro.step()
        w_pre = float(np.mean(cerebro.synapses.w[:]))
        n_syn = len(cerebro.synapses)
        print(f"    Sinapsis: {n_syn}  w_prom: {w_pre:.4f}")

        print("  Entrenamiento A (neurona 0, 30 rondas, drive=10)")
        cerebro.drive_amplitude = 10.0
        for ronda in range(30):
            for _ in range(5):
                cerebro.neurons.I_sensory[0] = 15.0
                cerebro.step()
            cerebro.neurons.I_sensory[0] = 0.0
            for _ in range(3): cerebro.step()
        w_post = float(np.mean(cerebro.synapses.w[:]))
        print(f"    w_post_entrenamiento: {w_post:.4f} (x{w_post/w_pre:.1f})")

        print("  Test (drive tonico=9, los pesos mandan)")
        cerebro.drive_amplitude = 9.0
        cerebro.drive_mode = 'tonic'
        mu_a, std_a = medir_3x(cerebro, canal=0)
        mu_b, std_b = medir_3x(cerebro, canal=2)
        delta_stdp = mu_a - mu_b
        print(f"    A={mu_a:.0f}±{std_a:.0f}  B={mu_b:.0f}±{std_b:.0f}  Δ={delta_stdp:+.0f}")

        resultados.append({
            'seed': seed, 'cond': 'STDP',
            'mu_a': mu_a, 'std_a': std_a,
            'mu_b': mu_b, 'std_b': std_b,
            'delta': delta_stdp, 'w_pre': w_pre, 'w_post': w_post, 'syn': n_syn,
        })
        logger.registrar_spikes(); logger.guardar(f"exp12_STDP_s{seed}")
        cerebro.save_brain(f"logs_brian2/exp12_STDP_s{seed}/cerebro.pkl")

        # --- CTRL ---
        cerebro2 = BrainBrian2(name=f"Exp12_CTRL_s{seed}", n_sensory=10, n_hidden=15,
                               n_motor=10, n_pfc=5, plasticity=False, drive_amplitude=0.0)
        cerebro2.force_state = 'AWAKE'
        logger2 = EventLogger(cerebro2, f"logs_brian2/exp12_CTRL_s{seed}")
        cerebro2._logger = logger2

        print("  Control (plasticidad OFF)")
        for _ in range(5): cerebro2.step()
        w2_pre = float(np.mean(cerebro2.synapses.w[:]))
        n2_syn = len(cerebro2.synapses)
        print(f"    Sinapsis: {n2_syn}  w_prom: {w2_pre:.4f}")

        cerebro2.drive_amplitude = 10.0
        for ronda in range(30):
            for _ in range(5):
                cerebro2.neurons.I_sensory[0] = 15.0
                cerebro2.step()
            cerebro2.neurons.I_sensory[0] = 0.0
            for _ in range(3): cerebro2.step()
        w2_post = float(np.mean(cerebro2.synapses.w[:]))

        cerebro2.drive_amplitude = 9.0
        cerebro2.drive_mode = 'tonic'
        mu2_a, std2_a = medir_3x(cerebro2, canal=0)
        mu2_b, std2_b = medir_3x(cerebro2, canal=2)
        delta_ctrl = mu2_a - mu2_b
        print(f"    A={mu2_a:.0f}±{std2_a:.0f}  B={mu2_b:.0f}±{std2_b:.0f}  Δ={delta_ctrl:+.0f}")

        resultados.append({
            'seed': seed, 'cond': 'CTRL',
            'mu_a': mu2_a, 'std_a': std2_a,
            'mu_b': mu2_b, 'std_b': std2_b,
            'delta': delta_ctrl, 'w_pre': w2_pre, 'w_post': w2_post, 'syn': n2_syn,
        })
        logger2.registrar_spikes(); logger2.guardar(f"exp12_CTRL_s{seed}")
        cerebro2.save_brain(f"logs_brian2/exp12_CTRL_s{seed}/cerebro.pkl")

    # Analisis
    print("\n" + "=" * 70)
    print("  RESULTADOS")
    print("=" * 70)
    print(f"\n  {'Seed':>5s} {'Cond':>5s} {'Syn':>5s} {'A':>8s} {'B':>8s} {'Δ':>8s} {'w_pre':>7s} {'w_post':>7s}")
    print(f"  {'-'*62}")
    for r in resultados:
        print(f"  {r['seed']:>5d} {r['cond']:>5s} {r['syn']:>5d} {r['mu_a']:>8.1f} {r['mu_b']:>8.1f} "
              f"{r['delta']:>+8.1f} {r['w_pre']:>7.4f} {r['w_post']:>7.4f}")

    deltas_stdp = [r['delta'] for r in resultados if r['cond'] == 'STDP']
    deltas_ctrl = [r['delta'] for r in resultados if r['cond'] == 'CTRL']
    pesos_stdp = [(r['w_post']/r['w_pre'] if r['w_pre']>0 else 1) for r in resultados if r['cond']=='STDP']
    pesos_ctrl = [(r['w_post']/r['w_pre'] if r['w_pre']>0 else 1) for r in resultados if r['cond']=='CTRL']

    mean_d = np.mean(deltas_stdp); mean_c = np.mean(deltas_ctrl)
    print(f"\n  Δ(A-B) STDP: {mean_d:+.1f}±{np.std(deltas_stdp):.1f}  CTRL: {mean_c:+.1f}±{np.std(deltas_ctrl):.1f}")
    print(f"  w_post/w_pre STDP: {np.mean(pesos_stdp):.1f}x  CTRL: {np.mean(pesos_ctrl):.1f}x")

    print("\n  --- CONCLUSION ---")
    if abs(mean_d) > 100 and abs(mean_c) < 10:
        print(f"  H1 SOPORTADA: Selectividad emergente detectada.")
        print(f"  STDP: Δ(A-B) = {mean_d:+.0f} ± {np.std(deltas_stdp):.0f} spikes")
        print(f"  CTRL: Δ(A-B) = {mean_c:+.0f} ± {np.std(deltas_ctrl):.0f} spikes")
        print(f"  El estimulo entrenado (A) produce {abs(mean_d/mean_c) if mean_c else abs(mean_d):.0f}x mas respuesta.")
        print(f"  La red desarrollo preferencia medible por la via entrenada.")
        print(f"  La variacion entre seeds ({np.min(deltas_stdp):.0f}-{np.max(deltas_stdp):.0f}) refleja")
        print(f"  diferencias en el cableado aleatorio inicial.")
    elif abs(mean_d) <= abs(mean_c) and abs(mean_d) < 10:
        print("  H0 NO RECHAZADA. Sin evidencia de selectividad.")
        print("  La red no responde diferente al estimulo entrenado vs novedoso.")
    else:
        print(f"  Efecto presente pero limitado (Δ STDP={mean_d:.0f} vs CTRL={mean_c:.0f}).")
        print("  La selectividad existe pero es debil.")

def experimento_13():
    separador("EXPERIMENTO 13: Torneo de cerebros")
    print("3 cerebros STDP compiten. El de mayor selectividad sobrevive.")
    print("Rondas: selectividad pura -> reentrenamiento -> presion de ruido\n")

    def entrenar(cerebro, rondas=30):
        cerebro.drive_amplitude = 10.0
        for _ in range(rondas):
            for _ in range(5):
                cerebro.neurons.I_sensory[0] = 15.0
                cerebro.step()
            cerebro.neurons.I_sensory[0] = 0.0
            for _ in range(3): cerebro.step()

    def medir_respuesta(cerebro, canal, duracion=5, ruido=0):
        spk_antes = cerebro.spike_mon.num_spikes
        for _ in range(duracion):
            cerebro.neurons.I_sensory[canal] = 15.0
            if ruido > 0:
                for i in range(cerebro.N_S):
                    cerebro.neurons.I_sensory[i] += ruido * np.random.randn()
            cerebro.step()
        cerebro.neurons.I_sensory[canal] = 0.0
        return cerebro.spike_mon.num_spikes - spk_antes

    def testear_selectividad(cerebro, ruido=0):
        cerebro.drive_amplitude = 9.0
        cerebro.drive_mode = 'tonic'
        resp_a = [medir_respuesta(cerebro, 0, 5, ruido) for _ in range(3)]
        for _ in range(3): cerebro.step()
        resp_b = [medir_respuesta(cerebro, 2, 5, ruido) for _ in range(3)]
        for _ in range(3): cerebro.step()
        delta = np.mean(resp_a) - np.mean(resp_b)
        return np.mean(resp_a), np.mean(resp_b), delta

    def crear_concursante(seed, name, plasticity=True):
        np.random.seed(seed)
        cerebro = BrainBrian2(name=name, n_sensory=10, n_hidden=15, n_motor=10, n_pfc=5,
                              plasticity=plasticity, drive_amplitude=0.0)
        cerebro.force_state = 'AWAKE'
        for _ in range(5): cerebro.step()
        return cerebro

    # ===== RONDA 1: Selectividad pura =====
    print("="*60)
    print("  RONDA 1 — Selectividad pura (entrenar 30 rondas, test A vs B)")
    print("="*60)

    concursantes = {}
    for s, name in [(101, "Alfa"), (102, "Beta"), (103, "Gamma")]:
        c = crear_concursante(s, name)
        w_pre = float(np.mean(c.synapses.w[:]))
        entrenar(c, 30)
        w_post = float(np.mean(c.synapses.w[:]))
        mu_a, mu_b, delta = testear_selectividad(c)
        concursantes[name] = {
            'cerebro': c, 'seed': s, 'w_pre': w_pre, 'w_post': w_post,
            'mu_a': mu_a, 'mu_b': mu_b, 'delta': delta,
        }
        print(f"  {name} (s{s}): w {w_pre:.3f}->{w_post:.3f}  A={mu_a:.0f} B={mu_b:.0f}  Δ={delta:+.0f}")

    # CTRL de referencia
    ctrl_ref = crear_concursante(201, "CTRL_ref", plasticity=False)
    w_pre_ctrl = float(np.mean(ctrl_ref.synapses.w[:]))
    entrenar(ctrl_ref, 30)
    w_post_ctrl = float(np.mean(ctrl_ref.synapses.w[:]))
    mu_a_ctrl, mu_b_ctrl, delta_ctrl = testear_selectividad(ctrl_ref)
    print(f"  CTRL (s201): w {w_pre_ctrl:.3f}->{w_post_ctrl:.3f}  A={mu_a_ctrl:.0f} B={mu_b_ctrl:.0f}  Δ={delta_ctrl:+.0f}")

    # Eliminar peor STDP
    eliminado_r1 = sorted(concursantes, key=lambda n: concursantes[n]['delta'])[0]
    print(f"\n  ELIMINADO R1: {eliminado_r1} (Δ={concursantes[eliminado_r1]['delta']:+.0f})")
    del concursantes[eliminado_r1]

    # ===== RONDA 2: Re-entrenamiento =====
    print("\n" + "="*60)
    print("  RONDA 2 — Re-entrenamiento (+20 rondas)")
    print("="*60)

    for name in concursantes:
        c = concursantes[name]['cerebro']
        w_pre_r2 = float(np.mean(c.synapses.w[:]))
        entrenar(c, 20)
        w_post_r2 = float(np.mean(c.synapses.w[:]))
        mu_a, mu_b, delta = testear_selectividad(c)
        concursantes[name].update({
            'w_post_r2': w_post_r2, 'delta_r2': delta,
            'mu_a_r2': mu_a, 'mu_b_r2': mu_b,
        })
        print(f"  {name}: w {w_pre_r2:.3f}->{w_post_r2:.3f}  A={mu_a:.0f} B={mu_b:.0f}  Δ={delta:+.0f}")

    eliminado_r2 = sorted(concursantes, key=lambda n: concursantes[n]['delta_r2'])[0]
    print(f"\n  ELIMINADO R2: {eliminado_r2} (Δ={concursantes[eliminado_r2]['delta_r2']:+.0f})")
    del concursantes[eliminado_r2]
    campeon_name = list(concursantes.keys())[0]
    campeon = concursantes[campeon_name]['cerebro']

    # ===== RONDA 3: Presion de ruido =====
    print("\n" + "="*60)
    print("  RONDA 3 — Bajo presion (ruido creciente)")
    print("="*60)

    ctrl_nuevo = crear_concursante(202, "CTRL_r3", plasticity=False)
    entrenar(ctrl_nuevo, 50)

    niveles_ruido = [0, 2, 4, 6]
    print(f"  {'Ruido':>6s} {'Campeon Δ':>10s} {'CTRL Δ':>10s}")
    print(f"  {'-'*30}")
    for ruido in niveles_ruido:
        _, _, d_camp = testear_selectividad(campeon, ruido=ruido)
        _, _, d_ctrl = testear_selectividad(ctrl_nuevo, ruido=ruido)
        print(f"  {ruido:>6.0f} {d_camp:>+10.0f} {d_ctrl:>+10.0f}")

    # Verificacion final
    _, _, delta_final = testear_selectividad(campeon)
    print(f"\n  CAMPEON: {campeon_name} (s{concursantes[campeon_name]['seed']})")
    print(f"  Selectividad final: Δ={delta_final:+.0f} spikes")
    print(f"  Pesos finales: {float(np.mean(campeon.synapses.w[:])):.4f}")

    if delta_final > 100:
        print("  >> Torneo VALIDO: el campeon discrimina claramente.")
    else:
        print("  >> Torneo FALLIDO: el campeon no discrimina mejor que el azar.")

    campeon.save_brain("cerebro_campeon.pkl")

def experimento_14():
    separador("EXPERIMENTO 14: Aprendizaje secuencial (A luego B)")
    print("Pregunta: Si Beta ya sabe A, ¿entrenar B interfiere con A?")
    print("Depende de la topologia de Beta (s102) — no esta programado.\n")

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
        spk_antes = cerebro.spike_mon.num_spikes
        for _ in range(duracion):
            cerebro.neurons.I_sensory[canal] = 15.0
            cerebro.step()
        cerebro.neurons.I_sensory[canal] = 0.0
        return cerebro.spike_mon.num_spikes - spk_antes

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
                  f"Δ(A,B)={mu_a-mu_b:+.0f}  Δ(A,C)={mu_a-mu_c:+.0f}")
        return mu_a, mu_b, mu_c

    # ===== Recrear Beta =====
    np.random.seed(102)
    cerebro = BrainBrian2(name="Beta_Seq", n_sensory=10, n_hidden=15, n_motor=10, n_pfc=5,
                          plasticity=True, drive_amplitude=0.0)
    cerebro.force_state = 'AWAKE'
    for _ in range(5): cerebro.step()

    # Pesos iniciales de las neuronas sensoriales 0, 2, 4 hacia capa oculta
    s0_weights_pre = cerebro.synapses.w[cerebro.synapses.i == 0][:]
    s2_weights_pre = cerebro.synapses.w[cerebro.synapses.i == 2][:]
    s4_weights_pre = cerebro.synapses.w[cerebro.synapses.i == 4][:]
    print(f"  Topologia Beta: {len(s0_weights_pre)}/{len(s2_weights_pre)}/{len(s4_weights_pre)} "
          f"conexiones S0/S2/S4 -> oculta")
    print(f"  Pesos iniciales S0: {float(np.mean(s0_weights_pre)):.4f}  "
          f"S2: {float(np.mean(s2_weights_pre)):.4f}  S4: {float(np.mean(s4_weights_pre)):.4f}")

    # ===== Fase 1: Entrenar A =====
    print("\n  --- FASE 1: Entrenar A (neurona 0, 50 rondas) ---")
    entrenar_estimulo(cerebro, canal=0, rondas=50)
    w_post_a = float(np.mean(cerebro.synapses.w[:, :]))
    s0_weights_post_a = cerebro.synapses.w[cerebro.synapses.i == 0][:]
    s2_weights_post_a = cerebro.synapses.w[cerebro.synapses.i == 2][:]
    print(f"  w_prom global: {w_post_a:.4f}")
    print(f"  w S0: {float(np.mean(s0_weights_post_a)):.4f}  "
          f"w S2: {float(np.mean(s2_weights_post_a)):.4f}  "
          f"(S2 NO fue entrenada)")

    # Test pre-B
    a_pre, b_pre, c_pre = test_triple(cerebro, "Pre-B ")

    # ===== Fase 2: Entrenar B =====
    print("\n  --- FASE 2: Entrenar B (neurona 2, 30 rondas) ---")
    entrenar_estimulo(cerebro, canal=2, rondas=30)
    w_post_b = float(np.mean(cerebro.synapses.w[:, :]))
    s0_weights_post_b = cerebro.synapses.w[cerebro.synapses.i == 0][:]
    s2_weights_post_b = cerebro.synapses.w[cerebro.synapses.i == 2][:]
    s4_weights_post_b = cerebro.synapses.w[cerebro.synapses.i == 4][:]
    print(f"  w_prom global: {w_post_b:.4f}")
    print(f"  w S0: {float(np.mean(s0_weights_post_b)):.4f}  "
          f"w S2: {float(np.mean(s2_weights_post_b)):.4f}  "
          f"w S4: {float(np.mean(s4_weights_post_b)):.4f}")

    # Test post-B
    a_post, b_post, c_post = test_triple(cerebro, "Post-B")

    # ===== ANALISIS =====
    print("\n" + "="*60)
    print("  RESULTADOS: Interferencia o independencia")
    print("="*60)
    print(f"  {'Metrica':<25s} {'Pre-B':>10s} {'Post-B':>10s} {'Cambio':>10s}")
    print(f"  {'-'*55}")
    print(f"  {'Respuesta a A':<25s} {a_pre:>10.0f} {a_post:>10.0f} {a_post-a_pre:>+10.0f}")
    print(f"  {'Respuesta a B':<25s} {b_pre:>10.0f} {b_post:>10.0f} {b_post-b_pre:>+10.0f}")
    print(f"  {'Respuesta a C (ctrl)':<25s} {c_pre:>10.0f} {c_post:>10.0f} {c_post-c_pre:>+10.0f}")
    print(f"  {'Δ(A,B) selectividad':<25s} {a_pre-b_pre:>+10.0f} {a_post-b_post:>+10.0f} "
          f"{(a_post-b_post)-(a_pre-b_pre):>+10.0f}")
    print(f"  {'w S0 (via entrenada A)':<25s} {float(np.mean(s0_weights_post_a)):>10.4f} "
          f"{float(np.mean(s0_weights_post_b)):>10.4f} "
          f"{float(np.mean(s0_weights_post_b)-np.mean(s0_weights_post_a)):>+10.4f}")
    print(f"  {'w S2 (via entrenada B)':<25s} {float(np.mean(s2_weights_post_a)):>10.4f} "
          f"{float(np.mean(s2_weights_post_b)):>10.4f} "
          f"{float(np.mean(s2_weights_post_b)-np.mean(s2_weights_post_a)):>+10.4f}")

    # Conclusion
    delta_selectividad = (a_post - b_post) - (a_pre - b_pre)
    delta_a = a_post - a_pre
    delta_c = c_post - c_pre

    print("\n  --- CONCLUSION ---")
    if delta_selectividad < -50:
        print(f"  INTERFERENCIA: entrenar B DAÑO la memoria de A (Δselectividad={delta_selectividad:+.0f}).")
        print(f"  Las vias de A y B comparten suficientes sinapsis para que B pise a A.")
    elif abs(delta_selectividad) < 50:
        print(f"  MEMORIA ESTABLE: entrenar B no afecto la discriminacion de A (Δ={delta_selectividad:+.0f}).")
        print(f"  Las vias de A y B son mayormente independientes en esta topologia.")
    else:
        print(f"  GENERALIZACION: entrenar B MEJORO la discriminacion (Δ={delta_selectividad:+.0f}).")

    if abs(delta_c) < 30:
        print(f"  La respuesta a C (no entrenado) se mantuvo estable — sin degradacion global.")
    else:
        print(f"  La respuesta a C cambio {delta_c:+.0f} — posible efecto colateral global.")

    cerebro.save_brain("cerebro_secuencial.pkl")

def experimento_15():
    separador("EXPERIMENTO 15: El misterio de la neurona C")
    print("Pregunta: ¿Por que una neurona NO entrenada (C) aumento su respuesta")
    print("cuando se entreno B? Esto contradice la prediccion de LTD global.\n")

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
        spk_antes = cerebro.spike_mon.num_spikes
        for _ in range(duracion):
            cerebro.neurons.I_sensory[canal] = 15.0
            cerebro.step()
        cerebro.neurons.I_sensory[canal] = 0.0
        return cerebro.spike_mon.num_spikes - spk_antes

    def test_triple(cerebro, label=""):
        a = [medir(cerebro, 0) for _ in range(3)]
        for _ in range(3): cerebro.step()
        b = [medir(cerebro, 2) for _ in range(3)]
        for _ in range(3): cerebro.step()
        c = [medir(cerebro, 4) for _ in range(3)]
        for _ in range(3): cerebro.step()
        mu_a, mu_b, mu_c = np.mean(a), np.mean(b), np.mean(c)
        if label:
            print(f"  {label}: A={mu_a:.0f} B={mu_b:.0f} C={mu_c:.0f} "
                  f"Δ(A,B)={mu_a-mu_b:+.0f} Δ(A,C)={mu_a-mu_c:+.0f}")
        return mu_a, mu_b, mu_c

    def pesos_por_neurona(cerebro, indices=[0,2,4]):
        pesos = {}
        for idx in indices:
            mask = cerebro.synapses.i[:] == idx
            w = cerebro.synapses.w[:][mask]
            pesos[f'S{idx}'] = {
                'n_syn': len(w), 'w_mean': float(np.mean(w)) if len(w) > 0 else 0,
                'w_std': float(np.std(w)) if len(w) > 0 else 0,
                'w_min': float(np.min(w)) if len(w) > 0 else 0,
                'w_max': float(np.max(w)) if len(w) > 0 else 0,
            }
        return pesos

    # ===== Crear Beta fresco =====
    np.random.seed(102)
    cerebro = BrainBrian2(name="Beta_Exp15", n_sensory=10, n_hidden=15, n_motor=10, n_pfc=5,
                          plasticity=True, drive_amplitude=0.0)
    cerebro.force_state = 'AWAKE'
    for _ in range(5): cerebro.step()

    print("Fase 0: Pesos iniciales (sin entrenamiento)")
    p0 = pesos_por_neurona(cerebro)
    for k, v in p0.items():
        print(f"  {k}: {v['n_syn']} syn, w={v['w_mean']:.4f}±{v['w_std']:.4f} [{v['w_min']:.4f}-{v['w_max']:.4f}]")

    a0, b0, c0 = test_triple(cerebro, "Baseline")

    # ===== Capturar timing de disparo durante 1 paso de entrenamiento =====
    print("\nFase 1: Analisis de timing de disparo (1 paso con drive=10, sin input)")
    cerebro.drive_amplitude = 10.0
    cerebro.drive_mode = 'oscillation'
    # Limpiar monitor de spikes
    spk_mon_antes = cerebro.spike_mon.num_spikes
    cerebro.step()
    spk_times = cerebro.spike_mon.t[spk_mon_antes:]
    spk_neurons = cerebro.spike_mon.i[spk_mon_antes:]

    # Primer spike de cada neurona sensorial 0, 2, 4
    for n in [0, 2, 4]:
        mask = spk_neurons == n
        if np.any(mask):
            first_t = float(np.min(spk_times[mask]) / ms)
            n_spikes = int(np.sum(mask))
            print(f"  Neurona S{n}: {n_spikes} spikes, primer spike a t={first_t:.1f}ms")
        else:
            print(f"  Neurona S{n}: 0 spikes")

    # ===== Entrenar SOLO B (sin entrenar A antes) =====
    print("\nFase 2: Entrenar SOLO B (neurona 2, 30 rondas)")
    entrenar_estimulo(cerebro, canal=2, rondas=30)

    p2 = pesos_por_neurona(cerebro)
    print("  Pesos post-entrenamiento B:")
    for k, v in p2.items():
        delta = v['w_mean'] - p0[k]['w_mean']
        print(f"  {k}: w={v['w_mean']:.4f}±{v['w_std']:.4f}  (Δ={delta:+.4f})")

    a_final, b_final, c_final = test_triple(cerebro, "Post-B  ")

    # ===== ANALISIS =====
    print("\n" + "="*60)
    print("  RESULTADOS: ¿Por que C subio?")
    print("="*60)
    print(f"  {'Neurona':>8s} {'Baseline':>10s} {'Post-B':>10s} {'Δ':>10s} {'Δw':>10s}")
    print(f"  {'-'*50}")
    print(f"  {'A (S0)':>8s} {a0:>10.0f} {a_final:>10.0f} {a_final-a0:>+10.0f} "
          f"{p2['S0']['w_mean']-p0['S0']['w_mean']:>+10.4f}")
    print(f"  {'B (S2)':>8s} {b0:>10.0f} {b_final:>10.0f} {b_final-b0:>+10.0f} "
          f"{p2['S2']['w_mean']-p0['S2']['w_mean']:>+10.4f}")
    print(f"  {'C (S4)':>8s} {c0:>10.0f} {c_final:>10.0f} {c_final-c0:>+10.0f} "
          f"{p2['S4']['w_mean']-p0['S4']['w_mean']:>+10.4f}")

    print("\n  --- CONCLUSION ---")
    delta_a = a_final - a0
    delta_b = b_final - b0
    delta_c = c_final - c0
    dw_a = p2['S0']['w_mean'] - p0['S0']['w_mean']
    dw_b = p2['S2']['w_mean'] - p0['S2']['w_mean']
    dw_c = p2['S4']['w_mean'] - p0['S4']['w_mean']

    if dw_c > 0:
        print(f"  C (S4) GANO peso (+{dw_c:.4f}) sin ser entrenada.")
        print(f"  Esto NO es esperado por STDP simple. Hipotesis:")
        print(f"  1. S4 dispara en fase favorable del I_ext respecto a sus targets")
        print(f"  2. S4 se beneficia del STDP global inducido por la actividad de B")
        print(f"  3. Los pesos de S4 NO son independientes — comparten targets con B")
    else:
        print(f"  C (S4) perdio peso ({dw_c:.4f}) como esperado. El aumento en respuesta")
        print(f"  comportamental (+{delta_c:.0f}) NO viene de cambio de pesos sino de otra causa.")
        print(f"  Posible: cambio en la excitabilidad de las neuronas post-sinapticas.")

    cerebro.save_brain("cerebro_exp15.pkl")

if __name__ == "__main__":
    print("=" * 60)
    print("  IA VIVA - EXPERIMENTOS BRIAN2 (v2)")
    print("=" * 60)

    experimento_1()
    print("\n" + "="*60 + "\n")
    experimento_2()
    print("\n" + "="*60 + "\n")
    experimento_3()
    print("\n" + "="*60 + "\n")
    experimento_4()
    print("\n" + "="*60 + "\n")
    experimento_5()
    print("\n" + "="*60 + "\n")
    experimento_6()
    print("\n" + "="*60 + "\n")
    experimento_7()
    print("\n" + "="*60 + "\n")
    experimento_8()
    print("\n" + "="*60 + "\n")
    experimento_9()
    print("\n" + "="*60 + "\n")
    experimento_10()
    print("\n" + "="*60 + "\n")
    experimento_11()
    print("\n" + "="*60 + "\n")
    experimento_12()
    print("\n" + "="*60 + "\n")
    experimento_13()
    print("\n" + "="*60 + "\n")
    experimento_14()
    print("\n" + "="*60 + "\n")
    experimento_15()

    print("\n" + "=" * 60)
    print("  TODOS LOS EXPERIMENTOS COMPLETADOS (1-15)")
    print("  Revisa logs_brian2/ para los CSV.")
    print("=" * 60)
