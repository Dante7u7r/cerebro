#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IA VIVA 3.0 - SIMULACIÃ“N CEREBRAL BIO-INSPIRADA CON EMERGENCIA DE INTELIGENCIA

Mejoras implementadas:
1. Diversidad neuronal (excitatoria / inhibitoria PV+ y SST+)
2. Plasticidad STDP (Spike-Timing-Dependent Plasticity)
3. Plasticidad a Corto Plazo STP (FacilitaciÃ³n y DepresiÃ³n sinÃ¡ptica)
4. Astrocitos activos (modulaciÃ³n sinÃ¡ptica, reciclaje de glutamato)
5. MielinizaciÃ³n y velocidad de conducciÃ³n variable
6. Gradientes morfÃ³genos y formaciÃ³n de capas corticales
7. Oscilaciones de red (LFP y espectrograma)
8. Homeostasis sinÃ¡ptica (scaling)
9. Ciclo sueÃ±o-vigilia y consolidaciÃ³n de memoria
10. NeuromodulaciÃ³n (Dopamina, Serotonina, Acetilcolina)
11. ComputaciÃ³n dendrÃ­tica (compartimentos soma/dendrita)
12. MicroglÃ­a y poda sinÃ¡ptica
13. Acoplamiento neurovascular (energÃ­a local)

Ejecutar: python semilla_cerebro.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import random
from collections import defaultdict, deque
import sys

# --- OptimizaciÃ³n vectorial (AMD/CPU) ---
def _cdist_np(A, B):
    """Distancia euclÃ­dea vectorizada (equivalente a scipy.spatial.distance.cdist)."""
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))

# ============================================================================
# CONFIGURACIÃ“N GLOBAL
# ============================================================================

np.random.seed(42)
random.seed(42)

# Constantes temporales
DT = 0.1                     # Paso de integraciÃ³n (ms)
SIGNAL_PERIOD = 127.7        # 7.83 Hz (Resonancia de Schumann)
SLEEP_CYCLE_PERIOD = 2000.0  # Periodo del ciclo sueÃ±o-vigilia (ms)
LAYER_HEIGHT = 100.0         # Altura total del espacio (Î¼m)
CORTICAL_DEPTH = 80.0        # Profundidad de la placa cortical (Î¼m)

# ============================================================================
# ENUMERACIONES
# ============================================================================

class CellType(Enum):
    STEM = 0
    NEURON = 1
    GLIA = 2
    APOPTOTIC = 3

class NeuronSubtype(Enum):
    EXCITATORY = 1      # GlutamatÃ©rgica (piramidal)
    INHIBITORY_PV = 2   # ParvalbÃºmina: inhibiciÃ³n rÃ¡pida al soma
    INHIBITORY_SST = 3  # Somatostatina: inhibiciÃ³n lenta a dendritas

class GliaSubtype(Enum):
    ASTROCYTE = 1
    OLIGODENDROCYTE = 2
    MICROGLIA = 3

class CellState(Enum):
    QUIESCENT = 0
    PROLIFERATIVE = 1
    DIFFERENTIATING = 2
    MIGRATING = 3

class BrainState(Enum):
    AWAKE = 0
    SLOW_WAVE_SLEEP = 1
    REM = 2

# Helper para retrocompatibilidad con subtipos inhibitorios
def is_inhibitory(subtype):
    """Retorna True si el subtipo neuronal es inhibitorio (PV+ o SST+)."""
    return subtype in (NeuronSubtype.INHIBITORY_PV, NeuronSubtype.INHIBITORY_SST)

# ============================================================================
# RED GENÃ‰TICA (Hes1/Ngn2/Mash1/Dlx)
# ============================================================================

class GeneRegulatoryNetwork:
    def __init__(self):
        # ParÃ¡metros de transcripciÃ³n
        self.k1, self.k2, self.k3, self.k4 = 0.3, 0.5, 0.4, 0.2
        self.d_m, self.d_p = 0.03, 0.02
        self.K_hes, self.K_ngn = 0.5, 0.5
        
        # ARNm
        self.H_mRNA = 0.8   # Hes1 (mantiene stemness)
        self.M_mRNA = 0.5   # Mash1 (pro-neuronal)
        self.N_mRNA = 0.6   # Ngn2 (pro-neuronal)
        self.D_mRNA = 0.4   # Dlx (pro-inhibitorio)
        
        # ProteÃ­nas
        self.H_protein = 0.7
        self.M_protein = 0.4
        self.N_protein = 0.5
        self.D_protein = 0.3
        
        self.age = 0.0
        self.time_since_last_division = 0.0

    def hill_rep(self, x, K, n=4): return 1.0 / (1.0 + (x/K)**n)
    def hill_act(self, x, K, n=4): return (x/K)**n / (1.0 + (x/K)**n)

    def update(self, dt, external_bias, morphogen_signal):
        # Hes1: autorepresiÃ³n
        dH_m = self.k1 * self.hill_rep(self.H_protein, self.K_hes) - self.d_m * self.H_mRNA
        # Mash1: reprimido por Hes1, activado por Ngn2
        rep = self.hill_rep(self.H_protein, self.K_hes)
        act = self.hill_act(self.N_protein, self.K_ngn)
        dM_m = self.k2 * (0.1 + 0.9*act) * rep - self.d_m * self.M_mRNA + external_bias*0.05
        # Ngn2: reprimido por Hes1
        dN_m = self.k3 * self.hill_rep(self.H_protein, self.K_hes) - self.d_m * self.N_mRNA
        # Dlx: inducido por Mash1, influenciado por morfÃ³geno (mÃ¡s en profundidad)
        dD_m = self.k4 * (0.2 + 0.8 * self.hill_act(self.M_protein, 0.5)) - self.d_m * self.D_mRNA
        dD_m += morphogen_signal * 0.1  # El gradiente promueve destino inhibitorio en zonas profundas

        # TraducciÃ³n
        dH_p = self.H_mRNA - self.d_p * self.H_protein
        dM_p = self.M_mRNA - self.d_p * self.M_protein
        dN_p = self.N_mRNA - self.d_p * self.N_protein
        dD_p = self.D_mRNA - self.d_p * self.D_protein

        # IntegraciÃ³n Euler
        self.H_mRNA += dH_m * dt
        self.M_mRNA += dM_m * dt
        self.N_mRNA += dN_m * dt
        self.D_mRNA += dD_m * dt
        self.H_protein += dH_p * dt
        self.M_protein += dM_p * dt
        self.N_protein += dN_p * dt
        self.D_protein += dD_p * dt

        for attr in ['H_mRNA','M_mRNA','N_mRNA','D_mRNA','H_protein','M_protein','N_protein','D_protein']:
            setattr(self, attr, max(0.0, getattr(self, attr)))
        self.age += dt
        self.time_since_last_division += dt

    def diff_score(self): return self.M_protein / (self.H_protein + 0.1)
    def glial_score(self): return self.H_protein / (self.M_protein + 0.1)
    def inhibitory_bias(self): return self.D_protein / (self.M_protein + self.N_protein + 0.1)

# ============================================================================
# SEÃ‘ALIZACIÃ“N DELTA-NOTCH
# ============================================================================

class DeltaNotch:
    def __init__(self):
        self.delta = 0.5; self.notch = 0.5; self.nicd = 0.1
        self.k_cleave = 0.8; self.k_hes = 0.5
        self.d_nicd = 0.1; self.d_delta = 0.05; self.d_notch = 0.05

    def update(self, dt, neighbor_delta, grn):
        cleavage = self.k_cleave * self.notch * neighbor_delta
        self.nicd += cleavage*dt - self.d_nicd*self.nicd*dt
        self.notch += (0.1 - self.d_notch*self.notch)*dt - cleavage*dt*0.5
        prod_delta = 0.15 * (1.0 - 0.8*grn.H_protein/(grn.H_protein+0.5))
        self.delta += (prod_delta - self.d_delta*self.delta)*dt
        grn.H_mRNA += self.k_hes * self.nicd * dt
        for attr in ['delta','notch','nicd']:
            setattr(self, attr, max(0.0, min(3.0, getattr(self, attr))))

# ============================================================================
# PAISAJE DE WADDINGTON
# ============================================================================

class WaddingtonLandscape:
    def __init__(self):
        self.x, self.y = 0.0, 0.0
        self.noise = 0.05

    def potential(self, x, y):
        stem = 2.0 * ((x+2)**2 + y**2)
        neuron = 1.5 * ((x-2)**2 + (y-2)**2)
        glia = 1.5 * ((x-2)**2 + (y+2)**2)
        barrier = 3.0 * np.exp(-0.5*x**2)
        return stem + neuron + glia + barrier

    def gradient(self, x, y, h=0.01):
        dx = (self.potential(x+h,y) - self.potential(x-h,y))/(2*h)
        dy = (self.potential(x,y+h) - self.potential(x,y-h))/(2*h)
        return dx, dy

    def update(self, dt, grn, morphogen):
        bias_x = -0.2*grn.H_protein + 0.3*(grn.M_protein+grn.N_protein)
        bias_y = 0.4*(grn.M_protein - grn.H_protein)
        # MorfÃ³geno empuja hacia destino inhibitorio (y negativo)
        bias_y -= 0.2 * morphogen * grn.D_protein
        gx, gy = self.gradient(self.x, self.y)
        self.x += (-gx + bias_x)*dt + self.noise*np.random.randn()*np.sqrt(dt)
        self.y += (-gy + bias_y)*dt + self.noise*np.random.randn()*np.sqrt(dt)

    def get_cell_type(self):
        if self.x < 0.5: return CellType.STEM
        elif self.y > 1.0: return CellType.NEURON
        elif self.y < -1.0: return CellType.GLIA
        else: return CellType.STEM

# ============================================================================
# SINAPSIS BIOLÃ“GICA CON STDP
# ============================================================================

class BiologicalSynapse:
    def __init__(self, pre_id, post_id, pre_subtype, delay=0.0, target_compartment='soma'):
        self.pre_id = pre_id
        self.post_id = post_id
        self.pre_subtype = pre_subtype
        self.weight = np.random.uniform(0.05, 0.2)
        self.delay = delay  # ms
        self.target_compartment = target_compartment  # 'soma' o 'dendrite'
        
        # Trazas STDP
        self.pre_trace = 0.0
        self.post_trace = 0.0
        self.last_pre_spike = -np.inf
        self.last_post_spike = -np.inf
        self.tau_trace = 20.0
        
        # Conductancias
        self.g_ampa = 0.0
        self.g_gaba = 0.0
        self.tau_ampa = 5.0
        self.tau_gaba = 10.0
        self.E_ampa = 0.0      # mV (excitatorio)
        self.E_gaba = -70.0    # mV (inhibitorio)
        
        # Para mielinizaciÃ³n
        self.myelination = 0.0  # 0-1, afecta velocidad
        
        # Plasticidad a Corto Plazo (STP)
        self.U_stp = 0.2 if pre_subtype == NeuronSubtype.EXCITATORY else 0.5
        self.u_stp = self.U_stp   # Probabilidad de liberaciÃ³n actual
        self.x_stp = 1.0          # Recursos vesiculares disponibles (0-1)
        self.tau_d = 200.0        # RecuperaciÃ³n de depresiÃ³n (ms)
        self.tau_f = 50.0         # Decaimiento de facilitaciÃ³n (ms)
        
    def update_conductances(self, dt):
        self.g_ampa *= np.exp(-dt/self.tau_ampa)
        self.g_gaba *= np.exp(-dt/self.tau_gaba)
        self.pre_trace *= np.exp(-dt/self.tau_trace)
        self.post_trace *= np.exp(-dt/self.tau_trace)
        # RecuperaciÃ³n STP
        self.x_stp += (1.0 - self.x_stp) * dt / self.tau_d
        self.u_stp += (self.U_stp - self.u_stp) * dt / self.tau_f
        
    def pre_spike(self, time):
        self.pre_trace = 1.0
        self.last_pre_spike = time
        # DinÃ¡mica STP: facilitaciÃ³n y depresiÃ³n
        self.u_stp = self.u_stp + self.U_stp * (1.0 - self.u_stp)
        release = self.u_stp * self.x_stp
        self.x_stp = max(0.0, self.x_stp - release)
        effective_weight = self.weight * release
        # Incremento de conductancia modulado por STP
        if self.pre_subtype == NeuronSubtype.EXCITATORY:
            self.g_ampa += effective_weight * 0.5
        else:
            self.g_gaba += effective_weight * 0.5
            
    def post_spike(self, time, da_modulation=1.0):
        self.post_trace = 1.0
        self.last_post_spike = time
        # STDP modulado por dopamina: PotenciaciÃ³n si post despuÃ©s de pre (LTP)
        dt_post_pre = time - self.last_pre_spike
        if 0 < dt_post_pre < 50:
            dw = 0.005 * np.exp(-dt_post_pre/17.0) * da_modulation
            self.weight = min(1.0, self.weight + dw)
        # DepresiÃ³n si pre despuÃ©s de post (LTD)
        elif -50 < dt_post_pre < 0:
            dw = -0.003 * np.exp(dt_post_pre/34.0) * da_modulation
            self.weight = max(0.0, self.weight + dw)
            
    def get_current(self, post_potential):
        I_ampa = self.g_ampa * (self.E_ampa - post_potential)
        I_gaba = self.g_gaba * (self.E_gaba - post_potential)
        return I_ampa + I_gaba

# ============================================================================
# ASTROCITO
# ============================================================================

class Astrocyte:
    def __init__(self, cell_id, position):
        self.id = cell_id
        self.pos = position.copy()
        self.ca_level = 0.0
        self.d_serine = 0.0
        self.radius = 40.0
        self.associated_synapses = []  # Lista de referencias a sinapsis cercanas
        self.glutamate_uptake_strength = 0.5
        # Acoplamiento neurovascular
        self.vasodilation_signal = 0.0
        
    def update(self, dt, neurons, synapses, local_energy=1.0):
        # Actividad glutamatÃ©rgica local
        local_activity = 0.0
        for n in neurons:
            if n.subtype == NeuronSubtype.EXCITATORY:
                dist = np.linalg.norm(n.pos - self.pos)
                if dist < self.radius:
                    local_activity += n.firing_rate * np.exp(-dist/self.radius)
        self.ca_level += dt * (0.05 * local_activity - 0.02 * self.ca_level)
        self.d_serine = 0.2 * self.ca_level  # Co-agonista NMDA
        
        # Recaptura de glutamato: acelera decaimiento de AMPA en sinapsis cercanas
        for syn in self.associated_synapses:
            if syn.pre_subtype == NeuronSubtype.EXCITATORY:
                syn.g_ampa *= (1.0 - self.glutamate_uptake_strength * dt / syn.tau_ampa)
        
        # Acoplamiento neurovascular: baja energÃ­a â†’ vasodilataciÃ³n
        if local_energy < 0.5:
            self.vasodilation_signal = min(1.0, self.vasodilation_signal + 0.1 * dt * (0.5 - local_energy))
        else:
            self.vasodilation_signal *= (1.0 - 0.05 * dt)

# ============================================================================
# OLIGODENDROCITO (MIELINIZACIÃ“N)
# ============================================================================

class Oligodendrocyte:
    def __init__(self, cell_id, position):
        self.id = cell_id
        self.pos = position.copy()
        self.radius = 50.0
        self.maturation = 0.0
        
    def update(self, dt, neurons):
        self.maturation = min(1.0, self.maturation + 0.001 * dt)
        
    def myelinate(self, synapse, distance):
        # A mayor maduraciÃ³n y cercanÃ­a, mÃ¡s mielinizaciÃ³n
        if self.maturation > 0.5:
            factor = self.maturation * np.exp(-distance/self.radius)
            synapse.myelination = min(1.0, synapse.myelination + 0.01 * factor)

# ============================================================================
# SISTEMA NEUROMODULADOR (Dopamina, Serotonina, Acetilcolina)
# ============================================================================

class NeuromodulatorSystem:
    """Sistema de neuromodulaciÃ³n global que modula plasticidad y excitabilidad."""
    def __init__(self):
        self.dopamine = 0.5       # SeÃ±al de novedad/recompensa
        self.serotonin = 0.5      # RegulaciÃ³n del Ã¡nimo y estado
        self.acetylcholine = 0.5  # AtenciÃ³n y excitabilidad
        self.tau_da = 200.0   # ms
        self.tau_5ht = 500.0  # ms
        self.tau_ach = 100.0  # ms
        
    def update(self, dt, brain_state, prediction_error):
        # Dopamina: impulsada por error de predicciÃ³n (novedad)
        da_target = 0.5 + 0.5 * np.tanh(abs(prediction_error) * 2.0)
        if brain_state == BrainState.SLOW_WAVE_SLEEP:
            da_target = 0.2
        self.dopamine += (da_target - self.dopamine) * dt / self.tau_da
        
        # Serotonina: tÃ³nica en vigilia, baja en REM
        if brain_state == BrainState.AWAKE:
            ht_target = 0.7
        elif brain_state == BrainState.SLOW_WAVE_SLEEP:
            ht_target = 0.4
        else:
            ht_target = 0.1
        self.serotonin += (ht_target - self.serotonin) * dt / self.tau_5ht
        
        # Acetilcolina: alta en vigilia activa, baja en sueÃ±o lento
        if brain_state == BrainState.AWAKE:
            ach_target = 0.8
        elif brain_state == BrainState.SLOW_WAVE_SLEEP:
            ach_target = 0.2
        else:
            ach_target = 0.7
        self.acetylcholine += (ach_target - self.acetylcholine) * dt / self.tau_ach
        
        self.dopamine = np.clip(self.dopamine, 0.0, 1.0)
        self.serotonin = np.clip(self.serotonin, 0.0, 1.0)
        self.acetylcholine = np.clip(self.acetylcholine, 0.0, 1.0)
        
    def stdp_modulation(self):
        """Dopamina modula la ventana STDP: alta DA = mÃ¡s plasticidad."""
        return 0.5 + self.dopamine  # Rango: 0.5 a 1.5
        
    def excitability_modulation(self):
        """ACh modula excitabilidad neural: alta ACh = umbral mÃ¡s bajo."""
        return -3.0 * self.acetylcholine  # Baja umbral hasta -3 mV

# ============================================================================
# MICROGLÃA (Poda SinÃ¡ptica)
# ============================================================================

class Microglia:
    """CÃ©lula inmune del SNC que patrulla y poda sinapsis dÃ©biles/inactivas."""
    def __init__(self, cell_id, position):
        self.id = cell_id
        self.pos = position.copy()
        self.radius = 45.0
        self.patrol_speed = 0.5  # Î¼m/ms
        self.pruning_threshold = 0.02  # Peso mÃ­nimo para conservar sinapsis
        self.activity_threshold = 500.0  # ms de inactividad mÃ¡xima
        self.phagocytosis_count = 0  # Sinapsis podadas
        
    def update(self, dt):
        """Movimiento de patrullaje aleatorio (modo vigilancia)."""
        self.pos += np.random.randn(3) * self.patrol_speed * dt
        self.pos[2] = np.clip(self.pos[2], 0, LAYER_HEIGHT)
        
    def prune(self, neurons, current_time):
        """Elimina sinapsis dÃ©biles o inactivas cerca de esta microglÃ­a."""
        pruned = 0
        for neuron in neurons:
            dist = np.linalg.norm(neuron.pos - self.pos)
            if dist > self.radius:
                continue
            surviving = []
            for syn in neuron.synapses_in:
                should_prune = False
                if syn.weight < self.pruning_threshold:
                    should_prune = True
                elif (current_time - syn.last_pre_spike > self.activity_threshold and
                      syn.last_pre_spike > 0):
                    should_prune = True
                if should_prune:
                    pruned += 1
                else:
                    surviving.append(syn)
            neuron.synapses_in = surviving
        self.phagocytosis_count += pruned
        return pruned

# ============================================================================
# CÃ‰LULA INTELIGENTE (MEJORADA)
# ============================================================================

@dataclass
class IntelligentCell:
    id: int
    pos: np.ndarray
    grn: GeneRegulatoryNetwork
    sig: DeltaNotch
    land: WaddingtonLandscape
    type: CellType = CellType.STEM
    subtype: Optional[Any] = None  # NeuronSubtype o GliaSubtype
    state: CellState = CellState.QUIESCENT
    energy: float = 150.0
    age: float = 0.0
    div_thresh: float = 30.0
    
    # Neurona
    v: float = -65.0              # Potencial de membrana (mV)
    v_rest: float = -65.0
    v_thresh: float = -50.0
    v_reset: float = -70.0
    refractory: float = 0.0
    tau_m: float = 20.0
    firing_rate: float = 0.0
    last_spike_time: float = -np.inf
    spike_times: List[float] = field(default_factory=list)
    
    synapses_in: List[BiologicalSynapse] = field(default_factory=list)
    synapses_out: List[BiologicalSynapse] = field(default_factory=list)
    total_synaptic_current: float = 0.0
    sensory_current: float = 0.0  # Corriente proveniente del cuerpo virtual
    
    # Homeostasis
    firing_rate_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_scaling_time: float = 0.0
    target_rate: float = 2.0  # Hz
    
    # Compartimento dendrÃ­tico
    v_dendrite: float = -65.0          # Potencial dendrÃ­tico (mV)
    tau_dendrite: float = 30.0         # Constante temporal dendrÃ­tica (ms)
    dendrite_spike_thresh: float = -40.0  # Umbral para spike dendrÃ­tico
    dendrite_coupling: float = 0.1     # Acoplamiento dendrita â†’ soma
    dendrite_refractory: float = 0.0
    
    # Historial
    history: List[Dict] = field(default_factory=list)
    
    def update_biology(self, dt, neighbor_delta, ext_signal, neurotrophic, morphogen, brain_state):
        self.age += dt
        self.grn.update(dt, ext_signal, morphogen)
        self.sig.update(dt, neighbor_delta, self.grn)
        self.land.update(dt, self.grn, morphogen)
        
        # EnergÃ­a base (ajustada para que no mueran de inaniciÃ³n rÃ¡pido)
        self.energy -= 0.1 * dt
        if self.state == CellState.PROLIFERATIVE: self.energy -= 2.0 * dt
        if self.state == CellState.QUIESCENT: self.energy += 0.3 * dt
        # RegeneraciÃ³n basal de energÃ­a (perfusiÃ³n sanguÃ­nea)
        if self.type == CellType.NEURON and self.energy < 200.0:
            self.energy += 0.1 * dt
        self.energy = max(0.0, min(200.0, self.energy))
        
        # DiferenciaciÃ³n
        if self.type == CellType.STEM:
            lt = self.land.get_cell_type()
            diff_score = self.grn.diff_score()
            if lt == CellType.NEURON and diff_score > 1.0:
                self.type = CellType.NEURON
                # Subtipo segÃºn Dlx y morfÃ³geno: PV+ o SST+
                if self.grn.inhibitory_bias() > 0.4 or morphogen > 0.5:
                    if self.grn.D_protein > 0.6 or np.random.rand() < 0.6:
                        self.subtype = NeuronSubtype.INHIBITORY_PV
                        self.tau_m = 10.0    # PV+ es rÃ¡pida
                        self.v_thresh = -45.0
                    else:
                        self.subtype = NeuronSubtype.INHIBITORY_SST
                        self.tau_m = 25.0    # SST+ es mÃ¡s lenta
                        self.v_thresh = -52.0
                else:
                    self.subtype = NeuronSubtype.EXCITATORY
                self.state = CellState.MIGRATING
            elif lt == CellType.GLIA and self.grn.glial_score() > 2.0:
                self.type = CellType.GLIA
                r = np.random.rand()
                if r < 0.6:
                    self.subtype = GliaSubtype.ASTROCYTE
                elif r < 0.85:
                    self.subtype = GliaSubtype.OLIGODENDROCYTE
                else:
                    self.subtype = GliaSubtype.MICROGLIA
            elif self.energy < 10.0:
                self.type = CellType.APOPTOTIC
                
        # MigraciÃ³n guiada por gradiente
        if self.state == CellState.MIGRATING and self.type == CellType.NEURON:
            if self.subtype == NeuronSubtype.EXCITATORY:
                # Excitatorias migran hacia arriba (hacia Reelin)
                self.pos[2] += 0.5 * dt * (1.0 - morphogen)
            else:
                # Inhibitorias (PV+ y SST+) migran tangencialmente
                self.pos[0] += np.random.randn() * 2.0 * dt
                self.pos[1] += np.random.randn() * 2.0 * dt
            if self.pos[2] > CORTICAL_DEPTH * 0.8:
                self.state = CellState.DIFFERENTIATING
                
        # Historial
        self.history.append({'type': self.type.value, 'v': self.v, 'rate': self.firing_rate})
        if len(self.history) > 500: self.history.pop(0)
        
    def _refractory_period(self):
        """Periodo refractario segÃºn subtipo neuronal."""
        if self.subtype == NeuronSubtype.INHIBITORY_PV:
            return 2.0   # PV+ dispara rÃ¡pido
        elif self.subtype == NeuronSubtype.INHIBITORY_SST:
            return 5.0
        else:
            return 5.0   # Piramidal estÃ¡ndar
    
    def update_neural(self, dt, time, input_current, brain_state, neuromod=None):
        if self.type != CellType.NEURON:
            return
            
        # Refractario
        if self.refractory > 0:
            self.refractory -= dt
            self.v = self.v_reset
            return
        if self.dendrite_refractory > 0:
            self.dendrite_refractory -= dt
            
        # ModulaciÃ³n de excitabilidad por neuromoduladores y energÃ­a (neurovascular)
        thresh_mod = 0.0
        if neuromod is not None:
            thresh_mod += neuromod.excitability_modulation()
        # Baja energÃ­a sube el umbral (mÃ¡s difÃ­cil disparar)
        energy_fraction = self.energy / 200.0
        thresh_mod += 5.0 * (1.0 - energy_fraction)
        effective_thresh = self.v_thresh + thresh_mod
            
        # Separar corrientes por compartimento (soma vs dendrita)
        I_soma = 0.0
        I_dendrite = 0.0
        for syn in self.synapses_in:
            if syn.target_compartment == 'dendrite':
                I_dendrite += syn.get_current(self.v_dendrite)
            else:
                I_soma += syn.get_current(self.v)
        
        # DinÃ¡mica dendrÃ­tica
        dv_dend = (-(self.v_dendrite - self.v_rest) + I_dendrite) / self.tau_dendrite
        self.v_dendrite += dv_dend * dt
        
        # Spike dendrÃ­tico: inyecciÃ³n de corriente fuerte al soma
        dendrite_current = 0.0
        if self.v_dendrite >= self.dendrite_spike_thresh and self.dendrite_refractory <= 0:
            self.v_dendrite = self.v_rest
            self.dendrite_refractory = 10.0
            dendrite_current = 15.0
        
        # Acoplamiento dendrita â†’ soma
        I_coupling = self.dendrite_coupling * (self.v_dendrite - self.v)
        self.total_synaptic_current = I_soma + I_coupling + dendrite_current + input_current + self.sensory_current
        
        # EcuaciÃ³n LIF
        dv = (-(self.v - self.v_rest) + self.total_synaptic_current) / self.tau_m
        self.v += dv * dt
        
        # Disparo
        if self.v >= effective_thresh:
            self.v = self.v_reset
            self.refractory = self._refractory_period()
            self.last_spike_time = time
            self.spike_times.append(time)
            if len(self.spike_times) > 100: self.spike_times.pop(0)
            # Notificar sinapsis salientes
            for syn in self.synapses_out:
                syn.pre_spike(time)
            # Consumo de energÃ­a por disparo (acoplamiento neurovascular)
            self.energy = max(0.0, self.energy - 1.0)
                
        # Tasa de disparo (ventana 100 ms)
        recent_spikes = [t for t in self.spike_times if time - t < 100.0]
        self.firing_rate = len(recent_spikes) * 10.0  # Hz
        self.firing_rate_history.append(self.firing_rate)
        
        # Homeostasis sinÃ¡ptica (cada 1000 ms)
        if time - self.last_scaling_time > 1000.0 and len(self.firing_rate_history) > 100:
            self.last_scaling_time = time
            avg_rate = np.mean(self.firing_rate_history)
            if avg_rate > 0:
                scale = self.target_rate / (avg_rate + 1e-6)
                scale = np.clip(scale, 0.8, 1.2)
                for syn in self.synapses_in:
                    syn.weight *= scale
                    
    def update_glia(self, dt, neurons, synapses, local_energy=1.0):
        if self.type != CellType.GLIA:
            return
        if self.subtype == GliaSubtype.ASTROCYTE:
            # Convertir en astrocito funcional
            astro = Astrocyte(self.id, self.pos)
            astro.ca_level = getattr(self, 'ca_level', 0.0)
            astro.vasodilation_signal = getattr(self, 'vasodilation_signal', 0.0)
            astro.update(dt, neurons, synapses, local_energy)
            # Guardar estado en atributos (simplificado)
            self.ca_level = astro.ca_level
            self.d_serine = astro.d_serine
            self.vasodilation_signal = astro.vasodilation_signal
            # Neurovascular: astrocito boost energÃ­a de neuronas cercanas
            if astro.vasodilation_signal > 0.1:
                for n in neurons:
                    dist = np.linalg.norm(n.pos - self.pos)
                    if dist < astro.radius:
                        n.energy = min(200.0, n.energy + 0.5 * dt * astro.vasodilation_signal)
        elif self.subtype == GliaSubtype.OLIGODENDROCYTE:
            oligo = Oligodendrocyte(self.id, self.pos)
            oligo.maturation = getattr(self, 'maturation', 0.0)
            oligo.update(dt, neurons)
            self.maturation = oligo.maturation
            self.radius = oligo.radius
        elif self.subtype == GliaSubtype.MICROGLIA:
            micro = Microglia(self.id, self.pos)
            micro.update(dt)
            self.pos = micro.pos.copy()
            
    def can_divide(self, dopamine=0.5):
        # En v5.0, la probabilidad de division aumenta con la Dopamina (exito alimenticio)
        # Si no hay exito, es muy dificil que el cerebro crezca.
        merit_factor = 0.3 + 0.7 * dopamine
        return (self.type == CellType.STEM and self.state == CellState.PROLIFERATIVE
                and self.energy > (self.div_thresh / merit_factor) 
                and self.grn.time_since_last_division > 30.0)
                
    def divide(self, new_id):
        self.energy -= 40.0
        self.grn.time_since_last_division = 0.0
        daughter = IntelligentCell(
            id=new_id, pos=self.pos + np.random.randn(3)*5.0,
            grn=GeneRegulatoryNetwork(), sig=DeltaNotch(), land=WaddingtonLandscape()
        )
        daughter.grn.H_protein = self.grn.H_protein * np.random.uniform(0.9,1.1)
        daughter.grn.M_protein = self.grn.M_protein * np.random.uniform(0.9,1.1)
        daughter.land.x = self.land.x + np.random.randn()*0.2
        daughter.land.y = self.land.y + np.random.randn()*0.2
        daughter.state = CellState.QUIESCENT
        daughter.energy = 60.0
        return daughter

# ============================================================================
# SIMULACIÃ“N PRINCIPAL
# ============================================================================

class BrainSimulation:
    def __init__(self):
        self.cells: List[IntelligentCell] = []
        self.time = 0.0
        self.next_id = 0
        self.dt = DT
        self.interaction_radius = 25.0
        self.ext_signal = 0.0
        self.step_count = 0
        self.brain_state = BrainState.AWAKE
        
        # CÃ©lulas iniciales (v5.0: Arquitectura por Capas)
        # Capa Sensorial (Abajo)
        for i in range(5):
            c = self._create_cell()
            c.pos = np.array([np.random.uniform(-40, 40), np.random.uniform(-40, 40), 5.0])
            c.type = CellType.NEURON
            c.subtype = NeuronSubtype.EXCITATORY
            c.energy = 200.0
            self.cells.append(c)
            
        # Capa Motora (Arriba)
        for i in range(4):
            c = self._create_cell()
            c.pos = np.array([np.random.uniform(-40, 40), np.random.uniform(-40, 40), 45.0])
            c.type = CellType.NEURON
            c.subtype = NeuronSubtype.EXCITATORY
            c.energy = 200.0
            self.cells.append(c)
            
        # Glia y STEM centrales
        for sub in [GliaSubtype.ASTROCYTE, GliaSubtype.ASTROCYTE, GliaSubtype.MICROGLIA]:
            g = self._create_cell()
            g.pos[2] = 25.0
            g.type = CellType.GLIA
            g.subtype = sub
            self.cells.append(g)
            
        stem = self._create_cell()
        stem.pos[2] = 25.0
        stem.state = CellState.PROLIFERATIVE
        self.cells.append(stem)
            
        self.compute_synaptic_connections() # Conectar de inmediato
        
        # Historial
        self.history = []
        self.max_history = 2000
        
        # SeÃ±al objetivo
        self.target_signal_history = []
        self.prediction_history = []
        self.error_history = []
        
        # LFP
        self.lfp_history = []
        self.lfp_electrode_pos = np.array([0.0, 0.0, CORTICAL_DEPTH/2])
        
        # Espectrograma
        self.spectrogram_data = []
        
        # Sistema neuromodulador
        self.neuromod = NeuromodulatorSystem()
        
        # Contador de poda microglial
        self.pruned_synapses_total = 0
        
        # Cache vectorizado de neighbor deltas (se recalcula en cada step)
        self._nd_cache = {}
        
    def _create_cell(self):
        c = IntelligentCell(
            id=self.next_id,
            pos=np.random.randn(3)*20 + np.array([0,0,10]),
            grn=GeneRegulatoryNetwork(),
            sig=DeltaNotch(),
            land=WaddingtonLandscape()
        )
        self.next_id += 1
        return c
        
    def morphogen_concentration(self, pos):
        # Gradiente lineal en Z (profundidad)
        z = pos[2]
        return 1.0 / (1.0 + np.exp(-0.1 * (z - CORTICAL_DEPTH/2)))
        
    def neighbor_delta(self, cell):
        """Wrapper mantenido por compatibilidad; la versiÃ³n rÃ¡pida es _compute_neighbor_deltas."""
        return self._nd_cache.get(cell.id, 0.0)

    def _compute_neighbor_deltas(self):
        """Calcula neighbor_delta para TODAS las cÃ©lulas de una vez (vectorizado)."""
        n = len(self.cells)
        if n == 0:
            self._nd_cache = {}
            return
        positions = np.array([c.pos for c in self.cells])      # (N, 3)
        deltas    = np.array([c.sig.delta for c in self.cells]) # (N,)
        # Matriz de distancias NÃ—N en una sola operaciÃ³n
        D = _cdist_np(positions, positions)                     # (N, N)
        mask = (D < self.interaction_radius) & (D > 0)          # excluir diagonal
        weights = np.where(mask, np.exp(-D / self.interaction_radius), 0.0)
        nd_all = weights @ deltas                               # (N,) suma ponderada
        self._nd_cache = {c.id: nd_all[i] for i, c in enumerate(self.cells)}
        
    def update_brain_state(self):
        # Ciclo circadiano: 70% vigilia, 20% sueno lento, 10% REM
        phase = self.time % SLEEP_CYCLE_PERIOD
        if phase < SLEEP_CYCLE_PERIOD * 0.7:
            self.brain_state = BrainState.AWAKE
        elif phase < SLEEP_CYCLE_PERIOD * 0.9:
            self.brain_state = BrainState.SLOW_WAVE_SLEEP
        else:
            self.brain_state = BrainState.REM
            
    def compute_synaptic_connections(self):
        """Reconfigura conexiones segÃºn proximidad y mielinizaciÃ³n."""
        neurons = [c for c in self.cells if c.type == CellType.NEURON]
        glia_oligo = [c for c in self.cells if c.type == CellType.GLIA and c.subtype == GliaSubtype.OLIGODENDROCYTE]
        
        # Limpiar sinapsis
        for n in neurons:
            n.synapses_in.clear()
            n.synapses_out.clear()
            
        if len(neurons) < 2:
            return
            
        # Crear nuevas sinapsis
        for pre in neurons:
            # Calcular distancias a otras neuronas
            for post in neurons:
                if pre.id == post.id: continue
                dist = np.linalg.norm(pre.pos - post.pos)
                # Radio de conexiÃ³n segÃºn subtipo
                if pre.subtype == NeuronSubtype.EXCITATORY:
                    max_dist = 80.0
                    p_connect = 0.3 * np.exp(-dist/40.0)
                elif pre.subtype == NeuronSubtype.INHIBITORY_PV:
                    max_dist = 50.0  # PV+ muy local
                    p_connect = 0.6 * np.exp(-dist/20.0)
                else:  # SST+
                    max_dist = 80.0  # SST+ mÃ¡s amplio
                    p_connect = 0.4 * np.exp(-dist/40.0)
                if dist < max_dist and np.random.rand() < p_connect:
                    # Calcular retraso por mielinizaciÃ³n
                    myelin_factor = 1.0
                    for oligo in glia_oligo:
                        d_oligo = np.linalg.norm(oligo.pos - (pre.pos + post.pos)/2)
                        if d_oligo < oligo.radius and hasattr(oligo, 'maturation'):
                            myelin_factor = 1.0 + 9.0 * oligo.maturation
                    conduction_velocity = 0.5 * myelin_factor
                    delay = dist / conduction_velocity
                    
                    # Compartimento destino: PV+â†’soma, SST+â†’dendrita, Excâ†’soma
                    if pre.subtype == NeuronSubtype.INHIBITORY_SST:
                        target_comp = 'dendrite'
                    else:
                        target_comp = 'soma'
                    
                    syn = BiologicalSynapse(pre.id, post.id, pre.subtype, delay, target_comp)
                    pre.synapses_out.append(syn)
                    post.synapses_in.append(syn)
                        
    def compute_lfp(self):
        lfp = 0.0
        for cell in self.cells:
            if cell.type != CellType.NEURON: continue
            dist = np.linalg.norm(cell.pos - self.lfp_electrode_pos)
            if dist < 100.0:
                # ContribuciÃ³n proporcional a corriente sinÃ¡ptica total
                lfp += cell.total_synaptic_current * np.exp(-dist/50.0)
        return lfp
        
    def step(self):
        self.time += self.dt
        self.step_count += 1
        self.update_brain_state()
        
        # SeÃ±al objetivo (senoidal + ruido) - solo en vigilia
        if self.brain_state == BrainState.AWAKE:
            target = np.sin(2 * np.pi * self.time / SIGNAL_PERIOD) + 0.1 * np.random.randn()
        else:
            target = 0.0  # Sin entrada durante sueÃ±o
        self.target_signal_history.append(target)
        
        # Reconfigurar sinapsis periÃ³dicamente (ELIMINADO: causaba pÃ©rdida de memoria)
        # if self.step_count % 100 == 0:
        #     self.compute_synaptic_connections()
            
        # Obtener neuronas y glÃ­a para actualizaciones
        neurons = [c for c in self.cells if c.type == CellType.NEURON]
        astrocytes = [c for c in self.cells if c.type == CellType.GLIA and c.subtype == GliaSubtype.ASTROCYTE]
        oligos = [c for c in self.cells if c.type == CellType.GLIA and c.subtype == GliaSubtype.OLIGODENDROCYTE]
        microglia = [c for c in self.cells if c.type == CellType.GLIA and c.subtype == GliaSubtype.MICROGLIA]
        
        # Actividad global (para neurotrofismo)
        total_activity = sum(n.firing_rate for n in neurons)
        neurotrophic = total_activity / (len(neurons) + 1) * 0.1
        
        # Actualizar neuromoduladores (cada 10 pasos para rendimiento)
        if self.step_count % 10 == 0:
            last_error = self.error_history[-1] if self.error_history else 0.0
            self.neuromod.update(self.dt * 10, self.brain_state, last_error)
        
        # EnergÃ­a local promedio (para neurovascular)
        avg_energy = np.mean([n.energy for n in neurons]) / 200.0 if neurons else 1.0
        
        # Actualizar astrocitos
        for astro in astrocytes:
            astro.update_glia(self.dt, neurons, [], avg_energy)
            
        # Actualizar oligodendrocitos y mielinizaciÃ³n
        for oligo in oligos:
            oligo.update_glia(self.dt, neurons, [])
            # Mielinizar sinapsis cercanas
            for n in neurons:
                for syn in n.synapses_out:
                    # PosiciÃ³n media del axÃ³n
                    pre_pos = next((c.pos for c in self.cells if c.id == syn.pre_id), None)
                    post_pos = next((c.pos for c in self.cells if c.id == syn.post_id), None)
                    if pre_pos is None or post_pos is None:
                        continue
                    mid = (pre_pos + post_pos) / 2
                    dist = np.linalg.norm(oligo.pos - mid)
                    if dist < oligo.radius:
                        syn.myelination = min(1.0, syn.myelination + 0.005 * oligo.maturation)
        
        # Actualizar microglÃ­a y poda sinÃ¡ptica (cada 500 pasos)
        for mg in microglia:
            mg.update_glia(self.dt, neurons, [])
        if self.step_count % 500 == 0 and microglia:
            for mg in microglia:
                micro = Microglia(mg.id, mg.pos)
                pruned = micro.prune(neurons, self.time)
                self.pruned_synapses_total += pruned
                         
        # Calcular neighbor_deltas y morfÃ³genos en bloque (vectorizado)
        self._compute_neighbor_deltas()
        positions_z = np.array([c.pos[2] for c in self.cells])
        morphs = 1.0 / (1.0 + np.exp(-0.1 * (positions_z - CORTICAL_DEPTH / 2)))

        # Actualizar cada cÃ©lula
        for i, cell in enumerate(self.cells):
            nd = self._nd_cache.get(cell.id, 0.0)
            morph = float(morphs[i])
            cell.update_biology(self.dt, nd, self.ext_signal, neurotrophic, morph, self.brain_state)
            
        # DinÃ¡mica neuronal
        # Decaer conductancias sinÃ¡pticas
        for cell in neurons:
            for syn in cell.synapses_in:
                syn.update_conductances(self.dt)
        # Actualizar entradas sinÃ¡pticas
        for cell in neurons:
            input_current = 0.0
            # En vigilia, aÃ±adir corriente de entrada sensorial
            if self.brain_state == BrainState.AWAKE:
                input_current += 2.0 * target
            # Actualizar con neuromoduladores
            cell.update_neural(self.dt, self.time, input_current, self.brain_state, self.neuromod)
            
        # DespuÃ©s del disparo, aplicar STDP modulado por dopamina
        da_mod = self.neuromod.stdp_modulation()
        for cell in neurons:
            if cell.last_spike_time == self.time:
                for syn in cell.synapses_in:
                    syn.post_spike(self.time, da_mod)
                    
        # Divisiones
        newborns = []
        for cell in self.cells:
            if cell.can_divide(self.neuromod.dopamine):
                daughter = cell.divide(self.next_id)
                self.next_id += 1
                newborns.append(daughter)
                cell.state = CellState.QUIESCENT
        self.cells.extend(newborns)
        
        # Eliminar apoptÃ³ticas
        self.cells = [c for c in self.cells if c.type != CellType.APOPTOTIC]
        if not self.cells:
            self.cells.append(self._create_cell())
            
        # SeÃ±al externa biolÃ³gica creciente
        if self.time > 100:
            self.ext_signal = min(1.0, self.ext_signal + 0.002)
            
        # LFP
        lfp = self.compute_lfp()
        self.lfp_history.append(lfp)
        if len(self.lfp_history) > 2000:
            self.lfp_history.pop(0)
            
        # Registro estadÃ­stico
        counts = {0:0, 1:0, 2:0}
        exc_count = 0
        inh_count = 0
        pv_count = 0
        sst_count = 0
        for cell in self.cells:
            counts[cell.type.value] += 1
            if cell.type == CellType.NEURON:
                if cell.subtype == NeuronSubtype.EXCITATORY: exc_count += 1
                elif cell.subtype == NeuronSubtype.INHIBITORY_PV:
                    inh_count += 1
                    pv_count += 1
                elif cell.subtype == NeuronSubtype.INHIBITORY_SST:
                    inh_count += 1
                    sst_count += 1
                
        pred = total_activity / (len(neurons) + 1)
        error = target - pred
        self.prediction_history.append(pred)
        self.error_history.append(error)
        # Cap historiales para evitar fuga de memoria en bucles largos
        _max_signal_hist = 5000
        if len(self.target_signal_history) > _max_signal_hist:
            self.target_signal_history = self.target_signal_history[-_max_signal_hist:]
        if len(self.prediction_history) > _max_signal_hist:
            self.prediction_history = self.prediction_history[-_max_signal_hist:]
        if len(self.error_history) > _max_signal_hist:
            self.error_history = self.error_history[-_max_signal_hist:]
        
        self.history.append({
            'time': self.time,
            'total': len(self.cells),
            'stem': counts[0], 'neuron': counts[1], 'glia': counts[2],
            'exc': exc_count, 'inh': inh_count,
            'pv': pv_count, 'sst': sst_count,
            'target': target, 'prediction': pred, 'error': error,
            'lfp': lfp,
            'brain_state': self.brain_state.value,
            'dopamine': self.neuromod.dopamine,
            'acetylcholine': self.neuromod.acetylcholine,
            'serotonin': self.neuromod.serotonin
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)




# ============================================================================
# MUNDO COMPARTIDO (La Arena Darwiniana - IA Viva 13.0)
# ============================================================================

class SharedWorld:
    WORLD_SIZE = 400.0
    
    def __init__(self):
        self.food_x = np.random.uniform(50, 350)
        self.food_y = np.random.uniform(50, 350)
        
        # Era 3: Objeto manipulable (Herramienta)
        self.tool_x = np.random.uniform(100, 300)
        self.tool_y = np.random.uniform(100, 300)
        
        self.era = 1 # 1: Competencia, 2: Cooperacion, 3: Herramientas
        self.step_count = 0
        self.bodies = [] # Lista de VirtualBody

    def update_world(self):
        self.step_count += 1
        
        # Transiciones de Era
        if self.step_count == 100000:
            print("\n[!] EL ECOSISTEMA HA CAMBIADO: ERA 2 (COOPERACION OBLIGADA) [!]", flush=True)
            self.era = 2
        elif self.step_count == 300000:
            print("\n[!] EL ECOSISTEMA HA CAMBIADO: ERA 3 (USO DE HERRAMIENTAS) [!]", flush=True)
            self.era = 3

    def register_body(self, body):
        self.bodies.append(body)

    def check_food_collision(self, body_x, body_y):
        dist = np.hypot(body_x - self.food_x, body_y - self.food_y)
        
        if self.era == 1:
            # Competencia pura: El primero que llega come
            if dist < 15.0:
                self.respawn_food()
                return True
        
        elif self.era == 2:
            # Cooperacion obligada: Ambos deben estar cerca (dist < 30)
            if len(self.bodies) == 2:
                d1 = np.hypot(self.bodies[0].x - self.food_x, self.bodies[0].y - self.food_y)
                d2 = np.hypot(self.bodies[1].x - self.food_x, self.bodies[1].y - self.food_y)
                if d1 < 30.0 and d2 < 30.0:
                    self.respawn_food()
                    return True # Ambos comen (la logica del VirtualBody procesara esto si retorna True y la comida cambia)
                    
        elif self.era == 3:
            # Herramientas: Si empujan la herramienta contra la comida
            tool_food_dist = np.hypot(self.tool_x - self.food_x, self.tool_y - self.food_y)
            if tool_food_dist < 20.0:
                self.respawn_food()
                self.respawn_tool()
                return "JACKPOT" # Super recompensa
                
            # Si llegan a la comida normal sin herramienta, tambien vale pero menos
            if len(self.bodies) == 2:
                d1 = np.hypot(self.bodies[0].x - self.food_x, self.bodies[0].y - self.food_y)
                d2 = np.hypot(self.bodies[1].x - self.food_x, self.bodies[1].y - self.food_y)
                if d1 < 30.0 and d2 < 30.0:
                    self.respawn_food()
                    return True

        return False
        
    def respawn_food(self):
        self.food_x = np.random.uniform(50, 350)
        self.food_y = np.random.uniform(50, 350)
        
    def respawn_tool(self):
        self.tool_x = np.random.uniform(100, 300)
        self.tool_y = np.random.uniform(100, 300)

# ============================================================================
# CUERPO VIRTUAL (bucle sensorimotor)
# ============================================================================

class VirtualBody:
    """Agente 2D simple controlado por las neuronas motoras del cerebro."""
    def __init__(self, world: SharedWorld):
        self.world = world
        self.world.register_body(self)
        self.x   = world.WORLD_SIZE / 2 + np.random.uniform(-50, 50)
        self.y   = world.WORLD_SIZE / 2 + np.random.uniform(-50, 50)
        self.angle = 0.0          # radianes
        self.speed = 0.0          # um/ms
        self.reward  = 0.0
        self.dist_to_food_prev = np.inf
        self.jackpot_reward = False

    def step(self, dt, motor_left: float, motor_right: float):
        """motor_left / motor_right: firing rate normalizada [0,1] de neuronas motoras."""
        v_l = np.clip(motor_left,  0, 1) * 5.0   # velocidad max 5 um/ms
        v_r = np.clip(motor_right, 0, 1) * 5.0
        linear  = (v_l + v_r) * 0.5
        angular = (v_r - v_l) * 0.04

        self.angle += angular * dt
        self.speed  = linear
        self.x = (self.x + np.cos(self.angle) * linear * dt) % self.world.WORLD_SIZE
        self.y = (self.y + np.sin(self.angle) * linear * dt) % self.world.WORLD_SIZE

        # Fisica de la Herramienta (Era 3)
        if self.world.era == 3:
            dist_to_tool = np.hypot(self.x - self.world.tool_x, self.y - self.world.tool_y)
            if dist_to_tool < 20.0:
                # Empujar herramienta
                force = linear * 0.5
                self.world.tool_x += np.cos(self.angle) * force * dt
                self.world.tool_y += np.sin(self.angle) * force * dt

        dist = np.hypot(self.x - self.world.food_x, self.y - self.world.food_y)
        
        # Recompensa incremental
        if self.dist_to_food_prev != np.inf:
            self.reward = np.clip((self.dist_to_food_prev - dist) * 0.1, -1, 1)
        self.dist_to_food_prev = dist

        # Interaccion con la comida
        collision_result = self.world.check_food_collision(self.x, self.y)
        if collision_result:
            self.dist_to_food_prev = np.inf # Reset para no afectar la recompensa despues del respawn
            if collision_result == "JACKPOT":
                self.reward = 10.0 # Recompensa masiva
                self.jackpot_reward = True
            else:
                self.reward = 1.0 # Llegaron a la comida

    def get_sensory_input(self):
        """Retina de 5 canales: Adapta su atencion segun la Era."""
        # En la Era 3, si la herramienta no ha chocado, priorizan ver la herramienta
        if self.world.era == 3:
            target_x, target_y = self.world.tool_x, self.world.tool_y
        else:
            target_x, target_y = self.world.food_x, self.world.food_y
            
        dx = target_x - self.x
        dy = target_y - self.y
        dist = np.hypot(dx, dy)
        angle_to_target = np.arctan2(dy, dx)
        rel_angle = (angle_to_target - self.angle + np.pi) % (2 * np.pi) - np.pi
        
        retina = np.zeros(5)
        angles = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])
        for i, a in enumerate(angles):
            diff = abs(rel_angle - a)
            retina[i] = np.exp(-diff * 5.0) * (1.0 - min(1.0, dist/400.0))
        return retina

# Parches sobre BrainSimulation (se inyectan en __init__ y step via monkey-patch)
_orig_BrainSim_init = BrainSimulation.__init__

def _new_BrainSim_init(self, name="Unnamed", world=None):
    _orig_BrainSim_init(self)
    self.name = name
    self.body = VirtualBody(world)
    self.history_food_dist = []
    self.consecutive_errors = 0
    self.last_dist = np.inf

    # IA Viva 6.0: Corteza Prefrontal (Supervisor)
    for i in range(3):
        c = self._create_cell()
        c.pos = np.array([np.random.uniform(-40, 40), np.random.uniform(-40, 40), 60.0])
        c.type = CellType.NEURON
        c.subtype = NeuronSubtype.INHIBITORY_PV
        c.energy = 200.0
        self.cells.append(c)

    # IA Viva 7.0 - Feature 1: Mapa de Curiosidad Espacial (rejilla 20x20)
    self._curiosity_grid = np.zeros((20, 20), dtype=np.float32)
    self._curiosity_decay = 0.9995  # Olvido lento

    # IA Viva 7.0 - Feature 2: Buffer de Memoria Episodica (para Sueno REM)
    self._episodic_buffer = deque(maxlen=200)
    self._best_da_threshold = 0.70  # Solo guarda momentos de alta dopamina

    # IA Viva 7.0 - Feature 3: Utilidad neuronal (Neuro-Evolucion)
    self._neuron_utility = {}  # id -> float

    # IA Viva 8.0: Núcleo Emocional (Resiliencia Cognitiva)
    self.frustration = 0.0   # Crece con errores
    self.resilience = 0.2    # Escudo contra poda (crece al superar frustración)
    self.courage = 0.2       # Audacia para hacer conexiones ciegas
    
    # IA Viva 9.0: Algoritmo Musk
    self._recent_novelty = 0.0
    self._idle_steps = 0


_orig_step = BrainSimulation.step

def _new_step(self):
    """Extiende step() original con cuerpo virtual, plasticidad estructural
    incremental y actualiza neuromodulacion por recompensa."""
    _orig_step(self)

    neurons  = [c for c in self.cells if c.type == CellType.NEURON]
    n_total  = len(neurons) + 1

    # ── Codificacion sensorial → Capa Sensorial (Z < 15) ─────────────────────
    # Limpiamos las corrientes
    for n in neurons:
        n.sensory_current = 0.0

    if self.brain_state == BrainState.AWAKE:
        retina = self.body.get_sensory_input()
        sensory_layer = [n for n in neurons if n.pos[2] < 15.0]
        for i, sn in enumerate(sensory_layer):
            # Cada neurona sensorial se sintoniza a uno de los 5 canales de la retina
            channel = i % 5
            sn.sensory_current = retina[channel] * 800.0
            
    # ── Place Cells → Capa de Asociacion (15 < Z < 35) ──────────────────────
    assoc_layer = [n for n in neurons if 15.0 <= n.pos[2] < 35.0]
    for n in assoc_layer:
        if not hasattr(n, 'place_center'):
            n.place_center = np.random.uniform(50, 350, 2)
            n.place_radius = 60.0
        dist_sq = (self.body.x - n.place_center[0])**2 + (self.body.y - n.place_center[1])**2
        if dist_sq < n.place_radius**2:
            n.sensory_current += 15.0
            
    # ── Decodificacion motora ← Capa Motora (35 < Z < 55) ────────────────────────
    motor_layer = [n for n in neurons if 35.0 <= n.pos[2] <= 55.0]
    n_m = len(motor_layer)
    
    # Basal noise: Siempre un poco de movimiento para que no se quede estatico
    noise_L = np.random.uniform(0, 0.05)
    noise_R = np.random.uniform(0, 0.05)
    
    if n_m >= 2:
        n_half = n_m // 2
        motor_L = sum(n.firing_rate for n in motor_layer[:n_half]) / (n_half * 100.0 + 1e-6) + noise_L
        motor_R = sum(n.firing_rate for n in motor_layer[n_half:]) / (n_half * 100.0 + 1e-6) + noise_R
    else:
        motor_L = motor_R = sum(n.firing_rate for n in neurons) / (len(neurons)*100.0 + 1e-6) + noise_L
        
    self.body.step(self.dt, motor_L, motor_R)

    # ── Modular dopamina por recompensa del cuerpo ───────────────────────────
    self.neuromod.dopamine = float(np.clip(
        self.neuromod.dopamine + self.body.reward * 0.05, 0.0, 1.0
    ))

    # ── Plasticidad estructural incremental (cada 200 pasos) ─────────────────
    if self.step_count % 200 == 0 and len(neurons) >= 2:
        _incremental_structural_plasticity(self, neurons)

    # ── Registrar distancia al alimento ──────────────────────────────────────
    dist = np.hypot(self.body.x - self.body.world.food_x, self.body.y - self.body.world.food_y)
    self.history_food_dist.append(dist)
    if len(self.history_food_dist) > 1000:
        self.history_food_dist.pop(0)

    # ── IA Viva 7.0/9.0 - Feature 1: Curiosidad Espacial ─────────────────────────
    # Determinar cuadrante actual (rejilla 20x20 sobre el mundo)
    grid_x = int(min(19, self.body.x / self.body.world.WORLD_SIZE * 20))
    grid_y = int(min(19, self.body.y / self.body.world.WORLD_SIZE * 20))
    visit_level = self._curiosity_grid[grid_x, grid_y]
    self._recent_novelty = 0.0
    if visit_level < 0.15:  # Zona poco explorada
        self._recent_novelty = 0.08 * (1.0 - visit_level / 0.15)
        self.neuromod.dopamine = min(1.0, self.neuromod.dopamine + self._recent_novelty)
    self._curiosity_grid[grid_x, grid_y] = min(1.0, visit_level + 0.05)
    self._curiosity_grid *= self._curiosity_decay  # Olvido lento

    # ── IA Viva 6.0/8.0/9.0: Motor de Errores y Gestión Emocional ────────────────
    if dist > self.last_dist + 0.05:
        # Musk Step 1: Make requirement less dumb. No hay error si hay alta curiosidad!
        if self._recent_novelty < 0.02: 
            self.consecutive_errors += 1
    elif dist < self.last_dist - 0.05:
        self.consecutive_errors = max(0, self.consecutive_errors - 2)
    self.last_dist = dist
    
    # Detección de parálisis (para Musk Step 2)
    if abs(dist - getattr(self, '_last_dist_idle', dist)) < 0.01:
        self._idle_steps += 1
    else:
        self._idle_steps = 0
    self._last_dist_idle = dist
    
    if self.consecutive_errors > 25:
        self.frustration = min(1.0, self.frustration + 0.1)
        punishment_factor = 0.85 + (0.15 * self.resilience)
        self.neuromod.dopamine *= (0.5 + 0.3 * self.resilience)
        
        for n in neurons:
            if self.time - n.last_spike_time < 60.0:
                for syn in n.synapses_in:
                    syn.weight *= punishment_factor
        self.consecutive_errors = 0

    # ── IA Viva 7.0 - Feature 2: Memoria Episodica (buffer REM) ──────────────
    # Guardar momento si la dopamina es alta (experiencia exitosa)
    if self.neuromod.dopamine > self._best_da_threshold:
        retina = self.body.get_sensory_input()
        self._episodic_buffer.append({
            'retina': retina.copy(),
            'dopamine': self.neuromod.dopamine,
            'time': self.time
        })
        # Evitar desbordamiento de RAM, pero darle suficiente memoria para evolucionar (20,000 sueños)
        if len(self._episodic_buffer) > 20000:
            self._episodic_buffer.pop(0)
    # Durante REM: reproducir memorias episodicas para consolidar aprendizaje
    if self.brain_state == BrainState.REM and len(self._episodic_buffer) > 0:
        # Seleccionar un recuerdo aleatorio del buffer
        memory = random.choice(self._episodic_buffer)
        replay_retina = memory['retina']
        sensory_layer = [n for n in neurons if n.pos[2] < 15.0]
        for i, sn in enumerate(sensory_layer):
            channel = i % 5
            # Replay suave (30% de la intensidad real) para no sobrescribir
            sn.sensory_current += replay_retina[channel] * 250.0
        # Modular dopamina durante replay para reforzar las sinapsis correctas
        self.neuromod.dopamine = min(1.0, self.neuromod.dopamine + 0.02)

    # ── IA Viva 9.0 - Musk Step 4: Accelerate (Mielinización) ────────────────
    # Recompensa emocional: Crecer a través de la adversidad y Acelerar
    if self.body.reward > 0:
        if self.frustration > 0.4:
            self.resilience = min(1.0, self.resilience + 0.02)
            self.courage = min(1.0, self.courage + 0.02)
        self.frustration *= 0.95  # Alivio
        
        # Mielinizar las rutas responsables del éxito para bajar la latencia a 0
        for n in neurons:
            if self.time - n.last_spike_time < 40.0:
                for syn in n.synapses_in:
                    syn.myelin = min(1.0, getattr(syn, 'myelin', 0.0) + 0.1)
                    syn.weight = min(1.5, syn.weight * (1.0 + syn.myelin * 0.05))

    # IA Viva 7.0 - Feature 3: Neuro-Evolucion (Seleccion Natural) ─────────
    # Actualizar utilidad de neuronas que dispararon antes del exito dopaminergico
    if self.neuromod.dopamine > 0.65:
        for n in neurons:
            if self.time - n.last_spike_time < 80.0:  # Disparo reciente
                self._neuron_utility[n.id] = min(2.0, self._neuron_utility.get(n.id, 0.5) + 0.01)
    else:
        # Decaimiento lento de utilidad para neuronas silenciosas
        for n in neurons:
            if n.id in self._neuron_utility:
                self._neuron_utility[n.id] *= 0.9995

    # Musk Step 2: Extreme Apoptosis si la red está en "Parálisis Zen" (Idle)
    mass_extinction = self._idle_steps > 1500
    if self.step_count % 500 == 0 or mass_extinction:
        to_kill = []
        for n in neurons:
            utility = self._neuron_utility.get(n.id, 0.5)
            # Poda ultra-agresiva si es inútil o si hay parálisis (suicidio masivo para resetear)
            # IA Viva 12.0: Eliminamos la protección incondicional a la PFC. La selección natural es implacable.
            if (utility < 0.1 and len(neurons) > 2) or (mass_extinction and np.random.rand() < 0.3):
                to_kill.append(n.id)
        if to_kill:
            for victim_id in to_kill:
                for cell in self.cells:
                    if cell.id == victim_id:
                        cell.type = CellType.APOPTOTIC
            if mass_extinction:
                print(f"[!] MUSK STEP 2: Paralisis detectada. {len(to_kill)} neuronas eliminadas.", flush=True)
                # IA Viva 10.0 - Reflejo Reptiliano (Espasmo de Escape)
                print(f"[!] IA VIVA 10.0: Reflejo Reptiliano Activado. Espasmo motor inyectado.", flush=True)
                self.body.angle += random.uniform(-np.pi, np.pi)
                self.body.x += random.uniform(-15, 15)
                self.body.y += random.uniform(-15, 15)
                self.body.x = np.clip(self.body.x, 0, self.body.world.WORLD_SIZE)
                self.body.y = np.clip(self.body.y, 0, self.body.world.WORLD_SIZE)
                self._idle_steps = 0

    # ── IA Viva 12.0: Morfogénesis Abierta (Evolución Darwiniana) ────────────
    if self.step_count % 100 == 0:
        total_synapses = sum(len(n.synapses_in) for n in neurons)
        systemic_panic = total_synapses < 2 or self.frustration > 0.8 or self._idle_steps > 3000
        
        if systemic_panic or len(neurons) < 4:
            mutations = random.randint(3, 5)
            for _ in range(mutations):
                c = self._create_cell()
                c.type = CellType.NEURON
                c.subtype = random.choice([NeuronSubtype.EXCITATORY, NeuronSubtype.INHIBITORY_PV])
                # Mutación caótica: coordenadas Z 100% aleatorias (sin concepto de Ojo o Músculo)
                c.pos = np.array([random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)])
                c.membrane_potential = -70.0
                c.synapses_in = []
                c.synapses_out = []
                self._neuron_utility[c.id] = 1.0 # Inmunidad inicial para que se pruebe
                self.cells.append(c)
                
            print(f"[+] IA VIVA 12.0: BROTE EVOLUTIVO. {mutations} células engendradas al azar. Esperando Selección Natural...", flush=True)
    # ── Guardar historia ──────────────────────────────────────────────────────
    if self.history:
        self.history[-1]['food_dist'] = dist
        self.history[-1]['reward']    = self.body.reward
        self.history[-1]['curiosity'] = float(self._curiosity_grid.mean())
        if len(self.history) > 5000:
            self.history.pop(0)
            
    # IA Viva 12.0/13.0: Auto-Guardado de Seguridad (Cada 100,000 pasos)
    if self.step_count % 100000 == 0:
        try:
            # En la Arena, guardamos en un archivo de "arena" para no tocar el ADN original
            save_path = f"arena_save_{self.name.lower()}.pkl"
            self.save_brain(save_path)
            print(f"[AUTO-SAVE] {self.name}: Progreso respaldado en {save_path}.", flush=True)
        except Exception:
            pass


BrainSimulation.__init__ = _new_BrainSim_init
BrainSimulation.step     = _new_step


def _incremental_structural_plasticity(sim, neurons):
    """Agrega y elimina sinapsis incrementalmente sin reconfigurar todo."""
    da_mod = sim.neuromod.stdp_modulation()
    # Poda: eliminar sinapsis debiles en neuronas activas
    for n in neurons:
        n.synapses_in = [s for s in n.synapses_in if s.weight > 0.01 * da_mod]

    # Formacion: buscar parejas sin conexion con alta correlacion de actividad
    if len(neurons) < 2:
        return
    # Ordenar por distancia; conectar las N parejas mas cercanas sin conexion
    existing = {(s.pre_id, s.post_id) for n in neurons for s in n.synapses_in}
    
    # IA Viva 10.0: Neurogénesis de Pánico (Raíz) y Anastomosis (Hongo)
    total_synapses = len(existing)
    panic_mode = total_synapses == 0 or sim.neuromod.dopamine < 0.05
    connected_ids = set(pre for pre, _ in existing).union(set(post for _, post in existing))
    
    candidates = []
    for i, pre in enumerate(neurons):
        for post in neurons[i+1:]:
            if (pre.id, post.id) in existing:
                continue
            dist = np.linalg.norm(pre.pos - post.pos)
            activity_corr = pre.firing_rate * post.firing_rate
            
            # Anastomosis Fúngica: Rescatar neuronas aisladas
            is_island = pre.id not in connected_ids or post.id not in connected_ids
            anastomosis = is_island and dist < 60.0
            
            # IA Viva 8.0/10.0: Salto de fe (Valor) o Pánico Radicular
            leap_of_faith = False
            if activity_corr == 0:
                if sim.courage > 0.4 and np.random.rand() < (sim.courage * 0.05):
                    leap_of_faith = True
                elif panic_mode and np.random.rand() < 0.2:
                    leap_of_faith = True
                
            if (dist < 80.0 and activity_corr > 0) or leap_of_faith or anastomosis:
                metric = dist / (activity_corr + 1e-6) if not (leap_of_faith or anastomosis) else dist * np.random.rand()
                candidates.append((metric, pre, post))
    candidates.sort(key=lambda x: x[0])
    
    # Musk Step 5: Automate (Memoria Muscular / Bypass).
    # Si la resiliencia es alta y el agente es un maestro, crear un atajo directo ojo -> motor
    automate_bypass = sim.resilience > 0.8 and sim.consecutive_errors == 0 and len(neurons) >= 4
    if automate_bypass:
        sensory = [n for n in neurons if n.pos[2] < 15]
        motor = [n for n in neurons if 35 <= n.pos[2] <= 55]
        if sensory and motor:
            s_n = random.choice(sensory)
            m_n = random.choice(motor)
            candidates.insert(0, (0.0, s_n, m_n)) # Prioridad absoluta
            
    for _, pre, post in candidates[:10]:
        dist = np.linalg.norm(pre.pos - post.pos)
        comp = 'dendrite' if pre.subtype == NeuronSubtype.INHIBITORY_SST else 'soma'
        syn  = BiologicalSynapse(pre.id, post.id, pre.subtype, dist * 0.002, comp)
        # Si es un Bypass de automatización, darle máxima fuerza y mielina instantánea
        if automate_bypass and pre.pos[2] < 15 and 35 <= post.pos[2] <= 55:
            syn.weight = 1.0
            syn.myelin = 1.0
        else:
            syn.weight = 0.05
        pre.synapses_out.append(syn)
        post.synapses_in.append(syn)


import pickle
import os

def save_brain(sim, path=None):
    """Guarda toda la estructura celular, sinapsis dinámicas y emociones."""
    if path is None:
        path = f"arena_save_{sim.name.lower()}.pkl"
    data = {
        'cells': sim.cells,
        'time': sim.time,
        'next_id': sim.next_id,
        'step_count': sim.step_count,
        'frustration': getattr(sim, 'frustration', 0.0),
        'resilience': getattr(sim, 'resilience', 0.2),
        'courage': getattr(sim, 'courage', 0.2),
        '_curiosity_grid': getattr(sim, '_curiosity_grid', None),
        '_episodic_buffer': getattr(sim, '_episodic_buffer', None),
        '_neuron_utility': getattr(sim, '_neuron_utility', None),
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"[SAVE] Memoria Profunda (Pickle) guardada en {path} ({len(sim.cells)} células, t={sim.time:.0f}ms)")

def load_brain(sim, path="brain_save.pkl"):
    """Restaura la memoria profunda y estructura."""
    if not os.path.exists(path):
        if os.path.exists("brain_save.npz"):
            path = "brain_save.npz"
        else:
            print(f"[LOAD] No se encontró {path}")
            return
            
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        sim.cells = data.get('cells', sim.cells)
        sim.time = data.get('time', sim.time)
        sim.next_id = data.get('next_id', sim.next_id)
        sim.step_count = data.get('step_count', sim.step_count)
        sim.frustration = data.get('frustration', getattr(sim, 'frustration', 0.0))
        sim.resilience = data.get('resilience', getattr(sim, 'resilience', 0.2))
        sim.courage = data.get('courage', getattr(sim, 'courage', 0.2))
        sim._curiosity_grid = data.get('_curiosity_grid', getattr(sim, '_curiosity_grid', None))
        sim._episodic_buffer = data.get('_episodic_buffer', getattr(sim, '_episodic_buffer', None))
        sim._neuron_utility = data.get('_neuron_utility', getattr(sim, '_neuron_utility', None))
        
        neurons = [c for c in sim.cells if c.type == CellType.NEURON]
        n_syn = sum(len(n.synapses_in) for n in neurons)
        print(f"[LOAD] Memoria Profunda restaurada desde {path} ({len(neurons)} neuronas, {n_syn} sinapsis)")
    else:
        # Fallback al viejo formato NPZ (Amnesia parcial)
        try:
            data = np.load(path)
            required = ('pre_ids', 'post_ids', 'weights')
            if not all(k in data.files for k in required):
                print(f"[LOAD] {path} no contiene claves esperadas {required}; se ignora.")
                return
            weight_map = {(int(p), int(q)): float(w)
                          for p, q, w in zip(data['pre_ids'], data['post_ids'], data['weights'])}
            for n in sim.cells:
                for s in n.synapses_in:
                    key = (s.pre_id, s.post_id)
                    if key in weight_map:
                        s.weight = weight_map[key]
            print(f"[LOAD] Formato antiguo cargado. {len(weight_map)} pesos restaurados desde {path}")
        except Exception as e:
            print(f"[LOAD] Error leyendo {path}: {e}. Se inicia desde cero.")

BrainSimulation.save_brain = save_brain
BrainSimulation.load_brain = load_brain


class BrainVisualizer:
    def __init__(self, sim):
        self.sim = sim
        self.trail_x, self.trail_y = [], []
        
        DARK = '#050510'; PANEL = '#0a0a1e'; BORDER = '#1a1a44'
        CYAN = '#00e5ff'; YELLOW = '#ffea00'; GRAY = '#555577'
        
        self.fig = plt.figure(figsize=(16, 7), facecolor=DARK)
        try:
            self.fig.canvas.manager.set_window_title('IA Viva 5.0 - Alto Rendimiento')
        except Exception:
            pass  # Algunos backends no exponen manager.set_window_title
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.1, left=0.02, right=0.98, top=0.90, bottom=0.05)
        
        def dark_ax(pos, title):
            ax = self.fig.add_subplot(pos)
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=GRAY, labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            ax.set_title(title, color=CYAN, fontsize=10, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.08, color=BORDER, linestyle='--')
            return ax

        # Panel 1 - Entorno
        self.ax_env = dark_ax(gs[0, 0], '🌍  Entorno Sensorimotor')
        self.ax_env.set_facecolor('#000008')
        self.ax_env.set_xlim(0, VirtualBody.WORLD_SIZE)
        self.ax_env.set_ylim(0, VirtualBody.WORLD_SIZE)
        self.ax_env.set_aspect('equal')
        self.trail_line,  = self.ax_env.plot([], [], '-', color=CYAN, lw=0.8, alpha=0.15, zorder=2)
        self.agent_aura,  = self.ax_env.plot([], [], 'o', color=CYAN, ms=22, alpha=0.08, zorder=3)
        self.agent_dot,   = self.ax_env.plot([], [], 'o', color='#ffffff', ms=7, markeredgecolor=CYAN, markeredgewidth=1.5, zorder=5)
        self.agent_arrow, = self.ax_env.plot([], [], '-', color=CYAN, lw=1.5, alpha=0.7, zorder=4)
        self.food_aura,   = self.ax_env.plot([], [], '*', color=YELLOW, ms=28, alpha=0.12, zorder=3)
        self.food_dot,    = self.ax_env.plot([], [], '*', color=YELLOW, ms=13, zorder=5)
        self.radar_lines  = []

        # Panel 2 - Cerebro 3D
        self.ax_3d = self.fig.add_subplot(gs[0, 1], projection='3d')
        self.ax_3d.set_facecolor(DARK)
        self.ax_3d.axis('off')
        self.ax_3d.view_init(elev=25, azim=45)
        self.ax_3d.set_title('🧠  Arquitectura Neural 3D', color=CYAN, fontsize=10, fontweight='bold', pad=10)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_key(self, event):
        if event.key == 's': self.sim.save_brain()

    def update(self, frame):
        for _ in range(150):  # Balance entre velocidad y fluidez
            self.sim.step()
            
        hist = self.sim.history
        if not hist: return

        body = self.sim.body
        neurons = [c for c in self.sim.cells if c.type == CellType.NEURON]
        current_time = self.sim.time
        nm = self.sim.neuromod

        # Entorno
        self.trail_x.append(body.x); self.trail_y.append(body.y)
        if len(self.trail_x) > 400: self.trail_x.pop(0); self.trail_y.pop(0)
        self.trail_line.set_data(self.trail_x, self.trail_y)
        self.agent_dot.set_data([body.x], [body.y])
        self.agent_aura.set_data([body.x], [body.y])
        self.agent_arrow.set_data(
            [body.x, body.x + 22 * np.cos(body.angle)],
            [body.y, body.y + 22 * np.sin(body.angle)]
        )
        self.food_dot.set_data([body.food_x], [body.food_y])
        self.food_aura.set_data([body.food_x], [body.food_y])

        retina = body.get_sensory_input()
        if not self.radar_lines:
            self.radar_lines = [self.ax_env.plot([], [], color='#00ff88', lw=1.5, alpha=0.3)[0] for _ in range(5)]
        for i, a in enumerate(np.array([-0.8, -0.4, 0.0, 0.4, 0.8]) * np.pi):
            intensity = float(retina[i])
            self.radar_lines[i].set_alpha(0.08 + 0.85 * intensity)
            self.radar_lines[i].set_linewidth(0.8 + 2.5 * intensity)
            beam = 20 + 160 * intensity
            ang = body.angle + a
            self.radar_lines[i].set_data([body.x, body.x + beam * np.cos(ang)], [body.y, body.y + beam * np.sin(ang)])

        # Cerebro 3D (actualización más frecuente para mayor fluidez)
        if frame % 3 == 0:
            self.ax_3d.clear()
            self.ax_3d.set_facecolor('#000008')
            self.ax_3d.axis('off')
            self.ax_3d.set_title('🧠  Arquitectura Neural 3D', color='#00e5ff', fontsize=10, fontweight='bold', pad=10)
            if neurons:
                xs = [n.pos[0] for n in neurons]
                ys = [n.pos[1] for n in neurons]
                zs = [n.pos[2] for n in neurons]
                clr = ['#00ff88' if n.pos[2] < 15 
                       else '#ff3366' if 35 <= n.pos[2] <= 55 
                       else '#b000ff' if n.pos[2] > 55  # Prefrontal (Supervisor)
                       else '#00e5ff' for n in neurons]
                firing = [current_time - n.last_spike_time < 12.0 for n in neurons]
                sz  = [60 + 180 * f for f in firing]
                alp = [1.0 if f else 0.35 for f in firing]
                self.ax_3d.scatter(xs, ys, zs, c=clr, s=[s*2 for s in sz], alpha=0.07)
                self.ax_3d.scatter(xs, ys, zs, c=clr, s=sz, alpha=alp, edgecolors='white', linewidths=0.4)
                
                id_pos = {n.id: n.pos for n in neurons}
                for n in neurons:
                    for s in n.synapses_in:
                        pp = id_pos.get(s.pre_id)
                        if pp is not None:
                            self.ax_3d.plot([pp[0], n.pos[0]], [pp[1], n.pos[1]], [pp[2], n.pos[2]], color='#aaaaff', alpha=min(0.55, s.weight * 4.5), lw=0.7)
                
                ws = VirtualBody.WORLD_SIZE
                self.ax_3d.text(ws*0.6, 0, 8,  'SENSOR',  color='#00ff88', fontsize=6, alpha=0.7)
                self.ax_3d.text(ws*0.6, 0, 25, 'MEMORIA', color='#00e5ff', fontsize=6, alpha=0.7)
                self.ax_3d.text(ws*0.6, 0, 42, 'MOTOR',   color='#ff3366', fontsize=6, alpha=0.7)
                self.ax_3d.text(ws*0.6, 0, 60, 'PFC/SUPV',color='#b000ff', fontsize=6, alpha=0.7)

        last = hist[-1]
        state_names = {0: '⏵ Despierto', 1: '🌙 Sueño Lento', 2: '💤 REM'}
        state_str = state_names.get(self.sim.brain_state.value, '?')
        dist_now = np.hypot(body.x - body.food_x, body.y - body.food_y)
        n_syn = sum(len(n.synapses_in) for n in neurons)
        self.fig.suptitle(
            f"IA Viva 5.0 (Alto Rendimiento)  ·  {state_str}  ·  t={current_time:.0f} ms  ·  "
            f"Neuronas: {last['neuron']}  ·  Sinapsis: {n_syn}  ·  DA: {nm.dopamine:.2f}  ·  "
            f"★ dist: {dist_now:.0f}   [S] = Guardar",
            color='#00e5ff', fontsize=11, fontweight='bold', y=0.96
        )
        return []

    def run(self):
        self.ani = FuncAnimation(self.fig, self.update, interval=20, cache_frame_data=False)
        plt.show()

# ============================================================================
# INICIO
# ============================================================================

if __name__ == "__main__":
    import time
    print("=" * 80)
    print("IA VIVA 13.0 - LA ARENA DARWINIANA (ALFA vs BETA)")
    print("Eras: 1. Competencia | 2. Cooperación | 3. Herramientas")
    print("=" * 80)
    
    # 1. Preparar el Mundo
    mundo = SharedWorld()
    
    # 2. Cargar a los Combatientes (Solo lectura de los originales)
    sim_alfa = BrainSimulation(name="Alfa", world=mundo)
    sim_beta = BrainSimulation(name="Beta", world=mundo)
    
    print("\n[CARGANDO ADN ORIGINARIOS...]")
    try:
        sim_alfa.load_brain("brain_save_alfa_laptop.pkl")
        sim_beta.load_brain("brain_save_beta_escritorio.pkl")
    except Exception as e:
        print(f"Error cargando cerebros: {e}")
        # Si no existen los archivos específicos, intentar el default
        sim_alfa.load_brain()
        sim_beta.load_brain()
        
    print(f"\n[INICIANDO SIMULACION MULTI-AGENTE EN {mundo.WORLD_SIZE}um x {mundo.WORLD_SIZE}um]")
    print("Presiona CTRL+C para detener y guardar el progreso de la Arena.\n")
    
    steps = 0
    batch_size = 1000
    last_time = time.time()
    
    try:
        while True:
            # Correr un lote de pasos entrelazados
            for _ in range(batch_size):
                mundo.update_world()
                sim_alfa.step()
                sim_beta.step()
            
            steps += batch_size
            now = time.time()
            fps = batch_size / (now - last_time + 1e-9)
            last_time = now
            
            # Métricas Alfa
            n_alfa = [c for c in sim_alfa.cells if c.type == CellType.NEURON]
            s_alfa = sum(len(n.synapses_in) for n in n_alfa)
            d_alfa = np.hypot(sim_alfa.body.x - mundo.food_x, sim_alfa.body.y - mundo.food_y)
            
            # Métricas Beta
            n_beta = [c for c in sim_beta.cells if c.type == CellType.NEURON]
            s_beta = sum(len(n.synapses_in) for n in n_beta)
            d_beta = np.hypot(sim_beta.body.x - mundo.food_x, sim_beta.body.y - mundo.food_y)
            
            era_names = {1: "COMPETENCIA", 2: "COOPERACION", 3: "HERRAMIENTAS"}
            current_era = era_names.get(mundo.era, "DESCONOCIDA")
            
            print(f"[{steps/1000:,.0f}K] ERA: {current_era} | ALFA (N:{len(n_alfa)} S:{s_alfa} D:{d_alfa:.0f}) | BETA (N:{len(n_beta)} S:{s_beta} D:{d_beta:.0f}) | FPS:{fps:.0f}")

    except KeyboardInterrupt:
        print("\n[!] ARENA DETENIDA POR EL USUARIO.")
        print("Guardando estado evolutivo de la Arena...")
        sim_alfa.save_brain() # Guardará en arena_save_alfa.pkl
        sim_beta.save_brain() # Guardará en arena_save_beta.pkl
        print("Cerebros guardados exitosamente. ADN original intacto.")
        print("Cerebro guardado exitosamente. ¡Hasta pronto!")
