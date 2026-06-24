#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IA VIVA - CEREBRO NEURONAL BIO-INSPIRADO

Simulacion neuronal con plasticidad Hebbiana (STDP),
neuromodulacion emocional, diferenciacion celular y memoria episodica.

Modo investigacion: observar, estimular y registrar.

Ejecutar: python asistente_cerebro.py
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import random
from collections import deque
import csv
import time as time_mod
import pickle
import os

# ============================================================================
# CONFIGURACION GLOBAL
# ============================================================================

np.random.seed(42)
random.seed(42)

DT = 0.1
SIGNAL_PERIOD = 127.7
SLEEP_CYCLE_PERIOD = 2000.0

# ============================================================================
# ENUMERACIONES
# ============================================================================

class CellType(Enum):
    STEM = 0
    NEURON = 1
    GLIA = 2
    APOPTOTIC = 3

class NeuronSubtype(Enum):
    EXCITATORY = 1
    INHIBITORY_PV = 2
    INHIBITORY_SST = 3

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

def is_inhibitory(subtype):
    return subtype in (NeuronSubtype.INHIBITORY_PV, NeuronSubtype.INHIBITORY_SST)

# ============================================================================
# RED GENETICA
# ============================================================================

class GeneRegulatoryNetwork:
    def __init__(self):
        self.k1, self.k2, self.k3, self.k4 = 0.3, 0.5, 0.4, 0.2
        self.d_m, self.d_p = 0.03, 0.02
        self.K_hes, self.K_ngn = 0.5, 0.5

        self.H_mRNA = 0.8
        self.M_mRNA = 0.5
        self.N_mRNA = 0.6
        self.D_mRNA = 0.4

        self.H_protein = 0.7
        self.M_protein = 0.4
        self.N_protein = 0.5
        self.D_protein = 0.3

        self.age = 0.0
        self.time_since_last_division = 0.0

    def hill_rep(self, x, K, n=4):
        return 1.0 / (1.0 + (x / K) ** n)

    def hill_act(self, x, K, n=4):
        return (x / K) ** n / (1.0 + (x / K) ** n)

    def update(self, dt, external_bias, morphogen_signal):
        dH_m = self.k1 * self.hill_rep(self.H_protein, self.K_hes) - self.d_m * self.H_mRNA
        rep = self.hill_rep(self.H_protein, self.K_hes)
        act = self.hill_act(self.N_protein, self.K_ngn)
        dM_m = self.k2 * (0.1 + 0.9 * act) * rep - self.d_m * self.M_mRNA + external_bias * 0.05
        dN_m = self.k3 * self.hill_rep(self.H_protein, self.K_hes) - self.d_m * self.N_mRNA
        dD_m = self.k4 * (0.2 + 0.8 * self.hill_act(self.M_protein, 0.5)) - self.d_m * self.D_mRNA
        dD_m += morphogen_signal * 0.1

        dH_p = self.H_mRNA - self.d_p * self.H_protein
        dM_p = self.M_mRNA - self.d_p * self.M_protein
        dN_p = self.N_mRNA - self.d_p * self.N_protein
        dD_p = self.D_mRNA - self.d_p * self.D_protein

        self.H_mRNA += dH_m * dt
        self.M_mRNA += dM_m * dt
        self.N_mRNA += dN_m * dt
        self.D_mRNA += dD_m * dt
        self.H_protein += dH_p * dt
        self.M_protein += dM_p * dt
        self.N_protein += dN_p * dt
        self.D_protein += dD_p * dt

        for attr in ['H_mRNA', 'M_mRNA', 'N_mRNA', 'D_mRNA',
                      'H_protein', 'M_protein', 'N_protein', 'D_protein']:
            setattr(self, attr, max(0.0, getattr(self, attr)))
        self.age += dt
        self.time_since_last_division += dt

    def diff_score(self):
        return self.M_protein / (self.H_protein + 0.1)

    def glial_score(self):
        return self.H_protein / (self.M_protein + 0.1)

    def inhibitory_bias(self):
        return self.D_protein / (self.M_protein + self.N_protein + 0.1)

# ============================================================================
# SENALIZACION DELTA-NOTCH
# ============================================================================

class DeltaNotch:
    def __init__(self):
        self.delta = 0.5
        self.notch = 0.5
        self.nicd = 0.1
        self.k_cleave = 0.8
        self.k_hes = 0.5
        self.d_nicd = 0.1
        self.d_delta = 0.05
        self.d_notch = 0.05

    def update(self, dt, neighbor_delta, grn):
        cleavage = self.k_cleave * self.notch * neighbor_delta
        self.nicd += cleavage * dt - self.d_nicd * self.nicd * dt
        self.notch += (0.1 - self.d_notch * self.notch) * dt - cleavage * dt * 0.5
        prod_delta = 0.15 * (1.0 - 0.8 * grn.H_protein / (grn.H_protein + 0.5))
        self.delta += (prod_delta - self.d_delta * self.delta) * dt
        grn.H_mRNA += self.k_hes * self.nicd * dt
        for attr in ['delta', 'notch', 'nicd']:
            setattr(self, attr, max(0.0, min(3.0, getattr(self, attr))))

# ============================================================================
# PAISAJE DE WADDINGTON
# ============================================================================

class WaddingtonLandscape:
    def __init__(self):
        self.x, self.y = 0.0, 0.0
        self.noise = 0.05

    def potential(self, x, y):
        stem = 2.0 * ((x + 2) ** 2 + y ** 2)
        neuron = 1.5 * ((x - 2) ** 2 + (y - 2) ** 2)
        glia = 1.5 * ((x - 2) ** 2 + (y + 2) ** 2)
        barrier = 3.0 * np.exp(-0.5 * x ** 2)
        return stem + neuron + glia + barrier

    def gradient(self, x, y, h=0.01):
        dx = (self.potential(x + h, y) - self.potential(x - h, y)) / (2 * h)
        dy = (self.potential(x, y + h) - self.potential(x, y - h)) / (2 * h)
        return dx, dy

    def update(self, dt, grn, morphogen):
        bias_x = -0.2 * grn.H_protein + 0.3 * (grn.M_protein + grn.N_protein)
        bias_y = 0.4 * (grn.M_protein - grn.H_protein)
        bias_y -= 0.2 * morphogen * grn.D_protein
        gx, gy = self.gradient(self.x, self.y)
        self.x += (-gx + bias_x) * dt + self.noise * np.random.randn() * np.sqrt(dt)
        self.y += (-gy + bias_y) * dt + self.noise * np.random.randn() * np.sqrt(dt)

    def get_cell_type(self):
        if self.x < 0.5:
            return CellType.STEM
        elif self.y > 1.0:
            return CellType.NEURON
        elif self.y < -1.0:
            return CellType.GLIA
        else:
            return CellType.STEM

# ============================================================================
# SINAPSIS BIOLOGICA CON STDP
# ============================================================================

class BiologicalSynapse:
    def __init__(self, pre_id, post_id, pre_subtype, delay=0.0, target_compartment='soma'):
        self.pre_id = pre_id
        self.post_id = post_id
        self.pre_subtype = pre_subtype
        self.weight = np.random.uniform(0.05, 0.2)
        self.delay = delay
        self.target_compartment = target_compartment

        self.pre_trace = 0.0
        self.post_trace = 0.0
        self.last_pre_spike = -np.inf
        self.last_post_spike = -np.inf
        self.tau_trace = 20.0

        self.g_ampa = 0.0
        self.g_gaba = 0.0
        self.tau_ampa = 5.0
        self.tau_gaba = 10.0
        self.E_ampa = 0.0
        self.E_gaba = -70.0

        self.myelination = 0.0

        self.U_stp = 0.2 if pre_subtype == NeuronSubtype.EXCITATORY else 0.5
        self.u_stp = self.U_stp
        self.x_stp = 1.0
        self.tau_d = 200.0
        self.tau_f = 50.0

    def update_conductances(self, dt):
        self.g_ampa *= np.exp(-dt / self.tau_ampa)
        self.g_gaba *= np.exp(-dt / self.tau_gaba)
        self.pre_trace *= np.exp(-dt / self.tau_trace)
        self.post_trace *= np.exp(-dt / self.tau_trace)
        self.x_stp += (1.0 - self.x_stp) * dt / self.tau_d
        self.u_stp += (self.U_stp - self.u_stp) * dt / self.tau_f

    def pre_spike(self, time):
        self.pre_trace = 1.0
        self.last_pre_spike = time
        self.u_stp = self.u_stp + self.U_stp * (1.0 - self.u_stp)
        release = self.u_stp * self.x_stp
        self.x_stp = max(0.0, self.x_stp - release)
        effective_weight = self.weight * release
        if self.pre_subtype == NeuronSubtype.EXCITATORY:
            self.g_ampa += effective_weight * 0.5
        else:
            self.g_gaba += effective_weight * 0.5

    def post_spike(self, time, da_modulation=1.0):
        self.post_trace = 1.0
        self.last_post_spike = time
        dt_post_pre = time - self.last_pre_spike
        if 0 < dt_post_pre < 50:
            dw = 0.005 * np.exp(-dt_post_pre / 17.0) * da_modulation
            self.weight = min(1.0, self.weight + dw)
        elif -50 < dt_post_pre < 0:
            dw = -0.003 * np.exp(dt_post_pre / 34.0) * da_modulation
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
        self.associated_synapses = []
        self.glutamate_uptake_strength = 0.5
        self.vasodilation_signal = 0.0

    def update(self, dt, neurons, synapses, local_energy=1.0):
        local_activity = 0.0
        for n in neurons:
            if n.subtype == NeuronSubtype.EXCITATORY:
                dist = np.linalg.norm(n.pos - self.pos)
                if dist < self.radius:
                    local_activity += n.firing_rate * np.exp(-dist / self.radius)
        self.ca_level += dt * (0.05 * local_activity - 0.02 * self.ca_level)
        self.d_serine = 0.2 * self.ca_level

        for syn in self.associated_synapses:
            if syn.pre_subtype == NeuronSubtype.EXCITATORY:
                syn.g_ampa *= (1.0 - self.glutamate_uptake_strength * dt / syn.tau_ampa)

        if local_energy < 0.5:
            self.vasodilation_signal = min(1.0, self.vasodilation_signal + 0.1 * dt * (0.5 - local_energy))
        else:
            self.vasodilation_signal *= (1.0 - 0.05 * dt)

# ============================================================================
# OLIGODENDROCITO
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
        if self.maturation > 0.5:
            factor = self.maturation * np.exp(-distance / self.radius)
            synapse.myelination = min(1.0, synapse.myelination + 0.01 * factor)

# ============================================================================
# SISTEMA NEUROMODULADOR
# ============================================================================

class NeuromodulatorSystem:
    def __init__(self):
        self.dopamine = 0.5
        self.serotonin = 0.5
        self.acetylcholine = 0.5
        self.tau_da = 200.0
        self.tau_5ht = 500.0
        self.tau_ach = 100.0

    def update(self, dt, brain_state, prediction_error):
        da_target = 0.5 + 0.5 * np.tanh(abs(prediction_error) * 2.0)
        if brain_state == BrainState.SLOW_WAVE_SLEEP:
            da_target = 0.2
        self.dopamine += (da_target - self.dopamine) * dt / self.tau_da

        if brain_state == BrainState.AWAKE:
            ht_target = 0.7
        elif brain_state == BrainState.SLOW_WAVE_SLEEP:
            ht_target = 0.4
        else:
            ht_target = 0.1
        self.serotonin += (ht_target - self.serotonin) * dt / self.tau_5ht

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
        return 0.5 + self.dopamine

    def excitability_modulation(self):
        return -3.0 * self.acetylcholine

# ============================================================================
# MICROGLIA
# ============================================================================

class Microglia:
    def __init__(self, cell_id, position):
        self.id = cell_id
        self.pos = position.copy()
        self.radius = 45.0
        self.patrol_speed = 0.5
        self.pruning_threshold = 0.02
        self.activity_threshold = 500.0
        self.phagocytosis_count = 0

    def update(self, dt):
        self.pos += np.random.randn(3) * self.patrol_speed * dt
        self.pos[2] = np.clip(self.pos[2], 0, 100)

    def prune(self, neurons, current_time):
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
                elif (current_time - syn.last_pre_spike > self.activity_threshold
                      and syn.last_pre_spike > 0):
                    should_prune = True
                if should_prune:
                    pruned += 1
                else:
                    surviving.append(syn)
            neuron.synapses_in = surviving
        self.phagocytosis_count += pruned
        return pruned

# ============================================================================
# CELULA INTELIGENTE
# ============================================================================

@dataclass
class IntelligentCell:
    id: int
    pos: np.ndarray
    grn: GeneRegulatoryNetwork
    sig: DeltaNotch
    land: WaddingtonLandscape
    type: CellType = CellType.STEM
    subtype: Optional[Any] = None
    state: CellState = CellState.QUIESCENT
    energy: float = 150.0
    age: float = 0.0
    div_thresh: float = 30.0

    v: float = -65.0
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
    sensory_current: float = 0.0

    firing_rate_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_scaling_time: float = 0.0
    target_rate: float = 2.0

    v_dendrite: float = -65.0
    tau_dendrite: float = 30.0
    dendrite_spike_thresh: float = -40.0
    dendrite_coupling: float = 0.1
    dendrite_refractory: float = 0.0

    history: List[Dict] = field(default_factory=list)

    def update_biology(self, dt, neighbor_delta, ext_signal, neurotrophic, morphogen, brain_state):
        self.age += dt
        self.grn.update(dt, ext_signal, morphogen)
        self.sig.update(dt, neighbor_delta, self.grn)
        self.land.update(dt, self.grn, morphogen)

        self.energy -= 0.1 * dt
        if self.state == CellState.PROLIFERATIVE:
            self.energy -= 2.0 * dt
        if self.state == CellState.QUIESCENT:
            self.energy += 0.3 * dt
        if self.type == CellType.NEURON and self.energy < 200.0:
            self.energy += 0.1 * dt
        self.energy = max(0.0, min(200.0, self.energy))

        if self.type == CellType.STEM:
            lt = self.land.get_cell_type()
            diff_score = self.grn.diff_score()
            if lt == CellType.NEURON and diff_score > 1.0:
                self.type = CellType.NEURON
                if self.grn.inhibitory_bias() > 0.4 or morphogen > 0.5:
                    if self.grn.D_protein > 0.6 or np.random.rand() < 0.6:
                        self.subtype = NeuronSubtype.INHIBITORY_PV
                        self.tau_m = 10.0
                        self.v_thresh = -45.0
                    else:
                        self.subtype = NeuronSubtype.INHIBITORY_SST
                        self.tau_m = 25.0
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

        if self.state == CellState.MIGRATING and self.type == CellType.NEURON:
            if self.subtype == NeuronSubtype.EXCITATORY:
                self.pos[2] += 0.5 * dt * (1.0 - morphogen)
            else:
                self.pos[0] += np.random.randn() * 2.0 * dt
                self.pos[1] += np.random.randn() * 2.0 * dt
            if self.pos[2] > 64.0:
                self.state = CellState.DIFFERENTIATING

        self.history.append({'type': self.type.value, 'v': self.v, 'rate': self.firing_rate})
        if len(self.history) > 500:
            self.history.pop(0)

    def _refractory_period(self):
        if self.subtype == NeuronSubtype.INHIBITORY_PV:
            return 2.0
        elif self.subtype == NeuronSubtype.INHIBITORY_SST:
            return 5.0
        return 5.0

    def update_neural(self, dt, time, input_current, brain_state, neuromod=None):
        if self.type != CellType.NEURON:
            return

        if self.refractory > 0:
            self.refractory -= dt
            self.v = self.v_reset
            return
        if self.dendrite_refractory > 0:
            self.dendrite_refractory -= dt

        thresh_mod = 0.0
        if neuromod is not None:
            thresh_mod += neuromod.excitability_modulation()
        energy_fraction = self.energy / 200.0
        thresh_mod += 5.0 * (1.0 - energy_fraction)
        effective_thresh = self.v_thresh + thresh_mod

        I_soma = 0.0
        I_dendrite = 0.0
        for syn in self.synapses_in:
            if syn.target_compartment == 'dendrite':
                I_dendrite += syn.get_current(self.v_dendrite)
            else:
                I_soma += syn.get_current(self.v)

        dv_dend = (-(self.v_dendrite - self.v_rest) + I_dendrite) / self.tau_dendrite
        self.v_dendrite += dv_dend * dt

        dendrite_current = 0.0
        if self.v_dendrite >= self.dendrite_spike_thresh and self.dendrite_refractory <= 0:
            self.v_dendrite = self.v_rest
            self.dendrite_refractory = 10.0
            dendrite_current = 15.0

        I_coupling = self.dendrite_coupling * (self.v_dendrite - self.v)
        self.total_synaptic_current = I_soma + I_coupling + dendrite_current + input_current + self.sensory_current

        dv = (-(self.v - self.v_rest) + self.total_synaptic_current) / self.tau_m
        self.v += dv * dt

        spiked = False
        if self.v >= effective_thresh:
            self.v = self.v_reset
            self.refractory = self._refractory_period()
            self.last_spike_time = time
            self.spike_times.append(time)
            if len(self.spike_times) > 100:
                self.spike_times.pop(0)
            for syn in self.synapses_out:
                syn.pre_spike(time)
            self.energy = max(0.0, self.energy - 1.0)
            spiked = True

        recent_spikes = [t for t in self.spike_times if time - t < 100.0]
        self.firing_rate = len(recent_spikes) * 10.0
        self.firing_rate_history.append(self.firing_rate)

        if time - self.last_scaling_time > 1000.0 and len(self.firing_rate_history) > 100:
            self.last_scaling_time = time
            avg_rate = np.mean(self.firing_rate_history)
            if avg_rate > 0:
                scale = self.target_rate / (avg_rate + 1e-6)
                scale = np.clip(scale, 0.8, 1.2)
                for syn in self.synapses_in:
                    syn.weight *= scale

        return spiked

    def update_glia(self, dt, neurons, synapses, local_energy=1.0):
        if self.type != CellType.GLIA:
            return
        if self.subtype == GliaSubtype.ASTROCYTE:
            astro = Astrocyte(self.id, self.pos)
            astro.ca_level = getattr(self, 'ca_level', 0.0)
            astro.vasodilation_signal = getattr(self, 'vasodilation_signal', 0.0)
            astro.update(dt, neurons, synapses, local_energy)
            self.ca_level = astro.ca_level
            self.d_serine = astro.d_serine
            self.vasodilation_signal = astro.vasodilation_signal
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
        merit_factor = 0.3 + 0.7 * dopamine
        return (self.type == CellType.STEM
                and self.state == CellState.PROLIFERATIVE
                and self.energy > (self.div_thresh / merit_factor)
                and self.grn.time_since_last_division > 30.0)

    def divide(self, new_id):
        self.energy -= 40.0
        self.grn.time_since_last_division = 0.0
        daughter = IntelligentCell(
            id=new_id,
            pos=self.pos + np.random.randn(3) * 5.0,
            grn=GeneRegulatoryNetwork(),
            sig=DeltaNotch(),
            land=WaddingtonLandscape()
        )
        daughter.grn.H_protein = self.grn.H_protein * np.random.uniform(0.9, 1.1)
        daughter.grn.M_protein = self.grn.M_protein * np.random.uniform(0.9, 1.1)
        daughter.land.x = self.land.x + np.random.randn() * 0.2
        daughter.land.y = self.land.y + np.random.randn() * 0.2
        daughter.state = CellState.QUIESCENT
        daughter.energy = 60.0
        return daughter

# ============================================================================
# BRAIN OBSERVER — Monitoreo en tiempo real
# ============================================================================

class BrainObserver:
    """Observa el estado interno del cerebro sin modificarlo."""

    def __init__(self, brain):
        self.brain = brain

    def _layer_activity(self, z_min, z_max, neurons):
        layer = [n for n in neurons if z_min <= n.pos[2] < z_max]
        active = [n for n in layer if n.last_spike_time == self.brain.time]
        total_current = sum(n.total_synaptic_current for n in layer)
        avg_weight = 0.0
        n_syn = 0
        for n in layer:
            for s in n.synapses_in:
                avg_weight += s.weight
                n_syn += 1
        if n_syn > 0:
            avg_weight /= n_syn
        return {
            'total': len(layer),
            'active': len(active),
            'firing_rate': sum(n.firing_rate for n in layer) / (len(layer) + 1),
            'avg_synaptic_current': total_current / (len(layer) + 1),
            'avg_weight': avg_weight,
        }

    def snapshot(self):
        neurons = [c for c in self.brain.cells if c.type == CellType.NEURON]
        stem_cells = [c for c in self.brain.cells if c.type == CellType.STEM]
        glia_cells = [c for c in self.brain.cells if c.type == CellType.GLIA]

        total_synapses = sum(len(n.synapses_in) for n in neurons)
        weights = [s.weight for n in neurons for s in n.synapses_in]
        avg_weight = np.mean(weights) if weights else 0.0
        max_weight = np.max(weights) if weights else 0.0
        min_weight = np.min(weights) if weights else 0.0

        return {
            'time': self.brain.time,
            'step': self.brain.step_count,
            'brain_state': self.brain.brain_state.name,
            'sensorial': self._layer_activity(0, 15, neurons),
            'oculta': self._layer_activity(15, 35, neurons),
            'motor': self._layer_activity(35, 55, neurons),
            'pfc': self._layer_activity(75, 200, neurons),
            'total_neurons': len(neurons),
            'total_stem': len(stem_cells),
            'total_glia': len(glia_cells),
            'total_synapses': total_synapses,
            'avg_weight': avg_weight,
            'max_weight': max_weight,
            'min_weight': min_weight,
            'dopamine': self.brain.neuromod.dopamine,
            'serotonin': self.brain.neuromod.serotonin,
            'acetylcholine': self.brain.neuromod.acetylcholine,
            'frustration': self.brain.frustration,
            'resilience': self.brain.resilience,
            'courage': self.brain.courage,
            'energy_avg': np.mean([c.energy for c in neurons]) if neurons else 0.0,
        }

    def print_snapshot(self):
        s = self.snapshot()
        print(f"\n{'='*60}")
        print(f"  t={s['time']:.1f}ms  paso={s['step']}  estado={s['brain_state']}")
        print(f"{'='*60}")
        for layer_name in ['sensorial', 'oculta', 'motor', 'pfc']:
            l = s[layer_name]
            print(f"  {layer_name:10s}: {l['total']:3d} celulas, "
                  f"{l['active']:3d} activas, "
                  f"rate={l['firing_rate']:.1f}Hz, "
                  f"w_prom={l['avg_weight']:.4f}")
        print(f"  {'─'*56}")
        print(f"  Neuronas: {s['total_neurons']}  Stem: {s['total_stem']}  Glia: {s['total_glia']}")
        print(f"  Sinapsis: {s['total_synapses']}  w=[{s['min_weight']:.4f}..{s['max_weight']:.4f}]")
        print(f"  DA={s['dopamine']:.3f}  5HT={s['serotonin']:.3f}  ACh={s['acetylcholine']:.3f}")
        print(f"  Fru={s['frustration']:.3f}  Res={s['resilience']:.3f}  Cou={s['courage']:.3f}")
        print(f"  Energia prom: {s['energy_avg']:.1f}")
        print(f"{'='*60}")

# ============================================================================
# STIMULUS CONTROLLER — Inyeccion controlada de estmulos
# ============================================================================

class StimulusController:
    """Inyecta estmulos en capas especificas del cerebro."""

    def __init__(self, brain):
        self.brain = brain

    def _get_layer(self, z_min, z_max):
        return [c for c in self.brain.cells
                if c.type == CellType.NEURON and z_min <= c.pos[2] < z_max]

    def inyectar_novedad(self, intensidad=0.5):
        """Inyecta senal de novedad -> dopamina sube."""
        self.brain.neuromod.dopamine = min(1.0,
            self.brain.neuromod.dopamine + intensidad * 0.3)

    def inyectar_error(self, severidad=0.5):
        """Inyecta senal de error -> frustracion crece."""
        self.brain.frustration = min(1.0,
            self.brain.frustration + severidad * 0.2)

    def inyectar_recompensa(self):
        """Recompensa -> consolida, alivia frustracion."""
        self.brain.neuromod.dopamine = min(1.0,
            self.brain.neuromod.dopamine + 0.4)
        self.brain.frustration *= 0.5
        self.brain.resilience = min(1.0,
            self.brain.resilience + 0.05)

    def inyectar_pulso(self, canal, voltaje, duracion_ms=10.0):
        """Inyecta corriente directa a una capa sensorial."""
        capa = self._get_layer(0, 15)
        for i, n in enumerate(capa):
            if i % 10 == canal % 10:
                n.sensory_current = voltaje
        pasos = int(duracion_ms / self.brain.dt)
        for _ in range(pasos - 1):
            self.brain.step()
        for n in capa:
            n.sensory_current = 0.0

    def inyectar_senal_temporal(self, frecuencia_hz, duracion_pasos=100, amplitud=400.0):
        """Inyecta senal periodica a capa sensorial."""
        capa = self._get_layer(0, 15)
        for paso in range(duracion_pasos):
            t = self.brain.time
            senal = amplitud * np.sin(2 * np.pi * frecuencia_hz * t / 1000.0)
            for n in capa:
                n.sensory_current = senal
            self.brain.step()
        for n in capa:
            n.sensory_current = 0.0

    def inyectar_senal_a_capa(self, z_min, z_max, voltaje, duracion_pasos=10):
        """Inyecta corriente constante a una capa arbitraria."""
        capa = [c for c in self.brain.cells
                if c.type == CellType.NEURON and z_min <= c.pos[2] < z_max]
        for _ in range(duracion_pasos):
            for n in capa:
                n.sensory_current = voltaje
            self.brain.step()
        for n in capa:
            n.sensory_current = 0.0

    def silencio(self, pasos=50):
        """Ejecuta pasos sin estmulos externos (estado basal)."""
        for _ in range(pasos):
            self.brain.step()

# ============================================================================
# EVENT LOGGER — Registro de eventos para analisis
# ============================================================================

class EventLogger:
    """Registra eventos del cerebro en CSV para analisis posterior."""

    def __init__(self, brain, directorio="logs"):
        self.brain = brain
        self.directorio = directorio
        self._spikes = []
        self._cell_events = []
        self._neuromod = []
        self._step_data = []
        self._snapshots = []
        os.makedirs(directorio, exist_ok=True)

    def registrar_spikes(self, neuronas_disparo):
        for n in neuronas_disparo:
            capa = 'sensorial' if n.pos[2] < 15 else \
                   'oculta' if n.pos[2] < 35 else \
                   'motor' if n.pos[2] < 55 else 'pfc'
            self._spikes.append({
                'time': self.brain.time,
                'neuron_id': n.id,
                'capa': capa,
                'subtype': n.subtype.name if n.subtype else 'NONE',
            })

    def registrar_evento_celula(self, tipo, celula_id, causa):
        self._cell_events.append({
            'time': self.brain.time,
            'evento': tipo,
            'cell_id': celula_id,
            'causa': causa,
        })

    def registrar_neuromod(self):
        self._neuromod.append({
            'time': self.brain.time,
            'dopamine': self.brain.neuromod.dopamine,
            'serotonin': self.brain.neuromod.serotonin,
            'acetylcholine': self.brain.neuromod.acetylcholine,
            'frustration': self.brain.frustration,
            'resilience': self.brain.resilience,
            'courage': self.brain.courage,
        })

    def registrar_paso(self, snapshot):
        self._step_data.append(snapshot)

    def registrar_snapshot(self, label):
        s = self.brain_observer.snapshot() if hasattr(self, 'brain_observer') else {}
        s['label'] = label
        self._snapshots.append(s)

    def _guardar_csv(self, datos, nombre):
        if not datos:
            return
        path = os.path.join(self.directorio, nombre)
        keys = datos[0].keys()
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(datos)

    def guardar(self, sufijo=""):
        ts = int(time_mod.time())
        base = f"{ts}_{sufijo}" if sufijo else str(ts)
        self._guardar_csv(self._spikes, f"{base}_spikes.csv")
        self._guardar_csv(self._cell_events, f"{base}_cell_events.csv")
        self._guardar_csv(self._neuromod, f"{base}_neuromod.csv")
        self._guardar_csv(self._step_data, f"{base}_steps.csv")
        n_spikes = len(self._spikes)
        n_cells = len(self._cell_events)
        print(f"[LOG] Guardado en {self.directorio}/ ({base}_*.csv)")
        print(f"      Spikes: {n_spikes}  Eventos celula: {n_cells}  "
              f"Neuromod: {len(self._neuromod)}  Pasos: {len(self._step_data)}")

    def limpiar(self):
        self._spikes.clear()
        self._cell_events.clear()
        self._neuromod.clear()
        self._step_data.clear()
        self._snapshots.clear()

# ============================================================================
# SIMULACION PRINCIPAL
# ============================================================================

class BrainSimulation:
    def __init__(self, name="Cerebro"):
        self.cells: List[IntelligentCell] = []
        self.time = 0.0
        self.next_id = 0
        self.dt = DT
        self.step_count = 0
        self.brain_state = BrainState.AWAKE
        self.name = name
        self._interaction_radius = 25.0

        for i in range(5):
            c = self._create_cell()
            c.pos = np.array([np.random.uniform(-40, 40),
                              np.random.uniform(-40, 40), 5.0])
            c.type = CellType.NEURON
            c.subtype = NeuronSubtype.EXCITATORY
            c.energy = 200.0
            self.cells.append(c)

        for i in range(4):
            c = self._create_cell()
            c.pos = np.array([np.random.uniform(-40, 40),
                              np.random.uniform(-40, 40), 45.0])
            c.type = CellType.NEURON
            c.subtype = NeuronSubtype.EXCITATORY
            c.energy = 200.0
            self.cells.append(c)

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

        self.compute_synaptic_connections()

        self.history = []
        self.max_history = 2000

        self.target_signal_history = []
        self.prediction_history = []
        self.error_history = []

        self.neuromod = NeuromodulatorSystem()
        self.pruned_synapses_total = 0
        self._nd_cache = {}

        self.frustration = 0.0
        self.resilience = 0.2
        self.courage = 0.2

        self._episodic_buffer = deque(maxlen=20000)
        self._best_da_threshold = 0.70
        self._neuron_utility = {}

    def _create_cell(self):
        c = IntelligentCell(
            id=self.next_id,
            pos=np.random.randn(3) * 20 + np.array([0, 0, 10]),
            grn=GeneRegulatoryNetwork(),
            sig=DeltaNotch(),
            land=WaddingtonLandscape()
        )
        self.next_id += 1
        return c

    def _compute_neighbor_deltas(self):
        n = len(self.cells)
        if n == 0:
            self._nd_cache = {}
            return
        positions = np.array([c.pos for c in self.cells])
        deltas = np.array([c.sig.delta for c in self.cells])

        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        D = np.sqrt(np.sum(diff ** 2, axis=-1))

        mask = (D < self._interaction_radius) & (D > 0)
        weights = np.where(mask, np.exp(-D / self._interaction_radius), 0.0)
        nd_all = weights @ deltas
        self._nd_cache = {c.id: nd_all[i] for i, c in enumerate(self.cells)}

    def update_brain_state(self):
        phase = self.time % SLEEP_CYCLE_PERIOD
        if phase < SLEEP_CYCLE_PERIOD * 0.7:
            self.brain_state = BrainState.AWAKE
        elif phase < SLEEP_CYCLE_PERIOD * 0.9:
            self.brain_state = BrainState.SLOW_WAVE_SLEEP
        else:
            self.brain_state = BrainState.REM

    def compute_synaptic_connections(self):
        neurons = [c for c in self.cells if c.type == CellType.NEURON]
        glia_oligo = [c for c in self.cells
                      if c.type == CellType.GLIA
                      and c.subtype == GliaSubtype.OLIGODENDROCYTE]

        for n in neurons:
            n.synapses_in.clear()
            n.synapses_out.clear()

        if len(neurons) < 2:
            return

        for pre in neurons:
            for post in neurons:
                if pre.id == post.id:
                    continue
                dist = np.linalg.norm(pre.pos - post.pos)
                if pre.subtype == NeuronSubtype.EXCITATORY:
                    max_dist = 80.0
                    p_connect = 0.3 * np.exp(-dist / 40.0)
                elif pre.subtype == NeuronSubtype.INHIBITORY_PV:
                    max_dist = 50.0
                    p_connect = 0.6 * np.exp(-dist / 20.0)
                else:
                    max_dist = 80.0
                    p_connect = 0.4 * np.exp(-dist / 40.0)

                if dist < max_dist and np.random.rand() < p_connect:
                    myelin_factor = 1.0
                    for oligo in glia_oligo:
                        d_oligo = np.linalg.norm(oligo.pos - (pre.pos + post.pos) / 2)
                        if d_oligo < oligo.radius and hasattr(oligo, 'maturation'):
                            myelin_factor = 1.0 + 9.0 * oligo.maturation
                    conduction_velocity = 0.5 * myelin_factor
                    delay = dist / conduction_velocity

                    if pre.subtype == NeuronSubtype.INHIBITORY_SST:
                        target_comp = 'dendrite'
                    else:
                        target_comp = 'soma'

                    syn = BiologicalSynapse(pre.id, post.id, pre.subtype, delay, target_comp)
                    pre.synapses_out.append(syn)
                    post.synapses_in.append(syn)

    def step(self):
        self.time += self.dt
        self.step_count += 1
        self.update_brain_state()

        neurons = [c for c in self.cells if c.type == CellType.NEURON]
        astrocytes = [c for c in self.cells
                      if c.type == CellType.GLIA
                      and c.subtype == GliaSubtype.ASTROCYTE]
        oligos = [c for c in self.cells
                  if c.type == CellType.GLIA
                  and c.subtype == GliaSubtype.OLIGODENDROCYTE]
        microglia = [c for c in self.cells
                     if c.type == CellType.GLIA
                     and c.subtype == GliaSubtype.MICROGLIA]

        total_activity = sum(n.firing_rate for n in neurons)
        neurotrophic = total_activity / (len(neurons) + 1) * 0.1

        if self.step_count % 10 == 0:
            last_error = self.error_history[-1] if self.error_history else 0.0
            self.neuromod.update(self.dt * 10, self.brain_state, last_error)

        avg_energy = np.mean([n.energy for n in neurons]) / 200.0 if neurons else 1.0

        for astro in astrocytes:
            astro.update_glia(self.dt, neurons, [], avg_energy)

        for oligo in oligos:
            oligo.update_glia(self.dt, neurons, [])
            for n in neurons:
                for syn in n.synapses_out:
                    pre_pos = next((c.pos for c in self.cells if c.id == syn.pre_id), None)
                    post_pos = next((c.pos for c in self.cells if c.id == syn.post_id), None)
                    if pre_pos is None or post_pos is None:
                        continue
                    mid = (pre_pos + post_pos) / 2
                    dist = np.linalg.norm(oligo.pos - mid)
                    if dist < oligo.radius:
                        syn.myelination = min(1.0, syn.myelination + 0.005 * oligo.maturation)

        for mg in microglia:
            mg.update_glia(self.dt, neurons, [])
        if self.step_count % 500 == 0 and microglia:
            for mg in microglia:
                micro = Microglia(mg.id, mg.pos)
                pruned = micro.prune(neurons, self.time)
                self.pruned_synapses_total += pruned

        self._compute_neighbor_deltas()
        positions_z = np.array([c.pos[2] for c in self.cells])
        morphs = 1.0 / (1.0 + np.exp(-0.1 * (positions_z - 40.0)))

        for i, cell in enumerate(self.cells):
            nd = self._nd_cache.get(cell.id, 0.0)
            morph = float(morphs[i])
            cell.update_biology(self.dt, nd, 0.0, neurotrophic, morph, self.brain_state)

        for cell in neurons:
            for syn in cell.synapses_in:
                syn.update_conductances(self.dt)

        neurons_disparo = []
        for cell in neurons:
            input_current = 0.0
            if self.brain_state == BrainState.AWAKE:
                input_current += 2.0 * np.sin(2 * np.pi * self.time / SIGNAL_PERIOD)
            spiked = cell.update_neural(self.dt, self.time, input_current,
                                        self.brain_state, self.neuromod)
            if spiked:
                neurons_disparo.append(cell)

        da_mod = self.neuromod.stdp_modulation()
        for cell in neurons_disparo:
            for syn in cell.synapses_in:
                syn.post_spike(self.time, da_mod)

        newborns = []
        for cell in self.cells:
            if cell.can_divide(self.neuromod.dopamine):
                daughter = cell.divide(self.next_id)
                self.next_id += 1
                newborns.append(daughter)
                cell.state = CellState.QUIESCENT
                if hasattr(self, '_logger'):
                    self._logger.registrar_evento_celula('neurogenesis', daughter.id, 'mitosis')
        self.cells.extend(newborns)

        prev_count = len(self.cells)
        self.cells = [c for c in self.cells if c.type != CellType.APOPTOTIC]
        apoptotic_count = prev_count - len(self.cells)
        if apoptotic_count > 0 and hasattr(self, '_logger'):
            self._logger.registrar_evento_celula('apoptosis', -1,
                f'{apoptotic_count} celulas eliminadas')
        if not self.cells:
            self.cells.append(self._create_cell())

        if hasattr(self, '_logger'):
            self._logger.registrar_spikes(neurons_disparo)

        pred = total_activity / (len(neurons) + 1)
        target = np.sin(2 * np.pi * self.time / SIGNAL_PERIOD)
        error = target - pred
        self.prediction_history.append(pred)
        self.error_history.append(error)

        self.history.append({
            'time': self.time,
            'total': len(self.cells),
            'neuron': len(neurons),
            'error': error,
            'brain_state': self.brain_state.value,
            'dopamine': self.neuromod.dopamine,
            'acetylcholine': self.neuromod.acetylcholine,
            'serotonin': self.neuromod.serotonin,
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)

        if self.brain_state == BrainState.REM and len(self._episodic_buffer) > 0:
            memory = random.choice(self._episodic_buffer)
            sensory_layer = [n for n in neurons if n.pos[2] < 15.0]
            for i, sn in enumerate(sensory_layer):
                if 'pattern' in memory:
                    channel = i % len(memory['pattern'])
                    sn.sensory_current += memory['pattern'][channel] * 250.0
            self.neuromod.dopamine = min(1.0, self.neuromod.dopamine + 0.02)

        return neurons_disparo

    def save_brain(self, path=None):
        if path is None:
            path = f"cerebro_save_{self.name.lower()}.pkl"
        data = {
            'cells': self.cells,
            'time': self.time,
            'next_id': self.next_id,
            'step_count': self.step_count,
            'frustration': self.frustration,
            'resilience': self.resilience,
            'courage': self.courage,
            '_episodic_buffer': self._episodic_buffer,
            '_neuron_utility': self._neuron_utility,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[SAVE] Cerebro guardado en {path}")

    def load_brain(self, path):
        if not os.path.exists(path):
            print(f"[LOAD] No se encontro {path}")
            return
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.cells = data.get('cells', self.cells)
        self.time = data.get('time', self.time)
        self.next_id = data.get('next_id', self.next_id)
        self.step_count = data.get('step_count', self.step_count)
        self.frustration = data.get('frustration', self.frustration)
        self.resilience = data.get('resilience', self.resilience)
        self.courage = data.get('courage', self.courage)
        self._episodic_buffer = data.get('_episodic_buffer', self._episodic_buffer)
        self._neuron_utility = data.get('_neuron_utility', self._neuron_utility)
        print(f"[LOAD] Cerebro cargado desde {path}")

# ============================================================================
# EJECUCION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  IA VIVA - MODO INVESTIGACION")
    print("=" * 60)

    cerebro = BrainSimulation(name="Investigacion")
    observer = BrainObserver(cerebro)
    estimulos = StimulusController(cerebro)
    logger = EventLogger(cerebro, directorio="logs")
    cerebro._logger = logger

    cerebro.print_snapshot = observer.print_snapshot

    print("\nCerebro inicializado. Ejecutando 100 pasos basales...")
    estimulos.silencio(100)
    observer.print_snapshot()

    print("\nInyectando senal temporal (5 Hz, 200 pasos)...")
    estimulos.inyectar_senal_temporal(5.0, duracion_pasos=200, amplitud=400.0)
    observer.print_snapshot()

    print("\nInyectando recompensa...")
    estimulos.inyectar_recompensa()
    observer.print_snapshot()

    print("\nInyectando error (severidad 0.8)...")
    estimulos.inyectar_error(0.8)
    observer.print_snapshot()

    print("\nSilencio (500 pasos)...")
    estimulos.silencio(500)
    observer.print_snapshot()

    print("\nInyectando novedad (0.9)...")
    estimulos.inyectar_novedad(0.9)
    observer.print_snapshot()

    print("\nSilencio final (1000 pasos)...")
    estimulos.silencio(1000)
    observer.print_snapshot()

    print("\nGuardando registros...")
    logger.guardar("experimento_01")

    print("\nGuardando cerebro...")
    cerebro.save_brain()

    print("\nListo. Revisa la carpeta logs/ para los CSV.")
