#ifndef CEREBRO_HPP
#define CEREBRO_HPP

#include "hardware_bridge.hpp"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <deque>

// Constantes globales de topología
const int N_SENSORY = 20;
const int N_HIDDEN = 30;
const int N_MOTOR = 30;
const int N_PFC = 20;
const int N_TOTAL = N_SENSORY + N_HIDDEN + N_MOTOR + N_PFC;

const double DT = 1.0; // ms
const double BATCH_MS = 500.0;
const double SLEEP_CYCLE_PERIOD = 20000.0;
const double SIGNAL_PERIOD = 127.7;

struct Neuron {
    double v;
    double v_dend;
    double v_thresh;
    double v_thresh_base;
    double tau_m; // ms
    double tau_dend; // ms
    double g_coupling;
    double E_ampa;
    double E_gaba;
    double v_rest;
    
    // Conductancias dinámicas
    double g_ampa_soma;
    double g_gaba_soma;
    double g_ampa_dend;
    double g_gaba_dend;

    // Umbral Adaptativo
    double delta_v_thresh;
    double tau_thresh; // ms

    // Ruido y CPG
    double noise_base;
    double is_sensory;
    double cpg_amplitude;
    double signal_period; // ms

    // Metabolismo y Emociones
    double energy;
    double firing_rate; // Hz
    double last_spike_time; // segundos
    double frustration;
    double resilience;

    // Neuromoduladores acoplados
    double da;
    double ser;
    double ach;

    // Físico / Red
    int layer_id; // 0=Sensorial, 1=Oculta, 2=Motor, 3=PFC
    int type;     // 1=Excitatoria, 4=Inhibitoria
    double x, y, z;
    double I_ext;
};

struct Synapse {
    int pre;
    int post;
    double w;
    double myelination;
    double is_excitatory;     // 1.0 o 0.0
    double target_is_dendrite; // 1.0 o 0.0
    double is_active;          // 1.0 o 0.0

    // STP (Tsodyks-Markram)
    double x_stp;
    double u_stp;
    double U_stp;
    double tau_d; // ms
    double tau_f; // ms

    // STDP
    double apre;
    double apost;
    double last_update_ms;

    // Delays
    int delay_steps;
    double base_delay_ms;
};

struct Astrocyte {
    int group_start;
    int group_end;
    double calcium;
};

class NeuromodulatorSystem {
public:
    double dopamine;
    double serotonin;
    double acetylcholine;
    double tau_da; // s
    double tau_5ht; // s
    double tau_ach; // s

    NeuromodulatorSystem();
    void update(double dt_sec, const std::string& brain_state, double prediction_error);
};

class SynapticScaler {
public:
    double target_sum_w;
    bool active;

    SynapticScaler(double target = 6.0, bool active_val = true);
    void scale(class BrainUnico& brain);
};

class GainController {
public:
    double target_rate;
    double alpha_gain;
    double v_offset;
    bool active;

    GainController(double target = 8.0, double alpha = 0.05, bool active_val = true);
    void adapt(class BrainUnico& brain, double motor_firing);
};

class EspacioGlobal {
public:
    double umbral;
    int ventana;
    double ganancia_broadcast;
    std::deque<double> fr_workspace_history;
    int steps_sobre_umbral;
    bool ignicion_activa;

    EspacioGlobal(double umbral_ini = 30.0, int vent = 100, double ganancia = 0.18);
    std::pair<double, bool> tick(class BrainUnico& brain, const std::vector<double>& firing_rates);
private:
    void adaptar_umbral();
};

struct EpisodicMemory {
    std::vector<double> sensory;
    double time_ms;
};

struct HistoryEntry {
    double time;
    int step;
    std::string state;
    double da;
    double ser;
    double ach;
    double w_mean;
    double w_max;
    int synapses;
    int spikes;
    double energy_mean;
    int pruned;
    int created;
    double prediction;
    double target;
    double workspace_fr;
    double workspace_umbral;
    bool ignicion;
};

class BrainUnico {
public:
    double time_ms;
    int step_count;
    std::string brain_state;
    int pruned_synapses;
    int created_synapses;
    
    double frustration;
    double resilience;
    double decay_factor;

    std::vector<Neuron> neurons;
    std::vector<Synapse> synapses;

    NeuromodulatorSystem neuromod;
    SynapticScaler scaler;
    GainController gain_control;
    EspacioGlobal workspace;
    std::unique_ptr<IHardwareBridge> hardware;

    // Distancias 3D precalculadas en memoria
    std::vector<double> dist_3d;

    // Búferes circulares de conductancias con delays
    double ring_ampa_soma[N_TOTAL][16];
    double ring_ampa_dend[N_TOTAL][16];
    double ring_gaba_soma[N_TOTAL][16];
    double ring_gaba_dend[N_TOTAL][16];
    int current_ring_step;

    std::deque<EpisodicMemory> episodic_buffer;
    std::vector<HistoryEntry> history;

    // Métricas del step actual
    int spikes_in_current_batch;

    // TD-Learning (Actor-Critic)
    double w_value[N_TOTAL];
    double last_state_value;
    double alpha_value;

    // Gating de Memoria de Trabajo (PBWM)
    double pfc_gates[N_TOTAL];

    // Mapa logístico caótico para el NCC
    double ncc_chaos_state;

    // Capas biológicas multiescala adicionales
    std::vector<Astrocyte> astrocytes;
    double cortisol;
    double melatonina;
    double v_if[20];

    BrainUnico();
    void step();
    std::string get_state_json();
    
    bool save_state(const std::string& filepath);
    bool load_state(const std::string& filepath);

private:
    void homeostasis();
    void structural_plasticity();
    void sleep_replay();
};

#endif // CEREBRO_HPP
