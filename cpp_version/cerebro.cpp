#include "cerebro.hpp"
#include "server.hpp"
#include "virtual_bridge.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Generadores aleatorios estáticos
static std::random_device rd;
static std::mt19937 gen(rd());
static std::normal_distribution<double> normal_dist(0.0, 1.0);
static std::uniform_real_distribution<double> rand_dist(0.0, 1.0);

// ============================================================================
// SISTEMA NEUROMODULADOR
// ============================================================================
NeuromodulatorSystem::NeuromodulatorSystem()
    : dopamine(0.5), serotonin(0.5), acetylcholine(0.5),
      tau_da(2.0), tau_5ht(8.0), tau_ach(1.0) {}

void NeuromodulatorSystem::update(double dt_sec, const std::string& brain_state, double prediction_error) {
    double da_target = 0.5 + 0.5 * std::tanh(std::abs(prediction_error) * 2.0);
    if (brain_state == "SLOW_WAVE_SLEEP") {
        da_target = 0.2;
    } else if (brain_state == "REM") {
        da_target = 0.6;
    }

    double ser_target = (brain_state == "AWAKE") ? 0.7 : ((brain_state == "SLOW_WAVE_SLEEP") ? 0.4 : 0.1);
    double ach_target = (brain_state == "AWAKE") ? 0.8 : ((brain_state == "SLOW_WAVE_SLEEP") ? 0.2 : 0.7);

    dopamine += (da_target - dopamine) * dt_sec / tau_da;
    serotonin += (ser_target - serotonin) * dt_sec / tau_5ht;
    acetylcholine += (ach_target - acetylcholine) * dt_sec / tau_ach;

    dopamine = std::max(0.0, std::min(dopamine, 1.0));
    serotonin = std::max(0.0, std::min(serotonin, 1.0));
    acetylcholine = std::max(0.0, std::min(acetylcholine, 1.0));
}

// ============================================================================
// REGULADORES HOMEOSTÁTICOS
// ============================================================================
SynapticScaler::SynapticScaler(double target, bool active_val)
    : target_sum_w(target), active(active_val) {}

void SynapticScaler::scale(BrainUnico& brain) {
    if (!active) return;
    
    for (int neuron_id = 0; neuron_id < N_TOTAL; ++neuron_id) {
        double sum_w = 0.0;
        std::vector<int> syn_indices;
        
        for (size_t s_idx = 0; s_idx < brain.synapses.size(); ++s_idx) {
            const auto& s = brain.synapses[s_idx];
            if (s.post == neuron_id && s.is_excitatory > 0.5 && s.is_active > 0.5) {
                sum_w += s.w;
                syn_indices.push_back((int)s_idx);
            }
        }
        
        // Escalado homeostático astrocítico unidireccional amortiguado (suave)
        if (sum_w > target_sum_w && !syn_indices.empty()) {
            double factor = target_sum_w / sum_w;
            // Coeficiente de amortiguación de 0.15 para evitar oscilaciones inestables en el aprendizaje
            double smooth_factor = 1.0 + 0.15 * (factor - 1.0);
            for (int idx : syn_indices) {
                brain.synapses[idx].w *= smooth_factor;
                brain.synapses[idx].w = std::max(0.02, std::min(brain.synapses[idx].w, 2.0));
            }
        }
    }
}

GainController::GainController(double target, double alpha, bool active_val)
    : target_rate(target), alpha_gain(alpha), v_offset(0.0), active(active_val) {}

void GainController::adapt(BrainUnico& brain, double motor_firing) {
    if (!active) return;
    
    double delta_offset = alpha_gain * (motor_firing - target_rate);
    v_offset = std::max(-5.0, std::min(v_offset + delta_offset, 10.0));
}

// ============================================================================
// ESPACIO GLOBAL DE TRABAJO
// ============================================================================
EspacioGlobal::EspacioGlobal(double umbral_ini, int vent, double ganancia)
    : umbral(umbral_ini), ventana(vent), ganancia_broadcast(ganancia),
      steps_sobre_umbral(0), ignicion_activa(false) {}

std::pair<double, bool> EspacioGlobal::tick(BrainUnico& brain, const std::vector<double>& firing_rates) {
    double actividad_ponderada = 0.0;
    double peso_total = 0.0;
    
    // Pesos por región: Sensory=0.2, Hidden=1.5, Motor=3.0, PFC=4.0
    double pesos[4] = {0.2, 1.5, 3.0, 4.0};

    for (int i = 0; i < N_TOTAL; ++i) {
        int layer = brain.neurons[i].layer_id;
        actividad_ponderada += pesos[layer] * firing_rates[i];
        peso_total += pesos[layer];
    }

    if (peso_total > 0.0) {
        actividad_ponderada /= peso_total;
    }

    fr_workspace_history.push_back(actividad_ponderada);
    if (fr_workspace_history.size() > (size_t)ventana * 2) {
        fr_workspace_history.pop_front();
    }
    adaptar_umbral();

    bool evento_nuevo = false;
    if (actividad_ponderada > umbral) {
        steps_sobre_umbral++;
        if (steps_sobre_umbral >= 2) {
            ignicion_activa = true;
            evento_nuevo = true;
        }
    } else {
        ignicion_activa = false;
        steps_sobre_umbral = 0;
    }

    return {actividad_ponderada, ignicion_activa};
}

void EspacioGlobal::adaptar_umbral() {
    if (fr_workspace_history.size() < (size_t)ventana) {
        return;
    }
    std::vector<double> recientes(fr_workspace_history.end() - ventana, fr_workspace_history.end());
    std::sort(recientes.begin(), recientes.end());
    int idx = (int)(0.90 * (recientes.size() - 1));
    double nuevo_umbral = recientes[idx];
    umbral = 0.95 * umbral + 0.05 * nuevo_umbral;
}

// ============================================================================
// CEREBRO ÚNICO
// ============================================================================
BrainUnico::BrainUnico()
    : time_ms(0.0), step_count(0), brain_state("AWAKE"),
      pruned_synapses(0), created_synapses(0), frustration(0.0), resilience(0.2),
      decay_factor(0.985), current_ring_step(0), spikes_in_current_batch(0),
      hardware(std::make_unique<VirtualBridge>()) {
    
    // Inicializar búferes de conductancias con delays
    for (int i = 0; i < N_TOTAL; ++i) {
        for (int j = 0; j < 16; ++j) {
            ring_ampa_soma[i][j] = 0.0;
            ring_ampa_dend[i][j] = 0.0;
            ring_gaba_soma[i][j] = 0.0;
            ring_gaba_dend[i][j] = 0.0;
        }
    }

    // Inicializar parámetros de TD-learning y gating
    last_state_value = 0.0;
    alpha_value = 0.02;
    ncc_chaos_state = 0.357;
    cortisol = 0.1;
    melatonina = 0.0;
    for (int i = 0; i < 20; ++i) {
        v_if[i] = 0.0;
    }
    
    astrocytes.resize(10);
    for (int i = 0; i < 10; ++i) {
        astrocytes[i].group_start = i * 10;
        astrocytes[i].group_end = (i + 1) * 10;
        astrocytes[i].calcium = 0.0;
    }

    for (int i = 0; i < N_TOTAL; ++i) {
        w_value[i] = rand_dist(gen) * 0.1;
        pfc_gates[i] = 1.0;
    }

    // 1. Configurar capas de las neuronas
    neurons.resize(N_TOTAL);
    for (int i = 0; i < N_TOTAL; ++i) {
        auto& n = neurons[i];
        if (i < N_SENSORY) {
            n.layer_id = 0; // Sensorial
        } else if (i < N_SENSORY + N_HIDDEN) {
            n.layer_id = 1; // Oculta
        } else if (i < N_SENSORY + N_HIDDEN + N_MOTOR) {
            n.layer_id = 2; // Motor
        } else {
            n.layer_id = 3; // PFC
        }

        // Tipo: Excitatoria por defecto
        n.type = 1; 
    }

    // 20% de las Ocultas y PFC son inhibitorias
    // Ocultas: 20 a 49 (30 neuronas). Las últimas 6 son inhibitorias
    for (int i = 44; i < 50; ++i) {
        neurons[i].type = 4;
    }

    // PFC: 80 a 99 (20 neuronas). Las últimas 4 son inhibitorias
    for (int i = 96; i < 100; ++i) {
        neurons[i].type = 4;
    }

    // 2. Posiciones 3D de visualización y conectividad
    for (int i = 0; i < N_TOTAL; ++i) {
        auto& n = neurons[i];
        n.x = rand_dist(gen) * 80.0 - 40.0;
        n.y = rand_dist(gen) * 80.0 - 40.0;
        if (n.layer_id == 0) n.z = 5.0;
        else if (n.layer_id == 1) n.z = 25.0;
        else if (n.layer_id == 2) n.z = 45.0;
        else n.z = 65.0;

        // Parámetros de membrana LIF
        n.v_rest = -65.0;
        n.v = -65.0 + normal_dist(gen) * 3.0;
        n.v_dend = -65.0 + normal_dist(gen) * 3.0;
        
        n.v_thresh_base = (n.type == 1) ? -55.0 : -57.0;
        n.v_thresh = n.v_thresh_base;
        n.tau_m = (n.type == 1) ? 15.0 : 10.0; // ms
        n.tau_dend = 30.0; // ms
        n.g_coupling = 0.35;
        n.E_ampa = 0.0;
        n.E_gaba = -75.0;

        n.g_ampa_soma = 0.0;
        n.g_gaba_soma = 0.0;
        n.g_ampa_dend = 0.0;
        n.g_gaba_dend = 0.0;

        n.delta_v_thresh = (n.type == 1) ? 0.35 : 0.20;
        n.tau_thresh = (n.type == 1) ? 60.0 : 30.0; // ms

        n.noise_base = 1.2;
        n.is_sensory = (n.layer_id == 0) ? 1.0 : 0.0;
        n.cpg_amplitude = 2.5;
        n.signal_period = SIGNAL_PERIOD; // ms

        n.energy = 1.0;
        n.firing_rate = 0.0;
        n.last_spike_time = -1.0;
        n.frustration = 0.0;
        n.resilience = 0.2;
        n.I_ext = 0.0;
    }

    // 3. Crear sinapsis con conectividad all-to-all (50*50 = 2500)
    synapses.reserve(N_TOTAL * N_TOTAL);
    dist_3d.reserve(N_TOTAL * N_TOTAL);
    
    for (int i = 0; i < N_TOTAL; ++i) {
        for (int j = 0; j < N_TOTAL; ++j) {
            Synapse s;
            s.pre = i;
            s.post = j;
            s.w = 0.0;
            s.myelination = 0.0;
            s.is_excitatory = (neurons[i].type == 1) ? 1.0 : 0.0;
            s.target_is_dendrite = (rand_dist(gen) < 0.5) ? 1.0 : 0.0;
            s.is_active = 0.0;

            // Parámetros STP
            s.U_stp = (neurons[i].type == 1) ? 0.12 : 0.5;
            s.tau_d = (neurons[i].type == 1) ? 80.0 : 100.0; // ms
            s.tau_f = (neurons[i].type == 1) ? 100.0 : 150.0; // ms
            s.x_stp = 1.0;
            s.u_stp = s.U_stp;

            s.apre = 0.0;
            s.apost = 0.0;
            s.last_update_ms = 0.0;

            // Retardos basados en distancia 3D
            double dx = neurons[i].x - neurons[j].x;
            double dy = neurons[i].y - neurons[j].y;
            double dz = neurons[i].z - neurons[j].z;
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            dist_3d.push_back(dist);
            s.base_delay_ms = dist; // Se calibrará con la máxima distancia más abajo
            synapses.push_back(s);
        }
    }

    // Calibrar delays con respecto a la máxima distancia
    double max_dist = 0.1;
    for (double d : dist_3d) {
        if (d > max_dist) max_dist = d;
    }

    for (size_t k = 0; k < synapses.size(); ++k) {
        double d_norm = dist_3d[k] / max_dist;
        synapses[k].base_delay_ms = 1.5 + 2.5 * d_norm; // Rango [1.5, 4.0] ms
        synapses[k].delay_steps = (int)std::round(synapses[k].base_delay_ms); // Discretizado a ms enteros
        if (synapses[k].delay_steps < 1) synapses[k].delay_steps = 1;
        if (synapses[k].delay_steps > 15) synapses[k].delay_steps = 15;
    }

    // 4. Activar topología inicial (conectividad difusa dispersa)
    for (size_t k = 0; k < synapses.size(); ++k) {
        auto& s = synapses[k];
        int pre_layer = neurons[s.pre].layer_id;
        int post_layer = neurons[s.post].layer_id;
        
        bool active = false;
        double p = rand_dist(gen);

        if (s.pre >= 90 && s.post >= 90) {
            // Conectividad recurrente interna densa y caótica en NCC
            if (p < 0.50 && s.pre != s.post) {
                s.is_active = 1.0;
                s.is_excitatory = (rand_dist(gen) < 0.40) ? 0.0 : 1.0;
                s.w = 0.3 + rand_dist(gen) * 0.3;
                s.myelination = 0.3;
            }
            continue;
        }

        if (pre_layer == 0 && post_layer == 1 && p < 0.35) active = true;
        else if (pre_layer == 1 && post_layer == 2 && p < 0.35) active = true;
        else if (pre_layer == 1 && post_layer == 3 && p < 0.25) active = false; // Deshabilitar PFC (trabajo memoria) para CSI
        else if (pre_layer == 3 && post_layer == 1 && p < 0.25) active = false;
        else if (pre_layer == 3 && post_layer == 2 && p < 0.25) active = false;
        else if (pre_layer == 1 && post_layer == 1 && p < 0.15) active = true;
        else if (pre_layer == 3 && post_layer == 3 && p < 0.25) active = false;
        else if (pre_layer == 2 && post_layer == 2 && p < 0.05) active = true;

        if (active && s.pre != s.post) { // Evitar autolazos
            s.is_active = 1.0;
            double scale = 1.0;
            if (pre_layer == 0 && post_layer == 1) scale = 0.5; // Amplificación sensorial-oculta (Calibrado a escala metabólica baja)
            if (pre_layer == 1 && post_layer == 2) scale = 0.5; // Amplificación oculta-motora (Calibrado a escala metabólica baja)
            s.w = scale * ((s.is_excitatory > 0.5) ? (0.8 + rand_dist(gen) * 0.6) : 1.2);
            s.myelination = 0.1 + rand_dist(gen) * 0.2;
        }
    }
}

void BrainUnico::step() {
    time_ms += BATCH_MS;
    step_count++;
    
    frustration = 0.0;
    
    // Determinar estado de sueño/vigilia (Forzar AWAKE constante para entrenamiento y clasificación de CSI)
    brain_state = "AWAKE";
    melatonina = 0.0;

    // Extraer tasas de disparo de los neurons del paso previo para el Workspace
    std::vector<double> current_rates(N_TOTAL);
    for (int i = 0; i < N_TOTAL; ++i) {
        current_rates[i] = neurons[i].firing_rate;
    }

    // 1. Tick del Espacio Global de Consciencia
    auto ws_result = workspace.tick(*this, current_rates);
    double fr_workspace = ws_result.first;
    bool ignicion = ws_result.second;

    // Sincronizar neuromoduladores a cada neurona
    for (int i = 0; i < N_TOTAL; ++i) {
        neurons[i].da = neuromod.dopamine;
        neurons[i].ser = neuromod.serotonin;
        neurons[i].ach = neuromod.acetylcholine;
        neurons[i].frustration = frustration;
        neurons[i].resilience = resilience;
        neurons[i].I_ext = 0.0;
    }

    double target_signal = 0.0;
    double prediction = 0.0;
    double error = 0.0;
    std::vector<double> sensor_currents(N_SENSORY, 0.0);

    if (brain_state == "AWAKE") {
        for (int i = 0; i < N_TOTAL; ++i) {
            if (neurons[i].is_sensory > 0.5) {
                neurons[i].cpg_amplitude = 2.5;
            }
        }

        // Bucle Cerrado: Calcular promedio de energía
        double sum_energy = 0.0;
        for (int i = 0; i < N_TOTAL; ++i) {
            sum_energy += neurons[i].energy;
        }
        double mean_energy = sum_energy / N_TOTAL;

        // Leer corrientes sensoriales del puente de hardware
        hardware->get_sensory_input(neurons, sensor_currents, time_ms, mean_energy);

        // TD-LEARNING (Actor-Critic) & PBWM (Gating de Memoria Prefrontal)
        double reward = hardware->get_dopamine_reward();
        
        // 1. Evaluar el valor estimado del estado actual V(t) basándose en actividad de PFC y Hidden
        double current_state_value = 0.0;
        for (int i = 20; i < 50; ++i) {
            current_state_value += w_value[i] * (neurons[i].firing_rate / 100.0);
        }
        for (int i = 80; i < 100; ++i) {
            current_state_value += w_value[i] * (neurons[i].firing_rate / 100.0);
        }

        // 2. Reward Shaping para guiar el aprendizaje dinámico
        double r_immediate = reward;
        VirtualBridge* vb = dynamic_cast<VirtualBridge*>(hardware.get());
        if (vb) {
            double current_dist = vb->entorno.distancia_a_comida();
            if (current_dist < vb->ultima_distancia) {
                r_immediate += 0.05 * std::max(0.0, 1.0 - current_dist / 40.0);
            } else if (current_dist > vb->ultima_distancia) {
                r_immediate -= 0.03 * std::max(0.0, 1.0 - current_dist / 40.0);
            }
            vb->ultima_distancia = current_dist;
        }

        // 3. Error de predicción por diferencia temporal (TD-Error / RPE)
        double gamma = 0.95;
        double delta = r_immediate + gamma * current_state_value - last_state_value;

        // 4. Actualizar los pesos del crítico (regla delta)
        for (int i = 20; i < 50; ++i) {
            w_value[i] = std::max(0.0, std::min(w_value[i] + alpha_value * delta * (neurons[i].firing_rate / 100.0), 1.5));
        }
        for (int i = 80; i < 100; ++i) {
            w_value[i] = std::max(0.0, std::min(w_value[i] + alpha_value * delta * (neurons[i].firing_rate / 100.0), 1.5));
        }

        // 5. Guardar valor de estado actual para el siguiente step
        last_state_value = current_state_value;

        // 6. Mapeo de Dopamina Fásica / Tónica basada en RPE
        // RPE positivo genera ráfagas de dopamina; RPE negativo genera caídas (dips)
        int ts = hardware->get_csi_true_state();
        double tt = hardware->get_csi_trial_timer();
        if (ts > 0 && tt < 10.0) {
            neuromod.dopamine = 1.0; // Dopamina supervisada alta durante entrenamiento de CSI
        } else {
            neuromod.dopamine = 0.5 + 0.5 * std::tanh(delta * 3.0);
        }

        // Si el agente comió (recompensa cruda), recargar energía y limpiar frustración
        if (reward > 0.0) {
            frustration = 0.0;
            for (int i = 0; i < N_TOTAL; ++i) {
                neurons[i].energy = std::min(1.0, neurons[i].energy + 0.30);
            }
        }

        // 7. Lógica de Gating (PBWM) para la memoria de trabajo del PFC
        // Si hay una señal de sorpresa positiva (delta > 0.12) y no está ciego, abrimos compuertas para memorizar
        bool blind = hardware->is_blind();
        double gate_val = (delta > 0.12 && !blind) ? 1.0 : 0.0;
        // Memoria de trabajo (gated): 80 a 89
        for (int i = 80; i < 90; ++i) {
            pfc_gates[i] = gate_val;
        }
        // PFC General (abierta por defecto): 90 a 99
        for (int i = 90; i < 100; ++i) {
            pfc_gates[i] = 1.0;
        }

        target_signal = sin(2.0 * M_PI * time_ms / SIGNAL_PERIOD);

        // Guardar en búfer episódico si la novedad/dopamina es alta
        if (neuromod.dopamine > 0.65) {
            EpisodicMemory mem;
            mem.time_ms = time_ms;
            mem.sensory.resize(N_SENSORY);
            for (int i = 0; i < N_SENSORY; ++i) {
                mem.sensory[i] = neurons[i].I_ext;
            }
            episodic_buffer.push_back(mem);
            if (episodic_buffer.size() > 1000) {
                episodic_buffer.pop_front();
            }
        }

        // Broadcast Consciente global
        if (ignicion) {
            double exceso = std::max(0.0, (fr_workspace - workspace.umbral) / std::max(1.0, workspace.umbral));
            double ganancia = workspace.ganancia_broadcast * (1.0 + exceso);
            
            for (int i = 0; i < N_TOTAL; ++i) {
                if (neurons[i].layer_id >= 1) { // Oculta, Motor, PFC
                    neurons[i].I_ext += ganancia * 3.5 * (neurons[i].firing_rate / 100.0);
                }
            }
        }
    } else if (brain_state == "SLOW_WAVE_SLEEP") {
        for (int i = 0; i < N_TOTAL; ++i) {
            if (neurons[i].is_sensory > 0.5) {
                neurons[i].cpg_amplitude = 1.0;
            }
        }
    } else { // REM
        for (int i = 0; i < N_TOTAL; ++i) {
            if (neurons[i].is_sensory > 0.5) {
                neurons[i].cpg_amplitude = 0.0;
            }
        }
        sleep_replay();
    }

    // 2. Correr la simulación numérica por 500 pasos de 1ms
    spikes_in_current_batch = 0;
    std::vector<int> spike_counts(N_TOTAL, 0);
    int astrocytes_spike_counts[10] = {0};

    // Vectores para indexación rápida de sinapsis pre y post
    static std::vector<std::vector<int>> pre_syn_list(N_TOTAL);
    static std::vector<std::vector<int>> post_syn_list(N_TOTAL);
    static bool lists_built = false;
    
    if (!lists_built) {
        for (size_t k = 0; k < synapses.size(); ++k) {
            pre_syn_list[synapses[k].pre].push_back((int)k);
            post_syn_list[synapses[k].post].push_back((int)k);
        }
        lists_built = true;
    }

    int true_state = hardware->get_csi_true_state();
    double trial_timer = hardware->get_csi_trial_timer();
    bool stdp_active = (true_state > 0 && trial_timer < 10.0);

    for (int ms_step = 0; ms_step < (int)BATCH_MS; ++ms_step) {
        double current_time_sec = (time_ms - BATCH_MS + ms_step) / 1000.0;

        // Actualizar el generador de corriente caótico (Mapa Logístico) para el NCC (Amplitud reducida a 1.5)
        ncc_chaos_state = 3.99 * ncc_chaos_state * (1.0 - ncc_chaos_state);

        // 1. Filtro Sensorial IF (digitalización de entradas visuales y hambre en spikes discretos)
        if (brain_state == "AWAKE") {
            for (int i = 0; i < N_SENSORY; ++i) {
                v_if[i] += sensor_currents[i] * 0.001; // dt = 1ms = 0.001s
                if (v_if[i] >= 1.0) {
                    v_if[i] = 0.0;
                    neurons[i].v = neurons[i].v_thresh + 1.0; // Forzar disparo directo de la neurona sensorial
                }
            }

            // Aprendizaje Guiado por Profesor (Teacher Forcing): primer 70% del ensayo (t < 7.0s)
            int true_state = hardware->get_csi_true_state();
            double trial_timer = hardware->get_csi_trial_timer();
            if (true_state >= 0) {
                if (trial_timer < 10.0) {  // Profesor activo en los primeros 10s del ensayo de 15s
                    // Limpiar corrientes externas y fijar guía/profesor (para neuronas motoras 50-79 y ocultas 20-43)
                    for (int i = 20; i < 44; ++i) {
                        neurons[i].I_ext = 0.0;
                    }
                    for (int i = 50; i < 80; ++i) {
                        neurons[i].I_ext = 0.0;
                    }
                    
                    if (true_state == 0) {
                        // Vacío: Excitar motor de vacío, inhibir resto de motor y todas las ocultas activas
                        for (int i = 50; i < 60; ++i) neurons[i].I_ext = 15.0;
                        for (int i = 60; i < 80; ++i) neurons[i].I_ext = -35.0;
                        for (int i = 20; i < 44; ++i) neurons[i].I_ext = -35.0;
                    } else if (true_state == 1) {
                        // Sujeto A: Excitar ocultas A (20-31) y motor A (60-69), inhibir resto
                        for (int i = 20; i < 32; ++i) neurons[i].I_ext = 12.0;
                        for (int i = 32; i < 44; ++i) neurons[i].I_ext = -35.0;
                        for (int i = 60; i < 70; ++i) neurons[i].I_ext = 15.0;
                        for (int i = 50; i < 60; ++i) neurons[i].I_ext = -35.0;
                        for (int i = 70; i < 80; ++i) neurons[i].I_ext = -35.0;
                    } else if (true_state == 2) {
                        // Sujeto B: Excitar ocultas B (32-43) y motor B (70-79), inhibir resto
                        for (int i = 20; i < 32; ++i) neurons[i].I_ext = -35.0;
                        for (int i = 32; i < 44; ++i) neurons[i].I_ext = 12.0;
                        for (int i = 70; i < 80; ++i) neurons[i].I_ext = 15.0;
                        for (int i = 50; i < 70; ++i) neurons[i].I_ext = -35.0;
                    }
                } else {
                    // Fase de evaluacion autonoma (t >= 10s): limpiar corrientes del profesor
                    for (int i = 20; i < 44; ++i) neurons[i].I_ext = 0.0;
                    for (int i = 50; i < 80; ++i) neurons[i].I_ext = 0.0;
                }
            }
        }

        // Inyectar señales acumuladas en los búferes de conductancias para este milisegundo
        for (int i = 0; i < N_TOTAL; ++i) {
            neurons[i].g_ampa_soma += ring_ampa_soma[i][current_ring_step];
            neurons[i].g_ampa_dend += ring_ampa_dend[i][current_ring_step];
            neurons[i].g_gaba_soma += ring_gaba_soma[i][current_ring_step];
            neurons[i].g_gaba_dend += ring_gaba_dend[i][current_ring_step];

            // Limpiar las ranuras para el próximo ciclo
            ring_ampa_soma[i][current_ring_step] = 0.0;
            ring_ampa_dend[i][current_ring_step] = 0.0;
            ring_gaba_soma[i][current_ring_step] = 0.0;
            ring_gaba_dend[i][current_ring_step] = 0.0;
        }

        // Evaluar dinámica de cada neurona
        std::vector<bool> current_step_spikes(N_TOTAL, false);
        for (int i = 0; i < N_TOTAL; ++i) {
            auto& n = neurons[i];

            bool is_clamped = false;
            if (trial_timer < 10.0) {
                if (true_state == 0) {
                    if ((i >= 20 && i < 44) || (i >= 60 && i < 80)) is_clamped = true;
                } else if (true_state == 1) {
                    if ((i >= 32 && i < 44) || (i >= 50 && i < 60) || (i >= 70 && i < 80)) is_clamped = true;
                } else if (true_state == 2) {
                    if ((i >= 20 && i < 32) || (i >= 50 && i < 70)) is_clamped = true;
                }
            }

            // CPG
            double I_cpg = n.cpg_amplitude * sin(2.0 * M_PI * current_time_sec / (n.signal_period / 1000.0)) * n.is_sensory;
            
            // Conductancias e Integración
            double I_syn_soma = n.g_ampa_soma * (n.E_ampa - n.v) + n.g_gaba_soma * (n.E_gaba - n.v);
            double I_syn_dend = n.g_ampa_dend * (n.E_ampa - n.v_dend) + n.g_gaba_dend * (n.E_gaba - n.v_dend);
            double I_coupling = n.g_coupling * (n.v_dend - n.v);

            // Umbral base efectivo metabólico, modulado por offsets astrocítico y de ganancia independientes
            double astro_offset = (astrocytes[i / 10].calcium > 0.35) ? 2.5 : 0.0;
            double gain_offset = (n.layer_id == 2) ? gain_control.v_offset : 0.0;
            double v_thresh_base_effective = n.v_thresh_base + 15.0 * std::pow(1.0 - n.energy, 2) + astro_offset + gain_offset;
            n.v_thresh += (v_thresh_base_effective - n.v_thresh) * 0.001 / (n.tau_thresh / 1000.0);

            // Ruido dinámico estocástico (modulado por Cortisol global y Frustración local)
            double frustration_factor = 1.0 + 1.2 * n.frustration;
            double cortisol_factor = 1.0 + 1.0 * cortisol;
            double ach_factor = 0.8 + 0.4 * n.ach;
            double energy_factor = 0.2 + 0.8 * n.energy;
            double xi = normal_dist(gen);
            double noise_term = n.noise_base * frustration_factor * cortisol_factor * ach_factor * energy_factor * std::sqrt(1.0 / (n.tau_m / 1000.0)) * xi * std::sqrt(0.001);

            // Integración de Euler con periodo refractario de 2 ms (0.002 s) para evitar la epilepsia y corriente caótica transitoria
            double I_chaos = (i >= 90 && i < 100) ? (ncc_chaos_state * 1.5) : 0.0;
            if (is_clamped) {
                n.v = n.v_rest;
            } else if (current_time_sec - n.last_spike_time < 0.002) {
                n.v = n.v_rest;
            } else {
                double dv = (-(n.v - n.v_rest) + I_syn_soma + I_coupling + n.I_ext + I_cpg + I_chaos) * 0.001 / (n.tau_m / 1000.0) + noise_term;
                n.v += dv;
            }

            double dv_dend = (-(n.v_dend - n.v_rest) + I_syn_dend) * 0.001 / (n.tau_dend / 1000.0);
            n.v_dend += dv_dend;

            // Decaimiento pasivo de conductancias
            n.g_ampa_soma *= (1.0 - 1.0 / 5.0); // tau_ampa = 5ms
            n.g_ampa_dend *= (1.0 - 1.0 / 5.0);
            n.g_gaba_soma *= (1.0 - 1.0 / 10.0); // tau_gaba = 10ms
            n.g_gaba_dend *= (1.0 - 1.0 / 10.0);

            // Verificar Spikes
            if (!is_clamped && n.v > n.v_thresh) {
                current_step_spikes[i] = true;
                n.v = n.v_rest;
                n.last_spike_time = current_time_sec;
                n.v_thresh = std::max(-60.0, std::min(n.v_thresh + n.delta_v_thresh, -35.0));
                
                spike_counts[i]++;
                spikes_in_current_batch++;
                astrocytes_spike_counts[i / 10]++;
            }
        }

        // Procesar plasticidad y propagación para los spikes ocurridos en este milisegundo
        double t_ms = time_ms - BATCH_MS + ms_step;

        for (int i = 0; i < N_TOTAL; ++i) {
            if (!current_step_spikes[i]) continue;

            // Pre-spike en neurona i
            for (int k : pre_syn_list[i]) {
                auto& s = synapses[k];
                
                // Decaimiento STDP y recuperación STP por evento (Event-driven)
                double dt = t_ms - s.last_update_ms;
                if (dt > 0.0) {
                    if (dt > 100.0) {
                        s.apre = 0.0;
                        s.apost = 0.0;
                    } else {
                        double factor = std::exp(-dt / 20.0);
                        s.apre *= factor;
                        s.apost *= factor;
                    }

                    // Recuperación pasiva de recursos y facilitación de la sinapsis (Tsodyks-Markram)
                    s.x_stp = 1.0 - (1.0 - s.x_stp) * std::exp(-dt / s.tau_d);
                    s.u_stp = s.U_stp + (s.u_stp - s.U_stp) * std::exp(-dt / s.tau_f);

                    s.last_update_ms = t_ms;
                }

                // Dinámica de activación de STP por el spike actual
                s.u_stp = std::max(0.0, std::min(s.u_stp + s.U_stp * (1.0 - s.u_stp), 1.0));
                double release = s.u_stp * s.x_stp;
                s.x_stp = std::max(0.0, std::min(s.x_stp - release, 1.0));

                double weight_factor = 1.0;
                if (s.pre >= 80 && s.post >= 80) {
                    weight_factor = 1.8; // Amplificar la recurrencia local en PFC para sostener memoria (100-neuron indices)
                } else if (s.pre < 80 && s.post >= 80) {
                    weight_factor = pfc_gates[s.post]; // Gating por Ganglios Basales (100-neuron indices)
                }

                // Inyección Híbrida del NCC a la corteza (0.05 Tónico + 0.35 Fásico)
                double chaos_modulation = 1.0;
                if (s.pre >= 90 && s.post < 80) {
                    chaos_modulation = 0.05 + 0.35 * frustration;
                }

                double effective_weight = s.w * release * (1.0 + 0.6 * s.myelination) * weight_factor * chaos_modulation * 2.5 * s.is_active;

                // Agendar en buffer circular correspondiente
                int slot = (current_ring_step + s.delay_steps) % 16;
                if (s.is_excitatory > 0.5) {
                    if (s.target_is_dendrite > 0.5) {
                        ring_ampa_dend[s.post][slot] += effective_weight;
                    } else {
                        ring_ampa_soma[s.post][slot] += effective_weight;
                    }
                } else {
                    if (s.target_is_dendrite > 0.5) {
                        ring_gaba_dend[s.post][slot] += effective_weight;
                    } else {
                        ring_gaba_soma[s.post][slot] += effective_weight;
                    }
                }

                // Apre + LTP (STDP)
                s.apre += 0.015 * s.is_excitatory;
                double da_post = neurons[s.post].da;
                double frustration_post = neurons[s.post].frustration;
                double ser_post = neurons[s.post].ser;
                
                // Modulación LTD por neuromodulación (Perfectamente balanceado a da = 0.5)
                double scale_factor = (((1.0 - da_post) / 0.5) + frustration_post) / (0.5 + 1.5 * ser_post);
                if (stdp_active) {
                    s.w = std::max(0.0, std::min(s.w + s.apost * s.is_excitatory * s.is_active * scale_factor, 2.0));
                }
            }

            // Post-spike en neurona i
            for (int k : post_syn_list[i]) {
                auto& s = synapses[k];
                
                // Decaimiento STDP por evento (Event-driven)
                double dt = t_ms - s.last_update_ms;
                if (dt > 0.0) {
                    if (dt > 100.0) {
                        s.apre = 0.0;
                        s.apost = 0.0;
                    } else {
                        double factor = std::exp(-dt / 20.0);
                        s.apre *= factor;
                        s.apost *= factor;
                    }
                    s.last_update_ms = t_ms;
                }

                // Apost + LTD (STDP)
                s.apost += -0.01575 * s.is_excitatory;
                double da_post = neurons[s.post].da;
                double ach_post = neurons[s.post].ach;
                double ser_post = neurons[s.post].ser;

                // Modulación LTP por neuromodulación (Perfectamente balanceado a da = 0.5)
                double scale_factor = (da_post / 0.5) * (1.0 + 2.5 * ach_post) / (0.5 + 1.5 * ser_post);
                if (stdp_active) {
                    s.w = std::max(0.0, std::min(s.w + s.apre * s.is_excitatory * s.is_active * scale_factor, 2.0));
                }
            }
        }

        // 2. Capa Glial: Astrocitos locales para prevenir la epilepsia (cada 20 ms)
        if (ms_step % 20 == 0) {
            for (int a = 0; a < 10; ++a) {
                double spikes_norm = (double)astrocytes_spike_counts[a];
                astrocytes[a].calcium = 0.95 * astrocytes[a].calcium + 0.05 * spikes_norm;
                astrocytes_spike_counts[a] = 0; // resetear contador
            }
        }

        current_ring_step = (current_ring_step + 1) % 16;
    }

    // 3. Traducir disparo de las 15 neuronas motoras (índices 25 a 39) en movimiento
    for (int i = 0; i < N_TOTAL; ++i) {
        neurons[i].firing_rate = spike_counts[i] / (BATCH_MS * 0.001); // Convertir a Hz
    }

    // Enviar comandos motores al hardware (solo si el cerebro está despierto)
    if (brain_state == "AWAKE") {
        std::vector<double> motor_firing_rates(30);
        for (int i = 50; i < 80; ++i) {
            motor_firing_rates[i - 50] = neurons[i].firing_rate;
        }
        hardware->send_motor_commands(motor_firing_rates, time_ms);
        hardware->update_environment(BATCH_MS * 0.001);
    }

    // Calcular disparo motor promedio para la homeostasis
    double motor_firing_sum = 0.0;
    int motor_count = 0;
    for (int i = 0; i < N_TOTAL; ++i) {
        if (neurons[i].layer_id == 2) {
            motor_firing_sum += neurons[i].firing_rate;
            motor_count++;
        }
    }
    double motor_firing = (motor_count > 0) ? (motor_firing_sum / motor_count) : 0.0;
    
    // Adaptación del umbral de ganancia
    gain_control.adapt(*this, motor_firing);

    prediction = motor_firing / 15.0;
    if (brain_state == "AWAKE") {
        error = hardware->get_prediction_error(target_signal, time_ms);
    } else {
        error = 0.0;
    }

    // Actualizar frustración y resiliencia
    if (brain_state == "AWAKE") {
        if (std::abs(error) > 0.35) {
            frustration = std::min(1.0, frustration + 0.06);
        } else {
            frustration = std::max(0.0, frustration - 0.04);
            resilience = std::min(1.0, resilience + 0.01);
        }
    } else {
        frustration = std::max(0.0, frustration - 0.02);
    }

    // Actualizar sistema endocrino neuromodulador y hormonas
    neuromod.update(BATCH_MS * 0.001, brain_state, error);
    cortisol = 0.95 * cortisol + 0.05 * frustration;

    // Consumo de energía celular metabólica
    for (int i = 0; i < N_TOTAL; ++i) {
        if (spike_counts[i] > 0) {
            neurons[i].energy = std::max(0.0, neurons[i].energy - 0.05);
        } else {
            neurons[i].energy = std::min(1.0, neurons[i].energy + 0.01);
        }
    }

    // Reguladores homeostáticos en NumPy equivalente
    if (step_count % 10 == 0) {
        homeostasis();
    }
    if (step_count % 20 == 0) {
        structural_plasticity();
    }

    // Mielinización dinámica
    if (neuromod.dopamine > 0.6) {
        for (size_t k = 0; k < synapses.size(); ++k) {
            auto& s = synapses[k];
            if (s.w > 0.1 && s.is_excitatory > 0.5) {
                s.myelination = std::min(1.0, s.myelination + 0.03);
                // recalcular delay basándose en la velocidad de la mielina
                s.delay_steps = (int)std::round(s.base_delay_ms / (1.0 + 3.0 * s.myelination));
                if (s.delay_steps < 1) s.delay_steps = 1;
                if (s.delay_steps > 15) s.delay_steps = 15;
            }
        }
    }

    // Registrar historial
    double w_sum = 0.0;
    double w_max = 0.0;
    int w_active_count = 0;
    for (const auto& s : synapses) {
        if (s.is_active > 0.5) {
            w_sum += s.w;
            if (s.w > w_max) w_max = s.w;
            w_active_count++;
        }
    }
    double w_mean = (w_active_count > 0) ? (w_sum / w_active_count) : 0.0;

    double energy_sum = 0.0;
    for (const auto& n : neurons) {
        energy_sum += n.energy;
    }
    double energy_mean = energy_sum / N_TOTAL;

    HistoryEntry entry;
    entry.time = time_ms;
    entry.step = step_count;
    entry.state = brain_state;
    entry.da = neuromod.dopamine;
    entry.ser = neuromod.serotonin;
    entry.ach = neuromod.acetylcholine;
    entry.w_mean = w_mean;
    entry.w_max = w_max;
    entry.synapses = w_active_count;
    entry.spikes = spikes_in_current_batch;
    entry.energy_mean = energy_mean;
    entry.pruned = pruned_synapses;
    entry.created = created_synapses;
    entry.prediction = prediction;
    entry.target = target_signal;
    entry.workspace_fr = fr_workspace;
    entry.workspace_umbral = workspace.umbral;
    entry.ignicion = ignicion;

    history.push_back(entry);
    if (history.size() > 1000) {
        history.erase(history.begin());
    }

    // Construir JSON y empujar al Servidor Web
    update_json_data(get_state_json());
}

void BrainUnico::homeostasis() {
    // Escalado homeostático lento de pesos
    for (size_t k = 0; k < synapses.size(); ++k) {
        auto& s = synapses[k];
        
        // Forzar peso a 0.0 si la sinapsis está inactiva (pruning real)
        if (s.is_active < 0.5) {
            s.w = 0.0;
            continue;
        }

        // Evitar el escalado multiplicativo basado en la tasa de disparo para la capa motora (output de clasificación)
        if (s.post >= 50 && s.post < 80) {
            s.w *= decay_factor;
            s.w = std::max(0.0, std::min(s.w, 2.0));
            continue;
        }

        double post_fr = neurons[s.post].firing_rate;
        
        // Multiplicativo: deprimir si está muy hiperactiva, potenciar si está hipoactiva
        if (post_fr > 8.0) {
            s.w *= (s.is_excitatory > 0.5) ? 0.98 : 1.02; // Hiperactiva (excitatorias bajan, inhibitorias suben)
        } else if (post_fr < 1.0) {
            s.w *= (s.is_excitatory > 0.5) ? 1.01 : 0.99; // Hipoactiva (excitatorias suben, inhibitorias bajan)
        }
        
        // Aplicar decaimiento por desuso pasivo
        s.w *= decay_factor;
        s.w = std::max(0.02, std::min(s.w, 2.0));
    }

    // Escalado astrocítico
    scaler.scale(*this);
}

void BrainUnico::structural_plasticity() {
    // 1. Poda de sinapsis débiles
    for (size_t k = 0; k < synapses.size(); ++k) {
        auto& s = synapses[k];
        if (s.is_excitatory > 0.5 && s.is_active > 0.5 && s.w < 0.02) {
            s.w = 0.0;
            s.is_active = 0.0;
            pruned_synapses++;
        }
    }

    // 2. Sinaptogénesis basada en distancia 3D (solo en proyecciones permitidas por la arquitectura)
    for (size_t k = 0; k < synapses.size(); ++k) {
        auto& s = synapses[k];
        if (s.is_active < 0.5 && s.is_excitatory > 0.5) {
            int pre_layer = neurons[s.pre].layer_id;
            int post_layer = neurons[s.post].layer_id;
            
            bool allowed = false;
            if (pre_layer == 0 && post_layer == 1) allowed = true;      // Sensory -> Hidden
            else if (pre_layer == 1 && post_layer == 2) allowed = true; // Hidden -> Motor
            else if (pre_layer == 1 && post_layer == 1) allowed = true; // Recurrente Hidden
            else if (pre_layer == 2 && post_layer == 2) allowed = true; // Recurrente Motor
            
            if (!allowed) continue;

            double dist = dist_3d[k];
            double p_conn = 0.005 * std::exp(-dist / 30.0);
            if (rand_dist(gen) < p_conn) {
                s.is_active = 1.0;
                s.w = 0.015 + rand_dist(gen) * 0.03; // Peso inicial bajo
                s.x_stp = 1.0;
                s.u_stp = s.U_stp;
                created_synapses++;
            }
        }
    }
}

void BrainUnico::sleep_replay() {
    if (episodic_buffer.empty()) return;

    // Seleccionar un recuerdo aleatorio
    std::uniform_int_distribution<size_t> mem_dist(0, episodic_buffer.size() - 1);
    const auto& memory = episodic_buffer[mem_dist(gen)];

    // Inyectar en neuronas sensoriales con ráfaga Poisson rápida
    for (int i = 0; i < N_SENSORY; ++i) {
        bool poisson_spike = rand_dist(gen) < 0.2;
        if (poisson_spike) {
            neurons[i].I_ext = memory.sensory[i] * 45.0;
        }
    }
    neuromod.dopamine = std::min(1.0, neuromod.dopamine + 0.02);
}

bool BrainUnico::save_state(const std::string& filepath) {
    std::ofstream f(filepath, std::ios::binary);
    if (!f.is_open()) return false;
    
    // Escribir metadatos generales
    f.write((char*)&time_ms, sizeof(time_ms));
    f.write((char*)&step_count, sizeof(step_count));
    
    size_t state_len = brain_state.length();
    f.write((char*)&state_len, sizeof(state_len));
    f.write(brain_state.c_str(), state_len);
    
    f.write((char*)&pruned_synapses, sizeof(pruned_synapses));
    f.write((char*)&created_synapses, sizeof(created_synapses));
    f.write((char*)&frustration, sizeof(frustration));
    f.write((char*)&resilience, sizeof(resilience));
    
    // Escribir neuronas
    size_t n_size = neurons.size();
    f.write((char*)&n_size, sizeof(n_size));
    for (const auto& n : neurons) {
        f.write((char*)&n.v, sizeof(n.v));
        f.write((char*)&n.v_dend, sizeof(n.v_dend));
        f.write((char*)&n.v_thresh, sizeof(n.v_thresh));
        f.write((char*)&n.v_thresh_base, sizeof(n.v_thresh_base));
        f.write((char*)&n.energy, sizeof(n.energy));
        f.write((char*)&n.firing_rate, sizeof(n.firing_rate));
        f.write((char*)&n.last_spike_time, sizeof(n.last_spike_time));
        f.write((char*)&n.g_ampa_soma, sizeof(n.g_ampa_soma));
        f.write((char*)&n.g_gaba_soma, sizeof(n.g_gaba_soma));
        f.write((char*)&n.g_ampa_dend, sizeof(n.g_ampa_dend));
        f.write((char*)&n.g_gaba_dend, sizeof(n.g_gaba_dend));
    }
    
    // Escribir sinapsis
    size_t s_size = synapses.size();
    f.write((char*)&s_size, sizeof(s_size));
    for (const auto& s : synapses) {
        f.write((char*)&s.w, sizeof(s.w));
        f.write((char*)&s.myelination, sizeof(s.myelination));
        f.write((char*)&s.is_active, sizeof(s.is_active));
        f.write((char*)&s.x_stp, sizeof(s.x_stp));
        f.write((char*)&s.u_stp, sizeof(s.u_stp));
        f.write((char*)&s.apre, sizeof(s.apre));
        f.write((char*)&s.apost, sizeof(s.apost));
        f.write((char*)&s.delay_steps, sizeof(s.delay_steps));
    }
    
    // Escribir neuromoduladores
    f.write((char*)&neuromod.dopamine, sizeof(neuromod.dopamine));
    f.write((char*)&neuromod.serotonin, sizeof(neuromod.serotonin));
    f.write((char*)&neuromod.acetylcholine, sizeof(neuromod.acetylcholine));
    
    // Reguladores y workspace
    f.write((char*)&scaler.active, sizeof(scaler.active));
    f.write((char*)&gain_control.active, sizeof(gain_control.active));
    f.write((char*)&gain_control.v_offset, sizeof(gain_control.v_offset));
    f.write((char*)&workspace.umbral, sizeof(workspace.umbral));
    
    // Entorno físico (si es puente virtual)
    VirtualBridge* vb = dynamic_cast<VirtualBridge*>(hardware.get());
    if (vb) {
        f.write((char*)&vb->entorno.x, sizeof(vb->entorno.x));
        f.write((char*)&vb->entorno.y, sizeof(vb->entorno.y));
        f.write((char*)&vb->entorno.theta, sizeof(vb->entorno.theta));
        f.write((char*)&vb->entorno.x_comida, sizeof(vb->entorno.x_comida));
        f.write((char*)&vb->entorno.y_comida, sizeof(vb->entorno.y_comida));
        f.write((char*)&vb->entorno.comidas_ingeridas, sizeof(vb->entorno.comidas_ingeridas));
    } else {
        double zero = 0.0;
        int izero = 0;
        f.write((char*)&zero, sizeof(zero));
        f.write((char*)&zero, sizeof(zero));
        f.write((char*)&zero, sizeof(zero));
        f.write((char*)&zero, sizeof(zero));
        f.write((char*)&zero, sizeof(zero));
        f.write((char*)&izero, sizeof(izero));
    }

    // Escribir variables biológicas multiescala adicionales
    f.write((char*)&cortisol, sizeof(cortisol));
    f.write((char*)&melatonina, sizeof(melatonina));
    f.write((char*)v_if, sizeof(v_if));
    for (int a = 0; a < 10; ++a) {
        f.write((char*)&astrocytes[a].calcium, sizeof(astrocytes[a].calcium));
    }
    
    std::cout << "[SAVE] Estado binario completo guardado en: " << filepath << "\n";
    return true;
}

bool BrainUnico::load_state(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) return false;
    
    f.read((char*)&time_ms, sizeof(time_ms));
    f.read((char*)&step_count, sizeof(step_count));
    
    size_t state_len = 0;
    f.read((char*)&state_len, sizeof(state_len));
    std::vector<char> state_chars(state_len + 1);
    f.read(state_chars.data(), state_len);
    state_chars[state_len] = '\0';
    brain_state = std::string(state_chars.data());
    
    f.read((char*)&pruned_synapses, sizeof(pruned_synapses));
    f.read((char*)&created_synapses, sizeof(created_synapses));
    f.read((char*)&frustration, sizeof(frustration));
    f.read((char*)&resilience, sizeof(resilience));
    
    size_t n_size = 0;
    f.read((char*)&n_size, sizeof(n_size));
    if (n_size != N_TOTAL) {
        return false;
    }
    neurons.resize(n_size);
    for (auto& n : neurons) {
        f.read((char*)&n.v, sizeof(n.v));
        f.read((char*)&n.v_dend, sizeof(n.v_dend));
        f.read((char*)&n.v_thresh, sizeof(n.v_thresh));
        f.read((char*)&n.v_thresh_base, sizeof(n.v_thresh_base));
        f.read((char*)&n.energy, sizeof(n.energy));
        f.read((char*)&n.firing_rate, sizeof(n.firing_rate));
        f.read((char*)&n.last_spike_time, sizeof(n.last_spike_time));
        f.read((char*)&n.g_ampa_soma, sizeof(n.g_ampa_soma));
        f.read((char*)&n.g_gaba_soma, sizeof(n.g_gaba_soma));
        f.read((char*)&n.g_ampa_dend, sizeof(n.g_ampa_dend));
        f.read((char*)&n.g_gaba_dend, sizeof(n.g_gaba_dend));
    }
    
    size_t s_size = 0;
    f.read((char*)&s_size, sizeof(s_size));
    synapses.resize(s_size);
    for (auto& s : synapses) {
        f.read((char*)&s.w, sizeof(s.w));
        f.read((char*)&s.myelination, sizeof(s.myelination));
        f.read((char*)&s.is_active, sizeof(s.is_active));
        f.read((char*)&s.x_stp, sizeof(s.x_stp));
        f.read((char*)&s.u_stp, sizeof(s.u_stp));
        f.read((char*)&s.apre, sizeof(s.apre));
        f.read((char*)&s.apost, sizeof(s.apost));
        f.read((char*)&s.delay_steps, sizeof(s.delay_steps));
        s.last_update_ms = time_ms;
    }
    
    f.read((char*)&neuromod.dopamine, sizeof(neuromod.dopamine));
    f.read((char*)&neuromod.serotonin, sizeof(neuromod.serotonin));
    f.read((char*)&neuromod.acetylcholine, sizeof(neuromod.acetylcholine));
    
    f.read((char*)&scaler.active, sizeof(scaler.active));
    f.read((char*)&gain_control.active, sizeof(gain_control.active));
    f.read((char*)&gain_control.v_offset, sizeof(gain_control.v_offset));
    f.read((char*)&workspace.umbral, sizeof(workspace.umbral));
    
    // Entorno físico (si es puente virtual)
    VirtualBridge* vb = dynamic_cast<VirtualBridge*>(hardware.get());
    if (vb) {
        f.read((char*)&vb->entorno.x, sizeof(vb->entorno.x));
        f.read((char*)&vb->entorno.y, sizeof(vb->entorno.y));
        f.read((char*)&vb->entorno.theta, sizeof(vb->entorno.theta));
        f.read((char*)&vb->entorno.x_comida, sizeof(vb->entorno.x_comida));
        f.read((char*)&vb->entorno.y_comida, sizeof(vb->entorno.y_comida));
        f.read((char*)&vb->entorno.comidas_ingeridas, sizeof(vb->entorno.comidas_ingeridas));
    } else {
        double dummy_d = 0.0;
        int dummy_i = 0;
        f.read((char*)&dummy_d, sizeof(dummy_d));
        f.read((char*)&dummy_d, sizeof(dummy_d));
        f.read((char*)&dummy_d, sizeof(dummy_d));
        f.read((char*)&dummy_d, sizeof(dummy_d));
        f.read((char*)&dummy_d, sizeof(dummy_d));
        f.read((char*)&dummy_i, sizeof(dummy_i));
    }

    // Leer variables biológicas multiescala adicionales
    f.read((char*)&cortisol, sizeof(cortisol));
    f.read((char*)&melatonina, sizeof(melatonina));
    f.read((char*)v_if, sizeof(v_if));
    for (int a = 0; a < 10; ++a) {
        f.read((char*)&astrocytes[a].calcium, sizeof(astrocytes[a].calcium));
    }
    
    std::cout << "[LOAD] Estado binario completo cargado desde: " << filepath << "\n";
    return true;
}

std::string BrainUnico::get_state_json() {
    std::ostringstream oss;
    oss << "{";
    oss << "\"time\":" << time_ms << ",";
    oss << "\"step\":" << step_count << ",";
    oss << "\"state\":\"" << brain_state << "\",";
    oss << "\"da\":" << neuromod.dopamine << ",";
    oss << "\"ser\":" << neuromod.serotonin << ",";
    oss << "\"ach\":" << neuromod.acetylcholine << ",";
    
    // Calcular w_mean
    double w_sum = 0.0;
    double w_max = 0.0;
    int w_active_count = 0;
    for (const auto& s : synapses) {
        if (s.is_active > 0.5 && s.w > 0.0) {
            w_sum += s.w;
            if (s.w > w_max) w_max = s.w;
            w_active_count++;
        }
    }
    double w_mean = (w_active_count > 0) ? (w_sum / w_active_count) : 0.0;
    oss << "\"w_mean\":" << w_mean << ",";
    oss << "\"synapses_active\":" << w_active_count << ",";
    oss << "\"synapses_pruned\":" << pruned_synapses << ",";
    oss << "\"synapses_created\":" << created_synapses << ",";
    
    // Calcular energy_mean
    double energy_sum = 0.0;
    for (const auto& n : neurons) {
        energy_sum += n.energy;
    }
    double energy_mean = energy_sum / N_TOTAL;
    oss << "\"energy_mean\":" << energy_mean << ",";
    
    double last_pred = 0.0, last_target = 0.0, last_workspace_fr = 0.0;
    if (!history.empty()) {
        last_pred = history.back().prediction;
        last_target = history.back().target;
        last_workspace_fr = history.back().workspace_fr;
    }
    oss << "\"prediction\":" << last_pred << ",";
    oss << "\"target\":" << last_target << ",";
    oss << "\"workspace_fr\":" << last_workspace_fr << ",";
    oss << "\"frustration\":" << frustration << ",";
    oss << "\"cortisol\":" << cortisol << ",";
    oss << "\"melatonina\":" << melatonina << ",";
    
    // Astrocytes calcium list
    oss << "\"astrocytes_calcium\":[";
    for (size_t a = 0; a < astrocytes.size(); ++a) {
        oss << astrocytes[a].calcium;
        if (a < astrocytes.size() - 1) oss << ",";
    }
    oss << "],";

    oss << "\"workspace_umbral\":" << workspace.umbral << ",";
    oss << "\"ignicion\":" << (workspace.ignicion_activa ? "true" : "false") << ",";
    
    oss << hardware->get_telemetry_json() << ",";
    
    // Lista de neuronas
    oss << "\"neurons\":[";
    for (int i = 0; i < N_TOTAL; i++) {
        const auto& n = neurons[i];
        oss << "{";
        oss << "\"layer\":" << n.layer_id << ",";
        oss << "\"x\":" << n.x << ",";
        oss << "\"y\":" << n.y << ",";
        oss << "\"z\":" << n.z << ",";
        oss << "\"firing\":" << n.firing_rate << ",";
        oss << "\"energy\":" << n.energy;
        oss << "}";
        if (i < N_TOTAL - 1) oss << ",";
    }
    oss << "],";
    
    // Lista de sinapsis activas (w > 0.01) para agilizar Three.js
    oss << "\"synapses\":[";
    bool first_syn = true;
    for (const auto& s : synapses) {
        if (s.is_active > 0.5 && s.w > 0.01) {
            if (!first_syn) oss << ",";
            oss << "{";
            oss << "\"pre\":" << s.pre << ",";
            oss << "\"post\":" << s.post << ",";
            oss << "\"w\":" << s.w << ",";
            oss << "\"exc\":" << (s.is_excitatory > 0.5 ? "true" : "false");
            oss << "}";
            first_syn = false;
        }
    }
    oss << "]";
    
    oss << "}";
    return oss.str();
}
