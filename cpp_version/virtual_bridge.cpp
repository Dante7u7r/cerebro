#include "virtual_bridge.hpp"
#include "cerebro.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <sstream>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static std::random_device rd;
static std::mt19937 gen(rd());

VirtualBridge::VirtualBridge(double limite_val)
    : entorno(limite_val), recompensa_dopamina(0.0), ultima_distancia(100.0), ultima_velocidad_motor(0.0),
      modo_ciego(false), blind_timer(0.0),
      csi_true_state(1), csi_predicted_state(0), csi_trial_timer(0.0), csi_trial_count(0), csi_correct_count(0),
      current_block_count(0), current_block_size(3) {
          motor_spikes_accum[0] = 0;
          motor_spikes_accum[1] = 0;
          motor_spikes_accum[2] = 0;
      }

// ---------------------------------------------------------------------------
// Selecciona la siguiente clase usando bloques semi-aleatorios.
// La misma clase se repite entre 2 y 5 veces (elegido al azar) antes de
// cambiar. Al cambiar, se elige aleatoriamente una de las otras dos clases.
// Esto da al STDP tiempo para consolidar sin ser completamente predecible.
// ---------------------------------------------------------------------------
void VirtualBridge::next_csi_state() {
    current_block_count++;
    if (current_block_count >= current_block_size) {
        // Agotar el bloque: cambiar a una clase distinta
        std::uniform_int_distribution<int> other_dist(1, 2);
        int offset = other_dist(gen); // 1 o 2 pasos adelante (nunca 0 = misma)
        csi_true_state = (csi_true_state + offset) % 3;

        // Nuevo tamaño de bloque: entre 2 y 5 ensayos
        std::uniform_int_distribution<int> size_dist(2, 5);
        current_block_size = size_dist(gen);
        current_block_count = 0;
    }
    // Si current_block_count < current_block_size: mantener la misma clase
}

void VirtualBridge::get_sensory_input(const std::vector<Neuron>& neurons,
                                       std::vector<double>& input_currents,
                                       double time_ms, double mean_energy) {
    double t_sec = time_ms / 1000.0;
    std::uniform_real_distribution<double> noise_dist(0.0, 2.0);

    for (int i = 0; i < 20; ++i) input_currents[i] = 0.0;

    // Señales Wi-Fi CSI omnidireccionales — siempre presentes.
    if (csi_true_state == 0) {
        // VACIO: sin señal (los canales permanecen en 0.0 nA)
    }
    else if (csi_true_state == 1) {
        // SUJETO A: gait lento 1.4 Hz en subportadoras 0-4
        for (int i = 0; i < 5; ++i) {
            double ph = i * 0.25;
            input_currents[i] = 28.0 + 18.0 * std::sin(2.0 * M_PI * 1.4 * t_sec + ph)
                                     +  6.0 * std::sin(2.0 * M_PI * 2.8 * t_sec + ph);
        }
    }
    else if (csi_true_state == 2) {
        // SUJETO B: gait rapido 2.5 Hz en subportadoras 5-9
        for (int i = 5; i < 10; ++i) {
            double ph = (i - 5) * 0.35;
            input_currents[i] = 28.0 + 18.0 * std::cos(2.0 * M_PI * 2.5 * t_sec + ph)
                                     +  6.0 * std::cos(2.0 * M_PI * 5.0 * t_sec + ph);
        }
    }

    // Propriocepcion y energia (canales 10-19)
    for (int i = 10; i < 20; ++i)
        input_currents[i] = 10.0 + noise_dist(gen);
}

void VirtualBridge::send_motor_commands(const std::vector<double>& motor_firing_rates,
                                         double time_ms) {
    if (motor_firing_rates.size() < 30) return;

    double fr0 = 0.0, fr1 = 0.0, fr2 = 0.0;
    for (int i = 0;  i < 10; ++i) fr0 += motor_firing_rates[i];
    for (int i = 10; i < 20; ++i) fr1 += motor_firing_rates[i];
    for (int i = 20; i < 30; ++i) fr2 += motor_firing_rates[i];
    fr0 /= 10.0; fr1 /= 10.0; fr2 /= 10.0;

    // Evaluacion autonoma: ultimos 5s del ensayo (t >= 10.0 de 15.0s totales)
    if (csi_trial_timer >= 10.0) {
        motor_spikes_accum[0] += fr0 * 5.0;
        motor_spikes_accum[1] += fr1 * 5.0;
        motor_spikes_accum[2] += fr2 * 5.0;
    }

    entorno.mover_agente(fr0, 4.0, fr1);
    ultima_velocidad_motor = 4.0;
}

void VirtualBridge::update_environment(double dt_sec) {
    // Sin modo ciego — senales Wi-Fi siempre presentes.
    blind_timer += dt_sec;
    modo_ciego = false;

    // Ensayos de 15 segundos (antes 10s).
    // Los primeros 10s el profesor actua; los ultimos 5s son evaluacion autonoma.
    csi_trial_timer += dt_sec;
    if (csi_trial_timer >= 15.0) {

        // --- Decodificacion por mayoria de spikes ---
        double s0 = motor_spikes_accum[0];
        double s1 = motor_spikes_accum[1];
        double s2 = motor_spikes_accum[2];
        double total = s0 + s1 + s2;

        double max_spikes = -1.0;
        int prediction = 0;
        double second_max = -1.0;
        for (int i = 0; i < 3; ++i) {
            double sv = (i==0)?s0:(i==1)?s1:s2;
            if (sv > max_spikes) { second_max = max_spikes; max_spikes = sv; prediction = i; }
            else if (sv > second_max) { second_max = sv; }
        }
        csi_predicted_state = prediction;

        // --- Recompensa proporcional al margen (Opcion 2) ---
        // El entorno escala la senal dopaminergica segun la confianza de la prediccion.
        // margin ∈ [0, 1]: que tan separado esta el ganador del segundo lugar.
        double margin = (total > 0.0) ? ((max_spikes - second_max) / total) : 0.0;
        margin = std::min(1.0, std::max(0.0, margin));

        if (csi_predicted_state == csi_true_state) {
            // Correcto: recompensa proporcional a la confianza
            // margen alto → recompensa fuerte; margen bajo (duda) → recompensa debil
            recompensa_dopamina = 0.3 + 0.7 * margin;  // rango [0.3, 1.0]
        } else {
            // Incorrecto: castigo proporcional a la arrogancia del error
            // margen alto en equivocado → castigo fuerte; casi empate → castigo suave
            recompensa_dopamina = -(0.2 + 0.6 * margin); // rango [-0.2, -0.8]
        }

        // --- Log ---
        csi_trial_count++;
        std::string name_true = (csi_true_state==0)?"VACIO":((csi_true_state==1)?"SUJETO A":"SUJETO B");
        std::string name_pred = (csi_predicted_state==0)?"VACIO":((csi_predicted_state==1)?"SUJETO A":"SUJETO B");
        int rs0=(int)std::round(s0), rs1=(int)std::round(s1), rs2=(int)std::round(s2);
        double da_log = recompensa_dopamina;

        if (csi_predicted_state == csi_true_state) {
            csi_correct_count++;
            std::cout << "[CSI ENSAYO " << csi_trial_count << "] CORRECTO | Verdadero: " << name_true
                      << " | Predicho: " << name_pred
                      << " | Spikes Vac:" << rs0 << " A:" << rs1 << " B:" << rs2
                      << " | Margen:" << (int)(margin*100) << "%"
                      << " | DA:" << (int)(da_log*100)/100.0
                      << " | Aciertos: " << csi_correct_count << "/" << csi_trial_count
                      << " (" << (int)(100.0*csi_correct_count/csi_trial_count) << "%)\n";
        } else {
            std::cout << "[CSI ENSAYO " << csi_trial_count << "] INCORRECTO | Verdadero: " << name_true
                      << " | Predicho: " << name_pred
                      << " | Spikes Vac:" << rs0 << " A:" << rs1 << " B:" << rs2
                      << " | Margen:" << (int)(margin*100) << "%"
                      << " | DA:" << (int)(da_log*100)/100.0
                      << " | Aciertos: " << csi_correct_count << "/" << csi_trial_count
                      << " (" << (int)(100.0*csi_correct_count/csi_trial_count) << "%)\n";
        }

        // --- Reset y siguiente clase ---
        motor_spikes_accum[0] = 0.0;
        motor_spikes_accum[1] = 0.0;
        motor_spikes_accum[2] = 0.0;
        csi_trial_timer = 0.0;

        next_csi_state(); // bloques semi-aleatorios
    }
}

double VirtualBridge::get_dopamine_reward() {
    double r = recompensa_dopamina;
    recompensa_dopamina = 0.0;
    return r;
}

double VirtualBridge::get_prediction_error(double target_signal, double time_ms) {
    return target_signal - (ultima_velocidad_motor / 15.0);
}

std::string VirtualBridge::get_telemetry_json() const {
    std::ostringstream oss;
    oss << "\"agent_x\":"    << entorno.x     << ",";
    oss << "\"agent_y\":"    << entorno.y     << ",";
    oss << "\"agent_theta\":" << entorno.theta << ",";

    double tx = 0.0, ty = 0.0;
    if (csi_true_state == 1) tx = -20.0;
    else if (csi_true_state == 2) tx = 20.0;

    oss << "\"food_x\":"     << tx               << ",";
    oss << "\"food_y\":"     << ty               << ",";
    oss << "\"meals_eaten\":" << csi_correct_count << ",";
    oss << "\"modo_ciego\":"  << (modo_ciego ? "true" : "false");
    return oss.str();
}

bool   VirtualBridge::is_blind()            const { return modo_ciego; }
int    VirtualBridge::get_csi_true_state()  const { return csi_true_state; }
double VirtualBridge::get_csi_trial_timer() const { return csi_trial_timer; }
