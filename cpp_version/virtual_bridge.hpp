#ifndef VIRTUAL_BRIDGE_HPP
#define VIRTUAL_BRIDGE_HPP

#include "hardware_bridge.hpp"
#include "entorno.hpp"

class VirtualBridge : public IHardwareBridge {
public:
    Entorno2D entorno;
    double recompensa_dopamina;
    double ultima_distancia;
    double ultima_velocidad_motor;

    VirtualBridge(double limite_val = 40.0);
    
    bool modo_ciego;
    double blind_timer;

    void get_sensory_input(const std::vector<Neuron>& neurons, std::vector<double>& input_currents, double time_ms, double mean_energy) override;
    void send_motor_commands(const std::vector<double>& motor_firing_rates, double time_ms) override;
    void update_environment(double dt_sec) override;
    double get_dopamine_reward() override;
    double get_prediction_error(double target_signal, double time_ms) override;
    std::string get_telemetry_json() const override;
    bool is_blind() const override;
    int get_csi_true_state() const override;
    double get_csi_trial_timer() const override;

    // Variables de clasificación Wi-Fi CSI
    int csi_true_state;         // 0 = vacío, 1 = Sujeto A, 2 = Sujeto B
    int csi_predicted_state;
    double csi_trial_timer;     // s
    int csi_trial_count;
    int csi_correct_count;
    double motor_spikes_accum[3];  // Acumuladores de spikes para cada clase

    // Control de bloques semi-aleatorios (Opción 1)
    int current_block_count;    // cuantos ensayos llevamos en el bloque actual
    int current_block_size;     // cuantos ensayos tiene este bloque (2-5, aleatorio)
    void next_csi_state();      // avanza al siguiente estado con logica de bloques
};

#endif // VIRTUAL_BRIDGE_HPP
