#ifndef HARDWARE_BRIDGE_HPP
#define HARDWARE_BRIDGE_HPP

#include <vector>
#include <string>

// Adelantar declaración
struct Neuron;

class IHardwareBridge {
public:
    virtual ~IHardwareBridge() = default;

    // Obtiene las corrientes de entrada sensoriales (10 entradas) basadas en el entorno
    virtual void get_sensory_input(const std::vector<Neuron>& neurons, std::vector<double>& input_currents, double time_ms, double mean_energy) = 0;

    // Envía comandos motores basados en las tasas de disparo de las neuronas motoras (15 entradas)
    virtual void send_motor_commands(const std::vector<double>& motor_firing_rates, double time_ms) = 0;

    // Actualiza el entorno físico (movimiento, reubicación de comida, etc.)
    virtual void update_environment(double dt_sec) = 0;

    // Obtiene la recompensa de dopamina si ocurrió un evento (ej. comer comida o alcanzar objetivo)
    virtual double get_dopamine_reward() = 0;

    // Obtiene el error de predicción motora comparado con la señal objetivo
    virtual double get_prediction_error(double target_signal, double time_ms) = 0;

    // Genera la cadena de texto JSON con el estado específico del entorno para telemetría
    virtual std::string get_telemetry_json() const = 0;

    // Retorna si el hardware se encuentra en modo ciego (ceguera sensorial)
    virtual bool is_blind() const = 0;

    // Retorna el estado real CSI de clasificación actual (0 = vacío, 1 = Sujeto A, 2 = Sujeto B)
    virtual int get_csi_true_state() const { return -1; }

    // Retorna el temporizador del ensayo actual en segundos
    virtual double get_csi_trial_timer() const { return 0.0; }
};

#endif // HARDWARE_BRIDGE_HPP
