#include "cerebro.hpp"
#include "server.hpp"
#include "virtual_bridge.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>
#include <fstream>
#include <cstdlib>

std::atomic<bool> sim_running(true);
std::atomic<bool> exit_requested(false);

void signal_handler(int signum) {
    if (sim_running) {
        std::cout << "\n[WARN] Interrupcion detectada. Deteniendo bucle de simulacion...\n";
        sim_running = false;
    } else {
        std::cout << "\n[WARN] Interrupcion detectada de nuevo. Apagando servidor...\n";
        exit_requested = true;
    }
}

int main() {
    std::signal(SIGINT, signal_handler);

    std::cout << "========================================================================\n";
    std::cout << "  INICIALIZANDO SIMULACION DE CEREBRO UNICO EN C++ NATIVO\n";
    std::cout << "  Total Neuronas: " << N_TOTAL << " (Sensorial: " << N_SENSORY 
              << ", Oculta: " << N_HIDDEN << ", Motor: " << N_MOTOR << ", PFC: " << N_PFC << ")\n";
    std::cout << "========================================================================\n";

    // Crear carpeta logs si no existe
#ifdef _WIN32
    std::system("mkdir logs 2>nul");
#else
    std::system("mkdir -p logs");
#endif

    std::string state_path = "./logs/cerebro_unico_state.bin";
    BrainUnico cerebro;
    
    // Cargar estado anterior si existe
    std::ifstream check_file(state_path, std::ios::binary);
    if (check_file.good()) {
        check_file.close();
        cerebro.load_state(state_path);
    } else {
        check_file.close();
        // Intentar carpeta padre por si corre desde subcarpeta
        std::string alt_path = "../logs/cerebro_unico_state.bin";
        std::ifstream check_file2(alt_path, std::ios::binary);
        if (check_file2.good()) {
            check_file2.close();
            state_path = alt_path;
            cerebro.load_state(state_path);
        }
    }

    // Iniciar servidor HTTP en el puerto 8000
    start_server(8000);

    // Bucle asincrono continuo de simulacion
    auto last_log_time = std::chrono::steady_clock::now();
    
    int max_steps = -1;
    char* max_steps_env = std::getenv("MAX_STEPS");
    if (max_steps_env) {
        max_steps = std::atoi(max_steps_env);
        std::cout << "[INFO] Limite de pasos detectado por entorno: " << max_steps << " pasos.\n";
    }
    
    while (sim_running) {
        cerebro.step();

        if (max_steps > 0 && cerebro.step_count >= max_steps) {
            std::cout << "\n[INFO] Simulacion alcanzo el limite de " << max_steps << " pasos. Guardando y saliendo...\n";
            sim_running = false;
        }

        // Imprimir telemetría en consola cada 500 ms de tiempo real
        auto now = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_time).count();
        if (elapsed_ms >= 500) {
            last_log_time = now;
            if (!cerebro.history.empty()) {
                const auto& h = cerebro.history.back();
                double ex = 0.0, ey = 0.0, ed = 0.0;
                int ecomidas = 0;
                VirtualBridge* vb = dynamic_cast<VirtualBridge*>(cerebro.hardware.get());
                if (vb) {
                    ex = vb->entorno.x;
                    ey = vb->entorno.y;
                    ed = vb->entorno.distancia_a_comida();
                    ecomidas = vb->entorno.comidas_ingeridas;
                }
                std::string blind_tag = "";
                if (cerebro.hardware->is_blind()) {
                    blind_tag = " [BLIND]";
                }
                std::cout << "[Paso " << h.step << "] t=" << (h.time / 1000.0) << "s | " 
                          << h.state << blind_tag << " | W=" << h.w_mean << " | DA=" << h.da 
                          << " | E=" << (int)(h.energy_mean * 100) << "% | Pos=(" 
                          << ex << "," << ey << ") d=" 
                          << ed << " | Spikes=" 
                          << h.spikes << " | Comidas=" << ecomidas << "\n";
            }
        }

        // Delay pequeño para controlar el consumo de CPU (50 ms por paso)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // Guardar estado final
    cerebro.save_state(state_path);

#ifdef NO_SERVER
    std::cout << "\n[OK] Simulacion finalizada y guardada. Saliendo...\n";
#else
    std::cout << "\n[OK] Simulacion pausada y guardada.\n";
    std::cout << "El servidor HTTP sigue activo en http://localhost:8000.\n";
    std::cout << "Presiona Ctrl+C de nuevo para apagar el servidor y salir.\n";

    while (!exit_requested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    stop_server();
    std::cout << "Cerebro apagado de forma segura.\n";
#endif
    return 0;
}
