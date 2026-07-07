#include "entorno.hpp"
#include <cmath>
#include <random>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Generador de números aleatorios estático para el entorno
static std::random_device rd;
static std::mt19937 gen(rd());

Entorno2D::Entorno2D(double limite_val)
    : limite(limite_val), x(0.0), y(0.0), theta(0.0), comidas_ingeridas(0) {
    reubicar_comida();
}

void Entorno2D::reubicar_comida() {
    std::uniform_real_distribution<double> dis(-limite * 0.7, limite * 0.7);
    x_comida = dis(gen);
    y_comida = dis(gen);
}

void Entorno2D::mover_agente(double fr_izq, double fr_fwd, double fr_der) {
    // 1. Calcular giros (sensibilidad de giro: 0.04 radianes por Hz)
    double delta_theta = (fr_izq - fr_der) * 0.04;
    theta = fmod(theta + delta_theta, 2.0 * M_PI);
    if (theta < 0.0) {
        theta += 2.0 * M_PI;
    }

    // 2. Calcular velocidad lineal (sensibilidad de velocidad: 1.5 unidades por Hz)
    double velocidad = fr_fwd * 1.5;

    // 3. Actualizar coordenadas físicas
    x += velocidad * cos(theta);
    y += velocidad * sin(theta);

    // 4. Bordes toroidales: si sale por un lado, aparece por el otro
    double L = 2.0 * limite;
    
    x = fmod(x + limite, L);
    if (x < 0.0) x += L;
    x -= limite;

    y = fmod(y + limite, L);
    if (y < 0.0) y += L;
    y -= limite;
}

double Entorno2D::distancia_a_comida() const {
    double dx = std::abs(x - x_comida);
    double dy = std::abs(y - y_comida);
    double L = 2.0 * limite;
    dx = std::min(dx, L - dx);
    dy = std::min(dy, L - dy);
    return std::sqrt(dx * dx + dy * dy);
}

double Entorno2D::angulo_relativo_a_comida() const {
    double dy = y_comida - y;
    double dx = x_comida - x;
    double L = 2.0 * limite;

    // Camino toroidal más corto
    if (std::abs(dx) > limite) {
        dx = dx - ((dx > 0.0) ? 1.0 : -1.0) * L;
    }
    if (std::abs(dy) > limite) {
        dy = dy - ((dy > 0.0) ? 1.0 : -1.0) * L;
    }

    double angulo_absoluto = std::atan2(dy, dx);
    double diff = angulo_absoluto - theta;

    // Normalizar a [-pi, pi]
    diff = fmod(diff + M_PI, 2.0 * M_PI);
    if (diff < 0.0) {
        diff += 2.0 * M_PI;
    }
    diff -= M_PI;

    return diff;
}
