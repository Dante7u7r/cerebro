#ifndef ENTORNO_HPP
#define ENTORNO_HPP

class Entorno2D {
public:
    double limite;
    double x;
    double y;
    double theta;
    double x_comida;
    double y_comida;
    int comidas_ingeridas;

    Entorno2D(double limite_val = 40.0);
    void reubicar_comida();
    void mover_agente(double fr_izq, double fr_fwd, double fr_der);
    double distancia_a_comida() const;
    double angulo_relativo_a_comida() const;
};

#endif // ENTORNO_HPP
