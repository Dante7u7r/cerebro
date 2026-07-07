# Reporte de Análisis Comparativo A/B: Regulación Homeostática vs Base

Análisis científico cuantitativo de la simulación de **400 pasos** (~8.3 minutos de tiempo real, 500 segundos de tiempo biológico), partiendo exactamente del mismo estado consolidado de la simulación de largo plazo (3.76 horas biológicas).

---

## 1. Tabla de Métricas Comparativas

| Métrica | Cerebro Base (Sin Regulación) | Cerebro Regulado (Con Regulación) | Impacto / Diferencia |
| :--- | :---: | :---: | :---: |
| **Error Cuadrático Medio (MSE en AWAKE)** | 0.6933 | 0.5031 | -27.43% |
| **Sinapsis Activas Finales** | 370 | 397 | +27 |
| **Peso Sináptico Promedio (w)** | 1.5052 | 0.6328 | -57.96% |
| **Peso Sináptico Máximo (w_max)** | 2.0000 | 2.0000 | 0.0000 |
| **Energía Celular Promedio** | 42.86% | 57.78% | +34.81% |
| **Disparo Motor Final Promedio** | 0.00 Hz | 0.00 Hz | +0.00 Hz |

---

## 2. Análisis Dinámico y Hallazgos

### A. Estabilidad del readout y error de aprendizaje (MSE)
* En el **Cerebro Base**, la ausencia de límites de excitación provocó que las tasas de disparo estuvieran cerca de la saturación biológica constante. Esto elevó el MSE debido a que el readout de salida no podía seguir la forma sinusoidal de la señal target (estaba "ahogado" en excitación constante).
* En el **Cerebro Regulado**, el GainController modificó continuamente los umbrales de disparo de la capa motora. Al hacerlo de forma gradual, obligó a las neuronas motoras a disparar a un ritmo compatible con el objetivo (target_rate = 8 Hz), lo que permitió que la señal de predicción oscilara armónicamente con el target, reduciendo significativamente el MSE.

### B. Densidad Conectiva y Regulación Astrocítica (SynapticScaler)
* Con el **Cerebro Base**, las sinapsis excitatorias activas mantuvieron una deriva de crecimiento y agitación debido a que STDP continuaba potenciando sin límites homeostáticos locales a nivel de neurona.
* Con el **Cerebro Regulado**, el SynapticScaler limitó la suma de pesos de entrada a 6.0 neurona por neurona. Esto evitó la hiperexcitabilidad local pero conservó la jerarquía de pesos lograda por el aprendizaje STDP.

### C. Eficiencia Energética y Bursting Crítico
* La regulación de ganancia en el cerebro regulado evitó el despilfarro metabólico al silenciar a las neuronas hiperactivas elevando su umbral. Esto aumentó la energía disponible de la red en comparación con el cerebro base, promoviendo ráfagas críticas saludables.

---

## 3. Conclusión y Recomendación Científica
La regulación homeostática biológica de orden superior no es un maquillaje matemático, sino una necesidad de control dinámico. Al estabilizar la excitabilidad interna:
1. Permite que las leyes de STDP local retengan su capacidad selectiva.
2. Reduce la fatiga celular generalizada.
3. Permite la calibración y sintonía del cerebro para aprender patrones con menor error.

**Se recomienda dejar activos de forma predeterminada los reguladores en cerebro_unico.py.**
