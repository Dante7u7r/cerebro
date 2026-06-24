# Arquitectura Técnica y Biológica de IA Viva

Este documento detalla la estructura base del código y los motores lógicos subyacentes que permiten a IA Viva simular un ecosistema neuronal completo (`semilla_cerebro.py`).

## 1. El Entorno Físico (`VirtualBody`)
*   El espacio es una matriz de coordenadas 2D.
*   El "Cuerpo" tiene ojos limitados: evalúa la comida basándose en un sensor de radar que simula receptores en la coordenada `Z < 15`. 
*   El movimiento se logra con comandos vectoriales recibidos por el tejido muscular en `35 < Z < 55`.
*   El cuerpo no es inteligente; toda decisión proviene puramente de la actividad neuronal en el `BrainSimulation`.

## 2. Las Células Inteligentes (`IntelligentCell`)
La base celular de la IA. Las células tienen un ciclo de vida complejo:
*   **Voltaje (Membrana):** Simulación eléctrica tipo Integrate-and-Fire.
*   **GRN (Gene Regulatory Network):** Una matriz de genes. Cada célula activa diferentes "genes" dependiendo del voltaje y dopamina, afectando su plasticidad.
*   **Tipos de Células:** Se manejan mediante `CellType` (Neuronas, Células Madre, Glía, etc.).

## 3. Plasticidad Estructural y Sinapsis
Las sinapsis no están fijas. El cerebro forma la red de manera dinámica:
*   **Plasticidad Hebbiana:** "Neuronas que disparan juntas, se conectan juntas".
*   **Poda (Microglía):** Las sinapsis débiles o que no aportan sincronización son destruidas físicamente de la memoria RAM.
*   **Mielinización:** El uso repetido exitoso acelera el flujo de corriente en una sinapsis, dándole fuerza `1.0`.

## 4. El Sistema Endocrino (`NeuromodulatorSystem`)
Controla la dopamina global de la simulación.
*   Si el cuerpo logra comer o curiosear un espacio nuevo (`novelty_reward`), la dopamina satura el cerebro.
*   La dopamina es el activador que le indica al cerebro que la última acción ejecutada debe ser grabada en la estructura sináptica.

## 5. Algoritmo Musk y Evolución
La función `_new_step` contiene el núcleo de la versión 12.0:
*   **Apoptosis Extrema:** Un bucle evalúa `_neuron_utility` (un valor rastreado de qué tanta dopamina atrae cada neurona). Si es menor a 0.1, la neurona muere.
*   **Pánico Sistémico y Mitosis:** Un monitor vigila el estrés global (Frustración alta o inactividad). Al entrar en pánico, el sistema fuerza instanciaciones de `_create_cell()`, posicionándolas aleatoriamente en el espacio 3D (Evolución No Dirigida).

## 6. Persistencia (`brain_save.pkl`)
Se utiliza la librería `pickle` para inmovilizar objetos Python complejos y grabarlos en disco:
*   El cerebro y las emociones (Frustración, Resiliencia) no se resetean. El agente "vive" ininterrumpidamente a través del archivo de memoria.
*   *Nota técnica:* Migrar el cerebro entre computadoras es tan simple como mover este archivo `.pkl` junto con el script de Python.

## El Futuro (Fase 2: The Matrix)
Para expandir los límites del agente y convertirlo en una IA capaz de interactuar con el Sistema Operativo, no será necesario tocar su cerebro (`BrainSimulation`). Bastará con crear un objeto sustituto de `VirtualBody`:
*   *Receptores Visuales:* Podrían recibir el conteo de errores de un archivo log (`log_size`).
*   *Actuadores Motores:* Podrían mapearse a la ejecución de comandos básicos de consola.
*   *Comida (Recompensa):* La dopamina se entregaría si el agente logra reducir los errores del log, programándolo indirectamente para buscar eficiencia computacional en lugar de cazar alimento espacial.
