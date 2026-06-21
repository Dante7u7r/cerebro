# ARGON PROJECT CONTEXT: cerebro-main
Generated: 2026-06-14 23:30:55.196811
Files: 22 | Connections: 3
Parser: tree-sitter

---

### cerebro_unico.py [PY | 887L | imp:0.76]
> Forzar backend de numpy en Windows para evadir bloqueos de compilador/AppLocker
- SYMBOLS: class:BrainHTTPServer, func:__init__, class:CustomHandler, func:end_headers, func:log_message, func:start, func:print, func:stop, class:NeuromodulatorSystem, func:update, class:BrainUnico, func:_get_brain_state, func:_homeostasis, func:_structural_plasticity, func:_sleep_replay, func:step, func:write_state_json, func:open, func:save_state, func:load_state
- DEPENDS: brian2, socketserver

### asistente_cerebro.py [PY | 1246L | imp:0.59]
> CONFIGURACION GLOBAL
- SYMBOLS: func:Hebbiana, class:import, class:CellType, class:NeuronSubtype, class:GliaSubtype, class:CellState, class:BrainState, func:is_inhibitory, class:GeneRegulatoryNetwork, func:__init__, func:hill_rep, func:hill_act, func:update, func:setattr, func:diff_score, func:glial_score, func:inhibitory_bias, class:DeltaNotch, class:WaddingtonLandscape, func:potential

### semilla_cerebro.py [PY | 1965L | imp:0.40]
- SYMBOLS: class:import, func:_cdist_np, class:CellType, class:NeuronSubtype, class:GliaSubtype, class:CellState, class:BrainState, func:is_inhibitory, class:GeneRegulatoryNetwork, func:__init__, func:hill_rep, func:hill_act, func:update, func:setattr, func:diff_score, func:glial_score, func:inhibitory_bias, class:DeltaNotch, class:WaddingtonLandscape, func:potential

### semilla_cerebro_original.py [PY | 1965L | imp:0.40]
- SYMBOLS: class:import, func:_cdist_np, class:CellType, class:NeuronSubtype, class:GliaSubtype, class:CellState, class:BrainState, func:is_inhibitory, class:GeneRegulatoryNetwork, func:__init__, func:hill_rep, func:hill_act, func:update, func:setattr, func:diff_score, func:glial_score, func:inhibitory_bias, class:DeltaNotch, class:WaddingtonLandscape, func:potential

### experimentos.py [PY | 340L | imp:0.36]
- SYMBOLS: func:separador, func:print, func:registrar_estado, func:experimento_1, func:experimento_2, func:experimento_3
- DEPENDS: asistente_cerebro

### run_timed_simulation.py [PY | 68L | imp:0.31]
> Asegurar que podemos importar cerebro_unico // 2 horas = 7200 segundos de tiempo real // Cargar cerebro con persistencia
- SYMBOLS: func:print, func:open
- DEPENDS: cerebro_unico

### run_long_simulation.py [PY | 25L | imp:0.31]
> Save long history
- SYMBOLS: func:print, func:open
- DEPENDS: cerebro_unico

### cerebro_brian2.py [PY | 856L | imp:0.18]
> CONFIGURACION GLOBAL // Mapeo de codegen target compatible con múltiples plataformas. // Si Cython y un compilador C++ no están listos, Brian2 cae automáticamente a numpy.
- SYMBOLS: class:NeuromodulatorSystem, func:__init__, func:update, func:stdp_modulation, class:BrainBrian2, func:_get_brain_state, func:step, func:save_brain, func:open, func:print, func:load_brain, class:BrainObserver, func:snapshot, func:print_snapshot, class:StimulusController, func:inyectar_novedad, func:inyectar_error, func:inyectar_recompensa, func:inyectar_pulso, func:inyectar_senal_temporal
- DEPENDS: brian2

### web_visualizer/app.js [JS | 330L | imp:0.06]
> Configuración del Frontend del Cerebro Único // Instancias de gráficos Chart.js // Paleta de colores de capas
- SYMBOLS: func:init3D, func:animate, func:requestAnimationFrame, func:initCharts, func:updateNetwork3D, func:updateDashboard, func:timeLabel, func:fetchData, func:setInterval

### INFORME_EXPERIMENTOS.md [MD | 334L | imp:0.05]
> IA Viva — Informe de Experimentos Brian2 // # 1. Setup Experimental // ## Arquitectura de la red
- SYMBOLS: class:directamente

### web_visualizer/index.css [CSS | 293L | imp:0.04]
> bg-main: #03030c; // bg-card: rgba(8, 8, 22, 0.85); // border-color: rgba(255, 255, 255, 0.06);

### analyze_long_sim.py [PY | 92L | imp:0.02]
> 1. Metricas Generales // 2. Análisis de Aprendizaje por Bloques (MSE de predicción durante AWAKE)
- SYMBOLS: func:print, func:open

### HISTORIA_EVOLUTIVA.md [MD | 89L | imp:0.01]
> Crónica Evolutiva: El Génesis de IA Viva // # La Fase Cartesiana (IA Viva 1.0 - 5.0) // # El Despertar del Sentir (IA Viva 6.0)
- SYMBOLS: func:distintas

### AGENTS.md [MD | 80L | imp:0.01]
> AGENTS.md - IA Viva // # Rama Original (semilla_cerebro.py) // # Rama Brian2 (cerebro_brian2.py) — Activa
- SYMBOLS: func:autónomos

### web_visualizer/index.html [HTML | 88L | imp:0.01]

### run_experiments.py [PY | 70L | imp:0.01]
> Runs all 19 experiments sequentially in separate processes to prevent Windows DLL lock issues.
- SYMBOLS: func:main, func:print

### MANUAL_ARQUITECTURA.md [MD | 42L | imp:0.01]
> Arquitectura Técnica y Biológica de IA Viva // # 1. El Entorno Físico (`VirtualBody`) // El espacio es una matriz de coordenadas 2D.
- SYMBOLS: func:completo, func:cerebro

### analyze_brain.py [PY | 23L | imp:0.01]
- SYMBOLS: class:import, class:CellType, func:open

### analyze_sim_results.py [PY | 25L | imp:0.01]
- SYMBOLS: func:print, func:open

### PROYECTO_IA_VIVA.md [MD | 27L | imp:0.00]
> IA VIVA: Ecosistemas de Inteligencia Emergente y Evolución Orgánica // # 1. Visión General // *IA Viva** es una plataforma de simulación neurobiológica avanzada donde agentes autónomos evolucionan arquitecturas cerebrales, comportamientos sociales y sistemas metabólicos sin intervención humana directa. A diferencia de las IAs tradicionales basadas en entrenamiento estático, IA Viva utiliza **Morfogénesis Evolutiva** para permitir que la inteligencia emerja de la necesidad de supervivencia.

### logs/cerebro_unico_state_final.json [JSON | 1L | imp:0.00]

### web_visualizer/sim_state.json [JSON | 1L | imp:0.00]
