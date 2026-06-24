# BITACORA — IA Viva Arena Darwiniana
Registro cronológico de cambios, arreglos y descubrimientos.

---

## FASE 0 — Cerebros base (pre-Fase 1)

Cuatro cerebros entrenados y guardados como `.pkl`:
| Archivo | Neuronas | Notas |
|---|---|---|
| `arena_save_alfa.pkl` | 235 | PFC de 97 neuronas, 20K memorias episodicas |
| `arena_save_beta.pkl` | 126 | Equilibrado |
| `arena_save_gamma.pkl` | 7 | Minimalista |
| `arena_save_delta.pkl` | 6 | El mas simple |

Todos usan: neuronas LIF, STDP, STP, homeostasis, astrocitos, microglía.
14 canales sensoriales de entrada, 11 canales motores de salida.

---

## FASE 1 — Cuerpos evolvables (morfologia emergente)

### Archivos creados

#### `morfoma.py`
Genoma morfologico. Define el cuerpo como grafo de segmentos + articulaciones + mapeo cerebro-cuerpo.
- `SegmentGene`: pieza fisica (forma, tamano, masa, friccion)
- `JointGene`: articulacion entre segmentos (con motor opcional)
- `MotorMap`: conecta canal de salida del cerebro a una articulacion
- `SensorMap`: conecta senal fisica a canal sensorial del cerebro
- Operaciones de mutacion con pesos: add_segment(0.18), add_motor_map(0.20), nada(0.50)
- `crossover()` estatico para reproduccion sexual (C2)
- B3: `rest_angle` mutable para resortes naturales en articulaciones

#### `cuerpo_evolvable.py`
Cuerpo pymunk construido desde un Morphome.
- `_build_from_morphome()`: construye cuerpos/shapes/joints/motores desde genes
- B3: usa `DampedRotarySpring` cuando `rest_angle != 0`
- B1: `_read_physics_sensor()` implementa sensor de contacto via `space.shape_query()`
- Empuje base: `forward = (left + right) * 250.0`, `turn = (right - left) * 0.25`
- A3: `_consume_nearby_food()` devuelve tipo de comida para recompensas diferenciadas
- B2: `fitness()` penaliza masa excesiva por encima de 1.5 baseline
- C1: contador `_predator_hits` penalizado en fitness

#### `mundo_fisico.py`
- A1: 65% de comida en clusters (4 centros, dispersion gaussiana), 35% aleatoria
- A2: obstaculos estaticos como segmentos pymunk
- A3: `FoodItem` con `.kind`: FOOD_NORMAL(+1), FOOD_RICH(+3), FOOD_TOXIC(-0.5)
- A4: `avanzar_era()` anade obstaculo, reduce comida, aumenta % toxicos
- Compatibilidad con cerebros: propiedades `food_solo_x/y`, `food_social_x/y`, `WORLD_SIZE`, `era`
- `quadrant_spawn(q)` para niching geografico
- `random_spawn(near=xy, radius=150)` para hijos cerca del padre

#### `depredador.py`
- Cuerpo pymunk simple que persigue al organismo mas cercano
- Parametros: speed=45.0, hit_radius=16.0, hit_cooldown=240
- `step()` devuelve lista de organismos golpeados

#### `visualizador_fase1.py`
- Pygame: organismos (color por linaje), comida (color por tipo), obstaculos, depredadores
- Panel lateral: top-5 fitness, descripcion morfoma del mejor, linajes activos
- Trails de movimiento (40 puntos por organismo)
- `draw(gen, step, organismos, depredadores)` devuelve False para cerrar

#### `arena_fase1.py`
Motor principal de evolucion.
- `_bootstrap_semilla_cerebro()`: truco exec() para cargar clases en globals() y poder desempaquetar pkl
- `Organismo`: dataclass cerebro + morphome + cuerpo + quadrant + lineage_color
- `construir_generacion_0()`: carga 4 pkl, asigna cuadrantes (C4), rellena con raiz
- `seleccion_por_torneo()`: C3, torneos de 3, mantiene mas diversidad que top-50% puro
- `reproducir()`: C2 crossover cuando se provee segundo padre
- `loop_evolutivo()`: loop principal con depredadores, avance de era, niching
- Sistema de checkpoints: guarda cerebros completos + morfomas cada N generaciones
- `--continue`: carga `checkpoints/latest.pkl` para continuar evolucion
- `--save_every N`: frecuencia de guardado (default 5 generaciones)
- `_archivar_checkpoints_viejos()`: al iniciar sin --continue, archiva checkpoints anteriores automaticamente

---

## BUGS ENCONTRADOS Y CORREGIDOS

### Bug 1 — Unpickle fallaba ("Can't get attribute 'IntelligentCell'")
**Causa:** Las clases de semilla_cerebro no estaban en el namespace `__main__`.
**Fix:** `_bootstrap_semilla_cerebro()` ejecuta semilla_cerebro.py con exec() en globals(), reemplazando `if __name__ == "__main__":` por `if False:`.

### Bug 2 — Fitness todo negativo desde generacion 0
**Causa:** Penalizacion B2 demasiado agresiva: `0.0008 * masa * pasos`.
**Fix:** Solo penalizar masa excesiva por encima de 1.5 con coeficiente 0.00015.

### Bug 3 — Depredador demasiado agresivo (colapso de poblacion)
**Causa:** Penalizacion de 2.0 por golpe + velocidad 70 causaba fitness negativo en todos.
**Fix:** 0.5 por golpe (tope en score*0.5+5), velocidad 45, cooldown 240 frames.

### Bug 4 — Background task silencioso contaminando checkpoints
**Causa:** Se lanzo un proceso en background con `--continue`. El usuario creyo empezar de cero corriendo sin `--continue`, pero el background task seguia escribiendo checkpoints con el estado antiguo (Era 10). Resultado: corrida aparentemente fresca que en realidad continuaba desde Era 10.
**Fix:** Funcion `_archivar_checkpoints_viejos()` se llama automaticamente al iniciar sin `--continue`. Mueve checkpoints viejos a `checkpoints_archive_TIMESTAMP/` y crea carpeta limpia.

### Bug 5 — Sistema de eras demasiado agresivo
**Causa:** `--eras_every 8` hacia que en 80 generaciones el mundo llegara a Era 10 (pocos recursos, muchos obstaculos). Los organismos no tenian tiempo de desarrollar motor maps antes de que el mundo se volviera brutal.
**Fix:** Parametro recomendado cambiado a `--eras_every 30`.

---

## DESCUBRIMIENTOS DE LAS SIMULACIONES

### Run 1 (gens 0-79, eras_every=8)
- **Beta domino completamente** — alfa, gamma y delta se extinguieron
- **Sensor maps evolucionaron solos**: de 0 a promedio 4.9 por organismo. Nadie lo programo, emergio por seleccion
- **Motor maps no evolucionaron**: `mmap=0` en casi todos. Los cuerpos tienen articulaciones pero el cerebro no las controla
- **Causa probable**: En mundo dificil (Era 10), el empuje base es suficiente para sobrevivir. Motor maps no dan ventaja visible, son eliminados por seleccion natural

### Run 2 (gens 80-164, continuacion contaminada con eras_every=30)
- **Brotes neuronales masivos**: los cerebros crecen neuronas constantemente por frustracion (no encuentran comida). Luego MUSK STEP las poda. Ciclo continuo de crecer-podar, analogo a neurogenesis bajo estres
- **Motor maps emergieron brevemente**: gen 81 mostro `mmap=2` en un organismo. Gen 82 varios con `mmap=1`. Luego fueron eliminados por seleccion
- **Sensor maps siguieron creciendo**: promedio 6.9 al final (gen 164)
- **Beta monopolio persistio**: 12/12 organismos beta
- **Era llego a 13** por contaminacion del run anterior

### Observacion general
Los organismos evolucionan una estrategia de **sentir mucho, moverse poco con intencion**. Muchos sensor maps, pocos motor maps. El empuje base del cerebro (left/right directo) es suficiente para desplazarse. Las articulaciones motorizadas anaden complejidad sin recompensa clara en el fitness actual.

El cerebro de beta (126 neuronas, equilibrado) resulto mas competitivo que alfa (235, complejo) en este entorno. Hipotesis: en mundos pobres con poca comida, la simplicidad computacional gana. Alfa puede tener ventaja en mundos con mas recursos y variabilidad.

---

### Run 3 (gens 0-299, limpio, eras_every=30) — 2026-04-28
Primera corrida verdaderamente limpia desde Era 0. 300 generaciones completas.

**Resultado inesperado — raiz gano a todos los cerebros entrenados:**
- Los 4 cerebros pkl (alfa, beta, gamma, delta) se extinguieron
- Gano `raiz-10`: cerebro vacio generado de cero sin entrenamiento previo
- Hipotesis: los cerebros pkl tienen pesos sinápticos moldeados para un cuerpo virtual diferente. Sus habitos aprendidos interfieren con el nuevo cuerpo fisico. Los cerebros raiz son tablas en blanco que aprenden desde cero via STDP

**Picos de fitness notables:**
- Gen 36: fitness 20.01 — alguien comio ~20 alimentos en 3000 pasos
- Gen 113: fitness 26.01 — record absoluto, comportamiento de navegacion real emergente
- Despues del pico, la poblacion converge demasiado rapido en el linaje ganador, pierde diversidad, y colapsa cuando el mundo sube de era

**Motor maps siguen en 0 — patron confirmado en 3 corridas:**
- El empuje base (left/right directo) es suficiente para moverse
- Motor maps no dan ventaja de fitness → seleccion los elimina
- Diagnostico definitivo: el fitness solo premia comer, no moverse eficientemente

**Sensor maps: patron consistente:**
- Promedio 4.9 al final, todos los organismos con 4-6 sensores
- Emergen solos sin programacion explicita en todas las corridas

### Analogia conceptual identificada (2026-04-28)
- `semilla_cerebro.py` = codigo genetico (ADN) — mecanismos de bajo nivel que no cambian
- `arena_fase1.py` = leyes fisicas del universo — consecuencias, no instrucciones
- `morfoma.py` = cromosomas — genes que codifican la forma del cuerpo
- `cuerpo_evolvable.py` = fenotipo — cuerpo fisico que emerge de los genes
- `mundo_fisico.py` = el planeta — recursos, obstaculos, condiciones
- `depredador.py` = presion de seleccion — la muerte que da forma a la evolucion
- `checkpoints/` = registro fosil — memoria de lo que sobrevivio

---

## MEJORAS IMPLEMENTADAS (2026-04-28)

### Mejora 1 — Bonus de acercamiento a comida (`cuerpo_evolvable.py`)
**Problema:** El fitness solo premiaba comer. Sin presion para moverse con direccion, los motor maps nunca tenian ventaja evolutiva.
**Solucion:** Cada step que el organismo se acerca a la comida mas cercana acumula +0.001 en `_approach_bonus`. Maximo ~3.0 en 3000 pasos (equivale a 3 alimentos normales). Esto crea presion de seleccion directa para motor maps que mejoren la navegacion.

---

---

## RUN 4 — PRIMER GRAN DESCUBRIMIENTO (2026-04-28)

### Configuracion
```bash
python arena_fase1.py --gens 300 --steps 4000 --pop 12 --food 70 --predators 1 --eras_every 30
```
Primera corrida con todas las mejoras activas:
- Bonus de acercamiento a comida (`cuerpo_evolvable.py`)
- Limpieza de cerebros pkl al cargar (pesos x0.2)
- M1: drift de clusters cada generacion (sigma=12px)
- M2: bursts transitorios de comida rica (prob=0.45 cada 30 steps)
- M3: rotacion de obstaculos al avanzar era
- M4: jitter en ratios de comida (+-8% cada gen)
- semilla_cerebro.py: merit_factor en can_divide, sine 0.3, ruido adaptativo

### Evolucion del fitness
| Gen | Best | Avg | Nota |
|-----|------|-----|------|
| 0 | 23 | 11 | inicio |
| 6 | 389 | 69 | crecimiento rapido |
| 7 | 84 | 37 | crash — clusters derivaron |
| 12 | 80 | 26 | crash — jitter de ratios |
| 14 | 617 | 118 | maximo previo (3 runs) |
| 20 | 704 | 124 | nuevo record |
| **25** | **1,130** | **247** | **RECORD ABSOLUTO — 43x gen 0** |
| 30 | 133 | 35 | crash — ERA 1 activo, mundo cambiado |
| 34 | 292 | 84 | recuperacion en progreso |

### DESCUBRIMIENTO PRINCIPAL — Motor Maps evolucionaron por primera vez

En 3 runs anteriores (mas de 500 generaciones combinadas), los motor maps (mmap) nunca sobrevivieron la seleccion. mmap=0 era una constante.

En Run 4, gen 34:
```
r-g34-004: segs=9 joints=8 motors=4 mmap=3 smap=3
r-g33-001: segs=9 joints=8 motors=4 mmap=3 smap=2
r-g33-000: segs=9 joints=8 motors=3 mmap=3 smap=2
```

**El cerebro controla activamente el cuerpo** via 3 canales motores mapeados. Simultaneamente, 3 canales sensoriales fisicos retroalimentan al cerebro. Se formo un bucle cerrado: cerebro → motores → movimiento → sensores → cerebro.

Progresion morfologica en Run 4:
| Gen | Segs | Joints | Motors | mmap | smap |
|-----|------|--------|--------|------|------|
| 1 (semilla) | 1 | 0 | 0 | 0 | 0 |
| 14 | 4 | 3 | 2 | 0 | 0 |
| 34 | **9** | **8** | **4** | **3** | **3** |

### Causa identificada del cambio
El **bonus de acercamiento** (+0.001 por step cuando el organismo se acerca a la comida) creo presion de seleccion directa sobre la capacidad de navegacion dirigida. Por primera vez, tener motores controlados por el cerebro daba ventaja de fitness medible sobre el empuje base (left/right ciego).

### Los crashes son parte del diseno
Los caidas bruscas de fitness (gen 7, 12, 30) son exactamente el comportamiento esperado:
- Los clusters de comida derivan → la memoria espacial antigua falla
- Los ratios de comida cambian → estrategias de alimentacion fijas se vuelven suboptimas
- La era avanza → obstaculos nuevos, menos comida
- Cada crash es seguido de recuperacion mas fuerte: seleccion bajo presion dinamica

### Analogia biologica
Lo que ocurrio en gen ~25-34 es el equivalente computacional de la **explosion cambrica**:
- Explosion de diversidad morfologica (1 → 9 segmentos)
- Aparicion de sistemas nerviosos que controlan el movimiento (mmap=0 → mmap=3)
- Feedback sensoriomotor emergente (el animal "siente" lo que hace)
- Todo sin instruccion explicita — solo seleccion natural sobre fisica

### Estado a gen 34
- Era: 1 (primera transicion superada)
- Mejor organismo: 9 segmentos, 8 articulaciones, 3 motor maps, 3 sensor maps
- Fitness max historico: 1,130 (gen 25)
- Poblacion entera en cuadrante 2 (niching geografico colapso — ver pendientes)
- Corrida continua con `--continue`

### Actualizacion gen 35-49 (2026-04-28)

**Fitness:**
| Gen | Best | Avg | Nota |
|-----|------|-----|------|
| 36 | 909 | 208 | mejor promedio hasta ese momento |
| 41 | **1,274** | 202 | **nuevo record absoluto** |
| 44 | 294 | 128 | consolidacion post-record |
| 47 | 654 | **162** | **promedio mas alto sostenido de toda la historia** |
| 49 | 383 | 94 | estable en era 1 |

**Morfologia consolidada — ya no crece explosivamente, se consolida:**
```
rx-g43-003: segs=12 joints=11 motors=5 mmap=5 smap=6
r-g44-000:  segs=12 joints=11 motors=5 mmap=6 smap=6
```
mmap=6 — seis canales del cerebro controlando motores activamente.
El siguiente salto morfologico se espera al avanzar a era 2 (aprox gen 60).

**Mejora implementada — sistema de utilidad neuronal sin tope (2026-04-28):**
El cap fijo en 2.0 fue reemplazado por presion metabolica competitiva (ATP budget).
- Antes: `utility = min(2.0, utility + 0.01)`
- Ahora: sin tope. El decay se acelera cuando la utilidad total supera `n_neuronas * 1.5`
- Efecto esperado: distribucion mas desigual — pocas neuronas muy especializadas (>2.0), muchas con utilidad baja
- Analogia biologica: neuronas compiten por glucosa/ATP, no hay un limite fisico fijo
- Aun no visible en checkpoints — necesita mas generaciones para manifestarse

**Conversacion conceptual registrada:**
- El tope 2.0 fue identificado como una simplificacion que pierde informacion de gradiente
- En biologia real, la "utilidad" es un estado fisico distribuido (densidad de receptores, mielinizacion, expresion genica) sin numero fijo
- La presion metabolica como regulador emergente es mas fiel al sistema real
- Observacion: durante alta dopamina (>0.65) solo hay crecimiento, nunca decay — riesgo teorico de inflacion de utilidad durante feasts prolongados, pero en practica la dopamina no se sostiene alta continuamente

---

## BUG CRITICO CORREGIDO — Utilidad neuronal runaway (2026-04-28/29)

**Problema detectado:** El sistema de presion metabolica tenia una asimetria fatal.
- Durante alta dopamina (>0.65): solo corre el bloque de crecimiento
- Durante baja dopamina: solo corre el bloque de decay
- Efecto: los organismos exitosos (alta dopamina sostenida) acumulaban utilidad indefinidamente
- Observado en gen 47: util_max=108.4, avg=81.7 — 53/53 neuronas por encima de 2.0
- Gen 54: degradacion a avg=70 porque el runaway eventual destruye plasticidad STDP

**Fix aplicado:** El decay metabolico se mueve FUERA del if/else — ahora corre SIEMPRE, independientemente del nivel de dopamina.
```python
# Crecimiento (solo con alta dopamina)
if dopamina > 0.65:
    ...crecimiento...

# Decay SIEMPRE activo (presion metabolica permanente)
decay_rate = 0.9995 ** metabolic_pressure
for n in neurons:
    utility[n.id] *= decay_rate
```

**Penalidad depredador aumentada** de 0.5 → 2.0 por golpe (antes el impacto era irrelevante para el fitness).
**Depredador por era** implementado: cada vez que el mundo avanza una era, se agrega un depredador nuevo.

---

## RUN 4 — Actualizacion gen 54-74 (2026-04-29)

Estado al momento de pausar la corrida para descanso de la PC (2 noches + 3 dias encendida):

| Gen | Best | Avg | Era | Nota |
|-----|------|-----|-----|------|
| 54 | ~70 | degradado | 4 | utilidad runaway en curso (antes del fix) |
| 69 | 297 | 81 | 3 | recuperacion post-fix |
| 70 | 374 | 98 | 3 | crecimiento sostenido |
| 71 | **671** | 138 | 3 | pico notable |
| 72 | 69 | 26 | **4** | crash brusco al entrar Era 4 — depredadores nuevos |
| 73 | 237 | 56 | 4 | recuperacion |
| 74 | 316 | 71 | 4 | consolidando |

Morfologias al gen 74 (Rama A): hasta **12 segmentos, 8 motor maps, 8 sensor maps** — las mas complejas de toda la historia de la simulacion.

---

## RAMIFICACION — Dos simulaciones paralelas (2026-04-29)

### Motivacion
Para comparar caminos evolutivos distintos, se lanzan dos ramas en paralelo:
- **Rama A**: continua desde gen 54 con todos los fixes
- **Rama B**: inicio fresco desde cero con el cerebro hibrido inyectado

### Implementacion tecnica
- Flag `--save_dir` agregado a `arena_fase1.py` — cada rama guarda en su propia carpeta sin colisionar
- `--continue` respeta el `--save_dir` al buscar `latest.pkl`

### Cerebro Hibrido (nuevo)
Script `crear_hibrido.py` fusiona las topologias de alfa + beta:
- **alfa**: 483 neuronas | **beta**: 245 neuronas → **hibrido: 728 neuronas**
- IDs de beta remapeados (offset = max_alfa_id + 1) para evitar colisiones
- Pesos escalados a x0.05 — memoria limpia, STDP recalibra desde casi-cero
- Curiosity grid y episodic buffer vaciados
- Guardado como `arena_save_hibrido.pkl`

**Bug adicional descubierto y corregido:** `crear_cerebro_desde_pkl` llamaba `.values()` sobre `brain.cells` que es una lista, no un dict. Esto hacia que la limpieza de pesos de los cerebros semilla (alfa/beta/gamma/delta) fallara silenciosamente en las corridas 1-4. Corregido con deteccion de tipo.

### Estado al pausar (2026-04-29)

| Rama | Checkpoint | Era | Best reciente | Morfologia dominante |
|------|-----------|-----|---------------|----------------------|
| A (`checkpoints/`) | gen 74 | 4 | 316 (gen 74) | 12 segs, 8 mmap, 8 smap |
| B (`checkpoints_fresh/`) | gen 59 | 7 | 398 (gen 57) | 4-5 segs, 5 mmap, 4 smap |

**Observacion inicial Rama B:** Avanza mas rapido en eras (era 7 vs era 4) con morfologias mas compactas pero igual de conectadas. Linaje dominante: `b-` (beta). Estrategia evolutiva distinta emergiendo desde el primer run — exactamente lo que se esperaba.

**Comandos para reanudar:**
```bash
# Rama A (continuar desde gen 74)
python arena_fase1.py --continue --gens 200 --steps 4000 --pop 12 --food 70 --predators 1 --eras_every 8 --save_dir checkpoints

# Rama B (continuar desde gen 59)
python arena_fase1.py --continue --gens 200 --steps 4000 --pop 12 --food 70 --predators 1 --eras_every 8 --save_dir checkpoints_fresh
```

---

## ESTADO ACTUAL (2026-04-29)

- **Simulaciones pausadas** — PC descansando despues de 2 noches + 3 dias de ejecucion continua
- Rama A: gen 74, era 4 — morfologias complejas (12 segs), fix metabolico aplicado
- Rama B: gen 59, era 7 — morfologias compactas, camino evolutivo distinto confirmado
- Cerebro hibrido (728n) listo en `arena_save_hibrido.pkl` para proxima Rama B o nueva rama
- Todos los checkpoints guardados y seguros

---

## ACTUALIZACION RAMAS — Gen 109 / Gen 139 (2026-04-29)

### Rama A — gen 109 | Era 8
| Gen | Best | Avg | Era | Nota |
|-----|------|-----|-----|------|
| 102 | 626 | 161 | 7 | pico fuerte |
| 103 | 195 | 76 | 7 | consolidacion |
| 104 | 119 | 31 | **8** | crash al entrar Era 8 |
| 105 | 119 | 37 | 8 | resistiendo |
| 106 | 285 | 61 | 8 | recuperacion |
| 107 | 275 | 71 | 8 | estable |
| 108 | 117 | 39 | 8 | oscilacion |
| 109 | 360 | 76 | 8 | recuperandose |

**Morfologia tope al gen 109:**
```
rx-g90-003: segs=12 joints=11 mmap=13 smap=11
rx-g86-002: segs=12 joints=11 mmap=11 smap=9
```
mmap llegó a **13** — el cerebro controla 13 canales motores activos (era 0 al inicio).
Record histórico de la rama: **1,273 pts (gen 41)**.

### Rama B — gen 139 | Era 17
| Gen | Best | Avg | Era | Nota |
|-----|------|-----|-----|------|
| 132 | 362 | 128 | 16 | sostenido |
| 133 | 396 | 127 | 16 | pico |
| 134 | 285 | 98 | 16 | estable |
| 135 | 392 | 124 | 16 | solido |
| 136 | 119 | 38 | **17** | crash al entrar Era 17 |
| 137 | 218 | 71 | 17 | recuperacion rapida |
| 138 | 246 | 83 | 17 | creciendo |
| 139 | 292 | 84 | 17 | consolidando |

**Morfologia tope al gen 139:**
```
b-g98-000:  segs=12 joints=11 mmap=12 smap=10
b-g97-002:  segs=12 joints=11 mmap=13 smap=10
```
Record histórico de la rama: **1,024 pts (gen 92)**.
Rama B corre mas rapido (gen 139 vs gen 109) porque los organismos mas simples generan menos carga computacional.

---

## DESCUBRIMIENTO — Convergencia morfologica entre ramas (2026-04-29)

**Hallazgo critico:** Ambas ramas, partiendo desde condiciones radicalmente distintas, convergieron a la misma morfologia optima:

| | Rama A | Rama B |
|---|---|---|
| Inicio | Gen 54, organismos complejos heredados | Gen 0, organismos frescos + hibrido |
| Era actual | 8 (presion moderada) | 17 (presion extrema) |
| Morfologia tope | 12 segs, 13 mmap, 11 smap | 12 segs, 13 mmap, 10 smap |

**Interpretacion:** Existe un "atractor morfologico" — la fisica del mundo 2D y las presiones selectivas empujan hacia el mismo plan corporal (12 segmentos, ~13 motor maps) independientemente del punto de partida. Es analogo a la **evolucion convergente** en biologia real: las aletas de peces, delfines y pinguinos emergen por separado porque son la solucion optima al mismo problema fisico.

**Rama B en Era 17 sigue funcional** con fitness de ~300 promedio — una resiliencia notable bajo presion maxima. Esto sugiere que las morfologias compactas de beta son genuinamente mas robustas a la escasez de recursos.

---

## ESTADO ACTUAL (2026-04-29)

- **Rama A**: gen 109, era 8, mmap=13, recuperandose del crash de era
- **Rama B**: gen 139, era 17, mmap=13, resistiendo presion extrema
- Convergencia morfologica confirmada entre ambas ramas — resultado inesperado y significativo

**Comandos para reanudar:**
```bash
# Rama A
python arena_fase1.py --continue --gens 200 --steps 4000 --pop 12 --food 70 --predators 1 --eras_every 8 --save_dir checkpoints

# Rama B
python arena_fase1.py --continue --gens 200 --steps 4000 --pop 12 --food 70 --predators 1 --eras_every 8 --save_dir checkpoints_fresh
```

---

## FASE 1.5 — Mundo Social (2026-04-29)

Implementacion completa del "mundo rico": arena mas grande, feromonas, zonas de friccion, visualizador pygame, colision entre organismos para cadena alimenticia.

### Cambios en `mundo_fisico.py`
- **F1.5-A**: Arena ampliada de 600x600 a 1200x1200 (4x area). Clusters aumentados de 4 a 6.
- **F1.5-B**: Cuadricula de feromonas 60x60 numpy array. Decay PHERO_DECAY=0.998 por step. Deposito al comer comida no toxica. Metodos: `deposit_pheromone()`, `sense_pheromone()`, `step_pheromones()`.
- **F1.5-D**: Zonas de friccion: 3 circulos con radio 80-140px que reducen la fuerza de los motores al 40% (FRICTION_ZONE_FACTOR=0.40). Se reposicionan al avanzar era.
- **F1.5-C**: `detect_predation()` v1: ratio masa 2x, radio 30px, muerte instantanea, recompensa fija +5. (Reemplazado en Fase 1.6.)
- `step_physics()` llama a `step_pheromones()` cada step.

### Cambios en `cuerpo_evolvable.py`
- **ShapeFilter con grupos**: cada organismo tiene `_group_id` unico. Segmentos del mismo organismo no colisionan entre si, pero SI con segmentos de otros organismos. Esto activa la cadena alimenticia fisica.
- Motor output multiplicado por `friction_factor` de la zona donde esta el organismo.
- Sensor `pheromone`: lee la cuadricula de feromonas en la posicion del centroide. Devuelve [0,1].
- Sensor `nearby_organism`: 3 sub-canales — distancia (0), angulo (1), tamano en segmentos (2) — al organismo mas cercano (excepto self).
- Contadores `_eaten_others` y `_was_eaten`.

### Cambios en `morfoma.py`
- `_op_add_sensor_map` actualizado con tipos `pheromone` (0.15) y `nearby_organism` (0.15).
- `_op_remove_segment` y `crossover` preservan sensores globales (pheromone, nearby_organism) que no dependen de indices de segmentos.

### Nuevo archivo: `visualizador_fase1.py`
- Escala adaptativa: MAX_WORLD_DISPLAY=800px, scale=min(1.0, 800/max(W,H)).
- Capa de feromonas: heat map ambar por celdas, solo si max > 0.1.
- Zonas de friccion: circulos purpura semi-transparentes.
- Obstaculos: lineas con grosor escalado.
- Organismos: circulos por segmento + punto blanco indicando angulo + lineas de joints + trail de 40 puntos.
- Depredadores: triangulos rojo-borde negro.
- Panel lateral: Top-5 fitness, morfoma del mejor, linajes por cuadrante.

### Semillas extraidas y reorganizacion
- `semillas/arena_save_omega.pkl` — Rama A gen 124, era 10: 31n, 12 segs, 13mmap, 13smap
- `semillas/arena_save_sigma.pkl` — Rama B gen 154, era 19: 39n, 12 segs, 15mmap, 10smap
- `semillas/arena_save_omega2.pkl`, `sigma2.pkl` — semillas secundarias
- `semillas/originales/` — alfa, beta, gamma, delta originales preservados
- `semillas/arena_save_hibrido.pkl` — cerebro fusionado alfa(483n) + beta(245n) = 728n
- Carpetas `tools/` y `docs/` creadas para reorganizacion del proyecto

### Smoke test Fase 1.5 — EXITOSO
Todos los modulos cargaron. Feromonas, friccion, colision entre organismos y visualizador funcionando correctamente.

---

## FASE 1.6 — Cadena Alimenticia Real + Obstaculos Vivos (2026-04-29)

Rediseno biologico de la depredacion y nuevo sistema de obstaculos dinamicos/moldeables.

### Motivacion
La implementacion v1 de `detect_predation` era demasiado permisiva (ratio 2x, muerte instantanea, sin cooldown). Se discutio en detalle la biologia de la depredacion y los principios de coexistencia antes de implementar. Objetivo: la depredacion debe ser rara, costosa y permitir coexistencia natural entre presas y cazadores.

### C2 — Cadena Alimenticia Refinada (`mundo_fisico.py`)

Cuatro reglas biologicas implementadas en `detect_predation()`:

| Regla | Parametro | Valor | Justificacion biologica |
|-------|-----------|-------|------------------------|
| 1. Ratio de masa | `mass_ratio` | **4x** (antes 2x) | Un lobo pesa ~8x un conejo. 4x es conservador pero exigente. |
| 2. Contacto sostenido | `contact_threshold` | **3 contactos consecutivos** | Simula atrapar y sujetar. Si la presa escapa, contador se resetea. |
| 3. Cooldown de saciedad | `cooldown_steps` | **100 steps** | Como un leon que descansa tras comer — no caza durante 100 steps. |
| 4. Recompensa proporcional | formula | `food_eaten + masa × 0.3` | Cazar algo grande y bien alimentado vale mas que cazar algo debil. |

- Radio de captura reducido de 30px a **18px** (necesitan estar muy cerca).
- Estado por organismo: `_predation_cooldown` (steps restantes, decrementado en `step_physics`) y `_pending_contact` (dict `id(presa) → contactos consecutivos`).
- Organismos en cooldown se saltan completamente en `detect_predation`.
- Contadores fuera de rango limpian la entrada del dict (la presa escapo).

**Mecanismos de coexistencia emergentes:**
- Presas mas pequeñas = menor costo energetico (B2) → se reproducen mas rapido
- Sensor `nearby_organism` permite evolucionar evasion antes del contacto
- Zonas de friccion penalizan mas a los masivos (depredadores lentos en esas zonas)
- Cooldown garantiza ventanas de seguridad para presas

### Modulo E — Obstaculos Vivos (`mundo_fisico.py`)

**Taxonomia de 4 tipos:**

| Tipo | Constante | ¿Se mueve? | ¿Se moldea? | Color |
|------|-----------|-----------|-------------|-------|
| Rigido | `OBS_RIGID` | No | No | Gris oscuro |
| Dinamico | `OBS_DYNAMIC` | Si (deriva) | No | Gris azulado |
| Plastico | `OBS_PLASTIC` | No | Si | Marron calido |
| Vivo | `OBS_LIVING` | Si (deriva) | Si | Ambar/naranja |

**Estructura interna de `Obstaculo` rediseñada:**
- `p1_orig, p2_orig`: posicion original (inmutable)
- `drift_x, drift_y`: traslacion solida acumulada (dinámicos/vivos)
- `p1_deform, p2_deform`: offset de deformacion por extremo (plasticos/vivos)
- `p1`, `p2` como properties: `p_final = p_orig + drift + deform`
- `vx, vy`: velocidad de deriva (0.3-0.6 px/step para dinamicos)
- `is_dynamic` y `is_plastic` como properties booleanas

**Distribucion garantizada** por algoritmo de deficit (no aleatoria):
- Para n=8: 3 rigidos, 1 dinamico, 3 plasticos, 1 vivo — todos los tipos presentes siempre.

**E1 — `step_obstacles()` (llamado en `step_physics`):**
- Avanza `drift_x/y` con `vx/vy` cada step
- Rebota en paredes (margin=30px): invierte la componente correspondiente sin aplicar el movimiento ese step
- Actualiza endpoints en pymunk con `unsafe_set_endpoints` + `space.reindex_static()`

**E2 — `try_mold_obstacles()` (llamado en `step_physics` por cada organismo):**
- Requiere `masa_total >= 8.0` (MOLD_MASS_THRESHOLD) Y `velocidad >= 8.0` (MIN_MOLD_SPEED)
- Si algun segmento del organismo esta a < 18px (MOLD_RADIUS) de un extremo plastico:
  - Acumula deformacion en direccion de la velocidad del organismo
  - Maximo 60px de deformacion (MAX_DEFORM) — el extremo no viaja indefinidamente
  - Actualiza pymunk con `unsafe_set_endpoints` + `reindex_static()`

**Comportamientos emergentes esperados (sin programacion):**
- Presas empujando extremos para cerrar una abertura → construccion de refugio
- Depredadores curvando obstaculos para acorralar → herding emergente
- Organismos acumulando comida en zonas delimitadas por obstaculos deformados

### E4 — Sensor `obstacle_endpoint` (`cuerpo_evolvable.py` + `morfoma.py`)

Nuevo tipo de sensor evolucionable:
- `target_idx=0`: distancia al extremo plastico mas cercano (normalizada, devuelve [0,1])
- `target_idx=1`: angulo relativo hacia ese extremo

Peso en mutaciones: 0.12 (igual que `nearby_organism`). Verificado que aparece en genomas tras ~500 generaciones.
Tratado como "sensor global" en `_op_remove_segment` y `crossover` — no invalida al eliminar segmentos.

### Visual por tipo (`visualizador_fase1.py`)
- Cada tipo dibujado en su color de `OBS_COLOR`
- Plasticos/vivos: circulos en extremos, color se intensifica (marron → rojo) proporcional a deformacion acumulada / MAX_DEFORM
- Dinamicos/vivos: flecha blanca en el centro apuntando hacia la direccion de deriva

### Verificacion
```
Distribucion tipos (n=8): rigid=3, dynamic=1, plastic=3, living=1  ✓
step_obstacles 20x: sin error  ✓
obstacle_endpoint en mutaciones (500 gens): aparece  ✓
detect_predation nueva signatura: OK  ✓
arena_fase1.py --smoke: Gen 00 en 32.5s, exitoso  ✓
```

---

## SIMULACIONES ACTIVAS (2026-04-29)

### Rama Depredadora (`checkpoints_rama_a`)
```bash
python arena_fase1.py --gens 300 --steps 3000 --pop 12 --food 80 --predators 2 --save_dir checkpoints_rama_a
```
Objetivo: presion alta de depredacion (2 depredadores externos + cadena alimenticia entre organismos). Espera morfologias de velocidad y evasion/caza emergentes.

### Rama Constructora (`checkpoints_rama_b`)
```bash
python arena_fase1.py --gens 300 --steps 3000 --pop 12 --food 80 --predators 0 --save_dir checkpoints_rama_b
```
Objetivo: sin depredadores externos. La cadena alimenticia entre organismos es la unica presion de caza. Mas espacio para que emerja moldeo de obstaculos plasticos. Espera comportamientos de construccion de nicho.

Ambas corren sin `--visual` para maximizar velocidad. Se visualizaran al observar logros.

---

## RESULTADOS NOCHE Gen 194 — Simulaciones Depredadora y Constructora (29-30 Abril 2026)

Ambas simulaciones corrieron toda la noche sin visual. Detenidas manualmente en Gen 194, Era 24.
Analisis con `analizar_checkpoint.py` sobre `checkpoint_gen0194.pkl` de cada rama.

### Configuracion al iniciar

| Parametro | Rama A (Depredadora) | Rama B (Constructora) |
|-----------|---------------------|----------------------|
| Depredadores | 2 | 0 |
| Food inicial | 80 | 80 |
| Steps/gen | 3000 | 3000 |
| Pop | 12 | 12 |
| Eras_every | 8 | 8 |

### Rama A (Depredadora) — Gen 194, Era 24

**Fitness:**
- Record absoluto: **499.44** (Gen 20) — alcanzado muy temprano, nunca superado en 194 gens
- Fitness promedio ultimas 20 gens: 175.39
- Gen 194: best=158.03, avg=66.03, worst=1.17
- Tendencia ultimas 5 gens: **-60.57** (descenso — Era 24 es muy dura)
- Gen mas notable: Gen 191 best=339.19 (pico tarde en la corrida)

**Morfologia de la poblacion (12 organismos):**
- Segmentos: min=10, avg=11.5, max=12
- Articulaciones: avg=10.5, **solo 2.4 con motor activo** (baja utilizacion)
- Motor maps: avg=8.3 (cerebro mapeado pero pocos joints activos)
- Sensor maps: avg=18.2
- Masa: bimodal — 3 organismos en rango 8-11, 9 organismos en rango 12-15 (posible split depredador/presa)

**Sensores en toda la poblacion:**
```
contact         x60  ████████████████████████████████████████
angle           x37  █████████████████████████████████████
pheromone       x37  █████████████████████████████████████
nearby_organism x36  ████████████████████████████████████
obstacle_endpoint x25 █████████████████████████
angvel          x24  ████████████████████████
```

**Estado neuronal:**
- Celulas totales: avg=84.7 (distribucion: ~75 NEURON + 3 GLIA + 2 STEM + 4 APOPTOTIC)
- Memoria episodica: avg=1895 entradas (max=5000 — algunos cerebros tienen memoria muy activa)
- Frustration avg: 0.452 | Resilience avg: 0.750 | Courage avg: 0.750
- Curiosity grid: **100% cobertura en todos los organismos** — exploran el mapa completo cada gen
- Especializacion neuronal: no detectada — IDs de alta utilidad difieren entre organismos

### Rama B (Constructora) — Gen 194, Era 24

**Fitness:**
- Record absoluto: **552.04** (Gen 143) — mas reciente y superior al de Rama A
- Fitness promedio ultimas 20 gens: 180.47
- Gen 194: best=177.36, avg=54.98, worst=-6.41
- Tendencia ultimas 5 gens: **-46.12** (descenso — mismo patron que A bajo presion de Era 24)
- Gen mas notable: Gen 183 best=397.42, Gen 187 best=380.01

**Morfologia de la poblacion (12 organismos):**
- Segmentos: **12 en el 100%** de organismos (saturacion completa)
- Articulaciones: todos con 11, **7.2 con motor activo** (3x mas que Rama A)
- Motor maps: avg=14.3 (rango 8-18) — mucho mas conectado que A
- Sensor maps: avg=21.6 (rango 20-22)
- Masa: uniforme en rango 12-15 (sin bifurcacion — sin presion de depredacion interna)

**Sensores en toda la poblacion:**
```
contact         x83  ████████████████████████████████████████████████████
angle           x53  █████████████████████████████████████████████████████
angvel          x51  ████████████████████████████████████████████████████
nearby_organism x36  ████████████████████████████████████
obstacle_endpoint x24 ████████████████████████
velocity        x12  ████████████
```
Notable: **mas propriocepcion** (angle+angvel=104 vs 61 en A) — los organismos sin depredadores desarrollaron mas conciencia de su propio cuerpo. Rama A en cambio desarrollo mas pheromona (comunicacion quimica).

**Estado neuronal:**
- Celulas totales: avg=119.1 (distribucion: ~114 NEURON + 3 GLIA + 2 STEM + 4 APOPTOTIC)
- **~40% mas neuronas activas que Rama A** (119 vs 84)
- Memoria episodica: avg=1061 entradas (menos que A — menor presion emocional)
- Frustration avg: 0.430 | Resilience avg: 0.682 | Courage avg: 0.682
- Curiosity grid: **100% cobertura en todos** — misma saturacion que A
- Especializacion neuronal: no detectada — IDs de alta utilidad son consistentes por organismo pero no compartidos entre organismos

### Comparativa A vs B — record historico

| Gen | Best_A | Best_B | Delta | Era |
|-----|--------|--------|-------|-----|
| 165 | 226.73 | 269.06 | +42 B | 20 |
| 173 | 313.29 | 224.02 | +89 A | 21 |
| 181 | 320.05 | 200.83 | +119 A | 22 |
| 183 | 197.28 | **397.42** | +200 B | 22 |
| 187 | 203.73 | **380.01** | +176 B | 23 |
| 191 | **339.19** | 276.34 | +63 A | 23 |
| 194 | 158.03 | 177.36 | +19 B | 24 |

Ganador de records absolutos: **Rama B — 552 vs 499**

### TERCERA CONVERGENCIA MORFOLOGICA CONFIRMADA

Ambas ramas convergieron independientemente a **12 segmentos + los 6 tipos de sensor** disponibles.
Esta es la tercera vez que se observa este patron (anteriormente con Omega y Sigma).

| | Rama A | Rama B |
|---|---|---|
| Linaje dominante | `r` (raiz) — 12/12 | `r` (raiz) — 12/12 |
| Morfologia tope | 12 segs, 10 mmap, 20 smap | 12 segs, 18 mmap, 22 smap |
| Tipos de sensor | 6/6 | 6/6 |

**Los 4 cerebros semilla originales (alfa, beta, gamma, delta) se extinguieron completamente** — la seleccion favorecio cerebros raiz frescos que aprenden desde cero sin habitos pre-entrenados.

### Observacion: Curiosity grid saturado (hallazgo nuevo)

Todos los organismos de ambas ramas tienen `curiosity_grid max=1.00 cobertura=100%`. El grid 20x20 (400 celdas) esta completamente visitado. Esto tiene dos implicaciones:
1. Los organismos ya no tienen incentivo de curiosidad para explorar zonas nuevas — el mundo es "conocido"
2. En Era 24 con food_target=50 y 30 obstaculos, explorar el 100% del mapa puede ser ineficiente (gastar steps en zonas sin comida)
3. Posible causa del descenso de fitness en eras altas: los organismos no se especializan geograficamente

### Observacion: Depredacion externa vs evolucion sostenida

- Rama A con 2 depredadores: record en Gen 20 (muy temprano), nunca superado
- Rama B sin depredadores: record en Gen 143 (mas tardio pero mas alto)
- **Hipotesis**: la depredacion externa constante genera presion erratica que dificulta la acumulacion de adaptaciones a largo plazo. Sin esa presion, la seleccion puede ser mas consistente y el fitness record crece mas.

### Investigacion: Canales espontaneos (resultado)

Se reviso si los cerebros estaban usando neuronas internas como canales motores espontaneos via `_neuron_utility`. Resultado:
- Los IDs de alta utilidad (A: n349,n666,n685,n964; B: n542,n587,n627,n669) son **consistentes dentro de cada organismo** pero **no compartidos entre organismos**
- Las utilidades de las neuronas top tienen el mismo valor en un organismo (e.g., todas 3.725) — indican circuitos sincronizados, no neuronas independientes
- **Conclusion**: no hay especializacion espontanea convergente visible. Los cerebros funcionan como redes integradas, no desarrollan por si solos neuronas dedicadas al control motor como "canales virtuales"
- Esto **no descarta** la idea — puede necesitar mas generaciones o presiones mas especificas

### Decisiones post-analisis

- Modulo F (reflexos corporales `ReflexGene`) viable como siguiente implementacion
- Canales espontaneos: investigacion continua, no implementar aun
- El curiosity grid saturado sugiere evaluar grids mas finos (40x40) o resetearlos por era
- La bifurcacion de masa en Rama A (3 livianos vs 9 pesados) merece seguimiento — posible inicio de especializacion depredador/presa interna

---

## PENDIENTES / IDEAS FUTURAS

- Modulo F — `ReflexGene`: arco reflejo directo sensor→motor sin pasar por el cerebro (evasion instantanea)
- Canales espontaneos: seguir investigando en corridas mas largas (>500 gens)
- Curiosity grid: evaluar si aumentar a 40x40 o resetear por era para mantener presion de exploracion
- Visualizar con `--visual` las morfologias de Gen 194 para ver comportamiento real
- Observar si emerge depredacion real entre organismos (3 contactos sostenidos con ratio 4x)
- Observar si algun organismo deforma visiblemente un obstaculo plastico (comparar p1/p2 vs p1_orig/p2_orig)
- Observar si emerge herding: depredador empujando obstaculo para acorralar presa
- Investigar si el linaje `b-` de Rama B tiene ventaja real por morfologia compacta o por azar
- Registrar si alguna morfologia descubre locomocion oscilatoria (como pez o gusano)
- Analizar si mmap puede superar 18 (maximo observado en B) o si hay un limite natural
- Puerto a Minecraft: objetivo de largo plazo post-Modulo F
