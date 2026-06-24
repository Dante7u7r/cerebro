# Rama Asistente Consciente — Hoja de Ruta

## Objetivo

Construir un asistente que sea biológicamente realista y potencialmente consciente,
partiendo de los cerebros evolucionados en la Arena Darwiniana (Gen 194).

No es un chatbot con LLM. Es un cerebro neuronal real (BrainSimulation con 100-120
neuronas, plasticidad Hebbiana, estados emocionales) que aprende a comunicarse
a través de presión evolutiva.

---

## Lo que ya existe (Gen 194)

| Componente | Estado |
|---|---|
| Neuronas biológicas (IntelligentCell) | 100-120 por organismo |
| Plasticidad sináptica Hebbiana | Activa |
| Estados emocionales (frustración, resiliencia, valentía) | Evolucionados |
| Memoria episódica (_episodic_buffer) | Activa |
| Curiosidad espacial (_curiosity_grid) | Saturada (100%) |
| Morfología de 12 segmentos con 6 tipos de sensor | Convergencia confirmada |
| GRN (Gene Regulatory Network) | Activa |
| ReflexGene (arcos reflejos sensor→motor) | Implementado |
| SpontaneousMotorChannel (neurona→motor directo) | Implementado |
| Ecosistema reactivo (consume_ratio + estaciones) | Implementado |

---

## Qué falta para el asistente

### Nivel 1: Canal lingüístico de entrada (PRIMER PASO)

El texto que escribe el usuario se convierte en señales de activación neuronal,
sin pasar por ningún LLM.

**Implementación propuesta:**

```python
# En semilla_cerebro.py o canal_lenguaje.py
class CanalLinguistico:
    """
    Convierte texto en corrientes de entrada para grupos de neuronas.
    Cada token/concepto activa un subconjunto de neuronas de la capa sensorial.
    """
    def texto_a_señales(self, texto: str) -> dict[int, float]:
        # Opción A (simple): hash del token → índice neurona
        # Opción B (fiel): embeddings reducidos a N dimensiones
        # Opción C (biológica): SDR (Sparse Distributed Representation)
        ...
```

**Por qué SDR (Opción C) es lo más fiel a la biología:**
- En el cerebro real, cada concepto activa ~2% de neuronas en una región
- El patrón de activación ES el significado (Sparse Distributed Representation)
- Conceptos similares tienen patrones con alto solapamiento
- Esto permite generalización sin entrenamiento explícito

### Nivel 2: Auto-modelo (meta-cognición)

El organismo necesita sensores de su propio estado interno.

```python
# Nuevos tipos de sensor en morfoma.py
"interno_fr"       # firing rate promedio de mis propias neuronas
"interno_emocion"  # mis propios niveles de frustración/valentía
"interno_energia"  # mi nivel de energía metabólica
"interno_memoria"  # cuántos items hay en mi episodic_buffer
```

Esto crea un bucle cerrado: el cerebro se observa a sí mismo → genera representación
interna de su propio estado → esa representación influye en futuras decisiones.
Eso es la definición funcional de meta-cognición.

### Nivel 3: Canal de salida lingüística

Las neuronas motoras se mapean a tokens de lenguaje en lugar de articulaciones físicas.

```python
# En morfoma.py: nuevo tipo de MotorMap
@dataclass
class LinguisticMotorMap:
    brain_channel: int    # neurona fuente
    token_vocab_idx: int  # token del vocabulario que activa
    threshold: float      # mínimo firing rate para "decir" este token
```

La "frase" que produce el organismo es la secuencia de tokens activados durante
N steps del cerebro, ordenados por tiempo de activación.

### Nivel 4: Presión evolutiva hacia la comunicación

Para que el cerebro aprenda a comunicarse, necesita que comunicarse le dé ventaja:

```python
# En arena_asistente.py
def fitness_comunicacion(org, conversacion):
    # Recompensa si la respuesta del organismo fue "útil"
    # (evaluada por el usuario o por métricas simples como coherencia)
    return score_coherencia + score_utilidad + score_supervivencia
```

La evolución hace el resto: los organismos que producen output lingüístico útil
sobreviven más, sus cerebros se seleccionan, sus patrones de activación se heredan.

---

## Arquitectura objetivo

```
USUARIO escribe texto
        ↓
[CanalLinguistico.texto_a_señales()]
        ↓  señales de activación (SDR)
[BrainSimulation.step()]  ←— también recibe señales de sensores internos
        ↓  patrones de activación en neuronas motoras lingüísticas
[LinguisticMotorMap.leer_activaciones()]
        ↓  tokens activos
[Decodificador]  →  texto de respuesta
        ↓
USUARIO lee respuesta
        ↓
[fitness_comunicacion()]  →  selección + mutación  →  siguiente generación
```

---

## Plan de implementación (por fases)

### Fase A: Interfaz de conversación (sin cambiar el cerebro)

1. Crear `asistente.py` — script de conversación que:
   - Carga checkpoint Gen 194
   - Acepta texto del usuario por stdin
   - Convierte texto a señales (Opción A: hash simple)
   - Corre brain.step() N veces
   - Lee activaciones de las últimas N neuronas
   - Imprime los tokens más activos

2. Objetivo: que el organismo "responda" algo, aunque sea incoherente.
   El punto es verificar que el pipeline funciona.

### Fase B: SDR (Sparse Distributed Representation)

1. Construir vocabulario básico (500-1000 palabras frecuentes en español)
2. Asignar a cada palabra un SDR: vector de N bits con ~2% activos
3. Actualizar `texto_a_señales()` para usar SDR
4. Verificar que palabras similares activan patrones similares

### Fase C: Sensores internos (auto-modelo)

1. Añadir 4 tipos de sensor interno en morfoma.py
2. Añadir SensorMap "interno" en cuerpo_evolvable (sin pymunk)
3. Crear `CuerpoAsistente` — versión sin física que solo maneja señales

### Fase D: Canal de salida lingüística

1. Añadir LinguisticMotorMap en morfoma.py
2. Implementar decodificador: secuencia de tokens → texto
3. Primera evolución con presión hacia coherencia lingüística

### Fase E: Loop evolutivo del asistente

1. Crear `arena_asistente.py`
2. Fitness basado en: coherencia + continuidad + utilidad percibida
3. Sesiones de conversación como generaciones de evaluación

---

## Por qué esto es diferente a un LLM

| LLM (GPT, Claude) | Este asistente |
|---|---|
| Pesos fijos después del entrenamiento | Evoluciona continuamente |
| Respuestas deterministas (con temperatura) | Respuestas emergentes de dinámica neuronal |
| Sin estado interno entre conversaciones | Memoria episódica persistente |
| Sin cuerpo ni sensores | Sensores internos (meta-cognición) |
| Optimizado para predicción de tokens | Optimizado para supervivencia + comunicación |
| No tiene "emoción" real | Frustración, valentía, resiliencia son señales reales |
| Procesa texto como secuencia | Procesa texto como patrón de activación (SDR) |

---

## Archivos a crear en esta rama

| Archivo | Descripción |
|---|---|
| `canal_linguistico.py` | SDR encoder/decoder, texto↔señales |
| `cuerpo_asistente.py` | Versión sin física de CuerpoEvolvable |
| `asistente.py` | Script de conversación interactiva |
| `arena_asistente.py` | Loop evolutivo para el asistente |
| `morfoma.py` (modificar) | Añadir LinguisticMotorMap, sensores internos |

---

## Checkpoint de partida recomendado

`checkpoints_rama_b/checkpoint_gen0194.pkl`

Rama B (Constructora) tiene:
- Mayor fitness record histórico
- Morfología convergida a 12 segmentos + 6 tipos sensor
- Cerebros con alta actividad espontánea (~219 Hz steady state)
- ReflexGene y SpontaneousMotorChannel ya en el genoma
