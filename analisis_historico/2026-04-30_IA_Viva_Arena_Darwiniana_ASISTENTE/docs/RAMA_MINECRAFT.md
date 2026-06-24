# Rama Minecraft — Estado y Continuación

## Qué se construyó

Un puente entre los cerebros evolucionados (.pkl) y un bot de Minecraft, usando:
- **Mineflayer 4.37** (Node.js) como cliente del juego
- **cerebro_minecraft.py** (Python) que carga el cerebro y lo conecta via socket TCP

### Arquitectura

```
[cerebro_minecraft.py]  ←— socket TCP 127.0.0.1:9876 —→  [bot.js / Mineflayer]
  carga .pkl                                                 vive en Minecraft
  restaura BrainSimulation                                  lee sensores del mundo
  inyecta señales sensor                                    ejecuta acciones
  llama brain.step()
  envía acción
```

### Protocolo socket (JSON lines)

```
Python → JS:  { "action": "forward"|"back"|"left"|"right"|"jump"|"attack"|"stop" }
JS → Python:  { "sensors": { contact, proximity, pain, light, taste, y, vx, vy, vz } }
```

---

## Cómo ejecutarlo

### Requisitos
- Node.js 18+
- Minecraft Java Edition 1.21.x (versión 1.21.11 confirmada)
- Python 3.12 + dependencias del proyecto

### Pasos

```bash
# 1. Instalar dependencias del bot
cd minecraft_bridge
npm install

# 2. Abrir Minecraft 1.21.11, crear mundo, abrir LAN en puerto 25565

# 3. Arrancar el bot (Terminal 1)
node bot.js 127.0.0.1 25565 Cerebro_IA 1.21.11

# 4. Arrancar el cerebro (Terminal 2)
cd ..
python minecraft_bridge/cerebro_minecraft.py --checkpoint checkpoints_rama_b/checkpoint_gen0194.pkl

# Opción: elegir organismo específico
python minecraft_bridge/cerebro_minecraft.py --checkpoint checkpoints_rama_b/checkpoint_gen0194.pkl --org 3
```

---

## Estado actual (30 Abril 2026)

| Componente | Estado |
|---|---|
| Bot conecta a Minecraft | OK |
| Socket Python↔JS | OK |
| Carga de cerebro desde .pkl | OK |
| Inyección de sensores | OK |
| Brain.step() corre | OK (con atributos faltantes parcheados) |
| Movimiento real en Minecraft | Bot se mueve, comportamiento básico |

### Atributos faltantes del pkl (ya parcheados en restore_brain)
El BrainSimulation serializado no tiene algunos atributos añadidos después de la serialización.
`restore_brain()` en cerebro_minecraft.py los inicializa con valores por defecto:
- `lfp_electrode_pos`, `lfp_history`
- `prediction_history`, `history`, `max_history`
- `neuromod`, `error_history`, `target_signal_history`
- `synapses`, `astrocytes`, `oligos`, `microglia_cells`

---

## Problema fundamental (por qué se abandonó esta rama)

El cerebro fue evolucionado para sobrevivir en una arena 2D con pymunk. Sus neuronas
aprendieron a controlar articulaciones físicas para comer comida. En Minecraft:

1. **No hay señal de recompensa** → el cerebro no puede aprender nada nuevo
2. **El mapeo sensor/motor es arbitrario** → las acciones no tienen relación con lo que el cerebro aprendió
3. **Sin loop evolutivo** → los organismos no mejoran entre generaciones

Lo que se necesita para que esto funcione de verdad es portar el loop evolutivo completo:

---

## Trabajo pendiente para continuar

### Paso 1: arena_minecraft.py (el más importante)

Equivalente de `arena_fase1.py` pero usando Mineflayer como física.

```python
# Pseudocódigo
for gen in range(N_GENS):
    for org in poblacion:
        bot = crear_bot(org)
        fitness = 0
        for step in range(3000):   # ~60 segundos a 20Hz
            sensores = bot.leer_sensores()
            org.cerebro.step()
            accion = calcular_accion(org)
            bot.ejecutar(accion)
            fitness += calcular_reward(sensores, org)
        org.fitness = fitness
        bot.desconectar()
    poblacion = seleccion_mutacion(poblacion)
    guardar_checkpoint(poblacion, gen)
```

### Paso 2: Definir fitness en Minecraft

Opciones:
- Distancia máxima recorrida desde spawn
- Comida recogida del suelo
- Tiempo de supervivencia contra mobs
- Combinación de los anteriores

### Paso 3: Múltiples bots simultáneos

Para acelerar la evolución, ejecutar N bots en N servidores locales (Paper headless)
en paralelo. Cada servidor en un puerto diferente (25565, 25566, ...).

### Paso 4: Re-mapeo fiel de sensores

Conectar los tipos de sensor del morfoma (contact, proximity, pain, light, taste, chemical)
a observaciones reales de Minecraft vía mineflayer:
- `contact` → colisión con bloque/entidad
- `proximity` → mob más cercano
- `pain` → cambio en bot.health
- `light` → world.getBlock(pos).light
- `taste` → items de comida en radio 5 bloques
- `chemical` → bioma actual (agua, lava, aire)

---

## Archivos de esta rama

```
minecraft_bridge/
  bot.js                  — Bot Mineflayer (Node.js)
  cerebro_minecraft.py    — Controlador Python del cerebro
  package.json            — Dependencias npm
  package-lock.json       — Lock file npm
  (node_modules/ — hacer npm install)
```

---

## Referencias

- Mineflayer docs: https://github.com/PrismarineJS/mineflayer
- minecraft-data versiones soportadas: node -e "console.log(Object.keys(require('minecraft-data/data.js').pc).slice(-5))"
