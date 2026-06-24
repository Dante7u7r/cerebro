# AGENTS.md - IA Viva

## Rama Original (semilla_cerebro.py)

```bash
python semilla_cerebro.py
```

Requiere: `numpy`, `matplotlib`.

Simulación 2D con 4 agentes autónomos (Alfa, Beta, Gamma, Delta) evolucionando arquitecturas cerebrales.

## Rama Brian2 (cerebro_brian2.py) — Activa

```bash
python cerebro_brian2.py
```

Requiere: `numpy`, **`brian2`** (`pip install brian2`).

**11 experimentos** de simulación neuronal con LIF nativo de Brian2 y STDP:

| Exp | Nombre | Pasos | Spikes | Propósito |
|-----|--------|-------|--------|-----------|
| 1 | Dopamina + Novedad | 180 | ~16k | DA responde a estímulos nuevos (gradiente suave) |
| 2 | Frustración | 240 | ~19k | Fru crece +0.16/ronda, recompensa la reduce a 0.5 |
| 3 | Ciclo sueño | 91 | ~7k | AWAKE→SWS→REM→AWAKE, buffer con 51 entradas |
| 4 | Plasticidad STDP | 260 | ~23k | Pesos +392% con estimulación repetida |
| 5 | Ruido Gaussiano | 70 | ~5.6k | Actividad 3x con ruido, energía cae → se recupera |
| 6 | Escalado (30 neuronas) | 40 | ~43k | 870 sinapsis, rendimiento 10s Cython |
| 7 | Secuencia temporal | 220 | ~26k | Red prefiere inversa (efecto de recencia STDP) |
| 8 | Privación sueño | 105 c/u | 9.5k/11k | Sin sueño: energía -41%, buffer compensa |
| 9 | Emoción vs memoria | 95 c/u | 8.5k/7.1k | Frustración consolida más que recompensa |
| 10 | Período crítico | 260 | ~24k | **Plasticidad 3.5x mayor en fase temprana** |
| 11 | Incertidumbre | 155 | ~21k | **DA correlaciona +0.65 con ruido, energia -60%** |

> Ver `INFORME_EXPERIMENTOS.md` para análisis detallado de cada experimento.

### Detalles técnicos

- **Batch**: 500ms por paso (`network.run(500*ms)`)
- **Spiking real**: I_ext = 15 de amplitud, v_thresh = -55 a -45 según subtipo
- **STDP**: on_pre/on_post nativo, Apre = 0.05, Apost ~ -0.0525
- **Neuromodulación**: Python-level (no Brian2), actualiza cada paso (500ms)
- **Monitoreo**: SpikeMonitor + StateMonitor (5 neuronas grabadas)
- **Logs**: CSV con spikes y neuromodulación en `logs_brian2/exp{N}/`
- **Persistencia**: Pickle del cerebro en `cerebro_brian2_save_{nombre}.pkl`
- **target**: `'cython'` (Cython 3.1.3, 2.4x más rápido que numpy; requiere `python3-dev` para compilar)

### Mecanismos biológicos conservados

- LIF con constante de tiempo por subtipo
- STDP asimétrico (potenciación pre→post, depresión post→pre)
- Ciclo sueño-vigilia (70% AWAKE, 20% SWS, 10% REM)
- Dopamina, serotonina, acetilcolina como neuromoduladores
- Frustración (error acumulado), resiliencia (recuperación), valentía (exploración)

## Archivos

| Archivo | Propósito |
|---------|-----------|
| `semilla_cerebro.py` | Simulación original 2D (~1965 líneas) |
| `semilla_cerebro_original.py` | Backup del original |
| `cerebro_brian2.py` | **Plataforma experimental activa** (1100 líneas, 11 experimentos) |
| `INFORME_EXPERIMENTOS.md` | Informe científico detallado de los 11 experimentos |
| `analyze_brain.py` | Analizador de brain_save.pkl |
| `arena_save_*.pkl` | Estado cerebral de simulación original |
| `cerebro_brian2_save_*.pkl` | Estado cerebral Brian2 |
| `mundo_arena.pkl` | Estado del mundo compartido |
| `logs_brian2/exp{N}/` | CSVs de cada experimento |
| `logs/` | Logs de experimentos Python puro |
| `*.md` | Documentación |

## Errores conocidos

- `numpy.min` secuestra a `builtins.min` cuando se usan arrays de numpy — usar ternarios o `np.clip` explícito
- `xi` (ruido blanco) solo funciona en ecuaciones diferenciales, no en subexpresiones
- Brian2 con `target='numpy'` es lento (~0.3s por step de 500ms); con `target='cython'` es 2.4x más rápido
- La memoria episódica requiere DA > 0.70 para registrar — rara vez se alcanza naturalmente
- No hay suite de pruebas ni CI — verificar cambios ejecutando `python cerebro_brian2.py`
