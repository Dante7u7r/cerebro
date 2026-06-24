#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BITÁCORA DE IA VIVA — ARENA DARWINIANA
Detecta logros evolutivos y actualiza BITACORA.md automáticamente.
Se puede llamar directamente o desde semilla_cerebro.py en cada auto-guardado.
"""

import os, sys, json, pickle
from datetime import datetime
import numpy as np

ESTADO_JSON  = "bitacora_estado.json"
BITACORA_MD  = "BITACORA.md"
PKL_FILES    = {
    "Alfa":  "arena_save_alfa.pkl",
    "Beta":  "arena_save_beta.pkl",
    "Gamma": "arena_save_gamma.pkl",
    "Delta": "arena_save_delta.pkl",
}

# ── Cargar clases de semilla_cerebro en __main__ para que pickle funcione ─────
# pickle busca las clases en el módulo donde fueron serializadas (__main__),
# así que hay que inyectarlas en globals() del script actual.
_clases_cargadas = False

def _cargar_clases():
    global _clases_cargadas
    if _clases_cargadas:
        return globals()
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "semilla_cerebro.py")
    with open(ruta, "r", encoding="utf-8") as f:
        src = f.read()
    src = src.replace('if __name__ == "__main__":', "if False:")
    exec(compile(src, ruta, "exec"), globals())
    _clases_cargadas = True
    return globals()

# ── Extraer métricas de un pkl ────────────────────────────────────────────────
def _metricas(data, ns):
    CellType    = ns["CellType"]
    NeuronSubtype = ns["NeuronSubtype"]
    cells   = data.get("cells", [])
    t       = data.get("time", 0)
    steps   = data.get("step_count", 0)
    neurons = [c for c in cells if c.type == CellType.NEURON]
    glia    = [c for c in cells if c.type == CellType.GLIA]
    stems   = [c for c in cells if c.type == CellType.STEM]

    exc   = sum(1 for n in neurons if n.subtype == NeuronSubtype.EXCITATORY)
    inh   = len(neurons) - exc
    ei_pct = exc / max(len(neurons), 1) * 100

    weights   = [s.weight      for n in neurons for s in n.synapses_in]
    myelins   = [s.myelination for n in neurons for s in n.synapses_in]
    total_syn = len(weights)
    density   = total_syn / max(len(neurons), 1)
    strong_pct = np.mean(np.array(weights) > 0.5) * 100 if weights else 0.0
    myel_pct   = np.mean(np.array(myelins) > 0.5) * 100 if myelins else 0.0
    mean_weight = float(np.mean(weights)) if weights else 0.0

    ep_buf  = data.get("_episodic_buffer", None)
    utility = data.get("_neuron_utility", {})
    live_ids = {c.id for c in cells if c.type == CellType.NEURON}
    ghost_pct = (sum(1 for k in (utility or {}) if k not in live_ids)
                 / max(len(utility or {}), 1) * 100)

    layer_s = sum(1 for n in neurons if n.pos[2] < 15)
    layer_m = sum(1 for n in neurons if 15 <= n.pos[2] < 35)
    layer_mo= sum(1 for n in neurons if 35 <= n.pos[2] < 55)
    layer_p = sum(1 for n in neurons if n.pos[2] >= 55)

    return {
        "tiempo_s":    round(t / 1000, 2),
        "steps":       steps,
        "neuronas":    len(neurons),
        "glia":        len(glia),
        "stem":        len(stems),
        "excitatorias":exc,
        "inhibitorias":inh,
        "ei_pct":      round(ei_pct, 1),
        "sinapsis":    total_syn,
        "densidad":    round(density, 2),
        "strong_pct":  round(strong_pct, 1),
        "myel_pct":    round(myel_pct, 1),
        "mean_weight": round(mean_weight, 4),
        "frustr":      round(data.get("frustration", 0), 3),
        "resil":       round(data.get("resilience", 0), 3),
        "valor":       round(data.get("courage", 0), 3),
        "memorias":    len(ep_buf) if ep_buf else 0,
        "ghost_pct":   round(ghost_pct, 1),
        "L_sensor":    layer_s,
        "L_memoria":   layer_m,
        "L_motor":     layer_mo,
        "L_pfc":       layer_p,
    }

# ── Definiciones de logros ────────────────────────────────────────────────────
def _detectar_logros(nombre, actual, prev_records):
    """Compara métricas actuales contra records históricos y devuelve logros nuevos."""
    logros = []
    ts     = datetime.now().isoformat(timespec="seconds")

    def logro(tipo, titulo, desc, extra=None):
        entry = {
            "id":          f"{nombre.lower()}_{tipo}_{ts}",
            "timestamp":   ts,
            "cerebro":     nombre,
            "tipo":        tipo,
            "titulo":      titulo,
            "descripcion": desc,
            "metricas":    actual,
        }
        if extra:
            entry.update(extra)
        logros.append(entry)

    pr = prev_records.get(nombre, {})

    # ── E/I ratio: primera vez excitatorio-dominante ──────────────────────────
    if actual["ei_pct"] > 50 and pr.get("ei_pct_max", 0) <= 50:
        logro("ei_flip",
              "🔀 Primer Giro E/I: Red Excitatorio-Dominante",
              f"Por primera vez en la historia de {nombre}, las neuronas excitatorias "
              f"superan a las inhibitorias ({actual['ei_pct']:.1f}% excitatorias). "
              f"Corrección espontánea de un desequilibrio estructural crónico.")

    # ── E/I balance biológico (>70% excitatorio) ─────────────────────────────
    if actual["ei_pct"] >= 70 and pr.get("ei_pct_max", 0) < 70:
        logro("ei_biologico",
              "🧬 Balance E/I Biológico Alcanzado (>70% Excitatorio)",
              f"{nombre} alcanzó un ratio excitatorio del {actual['ei_pct']:.1f}%, "
              f"dentro del rango de la corteza biológica real (~80/20).")

    # ── Nuevo record de densidad sináptica ────────────────────────────────────
    if actual["densidad"] > pr.get("densidad_max", 0) + 1.0:
        logro("densidad_record",
              f"🔗 Nuevo Record de Densidad Sináptica: {actual['densidad']:.1f} syn/neurona",
              f"{nombre} alcanzó {actual['densidad']:.1f} sinapsis por neurona "
              f"(anterior: {pr.get('densidad_max', 0):.1f}). "
              f"Red más conectada que nunca.")

    # ── Nuevo record de consolidación de pesos ────────────────────────────────
    if actual["strong_pct"] > pr.get("strong_pct_max", 0) + 5.0:
        logro("consolidacion_record",
              f"💪 Record de Consolidación: {actual['strong_pct']:.1f}% Pesos Fuertes",
              f"{nombre} tiene el {actual['strong_pct']:.1f}% de sus sinapsis con "
              f"peso >0.5, superando el máximo anterior de "
              f"{pr.get('strong_pct_max', 0):.1f}%.")

    # ── Poda masiva (pierde >40% neuronas) ───────────────────────────────────
    prev_n = pr.get("neuronas_ultimo", actual["neuronas"])
    if prev_n > 10 and actual["neuronas"] < prev_n * 0.6:
        drop = round((1 - actual["neuronas"] / prev_n) * 100, 1)
        logro("poda_masiva",
              f"✂️ Poda Masiva: -{drop}% de Neuronas",
              f"{nombre} se podó de {prev_n} a {actual['neuronas']} neuronas "
              f"({drop}% eliminado). Si la densidad aumentó, es evolución eficiente.")

    # ── Explosión de crecimiento (triplica neuronas) ──────────────────────────
    if actual["neuronas"] > prev_n * 3 and prev_n > 3:
        multi = round(actual["neuronas"] / prev_n, 1)
        logro("explosion",
              f"💥 Explosión Neuronal: x{multi} en una sesión",
              f"{nombre} pasó de {prev_n} a {actual['neuronas']} neuronas ({multi}x). "
              f"Brote masivo de neurogénesis.")

    # ── Nuevo record de mielinización ─────────────────────────────────────────
    if actual["myel_pct"] > pr.get("myel_pct_max", 0) + 3.0:
        logro("mielinizacion_record",
              f"⚡ Nuevo Record de Mielinización: {actual['myel_pct']:.1f}%",
              f"{nombre} alcanzó {actual['myel_pct']:.1f}% de sinapsis altamente "
              f"mielinizadas, acelerando las rutas más usadas.")

    # ── Utilidad limpia por primera vez ──────────────────────────────────────
    if actual["ghost_pct"] < 5.0 and pr.get("ghost_pct_min", 100) >= 20.0:
        logro("utility_limpia",
              "🧹 Selección Darwiniana Purificada",
              f"{nombre} redujo sus IDs fantasma en _neuron_utility a "
              f"{actual['ghost_pct']:.1f}%. La selección opera ahora sobre "
              f"la red real, no sobre generaciones muertas.")

    # ── Frustración superada (>0.7 → <0.1) ───────────────────────────────────
    if actual["frustr"] < 0.1 and pr.get("frustr_ultimo", 0) > 0.7:
        logro("frustracion_superada",
              "🌅 Frustración Superada",
              f"{nombre} cayó de frustración {pr.get('frustr_ultimo', 0):.2f} "
              f"a {actual['frustr']:.2f}. El agente encontró una estrategia viable.")

    # ── PFC desarrollada por primera vez (>20 neuronas prefrontales) ──────────
    if actual["L_pfc"] >= 20 and pr.get("L_pfc_max", 0) < 20:
        logro("pfc_desarrollada",
              f"🎯 Corteza Prefrontal Establecida: {actual['L_pfc']} neuronas",
              f"{nombre} desarrolló {actual['L_pfc']} neuronas en la capa PFC (Z>55), "
              f"consolidando capacidad supervisora y de control ejecutivo.")

    # ── Balance perfecto E/I (45–55%) ─────────────────────────────────────────
    if 45 <= actual["ei_pct"] <= 55 and not pr.get("tuvo_balance_perfecto", False):
        logro("balance_perfecto",
              f"⚖️ Balance E/I Perfecto ({actual['ei_pct']:.1f}% excitatorio)",
              f"{nombre} alcanzó un balance casi exacto entre neuronas excitatorias "
              f"e inhibitorias. Raro y valioso.")

    return logros

# ── Actualizar records tras análisis ─────────────────────────────────────────
def _actualizar_records(nombre, actual, records):
    r = records.setdefault(nombre, {})
    r["ei_pct_max"]        = max(r.get("ei_pct_max", 0),        actual["ei_pct"])
    r["densidad_max"]      = max(r.get("densidad_max", 0),       actual["densidad"])
    r["strong_pct_max"]    = max(r.get("strong_pct_max", 0),     actual["strong_pct"])
    r["myel_pct_max"]      = max(r.get("myel_pct_max", 0),       actual["myel_pct"])
    r["ghost_pct_min"]     = min(r.get("ghost_pct_min", 100),    actual["ghost_pct"])
    r["neuronas_max"]      = max(r.get("neuronas_max", 0),       actual["neuronas"])
    r["neuronas_ultimo"]   = actual["neuronas"]
    r["frustr_ultimo"]     = actual["frustr"]
    r["L_pfc_max"]         = max(r.get("L_pfc_max", 0),          actual["L_pfc"])
    if actual["ei_pct"] >= 45 and actual["ei_pct"] <= 55:
        r["tuvo_balance_perfecto"] = True

# ── Generar BITACORA.md ───────────────────────────────────────────────────────
def _generar_md(estado):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snap = estado.get("ultimo_snapshot", {})
    logros = estado.get("logros", [])
    records = estado.get("records", {})

    lineas = [
        "# 🧠 Bitácora — IA Viva Arena Darwiniana",
        "",
        f"> Última actualización: **{ts}**",
        "",
        "---",
        "",
        "## 📊 Estado Actual del Clan",
        "",
        "| Cerebro | Neuronas | Sinapsis | Densidad | E/I% | Fuertes | Mielina | Frustr | Memorias |",
        "|---------|----------|----------|----------|------|---------|---------|--------|----------|",
    ]
    for nombre in ["Alfa", "Beta", "Gamma", "Delta"]:
        m = snap.get(nombre)
        if not m:
            lineas.append(f"| {nombre} | — | — | — | — | — | — | — | — |")
            continue
        lineas.append(
            f"| **{nombre}** | {m['neuronas']} | {m['sinapsis']} | {m['densidad']:.1f}/n "
            f"| {m['ei_pct']:.0f}% exc | {m['strong_pct']:.1f}% | {m['myel_pct']:.1f}% "
            f"| {m['frustr']:.2f} | {m['memorias']:,} |"
        )

    lineas += ["", "### Capas Funcionales", "",
               "| Cerebro | Sensor (Z<15) | Memoria (15-35) | Motor (35-55) | PFC (Z>55) | Steps |",
               "|---------|--------------|-----------------|--------------|------------|-------|"]
    for nombre in ["Alfa", "Beta", "Gamma", "Delta"]:
        m = snap.get(nombre)
        if not m:
            lineas.append(f"| {nombre} | — | — | — | — | — |")
            continue
        lineas.append(
            f"| **{nombre}** | {m['L_sensor']} | {m['L_memoria']} | {m['L_motor']} "
            f"| {m['L_pfc']} | {m['steps']:,} |"
        )

    # Records históricos
    lineas += ["", "---", "", "## 🏅 Records Históricos", ""]
    for nombre in ["Alfa", "Beta", "Gamma", "Delta"]:
        r = records.get(nombre, {})
        if not r:
            continue
        def fmt(v): return f"{v:.1f}" if isinstance(v, (int, float)) else "—"
        lineas += [
            f"**{nombre}:** "
            f"max neuronas={r.get('neuronas_max', '—')}  |  "
            f"max densidad={fmt(r.get('densidad_max'))}/n  |  "
            f"max fuertes={fmt(r.get('strong_pct_max'))}%  |  "
            f"max mielina={fmt(r.get('myel_pct_max'))}%  |  "
            f"max E/I={fmt(r.get('ei_pct_max'))}%",
            ""
        ]

    # Logros
    lineas += ["---", "", f"## 🏆 Logros Detectados ({len(logros)} total)", ""]
    if not logros:
        lineas.append("*Aún no se han detectado logros. Sigue corriendo la simulación.*")
    else:
        for lg in reversed(logros):  # más recientes primero
            fecha = lg["timestamp"][:16].replace("T", " ")
            lineas += [
                f"### {lg['titulo']}",
                f"**Cerebro:** {lg['cerebro']}  |  **Fecha:** {fecha}  |  **Tipo:** `{lg['tipo']}`",
                "",
                lg["descripcion"],
                "",
                "**Métricas al momento del logro:**",
                (f"- Neuronas: {lg['metricas']['neuronas']} "
                 f"({lg['metricas']['excitatorias']} exc / {lg['metricas']['inhibitorias']} inh)"),
                f"- Sinapsis: {lg['metricas']['sinapsis']} ({lg['metricas']['densidad']:.1f}/n)",
                f"- Pesos fuertes: {lg['metricas']['strong_pct']:.1f}%  |  Mielinización: {lg['metricas']['myel_pct']:.1f}%",
                f"- Frustración: {lg['metricas']['frustr']:.3f}  |  Resiliencia: {lg['metricas']['resil']:.3f}",
                "",
                "---",
                "",
            ]

    lineas += [
        "## 📈 Historial de Snapshots",
        "",
        "| Fecha | Cerebro | Neuronas | Densidad | E/I% | Fuertes |",
        "|-------|---------|----------|----------|------|---------|",
    ]
    for snap_h in reversed(estado.get("snapshots", [])[-50:]):  # últimos 50
        fecha = snap_h["timestamp"][:16].replace("T", " ")
        for nombre in ["Alfa", "Beta", "Gamma", "Delta"]:
            m = snap_h.get(nombre)
            if not m:
                continue
            lineas.append(
                f"| {fecha} | {nombre} | {m['neuronas']} | {m['densidad']:.1f} "
                f"| {m['ei_pct']:.0f}% | {m['strong_pct']:.1f}% |"
            )

    return "\n".join(lineas) + "\n"

# ── Función principal ─────────────────────────────────────────────────────────
def actualizar(silent=False):
    ts = datetime.now().isoformat(timespec="seconds")

    # Cargar estado previo
    if os.path.exists(ESTADO_JSON):
        with open(ESTADO_JSON, "r", encoding="utf-8") as f:
            estado = json.load(f)
    else:
        estado = {"logros": [], "records": {}, "snapshots": []}

    # Cargar clases para unpickle
    try:
        ns = _cargar_clases()
    except Exception as e:
        if not silent:
            print(f"[BITACORA] Error cargando semilla_cerebro: {e}")
        return

    # Analizar cada cerebro
    snapshot_actual = {"timestamp": ts}
    nuevos_logros = []

    for nombre, pkl_path in PKL_FILES.items():
        if not os.path.exists(pkl_path):
            continue
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            m = _metricas(data, ns)
            snapshot_actual[nombre] = m

            # Detectar logros
            lg = _detectar_logros(nombre, m, estado["records"])
            nuevos_logros.extend(lg)

            # Actualizar records
            _actualizar_records(nombre, m, estado["records"])

        except Exception as e:
            if not silent:
                print(f"[BITACORA] Error leyendo {pkl_path}: {e}")

    # Guardar snapshot
    estado["snapshots"].append(snapshot_actual)
    if len(estado["snapshots"]) > 200:  # cap en 200 snapshots historicos
        estado["snapshots"] = estado["snapshots"][-200:]
    estado["ultimo_snapshot"] = {k: v for k, v in snapshot_actual.items() if k != "timestamp"}

    # Agregar nuevos logros
    for lg in nuevos_logros:
        estado["logros"].append(lg)
        if not silent:
            titulo_safe = lg['titulo'].encode('ascii', errors='replace').decode('ascii')
            print(f"[BITACORA] *** NUEVO LOGRO *** {lg['cerebro']}: {titulo_safe}")

    # Guardar estado
    with open(ESTADO_JSON, "w", encoding="utf-8") as f:
        json.dump(estado, f, ensure_ascii=False, indent=2)

    # Generar BITACORA.md
    md = _generar_md(estado)
    with open(BITACORA_MD, "w", encoding="utf-8") as f:
        f.write(md)

    if not silent:
        print(f"[BITACORA] Actualizada - {len(nuevos_logros)} logros nuevos. Ver {BITACORA_MD}")

if __name__ == "__main__":
    silent = "--silent" in sys.argv
    actualizar(silent=silent)
