#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisis profundo de checkpoints de IA Viva Arena Darwiniana.
Uso: python analizar_checkpoint.py <pkl_a> [pkl_b]
"""

import pickle
import sys
import os
from collections import defaultdict, Counter

_proj = os.path.dirname(os.path.abspath(__file__))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

try:
    import semilla_cerebro  # noqa: F401
    import morfoma          # noqa: F401
    import mundo_fisico     # noqa: F401
    import cuerpo_evolvable # noqa: F401
except ImportError as _ie:
    print(f"[WARN] {_ie}")

_PROJ_MODS = ["semilla_cerebro", "morfoma", "mundo_fisico",
              "cuerpo_evolvable", "arena_fase1", "depredador"]


class _Remapper(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            for mod in _PROJ_MODS:
                try:
                    m = sys.modules.get(mod) or __import__(mod)
                    if hasattr(m, name):
                        return getattr(m, name)
                except Exception:
                    pass
        return super().find_class(module, name)


def load(path):
    with open(path, "rb") as f:
        return _Remapper(f).load()


def sep(title=""):
    w = 70
    if title:
        pad = (w - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * (w - pad - len(title) - 2))
    else:
        print("=" * w)


def _cell_type_name(t):
    names = {0: "NEURON", 1: "SENSORY", 2: "GLIA", 3: "MOTOR"}
    try:
        return names.get(int(t), str(t))
    except Exception:
        return str(t)


def analizar_rama(data, label):
    sep(label)
    organismos = data.get("organismos", [])
    historial  = data.get("historial", [])
    gen        = data.get("gen", "?")
    era        = data.get("world_era", "?")
    n_food     = data.get("world_n_food", "?")
    n_obs      = data.get("world_n_obstacles", "?")

    print(f"Gen={gen}  Era={era}  Food_target={n_food}  Obstacles={n_obs}")
    print(f"Poblacion activa (proxima gen): {len(organismos)} organismos")
    print("NOTA: organismos en checkpoint = proxima generacion (aun sin correr)")

    # ── Historial completo ───────────────────────────────────────────────
    sep("Trayectoria de fitness (historial real)")
    if historial:
        all_fit = [h["best"]["fitness"] for h in historial if "best" in h]
        record_fit  = max(all_fit) if all_fit else 0
        record_gen  = next((h["gen"] for h in historial
                            if "best" in h and h["best"]["fitness"] == record_fit), "?")
        print(f"  Record absoluto: {record_fit:.2f} (Gen {record_gen})")
        print(f"  Fitness promedio ultimas 20 gens: "
              f"{sum(all_fit[-20:])/max(1,len(all_fit[-20:])):.2f}")

        print("\n  Ultimas 20 generaciones:")
        print("  Gen   Best_fit   Worst_fit  Avg_fit    Era  Ganador")
        for h in historial[-20:]:
            g    = h.get("gen", "?")
            best = h.get("best", {})
            fd   = h.get("fitness_distribution", [0])
            bf   = best.get("fitness", 0)
            wf   = min(fd) if fd else 0
            avg  = sum(fd) / len(fd) if fd else 0
            era_h= h.get("era", "?")
            nb   = best.get("nombre", "?")[:16]
            print(f"  {g:>4}  {bf:>9.2f}  {wf:>9.2f}  {avg:>9.2f}  {era_h:>3}  {nb}")

        # Tendencia de las ultimas 5 gens
        last5 = [h["best"]["fitness"] for h in historial[-5:] if "best" in h]
        if len(last5) >= 2:
            trend = last5[-1] - last5[0]
            print(f"\n  Tendencia ultimas 5 gens: {'+' if trend>=0 else ''}{trend:.2f}")

    # ── Morfologia de la poblacion ───────────────────────────────────────
    sep("Morfologia (proxima generacion — herencia directa)")
    seg_counts  = []
    joint_counts = []
    mmap_counts = []
    smap_counts = []
    sensor_hist = Counter()
    mass_totals = []
    motor_joint_counts = []

    for o in organismos:
        m = o.get("morphome")
        if m is None:
            continue
        segs   = getattr(m, "segments", [])
        joints = getattr(m, "joints", [])
        mmaps  = getattr(m, "motor_maps", [])
        smaps  = getattr(m, "sensor_maps", [])
        seg_counts.append(len(segs))
        joint_counts.append(len(joints))
        mmap_counts.append(len(mmaps))
        smap_counts.append(len(smaps))
        for sm in smaps:
            sensor_hist[getattr(sm, "sensor_type", "?")] += 1
        total_mass = sum(getattr(s, "mass", 1.0) for s in segs)
        mass_totals.append(total_mass)
        motor_j = sum(1 for j in joints if getattr(j, "has_motor", False))
        motor_joint_counts.append(motor_j)

    def stats(lst, name):
        if not lst:
            return
        mn, mx, avg = min(lst), max(lst), sum(lst) / len(lst)
        print(f"  {name}: min={mn:.1f}  avg={avg:.1f}  max={mx:.1f}")

    stats(seg_counts,         "Segmentos          ")
    stats(joint_counts,       "Articulaciones     ")
    stats(motor_joint_counts, "Joints con motor   ")
    stats(mmap_counts,        "Motor maps         ")
    stats(smap_counts,        "Sensor maps        ")
    stats(mass_totals,        "Masa total         ")

    print("\n  Distribucion segmentos:")
    for k, cnt in sorted(Counter(seg_counts).items()):
        print(f"    {k:>2} segs: {'#'*cnt} ({cnt})")

    print("\n  Sensores en la poblacion:")
    for stype, cnt in sorted(sensor_hist.items(), key=lambda x: -x[1]):
        bar = "#" * min(cnt, 50)
        print(f"    {stype:<22} {bar} x{cnt}")

    # ── Masa: bifurcacion ────────────────────────────────────────────────
    if mass_totals:
        sep("Distribucion de masas")
        buckets = defaultdict(int)
        for ms in mass_totals:
            b = int(ms // 4) * 4
            buckets[b] += 1
        for b in sorted(buckets):
            print(f"  masa [{b:>3}-{b+3}):  {'#'*buckets[b]} ({buckets[b]})")
        # Deteccion de bifurcacion (depredador/presa potencial)
        if len(mass_totals) > 3:
            avg_m = sum(mass_totals) / len(mass_totals)
            heavy = sum(1 for m in mass_totals if m > avg_m * 1.3)
            light = sum(1 for m in mass_totals if m < avg_m * 0.7)
            if heavy > 0 and light > 0:
                print(f"\n  *** BIFURCACION detectada: {heavy} pesados / {light} livianos ***")

    # ── Analisis neuronal ────────────────────────────────────────────────
    sep("Analisis neuronal (cerebros en la poblacion)")

    # Acumular estadisticas por organismo
    all_cell_counts = []
    all_neuron_counts = []
    all_glia_counts = []
    all_ep_lens = []
    all_frust = []
    all_resil = []
    all_courage = []
    shared_high_utility = Counter()  # neuronas con alta utilidad en multiples organismos

    for o in organismos:
        cerebro = o.get("cerebro", {})
        cells   = cerebro.get("cells", [])
        all_cell_counts.append(len(cells))
        neurons = [c for c in cells if _cell_type_name(getattr(c, "type", -1)) in ("NEURON", "SENSORY", "MOTOR")]
        glia    = [c for c in cells if _cell_type_name(getattr(c, "type", -1)) == "GLIA"]
        all_neuron_counts.append(len(neurons))
        all_glia_counts.append(len(glia))

        ep = cerebro.get("_episodic_buffer", [])
        all_ep_lens.append(len(ep))
        all_frust.append(cerebro.get("frustration", 0))
        all_resil.append(cerebro.get("resilience", 0))
        all_courage.append(cerebro.get("courage", 0))

        nu = cerebro.get("_neuron_utility", {})
        if nu:
            median_u = sorted(nu.values())[len(nu)//2]
            for nid, u in nu.items():
                if u > median_u * 2:  # mucho mas util que la mediana
                    shared_high_utility[nid] += 1

    def stats_f(lst, name):
        if not lst:
            return
        mn, mx, avg = min(lst), max(lst), sum(lst)/len(lst)
        print(f"  {name}: min={mn:.1f}  avg={avg:.1f}  max={mx:.1f}")

    stats_f(all_cell_counts,   "Celulas totales    ")
    stats_f(all_neuron_counts, "Neuronas (excl.glia)")
    stats_f(all_glia_counts,   "Celulas glia       ")
    stats_f(all_ep_lens,       "Episodios memoria  ")

    if all_frust:
        avg_f = sum(all_frust)/len(all_frust)
        avg_r = sum(all_resil)/len(all_resil)
        avg_c = sum(all_courage)/len(all_courage)
        print(f"\n  Estado emocional poblacion:")
        print(f"    frustration avg={avg_f:.3f}  resilience avg={avg_r:.3f}  courage avg={avg_c:.3f}")
        if avg_f > 0.8:
            print("    *** Frustration ALTA — mundo muy dificil para las capacidades actuales")
        if avg_r > 0.8:
            print("    *** Resilience alta — organismos persistentes frente a adversidad")

    # Neuronas con alta utilidad compartida (especializacion convergente)
    convergent = [(nid, cnt) for nid, cnt in shared_high_utility.items()
                  if cnt >= len(organismos) // 2]
    if convergent:
        print(f"\n  *** ESPECIALIZACION CONVERGENTE: {len(convergent)} neuronas con")
        print(f"      alta utilidad en >= {len(organismos)//2} organismos:")
        for nid, cnt in sorted(convergent, key=lambda x: -x[1])[:10]:
            print(f"      neurona_id={nid}  presente en {cnt}/{len(organismos)} cerebros")
    else:
        print("\n  Especializacion neuronal: no hay convergencia clara (neuronas diversas)")

    # Top 3 organismos: detalle neuronal
    print("\n  Detalle top 3 (por posicion en checkpoint, herederos de los mejores):")
    for o in organismos[:3]:
        nombre  = o.get("nombre", "?")
        padre   = o.get("parent", "?")
        cerebro = o.get("cerebro", {})
        cells   = cerebro.get("cells", [])
        nu      = cerebro.get("_neuron_utility", {})
        ep      = len(cerebro.get("_episodic_buffer", []))
        frust   = cerebro.get("frustration", 0)
        resil   = cerebro.get("resilience", 0)
        brain_n = cerebro.get("name", "?")
        # Distribucion de tipos celulares
        type_dist = Counter(_cell_type_name(getattr(c, "type", -1)) for c in cells)
        top_nu = sorted(nu.items(), key=lambda x: -x[1])[:5] if nu else []
        # Curiosity grid hotspot
        cgrid = cerebro.get("_curiosity_grid", None)
        hotspot = ""
        if cgrid is not None:
            try:
                import numpy as np
                arr = np.array(cgrid)
                hv = float(arr.max())
                hi = divmod(int(arr.argmax()), arr.shape[1])
                coverage = float((arr > 0).sum()) / arr.size
                hotspot = f"grid_max={hv:.2f}@{hi} cobertura={coverage:.0%}"
            except Exception:
                pass

        print(f"\n    [{nombre}] padre={padre}  brain={brain_n}")
        print(f"      celulas: {dict(type_dist)}  episodios={ep}")
        print(f"      emociones: frust={frust:.3f} resil={resil:.3f}")
        if top_nu:
            top_str = "  ".join(f"n{nid}:{u:.3f}" for nid, u in top_nu)
            print(f"      top utility: {top_str}")
        if hotspot:
            print(f"      curiosity: {hotspot}")

    sep()
    print()


def comparar_ramas(datasets, labels):
    sep("COMPARATIVA ENTRE RAMAS")
    hist_a = datasets[0].get("historial", [])
    hist_b = datasets[1].get("historial", [])
    gens_a = {h.get("gen"): h for h in hist_a}
    gens_b = {h.get("gen"): h for h in hist_b}
    all_gens = sorted(set(gens_a) | set(gens_b))

    print(f"{'Gen':>4}  {'BestFit_A':>10}  {'BestFit_B':>10}  {'Delta':>8}  Era")
    print("-" * 50)
    for g in all_gens[-30:]:
        ha = gens_a.get(g, {})
        hb = gens_b.get(g, {})
        fa = ha.get("best", {}).get("fitness", 0)
        fb = hb.get("best", {}).get("fitness", 0)
        delta = fb - fa
        era   = ha.get("era", hb.get("era", "?"))
        marker = " <--B" if delta > 50 else (" <--A" if delta < -50 else "")
        print(f"{g:>4}  {fa:>10.2f}  {fb:>10.2f}  {delta:>+8.2f}  {era}{marker}")

    # Records absolutos
    print()
    all_a = [h["best"]["fitness"] for h in hist_a if "best" in h]
    all_b = [h["best"]["fitness"] for h in hist_b if "best" in h]
    if all_a and all_b:
        ra = max(all_a)
        rb = max(all_b)
        ga = hist_a[all_a.index(ra)].get("gen", "?") if all_a else "?"
        gb = hist_b[all_b.index(rb)].get("gen", "?") if all_b else "?"
        print(f"  Record {labels[0]}: {ra:.2f} (Gen {ga})")
        print(f"  Record {labels[1]}: {rb:.2f} (Gen {gb})")
        winner = labels[0] if ra > rb else labels[1]
        print(f"  Ganador: {winner}")
    sep()


def main():
    paths = sys.argv[1:]
    if not paths:
        print("Uso: python analizar_checkpoint.py <pkl_a> [pkl_b]")
        sys.exit(1)

    labels = ["RAMA A (Depredadora)", "RAMA B (Constructora)"]
    datasets = []
    for i, path in enumerate(paths):
        label = labels[i] if i < len(labels) else f"RAMA {i+1}"
        print(f"\nCargando {path} ...")
        try:
            d = load(path)
            datasets.append(d)
        except Exception as e:
            print(f"  ERROR: {e}")
            datasets.append({})
            continue
        analizar_rama(d, label)

    if len(datasets) >= 2 and datasets[0] and datasets[1]:
        comparar_ramas(datasets, labels[:2])


if __name__ == "__main__":
    main()
