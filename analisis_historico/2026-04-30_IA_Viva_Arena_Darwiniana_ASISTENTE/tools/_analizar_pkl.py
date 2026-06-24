"""Analiza los .pkl guardados de la Arena Darwiniana."""
import sys, os, pickle
import numpy as np
from collections import Counter

# Los pkl fueron guardados con __main__ como módulo, así que corremos directamente
# el contenido de semilla_cerebro en __main__ para que unpickle funcione.
# Hacemos exec del script completo para poblar el namespace de __main__.
with open('semilla_cerebro.py', 'r', encoding='utf-8') as f:
    src = f.read()

# Sustituir el bloque __main__ para que no arranque la simulación
src = src.replace("if __name__ == \"__main__\":", "if False:")
exec(compile(src, 'semilla_cerebro.py', 'exec'), globals())

files = ['arena_save_alfa.pkl', 'arena_save_beta.pkl',
         'arena_save_gamma.pkl', 'arena_save_delta.pkl']

for fname in files:
    if not os.path.exists(fname):
        print(f"\n[SKIP] {fname} no encontrado")
        continue
    size_mb = os.path.getsize(fname) / 1e6
    print(f"\n{'='*60}")
    print(f"  {fname}  ({size_mb:.2f} MB)")
    print(f"{'='*60}")
    try:
        with open(fname, 'rb') as f:
            data = pickle.load(f)

        cells       = data.get('cells', [])
        t           = data.get('time', 0)
        steps       = data.get('step_count', 0)
        frustration = data.get('frustration', 0)
        resilience  = data.get('resilience', 0)
        courage     = data.get('courage', 0)
        ep_buf      = data.get('_episodic_buffer', None)
        utility     = data.get('_neuron_utility', {})
        grid        = data.get('_curiosity_grid', None)

        neurons = [c for c in cells if c.type == CellType.NEURON]
        glia    = [c for c in cells if c.type == CellType.GLIA]
        stems   = [c for c in cells if c.type == CellType.STEM]

        subtypes = Counter(str(c.subtype) for c in neurons)

        weights = [s.weight for n in neurons for s in n.synapses_in]
        myelins = [s.myelination for n in neurons for s in n.synapses_in]
        total_syn = len(weights)
        syn_density = total_syn / max(len(neurons), 1)
        hub_count = sum(1 for n in neurons if len(n.synapses_in) > 5)

        # Capas funcionales por altura Z
        layer_sensor  = [n for n in neurons if n.pos[2] < 15]
        layer_mem     = [n for n in neurons if 15 <= n.pos[2] < 35]
        layer_motor   = [n for n in neurons if 35 <= n.pos[2] < 55]
        layer_pfc     = [n for n in neurons if n.pos[2] >= 55]

        print(f"  Tiempo sim.    : {t/1000:.2f} s simulados  ({steps:,} steps)")
        print(f"  Células        : {len(cells)} total  |  {len(neurons)} neuronas  |  {len(glia)} glía  |  {len(stems)} stem")
        print(f"  Subtipos       : {dict(subtypes)}")
        print(f"  Capas          : sensor={len(layer_sensor)}  memoria={len(layer_mem)}  motor={len(layer_motor)}  PFC={len(layer_pfc)}")
        print(f"  Sinapsis       : {total_syn} total  /  {syn_density:.1f} por neurona  /  hubs(>5)={hub_count}")
        if weights:
            w = np.array(weights)
            m = np.array(myelins)
            print(f"  Pesos          : mean={w.mean():.4f}  max={w.max():.4f}  fuertes(>0.5)={np.mean(w>0.5)*100:.1f}%")
            print(f"  Mielinización  : mean={m.mean():.4f}  alta(>0.5)={np.mean(m>0.5)*100:.1f}%")
        print(f"  Emociones      : frustr={frustration:.3f}  resil={resilience:.3f}  valor={courage:.3f}")
        print(f"  Memorias epis. : {len(ep_buf) if ep_buf else 0}")
        if utility:
            vals = list(utility.values())
            print(f"  Utilidad neur. : n={len(vals)}  mean={np.mean(vals):.3f}  max={max(vals):.3f}  min={min(vals):.3f}")
        if grid is not None:
            explored = np.sum(grid > 0.01) / grid.size * 100
            hot = np.sum(grid > 0.3) / grid.size * 100
            print(f"  Mapa curiosidad: {explored:.1f}% visitado  zonas calientes={hot:.1f}%")
    except Exception as e:
        import traceback; traceback.print_exc()

print(f"\n{'='*60}")
print("Análisis completado.")
