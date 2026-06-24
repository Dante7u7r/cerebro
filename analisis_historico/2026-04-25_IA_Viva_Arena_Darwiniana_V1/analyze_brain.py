import pickle
from enum import Enum

class CellType(Enum):
    STEM = 0
    NEURON = 1
    GLIA = 2
    APOPTOTIC = 3

with open('brain_save.pkl', 'rb') as f:
    data = pickle.load(f)

cells = data.get('cells', [])
neurons = [c for c in cells if c.type == CellType.NEURON]
synapses = sum(len(n.synapses_in) for n in neurons)

print(f"Time: {data.get('time', 0) / 1000:.1f}s")
print(f"Total cells: {len(cells)}")
print(f"Neurons: {len(neurons)}")
print(f"Synapses: {synapses}")
print(f"Frustration: {data.get('frustration', 0):.2f}")
print(f"Resilience: {data.get('resilience', 0):.2f}")
print(f"Courage: {data.get('courage', 0):.2f}")
