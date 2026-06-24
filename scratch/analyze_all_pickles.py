#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
from types import ModuleType
from collections import Counter

# Rutas clave
USB_PROYECTOS = "/run/media/dante7u7r/7EE2-FCF9/01_Proyectos/Cerebro_IA_y_Asistente"
PC_WORKSPACE = "/home/dante7u7r/Documentos/cerebro-main"
ANALISIS_HISTORICO = "/home/dante7u7r/Documentos/cerebro-main/analisis_historico"
REPORTE_DEST = "/home/dante7u7r/.gemini/antigravity-ide/brain/78ea3765-a977-488a-9a4f-e0889fb4c40a/analisis_datos_pkl.md"

def mock_semilla_cerebro(semilla_path):
    """Crea un módulo mock en sys.modules para que pickle pueda deserializar las clases antiguas."""
    try:
        semilla_mod = ModuleType('semilla_cerebro')
        with open(semilla_path, 'r', encoding='utf-8') as f:
            src = f.read()
        # Modificar el arranque
        src = src.replace('if __name__ == "__main__":', 'if False:')
        exec(compile(src, 'semilla_cerebro.py', 'exec'), semilla_mod.__dict__)
        
        # Poner en sys.modules para resolver importación
        sys.modules['semilla_cerebro'] = semilla_mod
        sys.modules['__main__'] = semilla_mod
        return semilla_mod
    except Exception as e:
        print(f"[!] Error al mockear semilla_cerebro desde {semilla_path}: {e}")
        return None

def encontrar_semilla_cerebro_asociada(pickle_path):
    """Busca un archivo semilla_cerebro.py en la misma estructura o en la PC."""
    # Buscar en la misma carpeta del pickle o subiendo niveles
    dir_path = os.path.dirname(pickle_path)
    # Si está en el USB, a veces está en la versión extraída localmente en el PC
    nombre_base = os.path.basename(pickle_path)
    
    # Intentar buscar en la carpeta extraída homónima del PC
    # e.g., de "/run/.../Historico_IA_Viva/2026-05-12_IA_Viva.zip" (o pickle de esa fecha)
    # a "/home/.../analisis_historico/2026-05-12_IA_Viva_Arena_Darwiniana/IA_Viva_Arena_Darwiniana/semilla_cerebro.py"
    for root, dirs, files in os.walk(ANALISIS_HISTORICO):
        if "semilla_cerebro.py" in files:
            # Comprobar si la carpeta padre contiene fechas similares
            # Para simplificar, si encontramos un semilla_cerebro.py en el histórico, podemos usarlo
            return os.path.join(root, "semilla_cerebro.py")
            
    # Caída de respaldo: semilla_cerebro de producción en el PC
    pc_semilla = os.path.join(PC_WORKSPACE, "semilla_cerebro.py")
    if os.path.exists(pc_semilla):
        return pc_semilla
        
    return None

def analizar_pickle_antiguo(pickle_path, data):
    """Extrae métricas de un pickle de la era de IA Viva / Astrid (objetos Cell)."""
    cells = data.get('cells', [])
    t = data.get('time', 0)
    steps = data.get('step_count', 0)
    frustration = data.get('frustration', 0.0)
    resilience = data.get('resilience', 0.0)
    courage = data.get('courage', 0.0)
    ep_buf = data.get('_episodic_buffer', [])
    utility = data.get('_neuron_utility', {})
    grid = data.get('_curiosity_grid', None)

    # Identificar tipos de celdas
    # Comprobación de tipos flexible basada en strings de atributos
    neurons = []
    glia = []
    stems = []
    for c in cells:
        type_str = str(getattr(c, 'type', ''))
        if 'NEURON' in type_str:
            neurons.append(c)
        elif 'GLIA' in type_str:
            glia.append(c)
        elif 'STEM' in type_str:
            stems.append(c)

    subtypes = Counter(str(getattr(c, 'subtype', '')) for c in neurons)

    # Extraer pesos y mielina
    weights = []
    myelins = []
    for n in neurons:
        synapses = getattr(n, 'synapses_in', [])
        for s in synapses:
            weights.append(getattr(s, 'weight', 0.0))
            myelins.append(getattr(s, 'myelination', 0.0))

    total_syn = len(weights)
    syn_density = total_syn / max(len(neurons), 1)
    hub_count = sum(1 for n in neurons if len(getattr(n, 'synapses_in', [])) > 5)

    # Capas por altura Z
    layer_sensor = [n for n in neurons if getattr(n, 'pos', [0,0,0])[2] < 15]
    layer_mem = [n for n in neurons if 15 <= getattr(n, 'pos', [0,0,0])[2] < 35]
    layer_motor = [n for n in neurons if 35 <= getattr(n, 'pos', [0,0,0])[2] < 55]
    layer_pfc = [n for n in neurons if getattr(n, 'pos', [0,0,0])[2] >= 55]

    w_arr = np.array(weights) if weights else np.array([])
    m_arr = np.array(myelins) if myelins else np.array([])

    explored_curiosity = 0.0
    if grid is not None and hasattr(grid, 'size') and grid.size > 0:
        explored_curiosity = np.sum(grid > 0.01) / grid.size * 100

    return {
        'tipo': 'IA Viva / Astrid (Evolutivo)',
        'tiempo_sim': f"{t/1000:.2f} s ({steps:,} steps)",
        'celulas_total': len(cells),
        'neuronas_total': len(neurons),
        'glia_total': len(glia),
        'stems_total': len(stems),
        'subtipos': dict(subtypes),
        'capas': f"Sensorial={len(layer_sensor)} | Memoria={len(layer_mem)} | Motor={len(layer_motor)} | PFC={len(layer_pfc)}",
        'sinapsis_total': total_syn,
        'sinapsis_por_neurona': f"{syn_density:.1f}",
        'hubs_count': hub_count,
        'pesos_mean': f"{w_arr.mean():.4f}" if len(w_arr) > 0 else "0.0",
        'pesos_max': f"{w_arr.max():.4f}" if len(w_arr) > 0 else "0.0",
        'pesos_fuertes_pct': f"{np.mean(w_arr > 0.5) * 100:.1f}%" if len(w_arr) > 0 else "0.0%",
        'mielina_mean': f"{m_arr.mean():.4f}" if len(m_arr) > 0 else "0.0",
        'emociones': f"Frustración={frustration:.3f} | Resiliencia={resilience:.3f} | Valor={courage:.3f}",
        'recuerdos_episodicos': len(ep_buf) if ep_buf else 0,
        'curiosidad_explorada': f"{explored_curiosity:.1f}%" if grid is not None else "N/A"
    }

def analizar_pickle_brian2(pickle_path, data):
    """Extrae métricas de un pickle de la era de Cerebro Único (Brian2 vectorizado)."""
    t = data.get('time', 0.0)
    steps = data.get('step_count', 0)
    frustration = data.get('frustration', 0.0)
    resilience = data.get('resilience', 0.0)
    ep_buf = data.get('episodic_buffer', [])
    history = data.get('history', [])
    
    # Capas y tipos
    layer_indices = data.get('layer_indices')
    neuron_types = data.get('neuron_types')
    synapses_w = data.get('synapses_w')
    synapses_myel = data.get('synapses_myelination', [])
    
    n_total = len(layer_indices) if layer_indices is not None else 0
    n_sensor = int(np.sum(layer_indices == 0)) if layer_indices is not None else 0
    n_hidden = int(np.sum(layer_indices == 1)) if layer_indices is not None else 0
    n_motor = int(np.sum(layer_indices == 2)) if layer_indices is not None else 0
    n_pfc = int(np.sum(layer_indices == 3)) if layer_indices is not None else 0
    
    n_exc = int(np.sum(neuron_types == 1)) if neuron_types is not None else 0
    n_inh = int(np.sum(neuron_types == 4)) if neuron_types is not None else 0
    
    # Sinapsis activas (w > 0.01)
    w_arr = synapses_w[synapses_w > 0.01] if synapses_w is not None else np.array([])
    m_arr = np.array(synapses_myel)[synapses_w > 0.01] if len(synapses_myel) > 0 and synapses_w is not None else np.array([])
    
    total_syn = len(w_arr)
    syn_density = total_syn / max(n_total, 1)
    
    # Contar hubs (neurona destino con más de 5 sinapsis activas)
    hub_count = 0
    if synapses_w is not None and len(synapses_w) > 0:
        j_indices = data.get('synapses_j', [])
        if len(j_indices) > 0:
            active_j = j_indices[synapses_w > 0.01]
            counts = Counter(active_j)
            hub_count = sum(1 for nid, count in counts.items() if count > 5)

    return {
        'tipo': 'Cerebro Único (Brian2 Vectorial)',
        'tiempo_sim': f"{t/1000:.2f} s ({steps:,} steps)" if t > 100 else f"{t} ms ({steps:,} steps)",
        'celulas_total': n_total,
        'neuronas_total': n_total,
        'glia_total': 0, # Brian2 no modelaba células gliales como unidades discretas
        'stems_total': 0,
        'subtipos': {'Excitatoria': n_exc, 'Inhibitoria': n_inh},
        'capas': f"Sensorial={n_sensor} | Oculta={n_hidden} | Motor={n_motor} | PFC={n_pfc}",
        'sinapsis_total': total_syn,
        'sinapsis_por_neurona': f"{syn_density:.1f}",
        'hubs_count': hub_count,
        'pesos_mean': f"{w_arr.mean():.4f}" if len(w_arr) > 0 else "0.0",
        'pesos_max': f"{w_arr.max():.4f}" if len(w_arr) > 0 else "0.0",
        'pesos_fuertes_pct': f"{np.mean(w_arr > 0.5) * 100:.1f}%" if len(w_arr) > 0 else "0.0%",
        'mielina_mean': f"{m_arr.mean():.4f}" if len(m_arr) > 0 else "0.0",
        'emociones': f"Frustración={frustration:.3f} | Resiliencia={resilience:.3f} | Valor=N/A",
        'recuerdos_episodicos': len(ep_buf) if ep_buf else 0,
        'curiosidad_explorada': "N/A"
    }

def main():
    print("=== INICIANDO ANÁLISIS DE BASE DE DATOS PICKLE ===")
    
    # 1. Encontrar todos los pkl
    pkl_files = []
    # Escanear USB
    if os.path.exists(USB_PROYECTOS):
        for root, dirs, files in os.walk(USB_PROYECTOS):
            for file in files:
                if file.endswith(".pkl") and "sim_state" not in file:
                    pkl_files.append(os.path.join(root, file))
                    
    # Escanear PC (workspace y logs)
    if os.path.exists(PC_WORKSPACE):
        for root, dirs, files in os.walk(PC_WORKSPACE):
            # Omitir la carpeta de análisis histórico para no duplicar lecturas de copias extraídas
            if "analisis_historico" in root:
                continue
            for file in files:
                if file.endswith(".pkl") and "sim_state" not in file:
                    pkl_files.append(os.path.join(root, file))

    print(f"[i] Se encontraron {len(pkl_files)} archivos de datos .pkl.")
    
    resultados = {}
    
    # 2. Cargar cada pkl
    for pkl_path in pkl_files:
        nombre = os.path.basename(pkl_path)
        rel_path = os.path.relpath(pkl_path, "/")
        print(f"\n---> Analizando: {nombre}")
        
        # Encontrar semilla_cerebro asociada
        semilla_path = encontrar_semilla_cerebro_asociada(pkl_path)
        if semilla_path:
            mock_semilla_cerebro(semilla_path)
            
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                
            # Determinar tipo
            if 'cells' in data:
                # Tipo antiguo
                info = analizar_pickle_antiguo(pkl_path, data)
            elif 'layer_indices' in data:
                # Tipo Brian2
                info = analizar_pickle_brian2(pkl_path, data)
            else:
                # Desconocido o estructura básica
                info = {
                    'tipo': 'Datos Básicos / Desconocido',
                    'tiempo_sim': 'N/A',
                    'celulas_total': len(data) if isinstance(data, (dict, list)) else 'N/A',
                    'sinapsis_total': 'N/A',
                    'emociones': 'N/A'
                }
            
            info['tamano'] = f"{os.path.getsize(pkl_path) / 1e6:.2f} MB"
            info['ruta'] = pkl_path
            resultados[nombre] = info
            print(f"[+] Procesado exitosamente: {info['tipo']}")
            
        except Exception as e:
            print(f"[!] Error al procesar {nombre}: {e}")
            resultados[nombre] = {
                'tipo': 'ERROR DE CARGA',
                'tamano': f"{os.path.getsize(pkl_path) / 1e6:.2f} MB",
                'ruta': pkl_path,
                'error': str(e)
            }

    # 3. Escribir reporte Markdown
    with open(REPORTE_DEST, 'w', encoding='utf-8') as r:
        r.write("# Consolidación y Análisis de Checkpoints de Datos (.pkl)\n\n")
        r.write("Este reporte consolida las métricas de red y el perfil conductual de los cerebros evolutivos recuperados de la memoria flash de 2TB y de tu PC.\n\n")
        
        r.write("## 1. Tabla Resumen de Modelos\n\n")
        r.write("| Modelo / Archivo | Tipo de Estructura | Células (Neuronas) | Sinapsis Activas | Peso Promedio | Mielina Promedio | Perfil Emocional / Estado |\n")
        r.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for name, info in sorted(resultados.items()):
            if info['tipo'] == 'ERROR DE CARGA':
                r.write(f"| `{name}` | **Error de Carga** | N/A | N/A | N/A | N/A | *Error:* `{info.get('error')}` |\n")
            elif info['tipo'] == 'Datos Básicos / Desconocido':
                r.write(f"| `{name}` | Datos Básicos | N/A | N/A | N/A | N/A | Contiene {info['celulas_total']} elementos |\n")
            else:
                r.write(f"| `{name}` | {info['tipo']} | {info['neuronas_total']} | {info['sinapsis_total']} | {info['pesos_mean']} | {info['mielina_mean']} | {info['emociones']} |\n")
                
        r.write("\n## 2. Reporte de Análisis Detallado por Archivo\n\n")
        for name, info in sorted(resultados.items()):
            r.write(f"### 📄 {name}\n")
            r.write(f"*   **Ruta física:** `{info['ruta']}`\n")
            r.write(f"*   **Tamaño en disco:** {info['tamano']}\n")
            
            if info['tipo'] == 'ERROR DE CARGA':
                r.write(f"*   **Estado:** No se pudo deserializar.\n")
                r.write(f"*   **Error reportado:** `{info.get('error')}`\n\n")
                continue
                
            r.write(f"*   **Paradigma técnico:** {info['tipo']}\n")
            r.write(f"*   **Tiempo de simulación alcanzado:** {info['tiempo_sim']}\n")
            
            if info['tipo'] in ['IA Viva / Astrid (Evolutivo)', 'Cerebro Único (Brian2 Vectorial)']:
                r.write(f"*   **Población celular:**\n")
                r.write(f"    *   Células totales: {info['celulas_total']}\n")
                r.write(f"    *   Neuronas: {info['neuronas_total']}\n")
                r.write(f"    *   Células gliales: {info['glia_total']}\n")
                r.write(f"    *   Células madre (stem): {info['stems_total']}\n")
                r.write(f"*   **Estructura de capas:** `{info['capas']}`\n")
                r.write(f"*   **Perfil sináptico:**\n")
                r.write(f"    *   Sinapsis activas: {info['sinapsis_total']}\n")
                r.write(f"    *   Densidad sináptica: {info['sinapsis_por_neurona']} conexiones/neurona\n")
                r.write(f"    *   Neuronas superconectadas (Hubs > 5): {info['hubs_count']}\n")
                r.write(f"    *   Peso sináptico medio: {info['pesos_mean']} (Max: {info['pesos_max']})\n")
                r.write(f"    *   Sinapsis fuertes (>0.5): {info['pesos_fuertes_pct']}\n")
                r.write(f"    *   Mielinización media: {info['mielina_mean']}\n")
                r.write(f"*   **Parámetros biológicos/emocionales:**\n")
                r.write(f"    *   Perfil: `{info['emociones']}`\n")
                r.write(f"    *   Recuerdos grabados (sueño REM): {info['recuerdos_episodicos']}\n")
                if 'curiosidad_explorada' in info:
                    r.write(f"    *   Exploración del mapa (curiosidad): {info['curiosidad_explorada']}\n")
            r.write("\n---\n\n")
            
    print(f"\n[+] Reporte final consolidado y guardado en {REPORTE_DEST}")

if __name__ == "__main__":
    main()
