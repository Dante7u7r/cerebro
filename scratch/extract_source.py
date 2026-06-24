#!/usr/bin/env python3
import os
import subprocess

USB_PATH = "/run/media/dante7u7r/7EE2-FCF9/01_Proyectos/Cerebro_IA_y_Asistente"
DEST_BASE = "/home/dante7u7r/Documentos/cerebro-main/analisis_historico"

# Extensiones de código que queremos extraer
EXT_FILTERS = ["*.py", "*.md", "*.json", "*.html", "*.js", "*.css"]

def extraer_archivo_comprimido(file_path, dest_dir):
    """Extrae selectivamente archivos de código de un archivo comprimido usando 7z."""
    os.makedirs(dest_dir, exist_ok=True)
    
    # Construir comando 7z:
    # 7z x <comprimido> -o<destino> <filtros> -r -y
    cmd = [
        "7z", "x",
        file_path,
        f"-o{dest_dir}"
    ]
    # Añadir filtros
    cmd.extend(EXT_FILTERS)
    # Recursivo y responder "sí" a sobreescritura
    cmd.extend(["-r", "-y"])
    
    print(f"[i] Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"[+] Extracción completada para {os.path.basename(file_path)}")
    else:
        # A veces 7z devuelve error code 1 (warning) si algún filtro no coincide con nada,
        # lo cual es normal si un zip no contiene archivos de cierto tipo (ej. no contiene .css).
        # Comprobamos si efectivamente se extrajo algo.
        archivos_extraidos = []
        for root, dirs, files in os.walk(dest_dir):
            for file in files:
                archivos_extraidos.append(file)
        if archivos_extraidos:
            print(f"[+] Extracción parcial (warnings ignorados) para {os.path.basename(file_path)}. Archivos extraídos: {len(archivos_extraidos)}")
        else:
            print(f"[!] Error al extraer {os.path.basename(file_path)}: {result.stderr}")

def main():
    if not os.path.exists(USB_PATH):
        print(f"[!] ERROR: La ruta del USB {USB_PATH} no existe.")
        return

    print("=== INICIANDO EXTRACCIÓN SELECTIVA DE CÓDIGO HISTÓRICO ===")
    os.makedirs(DEST_BASE, exist_ok=True)

    # Buscar todos los archivos zip y rar
    archivos_a_procesar = []
    for root, dirs, files in os.walk(USB_PATH):
        # Evitar carpetas de proyectos activos descomprimidos en el USB, solo queremos el historial
        if "cerebro-main" in root or "red_critica" in root:
            continue
        for file in files:
            if file.endswith((".zip", ".rar")):
                archivos_a_procesar.append(os.path.join(root, file))

    # Ordenar alfabéticamente (por fecha, ya que tienen el prefijo AAAA-MM-DD)
    archivos_a_procesar.sort()

    print(f"[i] Se encontraron {len(archivos_a_procesar)} archivos de historial para extraer.")
    for file_path in archivos_a_procesar:
        nombre_sin_ext, _ = os.path.splitext(os.path.basename(file_path))
        
        # Carpeta de destino específica para esta versión
        dest_dir = os.path.join(DEST_BASE, nombre_sin_ext)
        print(f"\n---> Procesando: {os.path.basename(file_path)}")
        extraer_archivo_comprimido(file_path, dest_dir)

    print("\n=== PROCESO DE EXTRACCIÓN COMPLETADO ===")
    print(f"Todo el código histórico está en: {DEST_BASE}")

if __name__ == "__main__":
    main()
