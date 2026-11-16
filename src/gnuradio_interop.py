"""Módulo de interoperabilidad entre Python y GNU Radio (Fase 8).

Permite:
1. Exportar e importar señales en formato binario nativo complex64 (.c64 / .bin).
2. Localizar y orquestar el entorno GNU Radio (Radioconda).
3. Compilar flowgraphs .grc a scripts ejecutables con grcc.
4. Ejecutar simulaciones headless y validar las señales resultantes con el monitor
   y los clasificadores de modulaciones del proyecto.
"""
from pathlib import Path
import os
import subprocess
import numpy as np


# Rutas por defecto del entorno Radioconda en el sistema
DEFAULT_RADIOCONDA_PATH = Path(r"C:\Users\Luisr\radioconda")
DEFAULT_GRCC_PATH = DEFAULT_RADIOCONDA_PATH / "Scripts" / "grcc.exe"
DEFAULT_PYTHON_PATH = DEFAULT_RADIOCONDA_PATH / "python.exe"


def is_radioconda_available(radioconda_dir=None):
    """Verifica si el entorno Radioconda está disponible en el sistema."""
    base = Path(radioconda_dir) if radioconda_dir else DEFAULT_RADIOCONDA_PATH
    grcc = base / "Scripts" / "grcc.exe"
    py = base / "python.exe"
    return base.is_dir() and grcc.is_file() and py.is_file()


def save_c64(path, array):
    """Guarda un array complejo NumPy en formato binario complex64 (I/Q float32 interleaved).

    Args:
        path: Ruta del archivo destino (.c64 o .bin).
        array: Array 1D o 2D de muestras complejas.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array, dtype=np.complex64)
    arr.tofile(str(path))
    return path


def read_c64(path, count=-1, offset=0):
    """Lee un archivo binario complex64 producido por GNU Radio o Python.

    Args:
        path: Ruta del archivo .c64 o .bin.
        count: Número de muestras complejas a leer (-1 para leer todo).
        offset: Offset en número de muestras complejas (8 bytes cada una).

    Returns:
        np.ndarray de dtype np.complex64.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    file_bytes = path.stat().st_size
    if file_bytes % 8 != 0:
        raise ValueError(
            f"El tamaño del archivo ({file_bytes} bytes) no es múltiplo de 8 "
            "(cada muestra complex64 requiere 8 bytes: 4 float32 real + 4 float32 imag)."
        )

    byte_offset = offset * 8
    with open(path, "rb") as f:
        if byte_offset > 0:
            f.seek(byte_offset)
        data = np.fromfile(f, dtype=np.complex64, count=count)
    return data


def compile_grc(grc_path, output_dir=None, grcc_path=None):
    """Compila un flowgraph .grc a un script Python ejecutable usando grcc.

    Args:
        grc_path: Ruta del archivo .grc.
        output_dir: Directorio destino para el script .py (por defecto el mismo del .grc).
        grcc_path: Ruta al ejecutable grcc.exe.

    Returns:
        Path al script .py generado.
    """
    grc = Path(grc_path).resolve()
    if not grc.is_file():
        raise FileNotFoundError(f"Archivo .grc no encontrado: {grc}")

    out_dir = Path(output_dir).resolve() if output_dir else grc.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    grcc = Path(grcc_path) if grcc_path else DEFAULT_GRCC_PATH
    if not grcc.is_file():
        raise RuntimeError(f"Compilador grcc no encontrado en: {grcc}")

    cmd = [str(grcc), "-o", str(out_dir), str(grc)]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Error al compilar {grc.name} con grcc (código {proc.returncode}):\n{proc.stderr}\n{proc.stdout}"
        )

    expected_py = out_dir / f"{grc.stem}.py"
    if not expected_py.is_file():
        raise FileNotFoundError(f"No se generó el script esperado: {expected_py}")

    return expected_py


def run_headless_simulation(py_script_path, work_dir=None, python_exe=None, timeout_sec=30):
    """Ejecuta un script compilado headless de GNU Radio y espera su finalización.

    Args:
        py_script_path: Ruta al script Python generado por grcc.
        work_dir: Directorio de trabajo donde se generará la salida.
        python_exe: Intérprete Python de Radioconda con soporte gnuradio.
        timeout_sec: Tiempo máximo de espera en segundos.

    Returns:
        subprocess.CompletedProcess con stdout y stderr.
    """
    script = Path(py_script_path).resolve()
    if not script.is_file():
        raise FileNotFoundError(f"Script no encontrado: {script}")

    cwd = Path(work_dir).resolve() if work_dir else script.parent
    py = Path(python_exe) if python_exe else DEFAULT_PYTHON_PATH
    if not py.is_file():
        raise RuntimeError(f"Intérprete Python de GNU Radio no encontrado en: {py}")

    cmd = [str(py), str(script)]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Fallo al ejecutar {script.name} (código {proc.returncode}):\n{proc.stderr}\n{proc.stdout}"
        )
    return proc
