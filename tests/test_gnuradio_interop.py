"""Pruebas de interoperabilidad con GNU Radio (Fase 8)."""
from pathlib import Path
import numpy as np
import pytest

from src.gnuradio_interop import (
    save_c64,
    read_c64,
    is_radioconda_available,
    compile_grc,
    run_headless_simulation,
    DEFAULT_RADIOCONDA_PATH
)
from src.monitor import analyze


def test_c64_roundtrip(tmp_path):
    """Verifica que el guardado y lectura binaria complex64 sea bit a bit idéntica."""
    original = np.array([1.0 + 2.0j, -0.5 + 0.5j, 3.14 - 1.59j, 0.0 + 0.0j], dtype=np.complex64)
    file_path = tmp_path / "test_signal.c64"
    save_c64(file_path, original)

    assert file_path.stat().st_size == len(original) * 8

    recovered = read_c64(file_path)
    np.testing.assert_array_equal(original, recovered)


def test_c64_invalid_file_size(tmp_path):
    """Verifica que se lance ValueError si el archivo no es múltiplo de 8 bytes."""
    corrupted_file = tmp_path / "corrupted.c64"
    with open(corrupted_file, "wb") as f:
        f.write(b"1234567")  # 7 bytes, no divisible por 8

    with pytest.raises(ValueError, match="múltiplo de 8"):
        read_c64(corrupted_file)


def test_c64_partial_read(tmp_path):
    """Verifica lectura parcial con count y offset."""
    arr = np.arange(10, dtype=np.complex64) + 1j * np.arange(10, dtype=np.complex64)
    file_path = tmp_path / "stream.c64"
    save_c64(file_path, arr)

    chunk = read_c64(file_path, count=4, offset=3)
    assert len(chunk) == 4
    np.testing.assert_array_equal(chunk, arr[3:7])


@pytest.mark.skipif(not is_radioconda_available(), reason="Radioconda no disponible en este entorno")
def test_grc_compilation(tmp_path):
    """Verifica que los tres flowgraphs .grc compilen sin errores con grcc."""
    grc_dir = Path("grc")
    flowgraphs = [
        grc_dir / "flowgraph_headless_sim.grc",
        grc_dir / "flowgraph_tx_channel.grc",
        grc_dir / "flowgraph_rx_from_file.grc",
    ]
    for grc in flowgraphs:
        assert grc.is_file(), f"Falta flowgraph: {grc}"
        py_out = compile_grc(grc, output_dir=tmp_path)
        assert py_out.is_file(), f"No se generó script para {grc}"
        assert py_out.stat().st_size > 0


@pytest.mark.skipif(not is_radioconda_available(), reason="Radioconda no disponible en este entorno")
def test_headless_simulation_and_monitor(tmp_path):
    """Ejecuta la simulación headless de GNU Radio y valida la señal con el monitor."""
    grc_file = Path("grc") / "flowgraph_headless_sim.grc"
    py_script = compile_grc(grc_file, output_dir=tmp_path)

    # Ejecutar simulación en tmp_path
    run_headless_simulation(py_script, work_dir=tmp_path, timeout_sec=20)

    out_c64 = tmp_path / "grc_sim_output.c64"
    assert out_c64.is_file(), "El flowgraph no produjo grc_sim_output.c64"

    samples = read_c64(out_c64)
    assert len(samples) == 4096
    assert samples.dtype == np.complex64

    # Comprobar que no son todo ceros ni NaNs/Infs
    assert np.all(np.isfinite(samples))
    assert np.var(samples) > 1e-4

    # Pasar la señal generada por GNU Radio a través de nuestro monitor de señales
    result = analyze(samples[:256], fs_hz=8000)
    assert "spectrum" in result
    assert "total_power_bandwidth" in result["spectrum"]
    bw_hz = result["spectrum"]["total_power_bandwidth"]
    # El pulso RRC tiene un ancho de banda ocupado positivo
    assert 500.0 < bw_hz < 4000.0, f"Ancho de banda inesperado: {bw_hz} Hz"
