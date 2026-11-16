"""Pruebas del Monitor Integrado y CLI de señales desconocidas (Fase 7)."""
from pathlib import Path
import json
import numpy as np
import pytest

from src.monitor import load_signal, analyze_complete, analyze
from src.gnuradio_interop import save_c64
from src.snr_estimation import calibrate_noise_reference


def test_load_signal_various_formats(tmp_path):
    """Verifica que el cargador de señales soporte .c64, .npy, .npz y .csv."""
    test_iq = np.array([1.0 + 0.5j, -0.2 - 0.8j, 0.4 + 0.3j, -0.9 + 0.1j], dtype=np.complex64)

    # 1. Formato binario .c64
    c64_file = tmp_path / "test.c64"
    save_c64(c64_file, test_iq)
    loaded_c64 = load_signal(c64_file)
    np.testing.assert_allclose(loaded_c64, test_iq, rtol=1e-5)

    # 2. Formato NumPy .npy (1D complejo)
    npy_file = tmp_path / "test.npy"
    np.save(str(npy_file), test_iq)
    loaded_npy = load_signal(npy_file)
    np.testing.assert_allclose(loaded_npy, test_iq, rtol=1e-5)

    # 3. Formato NumPy .npy (2D real I/Q)
    npy_2d_file = tmp_path / "test_2d.npy"
    np.save(str(npy_2d_file), np.column_stack([test_iq.real, test_iq.imag]))
    loaded_npy_2d = load_signal(npy_2d_file)
    np.testing.assert_allclose(loaded_npy_2d, test_iq, rtol=1e-5)

    # 4. Formato comprimido .npz
    npz_file = tmp_path / "test.npz"
    np.savez(str(npz_file), iq=test_iq)
    loaded_npz = load_signal(npz_file)
    np.testing.assert_allclose(loaded_npz, test_iq, rtol=1e-5)

    # 5. Formato texto .csv
    csv_file = tmp_path / "test.csv"
    np.savetxt(str(csv_file), np.column_stack([test_iq.real, test_iq.imag]), delimiter=",")
    loaded_csv = load_signal(csv_file)
    np.testing.assert_allclose(loaded_csv, test_iq, rtol=1e-5)


def test_monitor_complete_five_outputs(tmp_path):
    """Verifica que analyze_complete entregue las 5 salidas obligatorias del curso."""
    rng = np.random.default_rng(42)
    # Señal sintética con 512 muestras
    noise = (rng.standard_normal(512) + 1j * rng.standard_normal(512)) * np.sqrt(0.01)
    tone = np.exp(1j * 2 * np.pi * 0.1 * np.arange(512))
    signal = tone + noise

    out_plot = tmp_path / "spec_test.png"
    result = analyze_complete(
        iq=signal,
        signal_id="tone_test",
        fs_hz=8000.0,
        default_threshold=0.5,
        plot_path=out_plot,
    )

    # 1. Detección
    assert "detection" in result
    assert result["detection"]["detected"] is True
    assert result["detection"]["status"] == "signal_present"

    # 2. Espectro y Ancho de Banda
    assert "spectrum" in result
    assert result["spectrum"]["bandwidth_99"] > 0
    assert result["spectrum"]["bandwidth_unit"] == "Hz"
    assert out_plot.is_file()

    # 3. SNR
    assert "snr" in result
    assert result["snr"]["status"] == "no_identificable_sin_supuestos"

    # 4. Modulación
    assert "classification" in result
    assert result["classification"]["modulation"] in ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "4PAM", "NO_SIGNAL")

    # 5. Confianza
    assert "confidence" in result["classification"]
    assert 0.0 <= result["classification"]["confidence"] <= 1.0


def test_monitor_with_noise_reference(tmp_path):
    """Verifica estimación de SNR cuando se suministra referencia de ruido."""
    rng = np.random.default_rng(123)
    noise_power = 0.05
    noise_ref = (rng.standard_normal(1000) + 1j * rng.standard_normal(1000)) * np.sqrt(noise_power / 2.0)
    ref_dict = calibrate_noise_reference(noise_ref)

    # Señal con SNR ~ 10 dB -> Potencia señal ~ 0.5
    signal = np.ones(256, dtype=np.complex64) * np.sqrt(0.5) + (rng.standard_normal(256) + 1j * rng.standard_normal(256)) * np.sqrt(noise_power / 2.0)

    result = analyze_complete(
        iq=signal,
        signal_id="snr_test",
        fs_hz=8000.0,
        default_threshold=0.01,
        noise_reference=ref_dict,
    )

    assert result["snr"]["snr_est_db"] is not None
    assert 7.0 < result["snr"]["snr_est_db"] < 13.0
