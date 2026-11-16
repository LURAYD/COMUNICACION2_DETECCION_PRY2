"""Verificaciones físicas del detector y del periodograma bilateral."""
import numpy as np
import pytest

from src.detection import calibrate_energy, detect_energy, energy_statistic
from src.spectrum import estimate_spectrum


def test_energia_y_calibracion_independiente():
    cal_rng, test_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(781).spawn(2)]
    make_noise = lambda rng: (rng.normal(size=(5000, 256))+1j*rng.normal(size=(5000, 256)))/np.sqrt(2)
    calibration = calibrate_energy(make_noise(cal_rng), .01)
    assert calibration["calibration_pfa"] <= .01
    measured = np.mean([detect_energy(row, calibration)["detected"] for row in make_noise(test_rng)])
    assert .004 <= measured <= .02  # Tolerancia de ensayo, no garantía universal.
    assert not detect_energy(np.zeros(256), calibration)["detected"]
    assert detect_energy(np.full(256, 2+1j), calibration)["detected"]
    assert energy_statistic([1+1j, -1-1j]) == pytest.approx(2)
    with pytest.raises(ValueError):
        detect_energy(np.ones(512), calibration)


def test_umbral_y_comparacion_estricta():
    cal = calibrate_energy([[1, 1], [2, 2], [3, 3], [4, 4]], .25)
    assert cal["threshold"] == 16
    assert not detect_energy([4, 4], cal)["detected"]
    assert detect_energy([4.1, 4.1], cal)["detected"]


@pytest.mark.parametrize("bad", [[], [[1]], [np.nan], [np.inf]])
def test_energia_invalida(bad):
    with pytest.raises(ValueError):
        energy_statistic(bad)


@pytest.mark.parametrize("frequency", [-125, 0, 250])
def test_tono_psd_parseval_unidades(frequency):
    fs, n = 1024, 256
    x = 2*np.exp(2j*np.pi*frequency*np.arange(n)/fs)
    result = estimate_spectrum(x, fs, nfft=1024)
    assert result["frequency"][np.argmax(result["psd"])] == pytest.approx(frequency)
    assert result["integrated_power"] == pytest.approx(4)
    assert result["frequency_unit"] == "Hz"
    assert result["df"] == 1
    assert result["enbw"] == pytest.approx(6)
    assert abs(result["fft"]).max() == pytest.approx(2)


def test_parseval_ventana_y_escala_normalizada():
    rng = np.random.default_rng(3)
    x = rng.normal(size=512)+1j*rng.normal(size=512)
    w = np.hanning(513)[:-1]
    norm = estimate_spectrum(x)
    physical = estimate_spectrum(x, 8000)
    assert norm["integrated_power"] == pytest.approx(np.sum(abs(x*w)**2)/np.sum(w*w))
    np.testing.assert_allclose(physical["frequency"], 8000*norm["frequency"])
    np.testing.assert_allclose(physical["psd"]*8000, norm["psd"])
    assert norm["frequency_unit"] == "cycles/sample"
    assert physical["total_power_bandwidth"] == pytest.approx(norm["total_power_bandwidth"]*8000)


def test_espectro_cero_y_invalidos():
    result = estimate_spectrum(np.zeros(256))
    assert result["integrated_power"] == 0
    assert result["total_power_bandwidth"] is None
    for kwargs in [{"fs_hz": 0}, {"fs_hz": np.nan}, {"nfft": 128}, {"power_fraction": 0}]:
        with pytest.raises(ValueError):
            estimate_spectrum(np.ones(256), **kwargs)


@pytest.mark.parametrize("nfft", [255, 256, 1024])
def test_ancho_observado_incluye_ruido_y_bordes_fisicos(nfft):
    rng = np.random.default_rng(101)
    x = np.exp(2j*np.pi*.1*np.arange(128))
    clean = estimate_spectrum(x, nfft=nfft)
    noisy = estimate_spectrum(x+3*(rng.normal(size=128)+1j*rng.normal(size=128)), nfft=nfft)
    assert noisy["total_power_bandwidth"] > clean["total_power_bandwidth"]
    assert -.5 <= noisy["total_power_bounds"][0] < noisy["total_power_bounds"][1] <= .5
    assert noisy["total_power_bandwidth"] <= 1
