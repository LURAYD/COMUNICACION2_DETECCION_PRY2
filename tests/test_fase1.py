"""Contratos de mapeo, importación y conformación de Fase 1."""
from itertools import product
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src import signals, waveform

MAPPERS = [("bpsk_modulation", 1), ("qpsk_modulation", 2),
           ("psk8_modulation", 3), ("qam16_modulation", 4),
           ("qam64_modulation", 6), ("pam4_modulation", 2)]


def test_tablas_exactas():
    def bits(k):
        return np.array(list(product([0, 1], repeat=k))).ravel()
    np.testing.assert_allclose(signals.bpsk_modulation(bits(1)), [1, -1])
    np.testing.assert_allclose(signals.qpsk_modulation(bits(2)),
                               np.array([1+1j, 1-1j, -1+1j, -1-1j])/np.sqrt(2))
    np.testing.assert_allclose(signals.psk8_modulation(bits(3)),
                               np.exp(1j*np.pi/4*np.array([0, 1, 3, 2, 7, 6, 4, 5])))
    for name, k, levels, scale in [
        ("qam16_modulation", 4, [3, 1, -3, -1], np.sqrt(10)),
        ("qam64_modulation", 6, [7, 5, 1, 3, -7, -5, -1, -3], np.sqrt(42)),
    ]:
        expected = np.array([i+1j*q for i in levels for q in levels])/scale
        np.testing.assert_allclose(getattr(signals, name)(bits(k)), expected)
    np.testing.assert_allclose(signals.pam4_modulation(bits(2)), np.array([3, 1, -3, -1])/np.sqrt(5))


@pytest.mark.parametrize("name,k", MAPPERS)
@pytest.mark.parametrize("dtype", [np.uint8, np.int64, np.float64, np.bool_])
def test_tipos_binarios_y_repetibilidad(name, k, dtype):
    bits = np.array(list(product([0, 1], repeat=k))).ravel()
    mapper = getattr(signals, name)
    np.testing.assert_array_equal(mapper(bits.astype(dtype)), mapper(bits))
    np.testing.assert_array_equal(mapper(bits), mapper(bits))


@pytest.mark.parametrize("name,k", MAPPERS)
@pytest.mark.parametrize("bad", [[], [[0, 1]], 0, [0.5], [np.nan], [np.inf], [-1], [2]])
def test_bits_invalidos(name, k, bad):
    with pytest.raises(ValueError):
        getattr(signals, name)(bad)


@pytest.mark.parametrize("name,k", MAPPERS[1:])
def test_grupos_incompletos(name, k):
    with pytest.raises(ValueError):
        getattr(signals, name)([0]*(k+1))


def test_import_sin_efectos_y_compatibilidad():
    code = "import sys; import src.signals, src.waveform, src.channels, src.test_signal; assert 'matplotlib.pyplot' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, check=True)
    assert result.stdout == result.stderr == ""
    assert signals.rrc_filter is waveform.rrc_filter
    assert signals.pulse_shape is waveform.pulse_shape


@pytest.mark.parametrize("kwargs", [
    {"beta": 0}, {"beta": -0.1}, {"beta": 1.1}, {"beta": np.nan},
    {"beta": np.inf}, {"beta": [0.35]}, {"beta": True}, {"beta": 1j},
    {"sps": 1}, {"sps": 8.0}, {"sps": True}, {"span": 0},
    {"span": 3}, {"span": 10.0}, {"span": True},
])
def test_parametros_rrc_invalidos(kwargs):
    with pytest.raises(ValueError):
        waveform.rrc_filter(**kwargs)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_enteros_numpy(dtype):
    np.testing.assert_array_equal(waveform.rrc_filter(sps=dtype(8), span=dtype(10)), waveform.rrc_filter())


def test_beta_positivo_muy_pequeno():
    with np.errstate(all="raise"):
        h = waveform.rrc_filter(beta=1e-20, sps=8, span=10)
    assert np.isfinite(h).all()
    np.testing.assert_allclose(h, h[::-1], atol=1e-14)
    assert np.sum(h*h) == pytest.approx(1)


@pytest.mark.parametrize("beta", [0.25, 0.5, 1.0])
def test_limites_rrc(beta):
    # OptiCommPy rrcFilterTaps, ecuaciones (2)-(3), Ts=1:
    # https://opticommpy.readthedocs.io/en/stable/_modules/optic/dsp/core.html#rrcFilterTaps
    # Relaciones eliminan la normalización discreta y verifican las ramas límite.
    h = waveform.rrc_filter(beta=beta)
    center = 40
    singular_index = center + round(8/(4*beta))
    limit_zero = 1 + beta*(4/np.pi - 1)
    limit_singular = beta/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/(4*beta))
                                    + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
    assert h[singular_index]/h[center] == pytest.approx(limit_singular/limit_zero)


@pytest.mark.parametrize("symbols", [[], [[1j]], [np.nan], [np.inf], [1+np.inf*1j]])
def test_simbolos_invalidos(symbols):
    with pytest.raises(ValueError):
        waveform.pulse_shape(symbols)


def test_impulso_y_retardo_adaptado():
    iq, h = waveform.pulse_shape([1j])
    np.testing.assert_allclose(iq[:len(h)], 1j*h)
    np.testing.assert_array_equal(iq[len(h):], np.zeros(7))
    matched = np.convolve(iq, h[::-1].conj())
    assert np.argmax(abs(iq)) == 40
    assert np.argmax(abs(matched)) == 80
    assert matched[80] == pytest.approx(1j)


def test_recuperacion_simbolos_con_retardo():
    symbols = signals.qpsk_modulation(np.random.default_rng(2026).integers(0, 2, size=128))
    iq, h = waveform.pulse_shape(symbols)
    received = np.convolve(iq, h[::-1].conj())
    recovered = received[80:80+len(symbols)*8:8]
    # Cota conservadora de la interferencia debida a los símbolos vecinos.
    response = np.convolve(h, h[::-1])
    residual = response[::8].copy()
    residual[len(residual)//2] = 0
    assert abs(recovered-symbols).max() <= abs(residual).sum()+1e-12
    assert abs(recovered-symbols).max() < .03


def test_espectro_rrc_aproxima_rc():
    h = waveform.rrc_filter()
    frequency = np.fft.fftshift(np.fft.fftfreq(65536, d=1/8))
    power = abs(np.fft.fftshift(np.fft.fft(h, 65536)))**2/8
    f = abs(frequency)
    ideal = np.where(f <= (1-.35)/2, 1.,
                     np.where(f <= (1+.35)/2, .5*(1+np.cos(np.pi/.35*(f-(1-.35)/2))), 0.))
    # Tolerancia de ingeniería para truncamiento a 10 símbolos.
    assert np.max(abs(power-ideal)) < 0.04
