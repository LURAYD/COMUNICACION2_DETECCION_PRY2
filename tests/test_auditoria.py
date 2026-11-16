"""Regresiones de Fase 0, convertidas a comprobaciones de Fase 1."""
from itertools import product

import numpy as np
import pytest

from src import signals
from src.channels import add_awgn


MAPPERS = [("bpsk_modulation", 1), ("qpsk_modulation", 2), ("psk8_modulation", 3),
           ("qam16_modulation", 4), ("qam64_modulation", 6),
           ("pam4_modulation", 2)]


@pytest.mark.parametrize("name,k", MAPPERS)
def test_constelacion_completa(name, k):
    labels = np.array(list(product([0, 1], repeat=k)))
    symbols = getattr(signals, name)(labels.ravel())
    assert symbols.shape == (2**k,)
    assert np.iscomplexobj(symbols)
    assert len(np.unique(symbols)) == 2**k
    assert np.mean(abs(symbols)**2) == pytest.approx(1)
    # Vecinos a distancia mínima deben diferir en un bit (Gray).
    distances = abs(symbols[:, None] - symbols[None, :])
    minimum = distances[distances > 1e-10].min()
    a, b = np.where(np.isclose(distances, minimum))
    assert np.all(np.sum(labels[a] != labels[b], axis=1) == 1)


@pytest.mark.parametrize("name,k", MAPPERS)
def test_rechazo_bits_invalidos(name, k):
    with pytest.raises(ValueError):
        getattr(signals, name)([2] * k)


def test_qpsk_disponible():
    assert callable(getattr(signals, "qpsk_modulation", None))


def test_bpsk_rechaza_vacio():
    with pytest.raises(ValueError):
        signals.bpsk_modulation([])


def test_bpsk_rechaza_matriz():
    with pytest.raises(ValueError):
        signals.bpsk_modulation([[0, 1]])


def test_bpsk_uint8():
    np.testing.assert_allclose(signals.bpsk_modulation(np.array([0, 1], dtype=np.uint8)), [1, -1])


@pytest.mark.parametrize("name,k", [("qam16_modulation", 4), ("qam64_modulation", 6)])
def test_qam_bits_float(name, k):
    np.testing.assert_allclose(getattr(signals, name)([0.] * k), getattr(signals, name)([0] * k))


@pytest.mark.parametrize("beta", [0.25, 0.35, 0.5, 1.0])
def test_rrc_propiedades(beta):
    h = signals.rrc_filter(beta=beta, sps=8, span=10)
    assert len(h) == 81
    assert np.isfinite(h).all()
    np.testing.assert_allclose(h, h[::-1], atol=1e-14)
    assert np.sum(h*h) == pytest.approx(1)
    matched = np.convolve(h, h[::-1])
    assert np.argmax(matched) == 80
    assert matched[80] == pytest.approx(1)
    # Tolerancia diagnóstica propuesta para truncamiento a 10 símbolos.
    residual = matched[::8].copy()
    residual[len(residual)//2] = 0
    assert abs(residual).max() < 0.03


def test_pulse_shape_longitud_y_componentes():
    symbols = np.array([1+1j, -1+1j, 1-1j])
    iq, h = signals.pulse_shape(symbols)
    assert len(iq) == len(symbols)*8 + len(h)-1
    up = np.zeros(len(symbols)*8, complex)
    up[::8] = symbols
    np.testing.assert_allclose(iq, np.convolve(up, h))
    assert np.any(iq.real) and np.any(iq.imag)


def test_awgn_snr_varianzas_y_reproducibilidad():
    x = np.ones(200_000, complex)
    y, noise = add_awgn(x, 10, np.random.default_rng(2026))
    y2, noise2 = add_awgn(x, 10, np.random.default_rng(2026))
    np.testing.assert_array_equal(y, y2)
    np.testing.assert_array_equal(noise, noise2)
    np.testing.assert_allclose(y, x+noise)
    assert 10*np.log10(np.mean(abs(x)**2)/np.mean(abs(noise)**2)) == pytest.approx(10, abs=0.08)
    assert np.var(noise.real) == pytest.approx(0.05, rel=0.02)
    assert np.var(noise.imag) == pytest.approx(0.05, rel=0.02)


@pytest.mark.parametrize("x", [[], [0j], [np.nan], [[1j]]])
def test_awgn_rechaza_entrada_invalida(x):
    with pytest.raises(ValueError):
        add_awgn(x, 10)
