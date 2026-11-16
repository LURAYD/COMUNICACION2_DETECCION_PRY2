"""Validación física de canales de Fase 2, sin entrenamiento ni dataset."""
import numpy as np
import pytest

from src.channels import add_awgn, apply_carrier_impairments, apply_multipath
from src.signals import qpsk_modulation
from src.waveform import pulse_shape


@pytest.mark.parametrize("snr_db", [-10, 0, 10, 20])
@pytest.mark.parametrize("seed", [2026, 73, 180])
def test_awgn_estadistico(snr_db, seed):
    x = np.full(200_000, 2+1j)
    y, noise = add_awgn(x, snr_db, np.random.default_rng(seed))
    expected = 5/10**(snr_db/10)
    np.testing.assert_allclose(y, x+noise)
    assert 10*np.log10(5/np.mean(abs(noise)**2)) == pytest.approx(snr_db, abs=.08)
    assert np.var(noise.real) == pytest.approx(expected/2, rel=.02)
    assert np.var(noise.imag) == pytest.approx(expected/2, rel=.02)
    assert abs(np.mean(noise)) < .01*np.sqrt(expected)
    assert abs(np.corrcoef(noise.real, noise.imag)[0, 1]) < .01


def test_awgn_reproducibilidad_y_escalado():
    x = np.array([1+1j, 2-1j]*1000)
    first = add_awgn(x, 7, np.random.default_rng(12))
    repeat = add_awgn(x, 7, np.random.default_rng(12))
    scaled = add_awgn(3*x, 7, np.random.default_rng(12))
    for a, b, c in zip(first, repeat, scaled):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(3*a, c)
    assert not np.array_equal(first[1], add_awgn(x, 7, np.random.default_rng(13))[1])


@pytest.mark.parametrize("bad", [[], [[1]], [np.nan], [np.inf]])
@pytest.mark.parametrize("function", [lambda x: add_awgn(x, 10),
                                     lambda x: apply_carrier_impairments(x, 8000),
                                     lambda x: apply_multipath(x, [1])])
def test_iq_invalido(bad, function):
    with pytest.raises(ValueError):
        function(bad)


@pytest.mark.parametrize("snr", [np.nan, np.inf, 1j, [10], True, '10', 4000, -4000])
def test_snr_invalida(snr):
    with pytest.raises(ValueError):
        add_awgn([1j], snr)


def test_awgn_potencia_y_rng_invalidos():
    for x in [[0j], [1e300], [1e-300]]:
        with pytest.raises(ValueError):
            add_awgn(x, 10)
    with pytest.raises(ValueError):
        add_awgn([1j], 10, rng=123)


def test_fase_fija_y_cfo_analitico():
    x = np.array([1+1j, 2, -1j, -3])
    np.testing.assert_allclose(apply_carrier_impairments(x, 8, phase_rad=np.pi/2), 1j*x, atol=1e-14)
    np.testing.assert_allclose(apply_carrier_impairments(np.ones(4), 8, cfo_hz=2), [1, 1j, -1, -1j], atol=1e-14)


@pytest.mark.parametrize("cfo", [-250, 0, 350])
def test_cfo_conserva_potencia_y_continuidad(cfo):
    rng = np.random.default_rng(4)
    x = rng.normal(size=1001)+1j*rng.normal(size=1001)
    whole = apply_carrier_impairments(x, 8000, .3, cfo, sample_offset=17)
    split = np.concatenate([apply_carrier_impairments(x[:413], 8000, .3, cfo, 17),
                            apply_carrier_impairments(x[413:], 8000, .3, cfo, 17+413)])
    np.testing.assert_array_equal(whole, split)
    np.testing.assert_allclose(abs(whole), abs(x), atol=1e-14)
    assert np.mean(abs(whole)**2) == pytest.approx(np.mean(abs(x)**2))


@pytest.mark.parametrize("kwargs", [{"fs_hz": 0}, {"fs_hz": -1}, {"fs_hz": np.inf},
    {"phase_rad": [0]}, {"phase_rad": np.nan}, {"cfo_hz": 1j}, {"cfo_hz": True},
    {"sample_offset": -1}, {"sample_offset": 1.5}, {"sample_offset": True}])
def test_portadora_parametros_invalidos(kwargs):
    parameters = {"fs_hz": 8000, **kwargs}
    with pytest.raises(ValueError):
        apply_carrier_impairments([1j], **parameters)


def test_fir_identidad_eco_ganancia_retardo():
    x = np.array([1+1j, 2-1j, -1])
    np.testing.assert_array_equal(apply_multipath(x, [1]), x)
    np.testing.assert_array_equal(apply_multipath(x, [2j]), 2j*x)
    assert np.mean(abs(apply_multipath(x, [2j]))**2) == pytest.approx(4*np.mean(abs(x)**2))
    np.testing.assert_array_equal(apply_multipath(x, [0, 0, 1]), np.r_[0j, 0j, x])
    expected = np.zeros(5, complex)
    expected[:3] = x
    expected[2:] += .5j*x
    np.testing.assert_array_equal(apply_multipath(x, [1, 0, .5j]), expected)
    np.testing.assert_array_equal(apply_multipath(x, [0]), np.zeros(3))


@pytest.mark.parametrize("taps", [[], [[1]], [np.inf], [np.nan]])
def test_taps_invalidos(taps):
    with pytest.raises(ValueError):
        apply_multipath([1j], taps)


def test_fir_desbordamiento():
    with np.errstate(all="raise"), pytest.raises(ValueError, match="rango numérico"):
        apply_multipath([1e300], [1e300])


def test_cadena_reproducible():
    def chain(seed):
        rng = np.random.default_rng(seed)
        symbols = qpsk_modulation(rng.integers(0, 2, 512))
        iq, _ = pulse_shape(symbols)
        faded = apply_multipath(iq, [1, 0, .3+.2j])
        rotated = apply_carrier_impairments(faded, 8000, .4, 125)
        y, noise = add_awgn(rotated, 10, rng)
        return y, noise, rotated
    first, second = chain(2026), chain(2026)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    y, noise, signal = first
    assert len(y) == 256*8+80+2
    np.testing.assert_allclose(y-noise, signal, atol=1e-14)
