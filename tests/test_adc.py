"""Pruebas unitarias del módulo ADC virtual (Fase 10)."""
import numpy as np
import pytest

from src.adc import evaluate_sqnr_db, quantize_component, quantize_iq


def test_adc_uniform_levels_and_bounds():
    rng = np.random.default_rng(20261111)
    iq = rng.normal(0.0, 0.8, size=(16, 256)) + 1j * rng.normal(0.0, 0.8, size=(16, 256))
    for bits in (4, 6, 8):
        q, info = quantize_iq(iq, bits=bits, full_scale=2.5, return_diagnostics=True)
        assert q.shape == iq.shape
        assert np.iscomplexobj(q)
        assert np.isfinite(q).all()
        assert info["levels"] == (1 << bits)
        assert np.isclose(info["delta"], 5.0 / (1 << bits))
        # Todos los valores cuantizados están estrictamente dentro de [-V_FS + Delta/2, V_FS - Delta/2]
        bound = 2.5 - 0.5 * info["delta"] + 1e-12
        assert np.max(np.abs(q.real)) <= bound
        assert np.max(np.abs(q.imag)) <= bound
        # El número de niveles únicos observados nunca excede 2^B
        assert len(np.unique(np.round(q.real, 10))) <= (1 << bits)
        assert len(np.unique(np.round(q.imag, 10))) <= (1 << bits)


def test_adc_sqnr_monotonic_with_bits_and_theoretical_6db_per_bit():
    rng = np.random.default_rng(42)
    # Señal uniforme sin clipping en [-2.0, 2.0] para probar régimen granular puro
    iq = rng.uniform(-2.0, 2.0, size=10000) + 1j * rng.uniform(-2.0, 2.0, size=10000)
    sqnrs = []
    for bits in (4, 5, 6, 8, 10):
        q = quantize_iq(iq, bits=bits, full_scale=2.0)
        res = evaluate_sqnr_db(iq, q)
        sqnrs.append(res["sqnr_db"])
    # Incrementar bits mejora estrictamente la SQNR (~6.02 dB por bit en régimen sin saturación)
    diffs = np.diff(sqnrs)
    assert np.all(diffs > 5.5)
    assert np.isclose(sqnrs[2] - sqnrs[1], 6.02, atol=0.25)


def test_adc_saturation_and_fixed_full_scale():
    # Verifica que full_scale sea fijo y no escale adaptativamente a valores extremos
    iq = np.array([0.1 - 0.1j, 10.0 - 12.0j, -8.0 + 0.2j], dtype=np.complex128)
    q, info = quantize_iq(iq, bits=6, full_scale=2.0, return_diagnostics=True)
    delta = 4.0 / 64.0
    max_level = 2.0 - 0.5 * delta
    assert np.isclose(q[1].real, max_level)
    assert np.isclose(q[1].imag, -max_level)
    assert np.isclose(q[2].real, -max_level)
    assert info["sample_saturation_rate"] == pytest.approx(2.0 / 3.0)
    assert info["clipping_rate_i"] == pytest.approx(2.0 / 3.0)
    assert info["clipping_rate_q"] == pytest.approx(1.0 / 3.0)


def test_adc_invalid_parameters():
    iq = np.ones(64, dtype=np.complex128)
    with pytest.raises(ValueError):
        quantize_iq(iq, bits=0, full_scale=2.0)
    with pytest.raises(ValueError):
        quantize_iq(iq, bits=True, full_scale=2.0)
    with pytest.raises(ValueError):
        quantize_iq(iq, bits=8, full_scale=0.0)
    with pytest.raises(ValueError):
        quantize_iq(iq, bits=8, full_scale=-1.5)
    with pytest.raises(ValueError):
        quantize_component(np.array([np.nan, 1.0]), bits=8, full_scale=2.0)
