"""Estimación condicionada a referencia observada, nunca al ruido oracle."""
import numpy as np
import pytest

from src.snr_estimation import calibrate_noise_reference, estimate_snr
from experiments.exp01_detection import wilson_interval


def test_estados_sin_estimacion():
    iq = np.ones(32)
    ref = calibrate_noise_reference(np.ones(100))
    assert estimate_snr(iq, detected=True)["status"] == "not_identifiable"
    assert estimate_snr(iq, detected=True)["reason"] == "missing_ref"
    assert estimate_snr(iq, ref, detected=False)["status"] == "no_signal"
    assert estimate_snr(iq, ref)["status"] == "detection_unknown"
    assert estimate_snr(iq, ref, True)["status"] == "nonpositive_signal_power"
    assert all(estimate_snr(iq, ref, detected=value)["snr_db"] is None for value in [None, False, True])
    assert estimate_snr(iq, {"noise_power_estimate": 1}, True)["status"] == "not_identifiable"
    with pytest.raises(ValueError):
        estimate_snr(iq, ref, detected=1)


def test_resta_potencias_y_deriva():
    ref = calibrate_noise_reference(np.ones(100))
    estimate = estimate_snr(np.full(256, np.sqrt(11)), ref, True)
    assert estimate["snr_db"] == pytest.approx(10)
    # Potencia señal10, ruido2: referencia vieja1 produce 10log10(11), no 10log10(5).
    drift = estimate_snr(np.full(256, np.sqrt(12)), ref, True)
    assert drift["snr_db"] > 10*np.log10(5)+3
    assert drift["status"] == "ok_under_stationarity_assumption"


def test_referencia_independiente_estadistica():
    rng_ref, rng_obs = [np.random.default_rng(s) for s in np.random.SeedSequence(7).spawn(2)]
    ref_noise = (rng_ref.normal(size=100000)+1j*rng_ref.normal(size=100000))/np.sqrt(2)
    ref = calibrate_noise_reference(ref_noise)
    n = 100000
    x = np.sqrt(10)*np.exp(2j*np.pi*.07*np.arange(n))
    observed = x+(rng_obs.normal(size=n)+1j*rng_obs.normal(size=n))/np.sqrt(2)
    assert estimate_snr(observed, ref, True)["snr_db"] == pytest.approx(10, abs=.08)
    with pytest.raises(ValueError):
        calibrate_noise_reference(np.zeros(10))


def test_wilson_limites():
    low, high = wilson_interval(0, 100)
    assert low == pytest.approx(0)
    assert high == pytest.approx(.0369934982)
    assert wilson_interval(50, 100) == pytest.approx([.4038315304, .5961684696])
    assert wilson_interval(100, 100)[1] == pytest.approx(1)
