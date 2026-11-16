"""SNR por referencia solo ruido observada, bajo estacionariedad explícita.

Requiere referencia independiente con igual ganancia, frecuencia de muestreo y
banda de observación. No usa señal limpia ni ruido de la misma trama. Una
deriva de ruido inadvertida sesga el resultado; no se detecta automáticamente.
"""
from numbers import Real

import numpy as np

from src.detection import energy_statistic


def calibrate_noise_reference(noise_iq, fs_hz=None, gain=1.0, window_len=256):
    """Estima potencia de un segmento observado y conocido como solo ruido."""
    if fs_hz is not None:
        if isinstance(fs_hz, (bool, np.bool_)) or not isinstance(fs_hz, Real) or not np.isfinite(fs_hz) or fs_hz <= 0:
            raise ValueError("fs_hz debe ser real, finito y positivo o None.")
        fs_hz = float(fs_hz)
    if isinstance(gain, (bool, np.bool_)) or not isinstance(gain, Real) or not np.isfinite(gain) or gain <= 0:
        raise ValueError("gain debe ser real, finito y positivo.")
    gain = float(gain)
    power = energy_statistic(noise_iq)
    if power <= 0:
        raise ValueError("La referencia debe tener potencia positiva.")
    n_samples = len(noise_iq)
    return {
        "noise_power_estimate": power,
        "n_samples": n_samples,
        "window_len": int(window_len),
        "n_windows": int(n_samples // int(window_len)),
        "fs_hz": fs_hz,
        "gain": gain,
        "method": "independent_observed_noise_reference",
    }


def estimate_snr(iq, reference=None, detected=None, fs_hz=None, gain=1.0):
    """Resta potencia de referencia a potencia observada; sin oracle.

    null identifica falta de condiciones para estimar; no se fuerza a cero
    una resta negativa. detected proviene del detector, no de la etiqueta.
    """
    observed = energy_statistic(iq)
    result = {
        "snr_db": None,
        "status": "not_identifiable",
        "reason": "missing_ref",
        "method": None,
        "observed_power": observed,
        "noise_power_estimate": None,
        "signal_power_estimate": None,
    }
    if detected is not None and not isinstance(detected, (bool, np.bool_)):
        raise ValueError("detected debe ser bool o None.")
    if detected is False or (isinstance(detected, np.bool_) and not detected):
        result["status"] = "no_signal"
        result["reason"] = "detector_negative"
        return result
    if detected is None:
        result["status"] = "detection_unknown"
        result["reason"] = "missing_detection_decision"
        return result
    if reference is None:
        return result
    if reference.get("method") != "independent_observed_noise_reference":
        result["reason"] = "unverified_external_reference"
        return result
    ref_fs = reference.get("fs_hz")
    if ref_fs is not None and fs_hz is not None and not np.isclose(float(ref_fs), float(fs_hz), rtol=1e-9, atol=0.0):
        result["status"] = "incompatible_reference"
        result["reason"] = "mismatched_fs_hz"
        return result
    ref_gain = reference.get("gain")
    if ref_gain is not None and gain is not None and not np.isclose(float(ref_gain), float(gain), rtol=1e-9, atol=0.0):
        result["status"] = "incompatible_reference"
        result["reason"] = "mismatched_gain"
        return result

    noise_power = reference["noise_power_estimate"]
    if (
        isinstance(noise_power, (bool, np.bool_))
        or not np.isscalar(noise_power)
        or not np.isrealobj(noise_power)
        or not np.isfinite(noise_power)
        or noise_power <= 0
    ):
        raise ValueError("Referencia de ruido inválida.")
    signal_power = observed - float(noise_power)
    result.update(
        noise_power_estimate=float(noise_power),
        signal_power_estimate=float(signal_power),
        method="independent_observed_noise_reference",
    )
    if signal_power <= 0:
        result["status"] = "nonpositive_signal_power"
        result["reason"] = "nonpositive_power_difference"
        return result
    result.update(
        snr_db=float(10.0 * (np.log10(signal_power) - np.log10(float(noise_power)))),
        status="ok_under_stationarity_assumption",
        reason=None,
    )
    return result
