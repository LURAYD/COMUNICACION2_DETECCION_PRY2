"""SNR por referencia solo ruido observada, bajo estacionariedad explícita.

Requiere referencia independiente con igual ganancia, frecuencia de muestreo y
banda de observación. No usa señal limpia ni ruido de la misma trama. Una
deriva de ruido inadvertida sesga el resultado; no se detecta automáticamente.
"""
import numpy as np

from src.detection import energy_statistic


def calibrate_noise_reference(noise_iq):
    """Estima potencia de un segmento observado y conocido como solo ruido."""
    power = energy_statistic(noise_iq)
    if power <= 0:
        raise ValueError("La referencia debe tener potencia positiva.")
    return {"noise_power_estimate": power, "n_samples": len(noise_iq),
            "method": "independent_observed_noise_reference"}


def estimate_snr(iq, reference=None, detected=None):
    """Resta potencia de referencia a potencia observada; sin oracle.

    null identifica falta de condiciones para estimar; no se fuerza a cero
    una resta negativa. detected proviene del detector, no de la etiqueta.
    """
    observed = energy_statistic(iq)
    result = {"snr_db": None, "status": "not_identifiable", "reason": "missing_ref", "observed_power": observed,
              "noise_power_estimate": None, "signal_power_estimate": None}
    if detected is not None and not isinstance(detected, (bool, np.bool_)):
        raise ValueError("detected debe ser bool o None.")
    if detected is False or isinstance(detected, np.bool_) and not detected:
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
    noise_power = reference["noise_power_estimate"]
    if isinstance(noise_power, (bool, np.bool_)) or not np.isscalar(noise_power) or not np.isrealobj(noise_power) or not np.isfinite(noise_power) or noise_power <= 0:
        raise ValueError("Referencia de ruido inválida.")
    signal_power = observed-noise_power
    result.update(noise_power_estimate=float(noise_power), signal_power_estimate=float(signal_power))
    if signal_power <= 0:
        result["status"] = "nonpositive_signal_power"
        result["reason"] = "nonpositive_power_difference"
        return result
    result.update(snr_db=float(10*(np.log10(signal_power)-np.log10(noise_power))),
                  status="ok_under_stationarity_assumption", reason=None)
    return result
