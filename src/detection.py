"""Detector de energía calibrado con observaciones independientes solo ruido."""
from numbers import Real

import numpy as np


def energy_statistic(iq):
    """Potencia media de muestras complejas; no es probabilidad de señal."""
    x = np.asarray(iq, dtype=np.complex128)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("IQ debe ser un vector no vacío y finito.")
    with np.errstate(over="ignore", invalid="ignore"):
        power = float(np.mean(abs(x)**2))
    if not np.isfinite(power):
        raise ValueError("La energía excede el rango numérico.")
    return power


def calibrate_energy(noise_windows, pfa=0.01):
    """Cuantil empírico conservador de ventanas conocidas solo ruido.

    Las filas son realizaciones independientes de calibración, todas de igual
    longitud. El umbral no se ajusta con test. P_FA real puede diferir del
    objetivo por tamaño finito o cambio de distribución/piso de ruido.
    """
    if isinstance(pfa, bool) or not isinstance(pfa, Real) or not np.isfinite(pfa) or not 0 < pfa < 1:
        raise ValueError("pfa debe estar estrictamente entre 0 y 1.")
    x = np.asarray(noise_windows, dtype=np.complex128)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("Se requieren al menos dos ventanas finitas de calibración.")
    powers = np.array([energy_statistic(row) for row in x])
    threshold = float(np.quantile(powers, 1-pfa, method="higher"))
    return {"threshold": threshold, "n_samples": int(x.shape[1]), "pfa_target": float(pfa),
            "n_calibration": int(x.shape[0]), "quantile_method": "higher",
            "calibration_pfa": float(np.mean(powers > threshold))}


def detect_energy(iq, calibration):
    """Solo observación y calibración: no recibe SNR ni ruido oracle."""
    power = energy_statistic(iq)
    if len(iq) != calibration["n_samples"]:
        raise ValueError("La ventana debe tener la longitud usada en calibración.")
    threshold = calibration["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("El umbral debe ser real, finito y no negativo.")
    return {"detected": bool(power > threshold), "energy": power, "threshold": float(threshold)}
