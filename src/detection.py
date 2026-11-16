"""Detector de energía calibrado con observaciones independientes solo ruido."""
from numbers import Real

import numpy as np
from scipy.stats import chi2


def _validate_positive_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} debe ser real, finito y positivo.")
    return float(value)


def _validate_nonnegative_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} debe ser real, finito y no negativo.")
    return float(value)


def energy_statistic(iq):
    """Potencia media de muestras complejas; no es probabilidad de señal."""
    x = np.asarray(iq, dtype=np.complex128)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("IQ debe ser un vector no vacío y finito.")
    with np.errstate(over="ignore", invalid="ignore"):
        power = float(np.mean(abs(x) ** 2))
    if not np.isfinite(power):
        raise ValueError("La energía excede el rango numérico.")
    return power


def calibrate_energy(noise_windows, pfa=0.01, fs_hz=None, gain=1.0):
    """Cuantil empírico conservador de ventanas conocidas solo ruido.

    Las filas son realizaciones independientes de calibración, todas de igual
    longitud. El umbral no se ajusta con test. P_FA real puede diferir del
    objetivo por tamaño finito o cambio de distribución/piso de ruido.
    """
    if isinstance(pfa, (bool, np.bool_)) or not isinstance(pfa, Real) or not np.isfinite(pfa) or not 0 < pfa < 1:
        raise ValueError("pfa debe estar estrictamente entre 0 y 1.")
    if fs_hz is not None:
        fs_hz = _validate_positive_real(fs_hz, "fs_hz")
    gain = _validate_positive_real(gain, "gain")
    x = np.asarray(noise_windows, dtype=np.complex128)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("Se requieren al menos dos ventanas finitas de calibración.")
    powers = np.array([energy_statistic(row) for row in x])
    threshold = float(np.quantile(powers, 1 - pfa, method="higher"))
    noise_mean = float(np.mean(powers))
    return {
        "threshold": threshold,
        "n_samples": int(x.shape[1]),
        "pfa_target": float(pfa),
        "n_calibration": int(x.shape[0]),
        "quantile_method": "higher",
        "calibration_pfa": float(np.mean(powers > threshold)),
        "noise_power_mean": noise_mean,
        "normalized_threshold": float(threshold / noise_mean) if noise_mean > 0 else None,
        "fs_hz": fs_hz,
        "gain": gain,
        "method": "empirical_quantile",
    }


def calibrate_detection_from_reference(
    noise_iq,
    window_len=256,
    pfa_file=0.01,
    num_file_windows=1,
    fs_hz=None,
    gain=1.0,
    method="auto",
):
    """Calibra umbrales por ventana y por archivo desde una referencia de solo ruido.

    Garantiza invariancia ante ganancia global g: al escalar señal y referencia por g,
    la potencia media y los umbrales escalan por |g|^2 y la estadística normalizada se conserva.
    """
    if isinstance(window_len, (bool, np.bool_)) or not isinstance(window_len, (int, np.integer)) or window_len < 2:
        raise ValueError("window_len debe ser un entero >= 2.")
    window_len = int(window_len)
    if isinstance(num_file_windows, (bool, np.bool_)) or not isinstance(num_file_windows, (int, np.integer)) or num_file_windows < 1:
        raise ValueError("num_file_windows debe ser un entero >= 1.")
    num_file_windows = int(num_file_windows)
    if isinstance(pfa_file, (bool, np.bool_)) or not isinstance(pfa_file, Real) or not np.isfinite(pfa_file) or not 0 < pfa_file < 1:
        raise ValueError("pfa_file debe estar estrictamente entre 0 y 1.")
    pfa_file = float(pfa_file)
    if fs_hz is not None:
        fs_hz = _validate_positive_real(fs_hz, "fs_hz")
    gain = _validate_positive_real(gain, "gain")

    arr = np.asarray(noise_iq, dtype=np.complex128)
    if not np.isfinite(arr).all():
        raise ValueError("La referencia de ruido debe ser finita.")
    if arr.ndim == 1:
        if arr.size < window_len:
            raise ValueError(f"La referencia requiere al menos {window_len} muestras.")
        n_win = arr.size // window_len
        windows = arr[: n_win * window_len].reshape(n_win, window_len)
        noise_mean = energy_statistic(arr)
    elif arr.ndim == 2:
        if arr.shape[0] < 1 or arr.shape[1] != window_len:
            raise ValueError(f"Se esperan ventanas de longitud {window_len}.")
        windows = arr
        noise_mean = float(np.mean([energy_statistic(row) for row in windows]))
    else:
        raise ValueError("noise_iq debe ser 1D o 2D.")

    if noise_mean <= 0:
        raise ValueError("La referencia de ruido debe tener potencia positiva.")

    powers = np.array([energy_statistic(row) for row in windows])
    alpha_win_for_file = float(1.0 - (1.0 - pfa_file) ** (1.0 / num_file_windows))

    use_empirical = (
        method == "empirical_quantile"
        or (method == "auto" and len(powers) >= 500 and alpha_win_for_file * len(powers) >= 5.0)
    )

    if use_empirical:
        if len(powers) < 2:
            raise ValueError("Se requieren al menos 2 ventanas para cuantil empírico.")
        threshold_per_window = float(np.quantile(powers, 1.0 - pfa_file, method="higher"))
        threshold_per_file = float(np.quantile(powers, 1.0 - alpha_win_for_file, method="higher"))
        chosen_method = "empirical_quantile"
    else:
        gamma_win = float(chi2.ppf(1.0 - pfa_file, df=2 * window_len) / (2.0 * window_len))
        gamma_file = float(chi2.ppf(1.0 - alpha_win_for_file, df=2 * window_len) / (2.0 * window_len))
        threshold_per_window = gamma_win * noise_mean
        threshold_per_file = gamma_file * noise_mean
        chosen_method = "chi2_scaled_reference"

    return {
        "threshold": float(threshold_per_window),
        "threshold_per_window": float(threshold_per_window),
        "threshold_per_file": float(threshold_per_file),
        "n_samples": window_len,
        "num_file_windows": num_file_windows,
        "pfa_target": pfa_file,
        "pfa_window_for_file": alpha_win_for_file,
        "n_calibration": int(len(powers)),
        "noise_power_mean": float(noise_mean),
        "normalized_threshold": float(threshold_per_file / noise_mean),
        "fs_hz": fs_hz,
        "gain": gain,
        "method": chosen_method,
    }


def detect_energy(iq, calibration):
    """Solo observación y calibración: no recibe SNR ni ruido oracle."""
    power = energy_statistic(iq)
    if len(iq) != calibration["n_samples"]:
        raise ValueError("La ventana debe tener la longitud usada en calibración.")
    threshold = _validate_nonnegative_real(calibration["threshold"], "threshold")
    noise_mean = calibration.get("noise_power_mean")
    if noise_mean is not None and isinstance(noise_mean, Real) and np.isfinite(noise_mean) and noise_mean > 0:
        norm_stat = float(power / float(noise_mean))
    elif threshold > 0:
        norm_stat = float(power / threshold)
    else:
        norm_stat = None
    return {
        "detected": bool(power > threshold),
        "energy": power,
        "threshold": threshold,
        "normalized_statistic": norm_stat,
    }


def _check_calibration_compatibility(cal_meta, window_len, fs_hz, gain):
    if cal_meta is None:
        return None
    if "n_samples" in cal_meta and int(cal_meta["n_samples"]) < 2:
        return "invalid_calibration_n_samples"
    if "n_samples" in cal_meta and "threshold" in cal_meta and int(cal_meta["n_samples"]) != window_len:
        return "mismatched_window_len"
    cal_fs = cal_meta.get("fs_hz")
    if cal_fs is not None and fs_hz is not None and not np.isclose(float(cal_fs), float(fs_hz), rtol=1e-9, atol=0.0):
        return "mismatched_fs_hz"
    cal_gain = cal_meta.get("gain")
    if cal_gain is not None and gain is not None and not np.isclose(float(cal_gain), float(gain), rtol=1e-9, atol=0.0):
        return "mismatched_gain"
    return None


def detect_energy_file(
    iq,
    window_len=256,
    calibration=None,
    noise_reference=None,
    manual_threshold=None,
    pfa_file=0.01,
    fs_hz=None,
    gain=1.0,
):
    """Evalúa detección por ventana y por archivo completo con control explícito de calibración."""
    x = np.asarray(iq, dtype=np.complex128)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("IQ debe ser un vector 1D no vacío y finito.")
    if isinstance(window_len, (bool, np.bool_)) or not isinstance(window_len, (int, np.integer)) or window_len < 2:
        raise ValueError("window_len debe ser un entero >= 2.")
    window_len = int(window_len)
    if isinstance(pfa_file, (bool, np.bool_)) or not isinstance(pfa_file, Real) or not np.isfinite(pfa_file) or not 0 < pfa_file < 1:
        raise ValueError("pfa_file debe estar estrictamente entre 0 y 1.")
    pfa_file = float(pfa_file)
    if fs_hz is not None:
        fs_hz = _validate_positive_real(fs_hz, "fs_hz")
    gain = _validate_positive_real(gain, "gain")
    if manual_threshold is not None:
        manual_threshold = _validate_nonnegative_real(manual_threshold, "manual_threshold")

    if calibration is not None:
        incompat = _check_calibration_compatibility(calibration, window_len, fs_hz, gain)
        if incompat is not None:
            return {
                "detected": None,
                "observed_energy": energy_statistic(x),
                "mean_energy": energy_statistic(x),
                "threshold": None,
                "threshold_per_window": None,
                "threshold_per_file": None,
                "normalized_statistic": None,
                "normalized_window_statistics": [],
                "window_energies": [],
                "active_window_indices": [],
                "num_windows_evaluated": 0,
                "pfa_file_target": pfa_file,
                "status": "indeterminate_incompatible_calibration",
                "calibration_status": incompat,
                "decision_rule": "file_max_energy",
            }

    if noise_reference is not None:
        if noise_reference.get("method") != "independent_observed_noise_reference":
            return {
                "detected": None,
                "observed_energy": energy_statistic(x),
                "mean_energy": energy_statistic(x),
                "threshold": None,
                "threshold_per_window": None,
                "threshold_per_file": None,
                "normalized_statistic": None,
                "normalized_window_statistics": [],
                "window_energies": [],
                "active_window_indices": [],
                "num_windows_evaluated": 0,
                "pfa_file_target": pfa_file,
                "status": "indeterminate_invalid_reference",
                "calibration_status": "invalid_noise_reference",
                "decision_rule": "file_max_energy",
            }
        noise_pow = noise_reference.get("noise_power_estimate")
        if isinstance(noise_pow, (bool, np.bool_)) or not isinstance(noise_pow, Real) or not np.isfinite(noise_pow) or noise_pow <= 0:
            return {
                "detected": None,
                "observed_energy": energy_statistic(x),
                "mean_energy": energy_statistic(x),
                "threshold": None,
                "threshold_per_window": None,
                "threshold_per_file": None,
                "normalized_statistic": None,
                "normalized_window_statistics": [],
                "window_energies": [],
                "active_window_indices": [],
                "num_windows_evaluated": 0,
                "pfa_file_target": pfa_file,
                "status": "indeterminate_invalid_reference",
                "calibration_status": "invalid_noise_reference",
                "decision_rule": "file_max_energy",
            }
        incompat_ref = _check_calibration_compatibility(noise_reference, window_len, fs_hz, gain)
        if incompat_ref is not None and incompat_ref != "mismatched_window_len":
            return {
                "detected": None,
                "observed_energy": energy_statistic(x),
                "mean_energy": energy_statistic(x),
                "threshold": None,
                "threshold_per_window": None,
                "threshold_per_file": None,
                "normalized_statistic": None,
                "normalized_window_statistics": [],
                "window_energies": [],
                "active_window_indices": [],
                "num_windows_evaluated": 0,
                "pfa_file_target": pfa_file,
                "status": "indeterminate_incompatible_calibration",
                "calibration_status": incompat_ref,
                "decision_rule": "file_max_energy",
            }

    if len(x) < window_len:
        obs_power = energy_statistic(x)
        if manual_threshold is not None:
            detected = bool(obs_power > manual_threshold)
            return {
                "detected": detected,
                "observed_energy": obs_power,
                "mean_energy": obs_power,
                "threshold": manual_threshold,
                "threshold_per_window": manual_threshold,
                "threshold_per_file": manual_threshold,
                "normalized_statistic": float(obs_power / manual_threshold) if manual_threshold > 0 else None,
                "normalized_window_statistics": [],
                "window_energies": [obs_power],
                "active_window_indices": [0] if detected else [],
                "num_windows_evaluated": 1,
                "pfa_file_target": None,
                "status": "signal_present" if detected else "noise_only_or_undetected",
                "calibration_status": "manual_threshold_unvalidated",
                "decision_rule": "short_signal_manual_threshold",
            }
        return {
            "detected": None,
            "observed_energy": obs_power,
            "mean_energy": obs_power,
            "threshold": None,
            "threshold_per_window": None,
            "threshold_per_file": None,
            "normalized_statistic": None,
            "normalized_window_statistics": [],
            "window_energies": [obs_power],
            "active_window_indices": [],
            "num_windows_evaluated": 0,
            "pfa_file_target": pfa_file,
            "status": "indeterminate_short_signal",
            "calibration_status": "insufficient_samples_for_window_len",
            "decision_rule": "file_max_energy",
        }

    num_windows = len(x) // window_len
    window_energies = [energy_statistic(x[k * window_len : (k + 1) * window_len]) for k in range(num_windows)]
    max_energy = float(max(window_energies))
    mean_energy = float(np.mean(window_energies))

    if calibration is None and noise_reference is None and manual_threshold is None:
        return {
            "detected": None,
            "observed_energy": max_energy,
            "mean_energy": mean_energy,
            "threshold": None,
            "threshold_per_window": None,
            "threshold_per_file": None,
            "normalized_statistic": None,
            "normalized_window_statistics": [],
            "window_energies": window_energies,
            "active_window_indices": [],
            "num_windows_evaluated": num_windows,
            "pfa_file_target": pfa_file,
            "status": "indeterminate_uncalibrated",
            "calibration_status": "missing_calibration",
            "decision_rule": "file_max_energy",
        }

    noise_power_mean = None
    if calibration is not None:
        threshold_per_window = _validate_nonnegative_real(calibration["threshold"], "threshold")
        noise_power_mean = calibration.get("noise_power_mean")
        if noise_power_mean is not None and noise_power_mean > 0:
            noise_power_mean = float(noise_power_mean)
            if num_windows == 1:
                threshold_per_file = threshold_per_window
            else:
                alpha_win = float(1.0 - (1.0 - pfa_file) ** (1.0 / num_windows))
                chi2_thresh = float(chi2.ppf(1.0 - alpha_win, df=2 * window_len) / (2.0 * window_len) * noise_power_mean)
                threshold_per_file = max(threshold_per_window, chi2_thresh)
        else:
            threshold_per_file = threshold_per_window
        cal_status = f"calibrated_{calibration.get('method', 'external_calibration')}"
    elif noise_reference is not None:
        noise_power_mean = float(noise_reference["noise_power_estimate"])
        alpha_win = float(1.0 - (1.0 - pfa_file) ** (1.0 / num_windows))
        gamma_win = float(chi2.ppf(1.0 - pfa_file, df=2 * window_len) / (2.0 * window_len))
        gamma_file = float(chi2.ppf(1.0 - alpha_win, df=2 * window_len) / (2.0 * window_len))
        threshold_per_window = gamma_win * noise_power_mean
        threshold_per_file = gamma_file * noise_power_mean
        cal_status = "calibrated_from_noise_reference"
    else:
        threshold_per_window = float(manual_threshold)
        threshold_per_file = float(manual_threshold)
        cal_status = "manual_threshold_unvalidated"

    active_indices = [idx for idx, e in enumerate(window_energies) if e > threshold_per_file]
    uncorrected_indices = [idx for idx, e in enumerate(window_energies) if e > threshold_per_window]
    detected = len(active_indices) > 0

    if noise_power_mean is not None and noise_power_mean > 0:
        norm_windows = [float(e / noise_power_mean) for e in window_energies]
        norm_stat = float(max_energy / noise_power_mean)
    elif threshold_per_file > 0:
        norm_windows = [float(e / threshold_per_file) for e in window_energies]
        norm_stat = float(max_energy / threshold_per_file)
    else:
        norm_windows = []
        norm_stat = None

    return {
        "detected": bool(detected),
        "observed_energy": max_energy,
        "mean_energy": mean_energy,
        "threshold": float(threshold_per_file),
        "threshold_per_window": float(threshold_per_window),
        "threshold_per_file": float(threshold_per_file),
        "noise_power_reference": noise_power_mean,
        "normalized_statistic": norm_stat,
        "normalized_window_statistics": norm_windows,
        "window_energies": window_energies,
        "active_window_indices": active_indices,
        "window_only_active_indices": uncorrected_indices,
        "num_windows_evaluated": num_windows,
        "pfa_file_target": pfa_file if manual_threshold is None else None,
        "status": "signal_present" if detected else "noise_only_or_undetected",
        "calibration_status": cal_status,
        "decision_rule": "file_max_energy_sidak" if num_windows > 1 and manual_threshold is None else "single_or_manual_window",
    }
