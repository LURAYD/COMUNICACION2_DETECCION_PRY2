"""Periodograma bilateral para I/Q: densidad de potencia y ejes explícitos.

Convención de densidad: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
Implementación NumPy, sin dependencia de SciPy.
"""
from numbers import Real

import numpy as np


def _occupied_bandwidth(frequency, density, fs, power_fraction):
    df = fs / len(frequency)
    total = float(np.sum(density) * df)
    if total <= 0:
        return None, None, total
    # Histograma periódico: el bin de Nyquist par se divide entre ambos
    # extremos. Los bordes físicos nunca exceden [-fs/2, fs/2]. Se
    # interpola masa uniforme dentro de cada bin, no resolución física.
    if len(frequency) % 2 == 0:
        edges = np.r_[-fs / 2, frequency + df / 2, fs / 2]
        dens = np.r_[density, density[0]]
    else:
        edges = np.r_[frequency - df / 2, fs / 2]
        dens = density
    cdf = np.r_[0.0, np.cumsum(dens * np.diff(edges))]
    if cdf[-1] <= 0:
        return None, None, 0.0
    cdf /= cdf[-1]
    tail = (1.0 - power_fraction) / 2.0
    bounds = [float(np.interp(tail, cdf, edges)), float(np.interp(1.0 - tail, cdf, edges))]
    return bounds[1] - bounds[0], bounds, total


def estimate_useful_signal_bandwidth(spec_res, noise_power_estimate=None, power_fraction=0.99):
    """Estima ancho de banda neto tras restar el piso plano de ruido si hay referencia."""
    if noise_power_estimate is None:
        return {
            "signal_bandwidth_est": None,
            "signal_bandwidth_bounds": None,
            "signal_bandwidth_status": "requires_noise_reference",
            "signal_bandwidth_method": None,
        }
    if (
        isinstance(noise_power_estimate, (bool, np.bool_))
        or not isinstance(noise_power_estimate, Real)
        or not np.isfinite(noise_power_estimate)
        or noise_power_estimate <= 0
    ):
        raise ValueError("noise_power_estimate debe ser real, finito y positivo.")

    freqs = spec_res["frequency"]
    psd = spec_res["psd"]
    df = spec_res["df"]
    fs = df * len(freqs)
    noise_psd_floor = float(noise_power_estimate) / fs
    excess_psd = np.maximum(psd - noise_psd_floor, 0.0)
    bw, bounds, excess_power = _occupied_bandwidth(freqs, excess_psd, fs, power_fraction)
    if bw is None or excess_power <= 0:
        return {
            "signal_bandwidth_est": None,
            "signal_bandwidth_bounds": None,
            "signal_bandwidth_status": "nonpositive_excess_signal_psd",
            "signal_bandwidth_method": "flat_noise_floor_subtraction_rectified_psd",
        }
    return {
        "signal_bandwidth_est": float(bw),
        "signal_bandwidth_bounds": bounds,
        "signal_bandwidth_status": "estimated_above_noise_floor",
        "signal_bandwidth_method": "flat_noise_floor_subtraction_rectified_psd",
    }


def estimate_spectrum(iq, fs_hz=None, nfft=None, power_fraction=0.99, noise_power_estimate=None):
    """FFT centrada con Hann periódica, sin quitar DC.

    PSD = |FFT(iq*w)|²/(fs*sum(w²)). Su integral corresponde a la potencia
    ponderada por w²; no exactamente a mean(|iq|²) de una trama arbitraria.
    Si fs_hz falta, fs=1 y el eje usa ciclos/muestra. Ancho por cuantiles
    incluye ruido y DC: es ancho de potencia TOTAL, no estimación de la banda
    de señal aislada. Zero-padding interpola la malla, no mejora resolución.
    """
    x = np.asarray(iq, dtype=np.complex128)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("IQ requiere al menos dos muestras complejas finitas.")
    if fs_hz is not None and (
        isinstance(fs_hz, (bool, np.bool_))
        or not isinstance(fs_hz, Real)
        or not np.isfinite(fs_hz)
        or fs_hz <= 0
    ):
        raise ValueError("fs_hz debe ser positiva y finita o None.")
    fs = 1.0 if fs_hz is None else float(fs_hz)
    if nfft is None:
        nfft = len(x)
    if isinstance(nfft, (bool, np.bool_)) or not isinstance(nfft, (int, np.integer)) or nfft < len(x):
        raise ValueError("nfft debe ser entero >= número de muestras.")
    if (
        isinstance(power_fraction, (bool, np.bool_))
        or not isinstance(power_fraction, Real)
        or not 0 < power_fraction <= 1
    ):
        raise ValueError("power_fraction debe estar en (0,1].")
    window = np.hanning(len(x) + 1)[:-1]
    window_power = float(np.sum(window**2))
    transformed = np.fft.fftshift(np.fft.fft(x * window, n=int(nfft)))
    frequency = np.fft.fftshift(np.fft.fftfreq(int(nfft), d=1.0 / fs))
    with np.errstate(over="ignore", invalid="ignore"):
        psd = abs(transformed) ** 2 / (fs * window_power)
    if not np.isfinite(psd).all():
        raise ValueError("La PSD excede el rango numérico.")
    df = fs / nfft
    bandwidth, bounds, total = _occupied_bandwidth(frequency, psd, fs, power_fraction)

    result = {
        "frequency": frequency,
        "psd": psd,
        "fft": transformed / np.sum(window),
        "frequency_unit": "Hz" if fs_hz is not None else "cycles/sample",
        "psd_unit": "amplitude^2/Hz" if fs_hz is not None else "amplitude^2/(cycles/sample)",
        "df": float(df),
        "enbw": float(fs * window_power / np.sum(window) ** 2),
        "integrated_power": total,
        "total_power_fraction": float(power_fraction),
        "total_power_bandwidth": bandwidth,
        "total_power_bounds": bounds,
        "bandwidth_method": "central_fraction_piecewise_uniform_periodic_bins_in_nyquist",
        "bandwidth_scope": "total_observed_power_includes_noise",
        "window": "periodic_hann",
        "n_samples": len(x),
        "nfft": int(nfft),
    }
    useful = estimate_useful_signal_bandwidth(result, noise_power_estimate=noise_power_estimate, power_fraction=power_fraction)
    result.update(useful)
    return result
