"""Periodograma bilateral para I/Q: densidad de potencia y ejes explícitos.

Convención de densidad: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
Implementación NumPy, sin dependencia de SciPy.
"""
from numbers import Real

import numpy as np


def estimate_spectrum(iq, fs_hz=None, nfft=None, power_fraction=.99):
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
    if fs_hz is not None and (isinstance(fs_hz, bool) or not isinstance(fs_hz, Real)
                              or not np.isfinite(fs_hz) or fs_hz <= 0):
        raise ValueError("fs_hz debe ser positiva y finita o None.")
    fs = 1. if fs_hz is None else float(fs_hz)
    if nfft is None:
        nfft = len(x)
    if isinstance(nfft, bool) or not isinstance(nfft, (int, np.integer)) or nfft < len(x):
        raise ValueError("nfft debe ser entero >= número de muestras.")
    if isinstance(power_fraction, (bool, np.bool_)) or not isinstance(power_fraction, Real) or not 0 < power_fraction <= 1:
        raise ValueError("power_fraction debe estar en (0,1].")
    window = np.hanning(len(x)+1)[:-1]
    window_power = float(np.sum(window**2))
    transformed = np.fft.fftshift(np.fft.fft(x*window, n=int(nfft)))
    frequency = np.fft.fftshift(np.fft.fftfreq(int(nfft), d=1/fs))
    with np.errstate(over="ignore", invalid="ignore"):
        psd = abs(transformed)**2/(fs*window_power)
    if not np.isfinite(psd).all():
        raise ValueError("La PSD excede el rango numérico.")
    df = fs/nfft
    total = float(np.sum(psd)*df)
    bounds = None
    bandwidth = None
    if total > 0:
        # Histograma periódico: el bin de Nyquist par se divide entre ambos
        # extremos. Los bordes físicos nunca exceden [-fs/2, fs/2]. Se
        # interpola masa uniforme dentro de cada bin, no resolución física.
        if nfft % 2 == 0:
            edges = np.r_[-fs/2, frequency+df/2, fs/2]
            density = np.r_[psd, psd[0]]
        else:
            edges = np.r_[frequency-df/2, fs/2]
            density = psd
        cdf = np.r_[0., np.cumsum(density*np.diff(edges))]
        cdf /= cdf[-1]
        tail = (1-power_fraction)/2
        bounds = [float(np.interp(tail, cdf, edges)), float(np.interp(1-tail, cdf, edges))]
        bandwidth = bounds[1]-bounds[0]
    return {"frequency": frequency, "psd": psd, "fft": transformed/np.sum(window),
            "frequency_unit": "Hz" if fs_hz is not None else "cycles/sample",
            "psd_unit": "amplitude^2/Hz" if fs_hz is not None else "amplitude^2/(cycles/sample)",
            "df": float(df), "enbw": float(fs*window_power/np.sum(window)**2),
            "integrated_power": total, "total_power_fraction": float(power_fraction),
            "total_power_bandwidth": bandwidth, "total_power_bounds": bounds,
            "bandwidth_method": "central_fraction_piecewise_uniform_periodic_bins_in_nyquist",
            "window": "periodic_hann", "n_samples": len(x), "nfft": int(nfft)}
