"""Canales discretos por trama; los parámetros verdaderos son de simulación."""
from numbers import Real

import numpy as np


def _complex_vector(values, name):
    x = np.asarray(values, dtype=np.complex128)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} debe ser un vector no vacío y finito.")
    return x


def _real_scalar(value, name):
    if (not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value)):
        raise ValueError(f"{name} debe ser un escalar real finito.")
    return float(value)


def add_awgn(iq, snr_db, rng=None):
    """Añade AWGN complejo, con SNR por potencia media de muestras de entrada.

    Devuelve (iq_con_ruido, ruido). Cada componente tiene varianza objetivo
    P_ruido/2. El ruido devuelto y snr_db son información oracle para evaluar
    el simulador; no son entradas del futuro monitor. No equivale a Eb/N0.
    Con rng=None la realización es no determinista; pasar Generator con semilla
    permite reproducibilidad. Una entrada de potencia cero se rechaza.
    """
    x = _complex_vector(iq, "IQ")
    snr_db = _real_scalar(snr_db, "SNR en dB")
    if rng is None:
        rng = np.random.default_rng()
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng debe ser un numpy.random.Generator.")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        signal_power = np.mean(np.abs(x)**2)
        snr_linear = np.power(10., snr_db/10)
        noise_power = signal_power/snr_linear
    if not np.isfinite(signal_power) or signal_power <= 0:
        raise ValueError("La potencia de IQ debe ser positiva y representable.")
    if not np.isfinite(snr_linear) or snr_linear <= 0 or not np.isfinite(noise_power) or noise_power <= 0:
        raise ValueError("La SNR solicitada produce una potencia de ruido no representable.")
    sigma = np.sqrt(noise_power)/np.sqrt(2)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("La escala del ruido no es representable.")
    noise = sigma*(rng.standard_normal(x.size) + 1j*rng.standard_normal(x.size))
    return x + noise, noise


def apply_carrier_impairments(iq, fs_hz, phase_rad=0, cfo_hz=0, sample_offset=0):
    """Aplica exp(j*(phase_rad + 2π*cfo_hz*n/fs_hz)) sin cambiar amplitud.

    fs_hz es muestras/segundo y debe ser positiva; cfo_hz está en Hz y
    phase_rad en radianes. n empieza en sample_offset, índice absoluto entero
    no negativo. Al procesar ventanas de una trama se mantiene el mismo
    phase_rad y se incrementa sample_offset; no se reinicia la fase.
    CFO fuera de Nyquist tiene aliasado discreto; no se limita implícitamente.
    Los índices se calculan en float64: por encima de 2**53 no se garantiza
    resolución de una muestra; fases muy grandes también pierden precisión.
    """
    x = _complex_vector(iq, "IQ")
    fs_hz = _real_scalar(fs_hz, "fs_hz")
    phase_rad = _real_scalar(phase_rad, "phase_rad")
    cfo_hz = _real_scalar(cfo_hz, "cfo_hz")
    if fs_hz <= 0:
        raise ValueError("fs_hz debe ser positiva.")
    if (not isinstance(sample_offset, (int, np.integer))
            or isinstance(sample_offset, (bool, np.bool_)) or sample_offset < 0):
        raise ValueError("sample_offset debe ser entero no negativo.")
    try:
        n = np.arange(x.size, dtype=float) + float(sample_offset)
    except OverflowError as error:
        raise ValueError("sample_offset excede el rango numérico.") from error
    with np.errstate(over="ignore", invalid="ignore"):
        phase = phase_rad + (2*np.pi*(cfo_hz/fs_hz))*n
    if not np.all(np.isfinite(phase)):
        raise ValueError("La fase calculada excede el rango numérico.")
    return x*np.exp(1j*phase)


def apply_multipath(iq, taps):
    """FIR estático por trama mediante convolución completa.

    taps[k] representa ganancia compleja del trayecto con retardo k muestras
    (k/fs_hz segundos si se conoce fs_hz). No renormaliza la ganancia ni recorta
    transitorios: la salida tiene len(iq)+len(taps)-1 muestras. Se permite un
    canal nulo. Procesar ventanas aisladas no conserva memoria FIR: aplicar a
    la trama completa antes de segmentar. No modela Rayleigh ni RF física.
    """
    x = _complex_vector(iq, "IQ")
    taps = _complex_vector(taps, "taps")
    with np.errstate(over="ignore", invalid="ignore"):
        y = np.convolve(x, taps, mode="full")
    if not np.all(np.isfinite(y)):
        raise ValueError("La salida FIR excede el rango numérico.")
    return y
