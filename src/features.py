"""Características de I/Q observable, sin sincronización ni parámetros oracle.

Escalas temporales prefijadas de 1/2/4/8 muestras; no estiman sps. Los
momentos se calculan tras centrar y normalizar potencia. El espectro es un
periodograma Hann periódico normalizado, en ciclos/muestra. Sin fit global.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


STAT_NAMES = ("dc_power_fraction", "envelope_mean", "envelope_std",
              "envelope_q10", "envelope_q25", "envelope_q50", "envelope_q75",
              "envelope_q90", "envelope_q99", "envelope_peak", "moment_abs4",
              "moment_abs6", "moment_abs8", "circularity", "c40_abs", "c42_real",
              "derotated_circularity", "derotated_c40_abs",
              "block64_circularity_mean", "block64_c40_abs_mean", "block64_c42_mean")
PHASE_NAMES = tuple(f"phase_resultant_{m}" for m in (1, 2, 4, 8))
DIFF_NAMES = tuple(name for lag in (1, 2, 4, 8)
                   for name in (f"autocorr_abs_lag{lag}", f"envelope_diff2_lag{lag}",
                                *(f"differential_resultant_m{m}_lag{lag}" for m in (1, 2, 4, 8))))
SPECTRAL_NAMES = ("spectral_entropy", "spectral_flatness", "spectral_peak_fraction",
                  "spectral_centroid", "spectral_rms_width", "spectral_bw90",
                  "spectral_bw99", "spectral_positive_fraction",
                  *(f"spectral_band_{i}" for i in range(8)))
FEATURE_NAMES = STAT_NAMES+PHASE_NAMES+DIFF_NAMES+SPECTRAL_NAMES
FEATURE_VERSION = "observable_v1"
CORE_INDICES = tuple(range(len(STAT_NAMES)+len(PHASE_NAMES)+len(DIFF_NAMES)))


def _cumulants(z):
    p = float(np.mean(abs(z)**2))
    if p <= np.finfo(float).eps:
        return (0., 0., 0.)
    m20 = np.mean(z*z)/p
    m40 = np.mean(z**4)/(p*p)
    m42 = np.mean(abs(z)**4)/(p*p)
    return float(abs(m20)), float(abs(m40-3*m20*m20)), float(m42-abs(m20)**2-2)


def extract_features(iq):
    """Vector finito de descriptores de una observación de al menos 64 IQ.

    C40=M40-3M20² y C42=M42-|M20|²-2M21², normalizados por M21².
    La corrección de portadora usa arg de autocorrelación de lag1 observada;
    no es sincronización perfecta. Una entrada constante/cero es degenerada
    pero produce descriptores finitos; detectar presencia es otra tarea.
    """
    x = np.asarray(iq, dtype=np.complex128)
    if x.ndim != 1 or len(x) < 64 or not np.isfinite(x).all():
        raise ValueError("IQ debe ser un vector finito de al menos 64 muestras.")
    scale = max(float(abs(x.real).max()), float(abs(x.imag).max()))
    raw = x.real/scale+1j*(x.imag/scale) if scale > 0 else np.zeros_like(x)
    raw_power = float(np.mean(abs(raw)**2))
    dc = float(abs(np.mean(raw))**2/max(raw_power, np.finfo(float).eps))
    z = raw-np.mean(raw)
    power = float(np.mean(abs(z)**2))
    z /= np.sqrt(max(power, np.finfo(float).eps))
    envelope = abs(z)
    quantiles = np.quantile(envelope, [.1, .25, .5, .75, .9, .99])
    circ, c40, c42 = _cumulants(z)
    # Para diferencias se preserva la media: la fase absoluta y un CFO
    # constante solo rotan cada producto, eliminado por su resultante.
    raw /= np.sqrt(max(raw_power, np.finfo(float).eps))
    carrier_step = float(np.angle(np.mean(raw[1:]*raw[:-1].conj())))
    derotated = raw*np.exp(-1j*carrier_step*np.arange(len(raw)))
    derotated -= np.mean(derotated)
    der_circ, der_c40, _ = _cumulants(derotated)
    blocks = [_cumulants(block-np.mean(block)) for block in z.reshape(-1, 64)] if len(z) % 64 == 0 else [_cumulants(z)]
    block_means = np.mean(blocks, axis=0)
    values = [dc, float(envelope.mean()), float(envelope.std()), *map(float, quantiles),
              float(envelope.max()), *(float(np.mean(envelope**k)) for k in (4, 6, 8)),
              circ, c40, c42, der_circ, der_c40, *map(float, block_means)]
    unit = np.zeros_like(z)
    np.divide(z, envelope, out=unit, where=envelope > np.finfo(float).eps)
    values.extend(float(abs(np.mean(unit**m))) for m in (1, 2, 4, 8))
    raw_envelope = abs(raw)
    for lag in (1, 2, 4, 8):
        product = raw[lag:]*raw[:-lag].conj()
        product_unit = np.zeros_like(product)
        np.divide(product, abs(product), out=product_unit, where=abs(product) > np.finfo(float).eps)
        values.extend([float(abs(product.mean())), float(np.mean((raw_envelope[lag:]-raw_envelope[:-lag])**2)),
                       *(float(abs(np.mean(product_unit**m))) for m in (1, 2, 4, 8))])
    w = np.hanning(len(z)+1)[:-1]
    psd = abs(np.fft.fftshift(np.fft.fft(z*w)))**2
    psd_total = float(psd.sum())
    if psd_total > np.finfo(float).eps:
        q = psd/psd_total
        f = np.fft.fftshift(np.fft.fftfreq(len(z)))
        centroid = float(np.sum(f*q))
        cdf = np.cumsum(q)
        bandwidths = []
        for tail in (.05, .005):
            low, high = np.searchsorted(cdf, [tail, 1-tail])
            bandwidths.append(float((high-low+1)/len(z)))
        values.extend([float(-np.sum(q*np.log(np.maximum(q, np.finfo(float).tiny)))/np.log(len(z))),
                       float(np.exp(np.mean(np.log(np.maximum(q, np.finfo(float).tiny))))*len(z)),
                       float(q.max()), centroid, float(np.sqrt(np.sum((f-centroid)**2*q))),
                       *bandwidths, float(q[f >= 0].sum()),
                       *(float(band.sum()) for band in np.array_split(q, 8))])
    else:
        values.extend([0.]*len(SPECTRAL_NAMES))
    features = np.asarray(values, dtype=np.float64)
    if len(features) != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise ValueError("Las características no son representables.")
    return features


def extract_feature_matrix(iq_windows):
    x = np.asarray(iq_windows, dtype=np.complex128)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("Se requiere una matriz no vacía de ventanas IQ.")
    return np.stack([extract_features(row) for row in x])


class IQFeatureTransformer(TransformerMixin, BaseEstimator):
    """Paso serializable del pipeline; no aprende estadísticas del dataset."""
    def __init__(self, n_samples=256):
        self.n_samples = n_samples

    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, X):
        x = np.asarray(X)
        if x.ndim != 2 or x.shape[1] != self.n_samples:
            raise ValueError(f"El modelo requiere ventanas de {self.n_samples} IQ.")
        return extract_feature_matrix(x)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)
