"""Cuantizador ADC virtual uniforme sobre componentes I/Q a escala fija (Fase 10).

Reglas de diseño sin leakage (GUIA_MAESTRA Secciones 2.3, 3.8 y Fase 10):
- Cuantiza independientemente I y Q con resolución de B bits sobre un rango
  fijo [-full_scale, +full_scale], sin ajustar full_scale por ventana ni usar
  estadísticas futuras de test.
- Convención: cuantizador uniforme simétrico mid-rise de L = 2^B niveles en
  [-full_scale, +full_scale], con paso Delta = 2 * full_scale / 2^B y niveles
  reconstruidos en los centros de celda:
      q(x) = (-full_scale + (k + 0.5) * Delta),  k in {0, ..., 2^B - 1}.
  Esta convención usa exactamente 2^B códigos binarios distintos y es simétrica
  respecto al origen.
- La función evaluate_sqnr_db calcula la SQNR contra la señal sin cuantizar
  EXCLUSIVAMENTE para diagnóstico/evaluación offline; nunca se entrega al
  clasificador ni al selector adaptativo.
"""
import numpy as np


def _validate_adc_params(bits, full_scale):
    if isinstance(bits, bool) or not isinstance(bits, (int, np.integer)) or bits < 1 or bits > 16:
        raise ValueError("bits debe ser un entero entre 1 y 16.")
    if (isinstance(full_scale, bool) or not isinstance(full_scale, (int, float, np.floating))
            or not np.isfinite(full_scale) or full_scale <= 0):
        raise ValueError("full_scale debe ser un real finito y estrictamente positivo.")
    return int(bits), float(full_scale)


def quantize_component(x, bits=8, full_scale=2.5):
    """Cuantiza un arreglo real con B bits uniformes mid-rise en [-full_scale, +full_scale]."""
    bits, full_scale = _validate_adc_params(bits, full_scale)
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("La componente de entrada debe ser finita y no vacía.")
    levels = 1 << bits
    delta = (2.0 * full_scale) / levels
    clipped_mask = np.abs(arr) > full_scale
    clipped = np.clip(arr, -full_scale, full_scale)
    indices = np.floor((clipped + full_scale) / delta)
    indices = np.clip(indices, 0, levels - 1)
    quantized = -full_scale + (indices + 0.5) * delta
    return quantized, clipped_mask, float(delta)


def quantize_iq(iq, bits=8, full_scale=2.5, return_diagnostics=False):
    """Cuantiza muestras complejas I/Q con un ADC virtual de B bits por rama.

    Parámetros
    ----------
    iq : array-like complejo (1D o 2D)
        Señal I/Q recibida antes del ADC.
    bits : int
        Resolución por rama I y Q (1 <= bits <= 16). Si bits is None, retorna copia sin cuantizar.
    full_scale : float
        Voltaje/amplitud de fondo de escala fijo V_FS > 0 por rama.
    return_diagnostics : bool
        Si True, retorna (quantized_iq, info_dict) con tasa de clipping y paso Delta.
    """
    z = np.asarray(iq, dtype=np.complex128)
    if z.ndim not in (1, 2) or z.size == 0 or not np.isfinite(z).all():
        raise ValueError("iq debe ser un arreglo complejo 1D o 2D finito y no vacío.")
    if bits is None:
        out = z.copy()
        if not return_diagnostics:
            return out
        return out, {
            "bits": None,
            "full_scale": float(full_scale),
            "delta": 0.0,
            "levels": None,
            "clipping_rate_i": 0.0,
            "clipping_rate_q": 0.0,
            "clipping_rate_iq": 0.0,
            "sample_saturation_rate": 0.0,
        }
    bits, full_scale = _validate_adc_params(bits, full_scale)
    qi, clip_i, delta = quantize_component(z.real, bits=bits, full_scale=full_scale)
    qq, clip_q, _ = quantize_component(z.imag, bits=bits, full_scale=full_scale)
    out = (qi + 1j * qq).astype(np.complex128)
    if not return_diagnostics:
        return out
    clip_any = clip_i | clip_q
    info = {
        "bits": bits,
        "full_scale": full_scale,
        "delta": delta,
        "levels": 1 << bits,
        "clipping_rate_i": float(np.mean(clip_i)),
        "clipping_rate_q": float(np.mean(clip_q)),
        "clipping_rate_iq": float(0.5 * (np.mean(clip_i) + np.mean(clip_q))),
        "sample_saturation_rate": float(np.mean(clip_any)),
    }
    return out, info


def evaluate_sqnr_db(reference_iq, quantized_iq):
    """Evalúa SQNR (dB) y MSE de cuantización usando la referencia pre-ADC.

    USO EXCLUSIVO DE EVALUACIÓN OFFLINE: requiere reference_iq (oracular respecto al ADC).
    """
    ref = np.asarray(reference_iq, dtype=np.complex128)
    quant = np.asarray(quantized_iq, dtype=np.complex128)
    if ref.shape != quant.shape or ref.size == 0 or not (np.isfinite(ref).all() and np.isfinite(quant).all()):
        raise ValueError("reference_iq y quantized_iq deben tener igual forma, ser no vacíos y finitos.")
    sig_power = float(np.mean(np.abs(ref) ** 2))
    err_power = float(np.mean(np.abs(ref - quant) ** 2))
    if sig_power <= np.finfo(float).eps:
        raise ValueError("La potencia de la señal de referencia debe ser positiva.")
    if err_power <= np.finfo(float).eps:
        sqnr_db = float("inf")
    else:
        sqnr_db = float(10.0 * np.log10(sig_power / err_power))
    return {
        "signal_power": sig_power,
        "quantization_error_power": err_power,
        "sqnr_db": sqnr_db,
    }
