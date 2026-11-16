"""Conformación RRC: tiempo normalizado al período de símbolo T=1.

Fórmula y límites: OptiCommPy, rrcFilterTaps, ecuaciones (1)-(3):
https://opticommpy.readthedocs.io/en/stable/_modules/optic/dsp/core.html#rrcFilterTaps
La energía discreta se normaliza después de truncar.
"""
from numbers import Real

import numpy as np


def rrc_filter(beta=0.35, sps=8, span=10):
    """
    Genera un filtro Root Raised Cosine (RRC).

    beta: factor de caída, 0 < beta <= 1
    sps: muestras por símbolo
    span: duración del filtro en símbolos (par)

    Retorna:
        coeficientes del filtro con energía unitaria.
    """

    # 1. Validar parámetros
    if (not isinstance(beta, Real) or isinstance(beta, (bool, np.bool_))
            or not np.isfinite(beta) or not (0 < beta <= 1)):
        raise ValueError("beta debe estar entre 0 y 1.")

    if not isinstance(sps, (int, np.integer)) or isinstance(sps, (bool, np.bool_)) or sps < 2:
        raise ValueError("sps debe ser entero >= 2.")

    if (
        not isinstance(span, (int, np.integer))
        or isinstance(span, (bool, np.bool_))
        or span <= 0
        or span % 2 != 0
    ):
        raise ValueError("span debe ser entero positivo y par.")

    # Evita aritmética sin signo con escalares NumPy.
    sps, span = int(sps), int(span)

    # 2. Eje temporal en unidades de símbolo
    t = np.arange(
        -span * sps // 2,
        span * sps // 2 + 1
    ) / sps

    # 3. Preparar los coeficientes
    h = np.zeros_like(t, dtype=float)

    # 4. Localizar puntos especiales de la fórmula
    center = np.isclose(t, 0.0)
    singular = np.isclose(
        np.abs(t),
        1 / (4 * beta)
    )

    regular = ~(center | singular)
    u = t[regular]

    # 5. Fórmula general del pulso RRC
    numerator = (
        np.sin(np.pi * u * (1 - beta))
        + 4 * beta * u
        * np.cos(np.pi * u * (1 + beta))
    )

    denominator = (
        np.pi * u
        * (1 - (4 * beta * u)**2)
    )

    h[regular] = numerator / denominator

    # 6. Valor límite cuando t = 0
    h[center] = 1 + beta * (4 / np.pi - 1)

    # 7. Valor límite cuando |t| = 1/(4*beta)
    if np.any(singular):
        h[singular] = (
            beta / np.sqrt(2)
            * (
                (1 + 2 / np.pi)
                * np.sin(np.pi / (4 * beta))
                +
                (1 - 2 / np.pi)
                * np.cos(np.pi / (4 * beta))
            )
        )

    # 8. Normalizar energía discreta del filtro
    h = h / np.sqrt(np.sum(h**2))

    return h


def pulse_shape(symbols, sps=8, beta=0.35, span=10):
    """
    Convierte símbolos complejos en muestras I/Q
    utilizando sobremuestreo y filtrado RRC.

    Conserva sps-1 ceros al final del tren de impulsos original. Por ello,
    len(iq) = len(symbols)*sps + len(h)-1. El retardo del transmisor
    es (len(h)-1)/2 muestras; con filtro adaptado se duplica.
    No compensa ni recorta automáticamente ese retardo.
    """

    symbols = np.asarray(
        symbols,
        dtype=np.complex128
    )

    if symbols.ndim != 1 or symbols.size == 0:
        raise ValueError(
            "Se requiere un vector no vacío de símbolos."
        )

    if not np.all(np.isfinite(symbols)):
        raise ValueError("Los símbolos deben ser finitos.")

    # 1. Diseñar filtro RRC
    h = rrc_filter(
        beta=beta,
        sps=sps,
        span=span
    )

    # 2. Insertar ceros entre símbolos
    upsampled = np.zeros(
        len(symbols) * sps,
        dtype=np.complex128
    )

    upsampled[::sps] = symbols

    # 3. Aplicar filtro
    iq = np.convolve(
        upsampled,
        h,
        mode="full"
    )

    return iq, h
