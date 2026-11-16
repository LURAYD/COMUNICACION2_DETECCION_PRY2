import numpy as np


def bpsk_modulation(bits):

    # Map bits to BPSK symbols: 0 -> -1, 1 -> 1
    bits = np.array(bits)
    
    if not np.all(np.logical_or(bits == 0, bits == 1)):
        raise ValueError("Input bits must be 0 or 1.")\

    s = 1 - 2 * bits  # Map 0 to 1 and 1 to -1
    return s.astype(complex)  # Return as complex numbers for consistency with other modulations

bits = [0, 1, 0, 1, 1, 0]  # Example input bits
symbol = bpsk_modulation(bits)

print("Input bits:", bits)
print("BPSK symbols:", symbol)

def psk8_modulation(bits):
    """
    Modula una secuencia de bits utilizando 8PSK.

    Parámetros:
        bits: secuencia binaria de longitud múltiplo de 3.

    Retorna:
        Arreglo NumPy de símbolos complejos 8PSK
        con energía unitaria.
    """

    # 1. Convertir entrada en arreglo NumPy
    bits = np.asarray(bits)

    # 2. Comprobar que sea un vector no vacío
    if bits.ndim != 1 or bits.size == 0:
        raise ValueError("Se requiere un vector no vacío.")

    # 3. Validar que contenga solamente 0 y 1
    if not np.all(np.isin(bits, [0, 1])):
        raise ValueError("Los bits deben ser 0 o 1.")

    # 4. Comprobar que la longitud sea múltiplo de 3
    if len(bits) % 3 != 0:
        raise ValueError(
            "La cantidad de bits debe ser múltiplo de 3."
        )

    # 5. Agrupar los bits de tres en tres
    grupos = bits.reshape(-1, 3)

    # 6. Tabla Gray: tres bits -> índice angular
    gray_map = {
        (0, 0, 0): 0,
        (0, 0, 1): 1,
        (0, 1, 1): 2,
        (0, 1, 0): 3,
        (1, 1, 0): 4,
        (1, 1, 1): 5,
        (1, 0, 1): 6,
        (1, 0, 0): 7
    }

    # 7. Obtener el índice angular de cada grupo
    indices = np.array([
        gray_map[tuple(grupo)]
        for grupo in grupos
    ])

    # 8. Calcular ángulos en radianes
    theta = 2 * np.pi * indices / 8

    # 9. Construir símbolos complejos
    symbols = np.exp(1j * theta)

    return symbols




def qam16_modulation(bits):
    """
    Modulación 16-QAM con mapeo Gray.

    Entrada:
        bits: secuencia binaria de longitud múltiplo de 4.

    Salida:
        símbolos complejos 16-QAM, con energía
        media unitaria para símbolos equiprobables.
    """

    # 1. Convertir a arreglo NumPy
    bits = np.asarray(bits)

    # 2. Validar que sea un vector no vacío
    if bits.ndim != 1 or bits.size == 0:
        raise ValueError("Se requiere un vector no vacío.")

    # 3. Validar los valores binarios
    if not np.all(np.isin(bits, [0, 1])):
        raise ValueError("Los bits deben ser 0 o 1.")

    # 4. Comprobar grupos completos de cuatro bits
    if bits.size % 4 != 0:
        raise ValueError(
            "La cantidad de bits debe ser múltiplo de 4."
        )

    # 5. Agrupar los bits de cuatro en cuatro
    groups = bits.reshape(-1, 4)

    # 6. Tabla Gray:
    # 00 -> +3, 01 -> +1, 10 -> -3, 11 -> -1
    levels = np.array([3, 1, -3, -1])

    # 7. Construir índices para I y Q
    index_I = 2 * groups[:, 0] + groups[:, 1]
    index_Q = 2 * groups[:, 2] + groups[:, 3]

    # 8. Obtener componentes de amplitud
    I = levels[index_I]
    Q = levels[index_Q]

    # 9. Construir símbolos complejos
    symbols = I + 1j * Q

    # 10. Normalizar la constelación
    symbols = symbols / np.sqrt(10)

    return symbols




def qam64_modulation(bits):
    """
    Modulación 64-QAM con mapeo Gray.

    Cada símbolo representa seis bits:
        3 bits para I
        3 bits para Q

    La constelación tiene energía media
    unitaria para símbolos equiprobables.
    """

    # 1. Convertir la entrada a un arreglo NumPy
    bits = np.asarray(bits)

    # 2. Comprobar que sea un vector no vacío
    if bits.ndim != 1 or bits.size == 0:
        raise ValueError(
            "Se requiere un vector no vacío de bits."
        )

    # 3. Validar que solo contenga 0 y 1
    if not np.all(np.isin(bits, [0, 1])):
        raise ValueError(
            "La secuencia debe contener solamente 0 y 1."
        )

    # 4. Validar la longitud
    if bits.size % 6 != 0:
        raise ValueError(
            "La cantidad de bits debe ser múltiplo de 6."
        )

    # 5. Agrupar los bits de seis en seis
    groups = bits.reshape(-1, 6)

    # 6. Niveles correspondientes a los índices
    # binarios: 000, 001, 010, 011,
    #           100, 101, 110, 111
    levels = np.array([
         7,  5,  1,  3,
        -7, -5, -1, -3
    ])

    # 7. Convertir los tres primeros bits
    # de cada grupo en un índice de I
    index_I = (
        4 * groups[:, 0]
        + 2 * groups[:, 1]
        + groups[:, 2]
    )

    # 8. Convertir los tres últimos bits
    # de cada grupo en un índice de Q
    index_Q = (
        4 * groups[:, 3]
        + 2 * groups[:, 4]
        + groups[:, 5]
    )

    # 9. Obtener las amplitudes I y Q
    I = levels[index_I]
    Q = levels[index_Q]

    # 10. Construir símbolos complejos
    symbols = I + 1j * Q

    # 11. Normalizar la energía media
    symbols = symbols / np.sqrt(42)

    return symbols




def pam4_modulation(bits):
    """
    Modulación 4-PAM con mapeo Gray.

    Entrada:
        bits: secuencia de 0 y 1.
              Debe tener longitud par.

    Salida:
        Símbolos complejos 4-PAM.
        La constelación tiene energía
        promedio unitaria.
    """

    # 1. Convertir los bits a NumPy
    bits = np.asarray(bits)

    # 2. Verificar que sea un vector no vacío
    if bits.ndim != 1 or bits.size == 0:
        raise ValueError(
            "Los bits deben formar un vector no vacío."
        )

    # 3. Validar los valores binarios
    if not np.all(np.isin(bits, [0, 1])):
        raise ValueError(
            "La entrada solo puede contener bits 0 y 1."
        )

    # 4. Verificar que la longitud sea par
    if bits.size % 2 != 0:
        raise ValueError(
            "Se requiere un número par de bits."
        )

    # 5. Agrupar los bits de dos en dos
    groups = bits.astype(np.int64).reshape(-1, 2)

    # 6. Tabla Gray:
    # 00 -> +3
    # 01 -> +1
    # 10 -> -3
    # 11 -> -1
    levels = np.array([3, 1, -3, -1])

    # 7. Convertir cada par en un índice
    indices = 2 * groups[:, 0] + groups[:, 1]

    # 8. Obtener las amplitudes
    symbols = levels[indices]

    # 9. Normalizar la energía promedio
    symbols = symbols / np.sqrt(5)

    # 10. Convertir a números complejos
    symbols = symbols.astype(np.complex128)

    return symbols




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
    if not (0 < beta <= 1):
        raise ValueError("beta debe estar entre 0 y 1.")

    if not isinstance(sps, int) or sps < 2:
        raise ValueError("sps debe ser entero >= 2.")

    if (
        not isinstance(span, int)
        or span <= 0
        or span % 2 != 0
    ):
        raise ValueError("span debe ser entero positivo y par.")

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
    """

    symbols = np.asarray(
        symbols,
        dtype=np.complex128
    )

    if symbols.ndim != 1 or symbols.size == 0:
        raise ValueError(
            "Se requiere un vector no vacío de símbolos."
        )

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




