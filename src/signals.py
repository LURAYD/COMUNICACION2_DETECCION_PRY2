"""Seis mapeadores binarios; constelaciones equiprobables de energía media 1.

Las tablas originales se conservan. Véase docs/MAPPING_MODULACIONES.md.
"""
import numpy as np

# Compatibilidad con la API anterior del proyecto.
from src.waveform import pulse_shape, rrc_filter


def _validate_bits(bits, bits_per_symbol):
    """Valida antes de convertir: evita truncar valores no binarios."""
    bits = np.asarray(bits)
    if bits.ndim != 1 or bits.size == 0:
        raise ValueError("Se requiere un vector no vacío de bits.")
    if not np.all(np.isin(bits, [0, 1])):
        raise ValueError("Los bits deben ser 0 o 1.")
    if bits.size % bits_per_symbol:
        raise ValueError(f"La cantidad de bits debe ser múltiplo de {bits_per_symbol}.")
    return bits.astype(np.int64)


def bpsk_modulation(bits):
    """BPSK: 0 → +1, 1 → −1; salida compleja."""
    bits = _validate_bits(bits, 1)
    return (1 - 2 * bits).astype(np.complex128)


def qpsk_modulation(bits):
    """Gray: b0 determina I y b1 determina Q; escala 1/sqrt(2)."""
    groups = _validate_bits(bits, 2).reshape(-1, 2)
    return ((1 - 2 * groups[:, 0]) + 1j * (1 - 2 * groups[:, 1])) / np.sqrt(2)


def psk8_modulation(bits):
    """
    Modula una secuencia de bits utilizando 8PSK.

    Parámetros:
        bits: secuencia binaria de longitud múltiplo de 3.

    Retorna:
        Arreglo NumPy de símbolos complejos 8PSK
        con energía unitaria.
    """

    bits = _validate_bits(bits, 3)

    # 2. Agrupar los bits de tres en tres
    grupos = bits.reshape(-1, 3)

    # 3. Tabla Gray: tres bits -> índice angular
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

    # 4. Obtener el índice angular de cada grupo
    indices = np.array([
        gray_map[tuple(grupo)]
        for grupo in grupos
    ])

    # 5. Calcular ángulos en radianes
    theta = 2 * np.pi * indices / 8

    # 6. Construir símbolos complejos
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

    bits = _validate_bits(bits, 4)

    # 2. Agrupar los bits de cuatro en cuatro
    groups = bits.reshape(-1, 4)

    # 3. Tabla Gray:
    # 00 -> +3, 01 -> +1, 10 -> -3, 11 -> -1
    levels = np.array([3, 1, -3, -1])

    # 4. Construir índices para I y Q
    index_I = 2 * groups[:, 0] + groups[:, 1]
    index_Q = 2 * groups[:, 2] + groups[:, 3]

    # 5. Obtener componentes de amplitud
    I = levels[index_I]
    Q = levels[index_Q]

    # 6. Construir símbolos complejos
    symbols = I + 1j * Q

    # 7. Normalizar la constelación
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

    bits = _validate_bits(bits, 6)

    # 2. Agrupar los bits de seis en seis
    groups = bits.reshape(-1, 6)

    # 3. Niveles correspondientes a los índices
    # binarios: 000, 001, 010, 011,
    #           100, 101, 110, 111
    levels = np.array([
         7,  5,  1,  3,
        -7, -5, -1, -3
    ])

    # 4. Convertir los tres primeros bits
    # de cada grupo en un índice de I
    index_I = (
        4 * groups[:, 0]
        + 2 * groups[:, 1]
        + groups[:, 2]
    )

    # 5. Convertir los tres últimos bits
    # de cada grupo en un índice de Q
    index_Q = (
        4 * groups[:, 3]
        + 2 * groups[:, 4]
        + groups[:, 5]
    )

    # 6. Obtener las amplitudes I y Q
    I = levels[index_I]
    Q = levels[index_Q]

    # 7. Construir símbolos complejos
    symbols = I + 1j * Q

    # 8. Normalizar la energía media
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

    bits = _validate_bits(bits, 2)

    # 2. Agrupar los bits de dos en dos
    groups = bits.astype(np.int64).reshape(-1, 2)

    # 3. Tabla Gray:
    # 00 -> +3
    # 01 -> +1
    # 10 -> -3
    # 11 -> -1
    levels = np.array([3, 1, -3, -1])

    # 4. Convertir cada par en un índice
    indices = 2 * groups[:, 0] + groups[:, 1]

    # 5. Obtener las amplitudes
    symbols = levels[indices]

    # 6. Normalizar la energía promedio
    symbols = symbols / np.sqrt(5)

    # 7. Convertir a números complejos
    symbols = symbols.astype(np.complex128)

    return symbols
