import numpy as np



def add_awgn(iq, snr_db, rng=None):
    """
    Agrega ruido blanco gaussiano complejo a una señal I/Q.

    Parámetros
    ----------
    iq : array
        Señal compleja de entrada.

    snr_db : float
        Relación señal-ruido deseada, en dB.
        Se define usando la potencia media de las
        muestras I/Q de entrada.

    rng : np.random.Generator, opcional
        Generador aleatorio para reproducibilidad.

    Retorna
    -------
    noisy_iq : array complejo
        Señal después del canal AWGN.

    noise : array complejo
        Ruido generado, útil para validar la simulación.
    """

    # 1. Convertir la entrada a números complejos
    x = np.asarray(iq, dtype=np.complex128)

    # 2. Validar la señal de entrada
    if x.ndim != 1 or x.size == 0:
        raise ValueError("IQ debe ser un vector no vacío.")

    if not np.all(np.isfinite(x)):
        raise ValueError("IQ contiene valores no finitos.")

    if not np.isscalar(snr_db) or not np.isfinite(snr_db):
        raise ValueError("SNR debe ser un valor finito.")

    # 3. Crear generador aleatorio si no se proporciona
    if rng is None:
        rng = np.random.default_rng()

    # 4. Calcular potencia media de la señal
    signal_power = np.mean(np.abs(x)**2)

    if signal_power == 0:
        raise ValueError(
            "La potencia de la señal no puede ser cero."
        )

    # 5. Convertir SNR de dB a escala lineal
    snr_linear = 10**(snr_db / 10)

    # 6. Calcular potencia de ruido necesaria
    noise_power = signal_power / snr_linear

    # 7. Generar ruido gaussiano complejo
    sigma = np.sqrt(noise_power / 2)

    noise = sigma * (
        rng.standard_normal(x.size)
        + 1j * rng.standard_normal(x.size)
    )

    # 8. Agregar ruido a la señal
    noisy_iq = x + noise

    return noisy_iq, noise
