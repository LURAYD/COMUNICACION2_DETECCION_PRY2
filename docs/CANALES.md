# Canales discretos de Fase 2

Convenciones de simulación en banda base. No representan un entorno RF medido. Los parámetros se fijan una vez por trama; los metadatos verdaderos y el ruido generado se reservan para diagnóstico/evaluación.

## Orden y unidades

```text
bits → símbolos → RRC → multipath FIR → fase/CFO → AWGN → observación I/Q
```

`fs_hz` es la frecuencia de muestreo en muestras/s; `cfo_hz` es la diferencia de frecuencia en Hz; `phase_rad` se expresa en radianes. Con tasa `Rs` en baud y `sps` muestras por símbolo, `fs_hz=sps*Rs`. No se aplica ADC en esta fase: es opcional y queda pospuesto.

## Fase y CFO

Para una trama que comienza en el índice absoluto `n0=sample_offset`:

```text
y[n] = x[n] * exp(j*(phase_rad + 2*pi*cfo_hz*(n+n0)/fs_hz))
```

La fase fija rota todos los puntos el mismo ángulo. CFO añade un incremento de `2*pi*cfo_hz/fs_hz` radianes por muestra. Como el factor complejo tiene módulo uno, no cambia la amplitud ni la potencia. CFO positivo gira en sentido antihorario y desplaza el espectro hacia frecuencias positivas.

La fórmula es la especialización a parámetros constantes del algoritmo publicado por [MathWorks, comm.PhaseFrequencyOffset, Algorithms](https://www.mathworks.com/help/comm/ref/comm.phasefrequencyoffset-system-object.html), consultado el 3 de noviembre de 2026. Es una referencia externa; no se ejecutó MATLAB. La API de este proyecto usa radianes.

`sample_offset` permite procesar porciones con continuidad: la segunda porción empieza en el número de muestras ya procesadas, manteniendo la misma fase inicial y CFO. Reiniciar ese índice a cero en cada ventana introduce discontinuidades artificiales. El simulador admite CFO finito; frecuencias separadas por múltiplos de fs son equivalentes en tiempo discreto. Para representar un desplazamiento sin aliasing, también debe caber el espectro desplazado dentro de ±fs/2.

## Multipath FIR

```text
y[n] = sum_k taps[k] * x[n-k]
```

Cada posición del vector taps es un retardo en **muestras**. El retardo físico es `k/fs_hz` segundos. Los coeficientes son ganancias complejas constantes durante la trama. La convolución es completa: `len(y)=len(x)+len(taps)-1`; se conservan transitorios.

El canal `[1]` es identidad. Un impulso de entrada devuelve taps. Un coeficiente único en k=D retrasa D muestras y aplica su ganancia. Varios coeficientes superponen réplicas y pueden generar interferencia entre símbolos. No se normalizan taps ni salida automáticamente: un FIR general puede amplificar o atenuar según su respuesta en frecuencia y la señal de entrada. Incluso normalizar `sum(abs(taps)^2)=1` no garantizaría potencia de salida idéntica para una señal correlacionada.

El canal nulo es un FIR válido, pero no puede pasarse a AWGN con SNR relativa a una señal de potencia cero. Tampoco hay aquí desvanecimiento Rayleigh, Doppler variable ni retardos fraccionales. Los retardos elegidos son ejemplos discretos; no se describen como reflejos de una habitación sin justificar distancias y fs.

## AWGN y qué significa la SNR

Se conserva el contrato `received, oracle_noise = add_awgn(iq, snr_db, rng)`. La potencia se mide sobre **todas las muestras de entrada**, incluidos transitorios, después de multipath/fase/CFO:

```text
Ps = mean(abs(iq)^2)
Pn = Ps / 10^(snr_db/10)
noise = sqrt(Pn/2) * (normal_I + j*normal_Q)
```

I y Q son gaussianas independientes de media cero y varianza objetivo Pn/2. Su suma compleja tiene potencia objetivo Pn. El ruido no se renormaliza para imponer una SNR muestral exacta; las realizaciones finitas fluctúan.

La convención de potencia medida sigue el enfoque documentado en [MathWorks, awgn, opción measured](https://www.mathworks.com/help/comm/ref/awgn.html). La documentación de [AWGN Channel](https://www.mathworks.com/help/comm/ug/awgn-channel.html) distingue SNR por muestras de Es/N0 y Eb/N0. No se usan estos nombres como equivalentes ni se aplica una conversión sin sus supuestos.

Al recalcular ruido en función de Ps por trama, variar la amplitud también cambia la potencia de ruido para mantener la SNR nominal. Esto modela un ensayo a SNR controlada; no un receptor con piso de ruido térmico fijo. La función rechaza potencia cero y escalas numéricas no representables. Las futuras observaciones solo ruido necesitarán un generador con potencia de ruido explícita.

## Coherencia, reproducibilidad y acceso a la verdad

La ruta recomendada genera la trama completa y aplica el canal completo una sola vez, antes de extraer ventanas. FIR por bloques necesita memoria/solapamiento; invocarlo de forma independiente no conserva las colas. AWGN por bloques vuelve a medir potencia y cambia la secuencia de consumo del generador aleatorio. `sample_offset` resuelve específicamente la continuidad de portadora, no convierte toda la cadena en un procesador streaming.

Se pasa un `numpy.random.Generator` explícito para obtener corridas reproducibles. Repetir el procedimiento con una semilla reproduce una realización; usar una sola realización repetida no crea observaciones independientes. Los tests y el diagnóstico usan varias semillas, pero no constituyen un dataset de entrenamiento/prueba.

SNR medida con `oracle_noise`, taps, fase y CFO verdaderos son información del simulador. El futuro monitor deberá recibir únicamente la observación I/Q y metadatos conocidos como fs, nunca esos campos para resolver sus salidas. Fase 2 no implementa ese monitor ni afirma haber probado todavía su separación de acceso.
