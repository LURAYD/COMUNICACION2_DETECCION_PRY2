# Detección, espectro y SNR condicionada

Fase 4, 4 de noviembre de 2026. Los métodos reciben observaciones, metadatos conocidos y calibración observada. Etiquetas, muestras limpias, ruido de la propia realización y parámetros verdaderos se reservan al experimento de evaluación.

## Detector de energía

La estadística es `T=mean(abs(iq)^2)`. El umbral gamma se obtiene del cuantil empírico 1−P_FA de ventanas **solo ruido de validación**, usando el orden estadístico superior (`method="higher"`); se detecta si `T>gamma`. La energía es una puntuación, no una probabilidad de que exista señal. La longitud de observación debe coincidir con la utilizada para calibrar.

P_FA nominal=0.01 no significa que exactamente el 1% de una nueva población produzca falsas alarmas. Se comprueba en ruido independiente y se informa un intervalo Wilson bilateral del 95%, según [NIST, sección 7.2.4.1](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm). Para k éxitos en n ensayos, con p=k/n y z=1.95996398454:

```text
centro = (p + z²/(2n))/(1+z²/n)
semiancho = z*sqrt(p*(1-p)/n + z²/(4n²))/(1+z²/n)
```

El intervalo describe la tasa para el umbral ya fijado; no incorpora por separado la incertidumbre de recalibrarlo en otra muestra. No se modifica gamma después de mirar test.

## Por qué E1 utiliza otro régimen de potencia

El smoke de Fase 3 tiene pocas observaciones noise-only y su piso de ruido varía por realización. No sirve para defender una calibración a P_FA=0.01 con un único umbral. E1 crea una validación y un test independientes con potencia de ruido fija y conocida solo por el simulador. El detector aprende su umbral desde las ventanas de ruido, no recibe esa potencia como verdad.

Para cada trama E1 se controla explícitamente la potencia de la ventana de señal a `Pn*10^(SNR/10)` y se añade ruido nuevo, sin normalizar su realización. Esto cambia la amplitud recibida a lo largo del barrido y separa SNR de deriva del piso de ruido. Las curvas son una validación controlada del detector; no prueban clasificación ni robustez general a variaciones desconocidas de ganancia.

ID y shift mantienen distintos CFO/canales, pero la potencia de señal final se iguala por SNR. Un detector de energía ideal ve principalmente esa potencia; no debe atribuirse a identificación de la modulación una diferencia aleatoria entre curvas.

## PSD y ancho de banda

Se usa un periodograma bilateral de I/Q complejo con ventana Hann periódica, sin quitar la media. No se duplica la energía de frecuencias positivas. La normalización es:

```text
PSD[k] = abs(FFT(iq*w)[k])² / (fs*sum(w²))
df = fs/nfft
sum(PSD)*df = sum(abs(iq*w)²)/sum(w²)
```

Es la potencia ponderada por la ventana; no tiene que coincidir exactamente con la media sin ventana de una trama arbitraria. El escalado y el tratamiento bilateral se contrastaron con la documentación oficial de [SciPy, periodogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html). Se implementa con NumPy, sin instalar ni afirmar haber ejecutado SciPy.

Si fs es conocida, el eje está en Hz y la densidad en amplitud²/Hz. Sin fs, se usa frecuencia en ciclos/muestra y amplitud²/(ciclos/muestra). El zero-padding hace más densa la rejilla, no mejora por sí solo la resolución física. La métrica de ancho es el intervalo central que contiene 99% de la potencia **total observada**, incluido ruido; no se presenta como ancho de la señal limpia. Para potencia cero se devuelve ancho no válido.

Al calcular los cuantiles se distribuye uniformemente la potencia dentro de cada bin de frecuencia. Con FFT de longitud par se reparte el bin de Nyquist entre ambos extremos del eje periódico. De esta forma los límites quedan dentro de la banda de adquisición.

## SNR con una referencia de ruido observable

Una sola mezcla `x=s+n` arbitraria no identifica de forma única sus potencias: distintos pares de señal y ruido pueden producir la misma observación. No se aplica un estimador de envolvente constante a QAM/PAM ni se usa la SNR del simulador.

La alternativa implementada requiere una referencia **noise-only observada e independiente**, con la misma ganancia, ancho de banda y potencia de ruido estacionaria que la observación actual. Su resumen se calcula una vez desde esas muestras; no se entrega el ruido generado para la propia señal ni un valor nominal verdadero.

```text
Pn_hat = mean(abs(referencia_noise_only)^2)
Ptotal_hat = mean(abs(observacion)^2)
Ps_hat = Ptotal_hat - Pn_hat
SNR_hat = 10*log10(Ps_hat/Pn_hat), solamente si Ps_hat>0
```

La suma de potencias se basa en ruido aditivo de media cero no correlacionado con la señal; el término cruzado de una realización finita no desaparece exactamente. No se necesita conocer su modulación.

FFTFormer, p. PDF 9 / impresa 1429, apartado *Measured Dataset*, describe calibrar potencia recibida contra mediciones de ruido de fondo. Se utiliza como antecedente de la disponibilidad de una referencia; esta fórmula sencilla y su contrato de validez son decisiones de ingeniería propias. No se adopta ni se reproduce su cifra de error. RML22, p. PDF 4 / impresa 7666, ecuaciones (8), aporta el modelo aditivo y la definición de SNR utilizados en las fases anteriores.

Sin referencia se devuelve `snr_db=null` y estado no identificable. Sin decisión de detección, no se afirma presencia; si la decisión indica ausencia, la SNR es N/A. Una resta de potencia no positiva también se marca inválida. Una detección falsa puede producir una SNR aparente: el detector no es un oracle de presencia.

## Evaluación y límites

La evaluación registra cuántas estimaciones son válidas, además de MAE, sesgo, RMSE y percentiles de error por régimen y SNR. A baja SNR, seleccionar las ventanas que superan el umbral puede elevar mucho el sesgo. Los intervalos percentiles al 95 % para esas tres métricas usan 2 000 remuestreos de los padres válidos con reemplazo y conservan fijo el umbral. Describen incertidumbre condicionada al número de detecciones válidas. Se omiten si hay cero o una estimación válida. Con pocos padres, incluso un intervalo calculable puede ser poco estable. Los percentiles 5/50/95 de los errores individuales describen la dispersión entre ventanas.

La verdad de comparación, obtenida de señal y ruido reales de cada ventana simulada, existe solo dentro del experimento. Las funciones desplegables no reciben esos componentes ni sus etiquetas. No hay todavía un monitor completo de cinco salidas.

Si el piso de ruido cambia después de calibrar, el detector y la resta de potencias pueden fallar. Se incluye un ensayo independiente duplicando esa potencia y conservando la referencia anterior. El método no puede identificar automáticamente toda deriva oculta; su validez sigue siendo condicional. Tampoco cubre interferencias no gaussianas, referencias contaminadas con señal, AGC desconocido ni RF real.
