# Fase 4 · Detección, espectro y SNR

**5 de noviembre de 2026.** E1 midió **50 falsas alarmas en 5 000 ventanas de ruido independientes**, un **1,00 %** con intervalo de Wilson del 95 % **[0,76 %, 1,32 %]**. El objetivo nominal era 1 %. El umbral de energía **1,152392455** se fijó con otras 5 000 ventanas de 256 muestras. Al duplicar la potencia del ruido y conservar el umbral se dispararon **5 000 de 5 000** ventanas. La estación receptora necesita calibrar con ruido representativo de su banda y de su ganancia.

## E1 en señales con modulación

Se generaron 9 600 padres independientes, con 100 por combinación de clase, canal y SNR. Cada fila de la tabla agrupa las seis modulaciones, 600 ventanas por régimen. ID usa CFO entre −100 y +100 Hz y canales con retardos máximos de 0, 2 o 5 muestras. Shift usa magnitudes de CFO de 200 a 300 Hz y retardos de 9 o 13 muestras. El experimento ajusta la potencia de señal después del canal y añade ruido complejo independiente de potencia media nominal 1. En consecuencia, el cambio de canal conserva la energía limpia media que ve el detector en cada SNR.

| SNR nominal (dB) | Detecciones ID | P_D ID | Detecciones shift | P_D shift |
|---:|---:|---:|---:|---:|
| −20 | 9/600 | 1,50 % | 9/600 | 1,50 % |
| −15 | 36/600 | 6,00 % | 17/600 | 2,83 % |
| −10 | 123/600 | 20,50 % | 122/600 | 20,33 % |
| −5 | 590/600 | 98,33 % | 589/600 | 98,17 % |
| 0, 5, 10 y 20, cada nivel | 600/600 | 100 % | 600/600 | 100 % |

A −10 dB, el IC Wilson del 95 % fue [17,46 %, 23,91 %] en ID y [17,31 %, 23,74 %] en shift. En cada punto con 600/600, el límite inferior fue 99,36 %. Los intervalos son puntuales y condicionados al umbral fijado. El detector mide energía. Con ruido blanco circular y la potencia limpia igualada, diferencias entre celdas pueden proceder del muestreo finito. [Curva e intervalos](../results/fase4/pd_vs_snr.png) · [Conteos por clase](../results/fase4/pd_por_clase.csv) · [Agregados](../results/fase4/pd_agregado.csv)

## Lectura de la SNR

La referencia se estimó en **0,9989814** con las 1 280 000 muestras observadas de la calibración. En una detección, la SNR se calcula restando esta referencia de la potencia recibida. La verdad de comparación usa las potencias realizadas de señal y ruido, separadas exclusivamente por el evaluador. La API devuelve SNR indeterminada si falta la referencia, y N/A si el detector indica ausencia.

| SNR nominal | Detecciones con SNR válida, ID / shift | Sesgo ID / shift (dB) | Error absoluto medio ID / shift (dB) |
|---:|---:|---:|---:|
| −20 dB | 9/600 · 9/600 | +13,12 · +12,96 | 13,12 · 12,96 |
| −10 dB | 123/600 · 122/600 | +3,07 · +3,03 | 3,07 · 3,03 |
| −5 dB | 590/600 · 589/600 | −0,03 · −0,13 | 1,00 · 1,04 |
| 0 dB | 600/600 · 600/600 | −0,03 · +0,01 | 0,54 · 0,53 |
| 10 dB | 600/600 · 600/600 | ≈0 · ≈0 | 0,25 · 0,26 |

A −20 dB, las pocas ventanas aceptadas tuvieron una fluctuación de energía suficientemente alta como para pasar el umbral. Interpretar estas estimaciones como representativas de todas las ventanas subestimaría el problema. A 0 dB, el IC percentil bootstrap del 95 % del error absoluto medio fue [0,51, 0,57] dB en ID y [0,50, 0,57] dB en shift. Cada intervalo remuestrea los padres válidos con calibración fija. [Errores e intervalos por régimen](../results/fase4/snr_errores_agregados.csv) · [Por clase](../results/fase4/snr_errores_por_clase.csv) · [Figura](../results/fase4/snr_error_condicional.png)

Con el doble de potencia de ruido, la referencia antigua atribuye parte del ruido a la señal. La tasa de falsas alarmas observada subió al 100 % [99,92 %, 100 %] y el sesgo de SNR en QPSK alcanzó **+4,75 dB a 0 dB** y **+3,17 dB a 10 dB**, 100 detecciones válidas por nivel. Una deriva desconocida también puede afectar a una captura del profesor sin que el estimador pueda distinguirla de energía de señal. [Ensayo de deriva](../results/fase4/snr_noise_drift.csv)

## Espectro

La captura ilustrativa QPSK a 10 dB tiene 256 muestras a 8 000 muestras/s, ventana Hann periódica y FFT de 1 024 puntos. La rejilla es de 7,8125 Hz y el ancho equivalente de ruido de la ventana es 46,875 Hz. El intervalo central que acumula el 99 % de la potencia **recibida**, de −3 499,29 a +3 472,59 Hz, mide **6 971,88 Hz**. La PSD muestra la señal concentrada cerca de cero y ruido a lo largo de la banda de adquisición. Eso explica el ancho grande aun cuando la parte central de la figura destaca a simple vista. Si falta la frecuencia de muestreo, la API conserva ciclos/muestra. [PSD y banda resaltada](../results/fase4/psd_ejemplo.png) · [FFT compleja y PSD tabuladas](../results/fase4/psd_ejemplo.csv)

## Ejecución y verificación

```powershell
.\.venv\Scripts\python.exe -m pytest -q -W error
.\.venv\Scripts\python.exe -m experiments.exp01_detection --config configs/exp01_detection.json --output results/e1_repeticion
```

**227 pruebas pasaron en 10,03 s** con las advertencias tratadas como errores. Las nuevas pruebas comprueban calibración con ruido independiente, decisión en el umbral, Parseval, tonos positivos y negativos, estados de SNR, reproducción en un ensayo reducido y uso de un archivo de observaciones sin manifiesto privado. Se revisaron las tres figuras contra los datos. La revisión independiente recalculó el umbral a partir de I y Q, encontró 9 600 identificadores únicos, verificó la SNR realizada de cada padre y cotejó nueve hashes de fuentes. La cola teórica de la energía en este AWGN al umbral medido fue 0,9514 %, dentro del intervalo observado. [Revisión independiente](../results/fase4/revision_orquestador.txt) · [Métricas, configuración, ramas de semilla y versiones](../results/fase4/metricas.json)

Una segunda ejecución completa con semilla 20261105 reprodujo byte a byte los doce archivos de configuración, métricas, tablas y figuras. El comando registrado en cada ejecución conserva su propia ruta de destino. [Hashes de reproducción](../results/fase4/reproducibilidad.json) · [Resultado de pytest](../results/fase4/tests.json)

El experimento y la configuración se encuentran en [exp01_detection.py](../experiments/exp01_detection.py) y [exp01_detection.json](../configs/exp01_detection.json). La interfaz de recepción integra [detección](../src/detection.py), [espectro](../src/spectrum.py) y [SNR condicionada](../src/snr_estimation.py) en [monitor.analyze](../src/monitor.py). Los supuestos, las páginas consultadas de FFTFormer y RML22 y las referencias de escalado e intervalos están en [Fundamentos](FUNDAMENTOS_FASE4.md). La siguiente fase de la guía es E2, la evaluación del clasificador convencional. Para un uso posterior del monitor en I/Q desconocida queda por establecer cómo adquirir una referencia representativa del ruido en el receptor.
