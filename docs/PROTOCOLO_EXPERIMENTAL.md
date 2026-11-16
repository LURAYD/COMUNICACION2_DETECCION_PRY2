# Protocolo experimental

Las secciones de Fase 3 describen el dataset de humo, versión 1 del 4 de noviembre de 2026. La última sección especifica el experimento independiente de detección de Fase 4.

## Unidad de independencia y partición

La unidad independiente es la **realización madre**, identificada por `parent_waveform_id`. Antes de generar símbolos, canal o ruido se asignan split, clase, SNR nominal y semilla madre. No se divide una colección de ventanas ya generadas.

Hay cuatro splits: `train`, `val`, `test` y `test_shift`. Cada uno contiene 48 padres con señal: seis clases × cuatro SNR × dos realizaciones. Se añaden ocho padres solo ruido, total 56 por split y 224 padres. Las clases son BPSK, QPSK, 8PSK, 16QAM, 64QAM y 4PAM, conservando el mapping de Fase 1. Noise-only es una etiqueta de presencia/ausencia para detección, no una séptima modulación.

Cada padre entrega una ventana de 512 muestras y su prefijo exacto de 256, incluidos el mismo canal, fase y ruido. Hay 448 ventanas, pero **224 unidades independientes**, no 448. Si se añaden después representaciones FFT, cuantización u otras versiones, deben conservar padre y split. No se emparejan tramas independientes para simular una observación ampliada.

La semilla raíz de la configuración se ramifica mediante `numpy.random.SeedSequence`: split → orden/semillas madre → parámetros/bits/ruido. Las filas se permutan dentro de cada split. Cambiar la semilla cambia las realizaciones; repetir semilla y configuración reproduce el contenido. Los IDs son administrativos y opacos. Ni IDs, nombres, filas, semillas ni rutas se usarán como características.

## Observación disponible y verdad del simulador

| Artefacto | Contenido y uso |
|---|---|
| `observations/{split}.npz` | Solo arreglos complejos `iq256` e `iq512`; sin etiquetas ni parámetros verdaderos. |
| `public_metadata.json` | `fs_hz` conocido y longitudes admitidas. No contiene SNR, clase, canal ni semillas. |
| `index.csv` | IDs, padre, split, fila y ruta; sirve para localizar registros, no como entrada al monitor. |
| `oracle/manifest.jsonl` | Presencia, clase, SNR, parámetros realizados, semillas y política de recorte; solo supervisión/evaluación. |
| `config.json`, `audit.json` | Procedimiento y evidencia de integridad; no son entradas del monitor. |

`load_observation(root, record_id, n_samples)` devuelve exclusivamente `{"iq": ..., "fs_hz": ...}` y no requiere leer oracle ni configuración. `load_truth(root)` es un acceso explícito y separado para supervisión/evaluación. Esta separación se prueba por comportamiento; no es una barrera de seguridad del sistema de archivos. No hay todavía un monitor o clasificador sobre el que afirmar ausencia de fuga en entrenamiento.

## Parámetros y soporte de canal

Configuración reproducible: `configs/dataset_smoke.json`. Se utiliza JSON de biblioteca estándar para evitar otra dependencia; sustituye el YAML sugerido por la guía sin cambiar el contenido requerido.

| Variable | Train/val/test | Test_shift |
|---|---|---|
| Rs / sps / fs | 1000 baud / 8 / 8000 muestras/s | Igual |
| Símbolos por padre / span RRC | 128 / 10 símbolos | Igual |
| Beta | Elección entre 0.25, 0.35 y 0.50 | Igual |
| Amplitud / fase inicial | Uniforme [0.5,1.5) / [−pi,pi) rad | Igual |
| SNR nominal | −10, 0, 10, 20 dB, estratificada | Igual |
| CFO | Uniforme [−100,100) Hz | Signo aleatorio, magnitud [200,300) Hz |
| Retardo máximo FIR | Banco de 0, 2 o 5 muestras | Banco de 9 o 13 muestras |

Cada trama con señal usa parámetros, bits y ruido propios. Los bancos FIR exactos están en el JSON; su ganancia no se normaliza automáticamente. `test_shift` cambia simultáneamente soporte de CFO y canal, por lo que no aísla causalmente cuál de los dos explica un eventual deterioro. Se reserva para E6; no se utilizará para escoger modelos, umbrales o hiperparámetros. Val/test comparten distribución nominal con train, pero no realizaciones.

Estas distribuciones son decisiones de ingeniería del proyecto. No son una reproducción de ETU70, Rician, Rayleigh ni de los parámetros completos de RML22. Con fs=8000, cada muestra equivale a 125 microsegundos; los retardos son virtuales y no describen una geometría de habitación.

## Generación y transitorios

Cadena: bits aleatorios → símbolos → RRC → amplitud → multipath → fase/CFO → AWGN → recorte. Se genera la trama entera una sola vez, preservando fase y ruido al extraer los prefijos. No se aplica filtro adaptado ni sincronización oracle a la observación del futuro clasificador.

El recorte comienza en `Lrrc-1 + Lcanal-1 + timing_offset_samples`, con offset entero uniforme en 0…sps−1. Es un margen conservador que deja atrás la memoria completa de ambos FIR. El final de las 512 muestras debe quedar antes del último impulso de la secuencia de símbolos; se valida al leer configuración y al auditar. Así no se confunden ventanas de arranque o cola con señal estable. El offset aleatorio cambia la posición de muestreo respecto de los símbolos; no modela jitter fraccional ni SRO.

AWGN mantiene la definición de Fase 2: potencia de **la trama completa** después del canal. La SNR local de la ventana puede diferir de la nominal por fluctuación estadística y por recortar transitorios. Se guardan ambas solo en oracle, sin reajustar las muestras a posteriori. No se normaliza cada ventana por energía ni se convierte SNR a Eb/N0.

## Observaciones solo ruido

Se generan gaussianas complejas directamente, sin modular bits ni llamar a `add_awgn` con ceros. La potencia objetivo explícita es:

```text
Pn = Aref^2 / (sps * 10^(r/10))
noise = sqrt(Pn/2) * (normal_I + j*normal_Q)
```

`Aref` y `r` son variables auxiliares para elegir la escala de ruido con el mismo rango aproximado del experimento con señal. No representan una señal oculta ni una SNR real. `modulation`, `snr_true_db` y `snr_measured_window_db` son nulos cuando `has_signal=false`. La potencia objetivo y la medida se conservan en oracle.

Esta distribución aproxima el rango del ruido de signal-present, pero no coincide exactamente después de la variación de energía por FIR y trama. Es una limitación explícita del smoke; no fundamenta una evaluación definitiva de detección o de piso térmico fijo.

## Qué verifica la auditoría

IDs y semillas sin duplicados entre padres; correspondencia índice/oracle; filas y rutas coherentes; archivos de observación con solo I/Q finita; longitudes 256/512; igualdad exacta de prefijos; balance por clase/SNR; presencia de noise-only con etiquetas nulas; soportes de canal y recorte coherentes con configuración. Los tests alteran datos o metadatos para comprobar que las inconsistencias se rechazan, además de repetir la generación.

Se registran hashes de configuración, fuentes y observaciones, junto con versiones. La ausencia de IDs duplicados y la jerarquía de RNG no constituyen una prueba matemática de independencia estadística absoluta; evitan las reutilizaciones accidentales especificadas.

## Uso posterior y límites

El smoke es demasiado pequeño para entrenar y comparar modelos, estimar una tasa de falsa alarma de 0.01 o medir generalización con precisión. Se deberá aumentar el número de realizaciones con una configuración versionada antes de esos experimentos. `val` servirá para calibración/selección y `test` para evaluación final; si se usa `val` para varios ajustes importantes, deberán reservarse subconjuntos independientes por padre. No se ajustará nada mirando `test_shift`.

## Fase 4 · E1 de detección

E1 usa realizaciones propias. El smoke tiene ocho ventanas de ruido en validación y su potencia de ruido cambia por trama. Para medir falsas alarmas cerca de 1 %, E1 fija el piso de ruido del simulador y separa 5 000 ventanas de calibración de otras 5 000 de evaluación. Cada ventana tiene 256 muestras. El detector usa el cuantil superior empírico de energía de la calibración. La SNR usa la potencia de esa misma calibración como referencia observada.

Para cada combinación de seis modulaciones, dos regímenes de canal y ocho SNR nominales se generan 100 padres independientes. Se aplica RRC, FIR y portadora a la trama completa, se extrae una ventana estable y se ajusta su potencia limpia al valor de la celda. Se suma ruido nuevo sin reescalar la realización. El evaluador calcula la SNR realizada con las dos componentes de esa ventana. El análisis recibe únicamente la mezcla y su calibración. Un último ensayo duplica la potencia del ruido y conserva los parámetros de calibración anteriores para medir la sensibilidad a la deriva.

Los intervalos de Wilson se calculan para el umbral ya fijado. El error de SNR se evalúa en las detecciones con estimación válida. Se muestra cuántas ventanas contribuyen a cada cifra para exponer el sesgo de selección en ventanas débiles. [Métodos y fuentes](FUNDAMENTOS_FASE4.md) · [Resultados E1](FASE4_VALIDACION.md)

## Fase 5 · Protocolo de E2 antes de abrir la prueba

E2 evalúa la clasificación de **seis modulaciones en tramas que contienen señal**. La etiqueta de presencia de la simulación delimita ese experimento. La tasa de detección de E1 y el acierto de clasificación de E2 tienen denominadores y regímenes de ruido diferentes. Clasificar sin etiqueta una captura nueva requiere que el llamante establezca antes que hay señal.

Se usará el generador de Fase 3 con otra semilla y 100 padres por combinación de clase y SNR en cada partición. Los ocho niveles nominales son −20, −15, −10, −5, 0, 5, 10 y 20 dB. Así, cada partición reúne 4 800 señales y 64 padres de solo ruido. Los dos modelos de longitud usan los mismos padres y el de 256 recibe el prefijo de la observación de 512. Los campos de SNR, canal, clase, rutas y semillas sirven exclusivamente para supervisión, agrupación de métricas o procedencia.

La elección se hará por F1 macro en validación, sobre candidatos y desempate declarados de antemano. Se ajustan escalas poblacionales y parámetros de modelo en entrenamiento. Se fijarán los dos modelos antes de calcular predicciones de prueba. Se conservarán IDs y hashes para reutilizar las mismas muestras en la futura comparación con el modelo avanzado. La prueba con cambio de canal se reserva para E6.

Como las ventanas están emparejadas, la diferencia de exactitud se calculará por padre. El signo positivo indicará que pasar de 256 a 512 convierte más errores en aciertos que aciertos en errores. Para su intervalo al 95 % se remuestrearán padres dentro de las 48 celdas clase/SNR, manteniendo cada pareja junta. La latencia de CPU con modelo cargado se informará por separado del tiempo virtual de adquisición: 32 y 64 ms a 8 000 muestras/s.
