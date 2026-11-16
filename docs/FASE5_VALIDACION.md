# Fase 5 · Clasificación convencional de modulaciones (E2)

**Cierre del 6 de noviembre de 2026.** Los dos modelos elegidos en validación fueron SVM RBF con C=10. Sobre los mismos 4 800 padres de prueba, el modelo de 256 muestras obtuvo **43,69 % de exactitud y F1 macro 0,4377**. El de 512 obtuvo **46,60 % y F1 macro 0,4637**. Los modelos, las predicciones, los tiempos y las matrices se guardaron en `results/fase5`.

| Resultado sobre test | 256 muestras | 512 muestras |
|---|---:|---:|
| Características seleccionadas | 49 temporales | 65 temporales y espectrales |
| Exactitud | 43,69 % | 46,60 % |
| Precision macro | 0,4396 | 0,4630 |
| Recall macro | 0,4369 | 0,4660 |
| F1 macro | 0,4377 | 0,4637 |
| F1 macro en validación, usado para seleccionar | 0,4282 | 0,4741 |

## Resultados por SNR y comparación pareada

| SNR nominal (dB) | Exactitud 256 | Exactitud 512 | Padres por longitud |
|---:|---:|---:|---:|
| −20 | 16,67 % | 16,33 % | 600 |
| −15 | 17,83 % | 14,33 % | 600 |
| −10 | 17,00 % | 16,83 % | 600 |
| −5 | 20,83 % | 22,67 % | 600 |
| 0 | 44,67 % | 55,67 % | 600 |
| 5 | 64,67 % | 72,00 % | 600 |
| 10 | 82,00 % | 86,83 % | 600 |
| 20 | 85,83 % | 88,17 % | 600 |

La mejora media de 512 respecto de 256 fue **2,92 puntos porcentuales**, con intervalo percentil bootstrap del 95 % **[1,46, 4,35] puntos**. En 696 padres, 512 corrigió un error de 256. En otros 556 ocurrió lo contrario. El remuestreo mantuvo las parejas juntas dentro de cada celda clase/SNR, con 10 000 réplicas y semilla 20261505. El intervalo condiciona al dataset y a los dos modelos fijados. La comparación incluye el cambio de longitud y de grupo de características elegido para cada pipeline. [Cálculo pareado](../results/fase5/comparison_paired.json) · [Curva por SNR](../results/fase5/accuracy_by_snr.png)

A SNR de −20 a −10 dB, el acierto está cerca de elegir entre seis clases equiprobables. Las matrices agregadas muestran confusión frecuente entre 16QAM y 64QAM, y entre QPSK y 8PSK. A 20 dB todavía hay errores, con exactitudes de 85,83 % y 88,17 %. Estos resultados delimitan la utilidad del clasificador convencional en las ventanas y canales ensayados. [Matriz 256](../results/fase5/confusion_256.png) · [Matriz 512](../results/fase5/confusion_512.png) · [Precision, recall y F1 por clase y SNR](../results/fase5/test_metrics_by_class.csv)

## Población y selección

E2 mide la probabilidad de acierto **dado que hay una señal de una de las seis modulaciones**. La presencia y la modulación verdadera solo determinan las filas que entran en este experimento y los objetivos supervisados. En despliegue, el clasificador recibe un vector I/Q con la longitud de su modelo. La detección de ausencia corresponde al detector de Fase 4.

La semilla 2026110502 produjo 19 456 padres con el generador ya validado en Fase 3. Cada partición reúne 4 800 señales y 64 observaciones de solo ruido. Hay 100 padres por modulación y SNR nominal (−20 a 20 dB en ocho niveles). Las particiones se asignaron antes de generar los prefijos. La auditoría comprobó padres disjuntos y 19 456 prefijos exactos. Los modelos de 256 y 512 muestras observan aproximadamente 32 y 64 períodos de símbolo, respectivamente, con Fs=8 000 muestras/s y ocho muestras por símbolo. [Configuración del dataset](../configs/dataset_e2.json) · [Protocolo experimental](PROTOCOLO_EXPERIMENTAL.md)

Se fijaron cinco candidatos por longitud en la configuración del experimento: un bosque de 200 árboles y SVM radial con C=1 o C=10 sobre el conjunto completo, más bosque y SVM C=10 sobre el conjunto que excluye espectro. El bosque permite particiones no lineales en las características. La SVM radial compara distancias y necesita escalas comparables. Su estandarización aprende de los 4 800 padres de entrenamiento. Aparte de ese ajuste poblacional, cada ventana se centra y normaliza con estadísticas calculadas en sus propias muestras disponibles. La selección maximiza F1 macro en validación, con desempate por el orden definido en la configuración. Se congela cada ganador entrenado sin incorporar validación al ajuste. [Configuración de candidatos](../configs/exp02_conventional.json) · [Prevención de fuga en ajuste y transformación](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)

## Qué mide el clasificador

| Familia de características | Información observable y límite |
|---|---|
| Amplitud y momentos | Media, dispersión y cuantiles de la envolvente normalizada describen la distribución de amplitud. Los momentos de órdenes 4, 6 y 8 responden a colas y picos, con mayor variabilidad en ventanas cortas. |
| Circularidad y cumulantes | Con la señal centrada `z`, M20=promedio(z²), M21=promedio(\|z\|²), M40=promedio(z⁴) y M42=promedio(\|z\|⁴). Se usan \|M20\|/M21, \|M40−3M20²\|/M21² y (M42−\|M20\|²−2M21²)/M21². Algunas variantes calculan promedios por bloques de 64 muestras o compensan una pendiente de fase estimada en la propia ventana. |
| Fase y retardos | Se calculan resultantes de fase y diferencias a retardos prefijados de 1, 2, 4 y 8 muestras. En un producto diferencial, una rotación global se cancela y un CFO constante añade una fase común para ese retardo. La magnitud de la resultante elimina esa fase común. El resto de características puede seguir cambiando por CFO y canal. |
| Espectro | La distribución relativa de potencia de una FFT con Hann se resume mediante entropía, planitud, centro, anchura y energía por subbanda. La frecuencia se expresa en ciclos/muestra. Ruido y desplazamiento de portadora también modifican estas medidas. |

El conjunto temporal tiene 49 características y añadir las 16 espectrales da 65. El extractor común calcula las 65 antes de aplicar la selección. Las mediciones de inferencia incluyen ese coste en ambas alternativas. Aplicar los mismos retardos a una captura con otra relación entre muestreo y tasa de símbolos puede cambiar su utilidad. La mezcla de clases, ruido, canal y sincronización del simulador delimitan el alcance de cualquier cifra de E2. [Extractor](../src/features.py) · [Pipelines y API de observación](../src/conventional.py) · [Papers originales consultados y páginas](MATRIZ_LITERATURA.md)

La comparación de grupos tuvo resultados distintos según longitud. En validación con 256, SVM C=10 pasó de F1 0,4205 con 65 características a 0,4282 al excluir espectro. Con 512, el grupo completo dio 0,4741 frente a 0,4719 para el temporal. Se aplicó la regla fijada, incluso para esa diferencia pequeña. [Resultados de los diez candidatos](../results/fase5/candidate_validation.csv)

## Costo medido en CPU

Se utilizó Intel Core i7-13620H con un hilo de las bibliotecas numéricas, Python 3.11.0 y scikit-learn 1.7.2. La latencia de pared se midió con modelo cargado, calentamiento previo, una ventana por llamada, 128 ventanas distintas y cinco repeticiones por longitud. Incluye extracción y predicción.

| Medida | 256 | 512 |
|---|---:|---:|
| Latencia total mediana | 5,22 ms | 6,57 ms |
| Latencia total percentil 95 | 8,95 ms | 9,02 ms |
| Extracción, mediana | 3,58 ms | 4,82 ms |
| Clasificador, mediana | 1,64 ms | 1,73 ms |
| Ajuste del ganador, con características ya calculadas | 3,05 s | 3,23 s |
| Extracción de las 4 800 ventanas train | 16,35 s | 22,82 s |
| Adquisición virtual N/fs | 32 ms | 64 ms |
| Archivo de modelo comprimido | 1 614 101 bytes | 1 981 566 bytes |

La mediana de una suma se calcula sobre las sumas medidas y puede diferir de sumar las medianas por etapa. El registro conserva también tiempo de CPU del proceso, cuya resolución en este Windows produjo valores cuantizados de 0 y 15,625 ms por llamada. Para describir la latencia se usan los intervalos de pared anteriores. Los tiempos varían con la carga del equipo. [Mediciones individuales](../results/fase5/latency_batch1.csv) · [Métricas y entorno](../results/fase5/metricas.json)

## Comprobaciones y reproducción

**232 pruebas pasaron en 23,16 s** con advertencias tratadas como errores. Cubren invariancia de características a ganancia y fase constante, cumulantes de constelaciones conocidas, invariancia de las resultantes diferenciales frente a CFO, escala aprendida solo en train, persistencia y orden de selección/evaluación. La revisión independiente reconstruyó exactitud, precision, recall y F1 desde las 9 600 filas de predicción emparejada, cotejó hashes y verificó la elección por validación. Se inspeccionaron las dos matrices y la curva de exactitud. [Pruebas](../results/fase5/tests.json) · [Revisión independiente](../results/fase5/revision_orquestador.txt)

La recarga conservó las predicciones de validación. Una comprobación adicional reentrenó exclusivamente los ganadores sobre train con sus semillas originales y obtuvo estados de selección, escala y SVM idénticos, junto con las 4 800 predicciones de entrenamiento por longitud. Los modelos guardados conservaron sus hashes. La demo contiene seis padres nuevos, semilla 2026110504, y llamadas a la API con I/Q y fs. [Comprobación de artefactos](../results/fase5/artifact_validation.json) · [Predicciones de la demo](../results/fase5/demo_predictions.json)

Para reproducir el experimento en una carpeta nueva desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-fase5.txt
.\.venv\Scripts\python.exe -m pytest -q -W error
.\.venv\Scripts\python.exe -m experiments.exp02_conventional --config configs/exp02_conventional.json --output results/e2_repeticion
```

El comando reutiliza `data/e2_dataset` si coincide con la configuración. Si esa carpeta está ausente, regenera el dataset. Las semillas reproducen datos y ajuste. Los tiempos medidos dependen de la ejecución.

Los dos artefactos listos para inferencia están en [conventional_256.joblib](../results/fase5/conventional_256.joblib) y [conventional_512.joblib](../results/fase5/conventional_512.joblib). Su API requiere una señal de una de las clases conocidas y devuelve una etiqueta sin confianza calibrada. La evaluación de entradas de solo ruido, clases desconocidas y cambio de canal corresponde a etapas posteriores. `test_shift` permanece reservado y la siguiente fase de la guía es la comparación con un modelo avanzado sobre estos mismos padres.
