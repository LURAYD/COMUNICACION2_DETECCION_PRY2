# Evaluación independiente final de los cuatro clasificadores

La evaluación del holdout nuevo **no demuestra una mejora de la CNN con rotación de fase frente a la original**. El aumento de exactitud es pequeño y sus intervalos pareados incluyen cero. Los SVM conservan mejores resultados agregados; todas las alternativas se degradan ante cambio de canal. No se reentrenó ni ajustó ningún modelo después de abrir esta reserva.

| Modelo congelado | Exactitud ID | F1 macro ID | Exactitud cambio de canal | F1 macro cambio |
|---|---:|---:|---:|---:|
| CNN original, época 46 | 40,75% | 0,40778 | 19,02% | 0,17285 |
| CNN con rotación, época 24 | 41,54% | 0,41698 | 19,48% | 0,14835 |
| SVM 256 | 43,27% | 0,43209 | 26,81% | 0,26911 |
| SVM 512 | 47,21% | 0,47031 | 29,06% | 0,28718 |

Cada régimen reúne 4 800 señales presentes, con 100 padres por clase/SNR. Las cuatro variantes observan las mismas realizaciones; SVM512 recibe el prefijo extendido. No es una evaluación del detector, de clases desconocidas ni del monitor completo. Precision, recall y F1 por clase están en `results/fase6_holdout_final/metrics.json`; resultados por SNR en `by_snr.csv`; todas las predicciones y ocho matrices de conteos están guardadas.

## Comparación pareada

Intervalos percentiles del 95% para diferencia de **exactitud**, en puntos porcentuales: 10 000 réplicas, semilla 2026110610, padres remuestreados dentro de cada celda clase/SNR y conservando parejas. El cálculo multinomial de diferencias −1/0/+1 es equivalente a ese bootstrap para la media de diferencias. No son intervalos de F1.

| Diferencia A−B | ID: diferencia [IC95%] | Cambio: diferencia [IC95%] |
|---|---:|---:|
| CNN rotación − CNN original, primaria | +0,79 [−0,60; +2,17] | +0,46 [−0,71; +1,67] |
| CNN original − SVM256 | −2,52 [−4,06; −0,94] | −7,79 [−9,27; −6,29] |
| CNN rotación − SVM256 | −1,73 [−3,23; −0,27] | −7,33 [−8,75; −5,87] |
| CNN rotación − SVM512 | −5,67 [−7,12; −4,17] | −9,58 [−11,02; −8,17] |
| SVM512 − SVM256 | +3,94 [+2,50; +5,35] | +2,25 [+0,83; +3,71] |

En ID, la variante corrige 653 errores de la CNN original y rompe 615 aciertos: saldo de 38 padres. Con cambio de canal corrige 502 y rompe 480: saldo de 22. El F1 macro bajo cambio empeora pese al pequeño aumento de accuracy. La matriz muestra concentración de la variante en QPSK y 64QAM y escasos aciertos de BPSK, 8PSK y 4PAM. Por eso no se interpreta ese pequeño aumento como robustez general.

Las curvas muestran que a SNR alta ID los SVM superan a ambas CNN; a SNR intermedia las CNN pueden superar SVM256. Esta descripción del test no se convierte en una política seleccionada usando SNR verdadera. Con cambio de canal las CNN permanecen cerca del nivel de azar agregado incluso a SNR alta. El shift combina CFO fuera del rango de entrenamiento y multipath distinto; no permite separar cuál causa el deterioro.

## Integridad y límites

Se verificó la reserva semilla 2026110609 contra todos sus hashes antes de inferir. La configuración física coincide con E2 salvo semilla. Los padres son disjuntos; ninguna ventana IQ256 de evaluación es idéntica a una de cualquier split antiguo. La auditoría confirma prefijos y particiones independientes. Se congelaron rutas y SHA256 de cuatro modelos antes de inferencia en `protocol_frozen.json`; `completed.json` confirma que los modelos no cambiaron. No se usaron train/val del nuevo dataset.

La apertura se autorizó después del ajuste, que solo usó train/val antiguos. El holdout fue independiente para esta evaluación, pero **ya no está sellado para futuros desarrollos**. La reserva original se conserva como registro histórico; el estado actual es `results/fase6_holdout_final/ESTADO_REANUDACION.md`.

Los IC condicionan a estos modelos fijos y al simulador. Una semilla de entrenamiento no estima variabilidad entre entrenamientos. No incluyen incertidumbre del canal físico ni RF real. Las comparaciones SVM son descriptivas, sin corrección por multiplicidad. Las cifras no demuestran novedad ni utilidad de una política de riesgo. La latencia no se volvió a medir: las mediciones anteriores describen el mismo hardware y arquitectura, pero no son parte de esta prueba de exactitud.

## Verificación y reproducción

Dos pruebas pertinentes del bootstrap pasaron; comprueban parejas idénticas y diferencia determinista por estrato. La verificación desde CSV reconstruyó exactamente las métricas y las matrices de los ocho grupos y confirmó 38 400 predicciones y emparejamiento por padre (`verification.json`). Se inspeccionaron las cuatro figuras. Se corrigieron márgenes de las matrices a partir de conteos guardados, sin repetir inferencia.

```powershell
.\.venv\Scripts\python.exe -m experiments.exp06_final_holdout --output results/holdout_reproduccion
.\.venv\Scripts\python.exe -m experiments.verify_final_holdout
```

El primer comando reproduce modelos congelados sobre el mismo holdout; no constituye otra prueba independiente. El segundo verifica la carpeta final original. El programa rechaza sobrescribir una carpeta existente. [Protocolo](FASE6_HOLDOUT_PROTOCOLO.md).

## Decisión y siguiente fase

Conservar CNN original como referencia avanzada y la variante de fase como ensayo documentado; **no promover la variante como mejora probada ni sustituir los artefactos originales**. Mantener SVM como referencia convencional principal. El siguiente paso propuesto por la guía es integrar el monitor y diseñar calibración de confianza con datos de desarrollo independientes de este holdout. Eso requiere su propia validación: no se realizó aquí. ADC permanece secundario. No se ejecutó apagado.
