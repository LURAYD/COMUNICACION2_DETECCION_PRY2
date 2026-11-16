# Fase 3: dataset reproducible y separación por realización

Fecha de ejecución final: 4 de noviembre de 2026. Estado: completada. Se retomaron las candidatas guardadas tras una interrupción por límite de uso y se aplicó la propuesta previamente mostrada. No se entrenaron modelos ni se inició Fase 4.

## Cambios y entregables

- `src/dataset.py`: planificación por split/padre, generación, lectura pública separada y auditoría.
- `configs/dataset_smoke.json`: parámetros versionados sin dependencias adicionales.
- `tests/test_dataset.py`: 15 casos nuevos, incluidas alteraciones que deben ser rechazadas.
- `.gitignore`: excluye datos regenerables en `data/`.
- `docs/PROTOCOLO_EXPERIMENTAL.md`, `docs/DECISIONES.md`, `docs/MATRIZ_LITERATURA.md` y `README.md`: protocolo, fuentes y ejecución.
- `data/fase3_smoke/`: observaciones NPZ, índice CSV, metadata pública, manifiesto oracle JSONL, configuración y auditoría.
- `results/fase3/`: logs, `audit.json`, `conteos.csv`, `reproducibilidad.json`, respaldo de `.gitignore` y diff aplicado.

El diff previo está en `docs/PROPUESTA_FASE3.diff`; el aplicado en `results/fase3/cambios.diff`. Se conservaron los módulos funcionales de Fases 1 y 2.

## Resultados reales

| Split | Padres con señal | Solo ruido | Total padres | Ventanas 256/512 |
|---|---:|---:|---:|---:|
| train | 48 | 8 | 56 | 112 |
| val | 48 | 8 | 56 | 112 |
| test | 48 | 8 | 56 | 112 |
| test_shift | 48 | 8 | 56 | 112 |
| Total | 192 | 32 | 224 | 448 |

En cada split, cada una de las seis clases tiene exactamente dos realizaciones por SNR nominal −10/0/10/20 dB. Se comprobaron 224 padres distintos, 672 streams de parámetros/bits/ruido distintos, cero padres compartidos entre splits y prefijos exactos en todos los registros. La auditoría también rechaza I/Q512 idéntica entre padres.

El dataset ocupa **3,005,629 bytes**; sus cuatro archivos I/Q ocupan 2,754,536 bytes. Los **nueve archivos** generados son idénticos por SHA-256 a la segunda ejecución, incluyendo observaciones, manifiestos y auditoría. Cambiar la semilla cambia las muestras; cambiar la configuración cambia también el espacio de IDs de padres, según los tests.

`test_shift` usa magnitudes CFO en [200,300) Hz y canales con retardos máximos 9/13 muestras; ID usa CFO en [−100,100) Hz y retardos máximos 0/2/5. Es un cambio de soporte real, no solo otra semilla. Configuración común: Rs=1000 baud, sps=8, fs=8000 Hz, 128 símbolos, beta variable y amplitud/fase variables.

La SNR local medida de las ventanas va de −10.2890 a 21.1745 dB en el conjunto. Puede diferir de la nominal porque AWGN se escala con potencia de la trama completa y luego se recorta una ventana estable, además de la fluctuación estadística. Se conservan ambos valores en oracle; no se reajusta la ventana para imponer igualdad.

## Ejecución y pruebas

```powershell
.\.venv\Scripts\python.exe -m pytest -q -W error
# Agente de código: 206 passed in 9.74s
# Revisión independiente: 206 passed in 5.00s

.\.venv\Scripts\python.exe -m src.dataset --config configs/dataset_smoke.json --output data/fase3_smoke
.\.venv\Scripts\python.exe -m src.dataset --config configs/dataset_smoke.json --output tmp/fase3_repeticion
```

No hubo fallos, xfail ni advertencias. Los tests verifican balance, reproducción, semillas, soporte shift, recorte y ruido solo mediante estadística agregada. También corrompen padre, fila, ruta, prefijo, streams, metadata, recorte, taps shift o I/Q duplicada y comprueban rechazo. Una copia que contiene solo observaciones, índice y metadata pública demuestra que `load_observation` funciona sin oracle ni configuración y devuelve únicamente I/Q y fs.

Logs en `results/fase3/pytest.txt`, `revision_orquestador.txt`, `generacion.txt` y `repeticion.txt`; hashes y comparaciones en `reproducibilidad.json`. El entorno sigue siendo Python 3.11.0 y NumPy 2.4.6. No se instalaron dependencias.

## Fuentes consultadas y decisiones propias

Se revisaron visualmente los PDF originales de **RML22**, pp. PDF 4 y 8 (impresas 7666/7670), y **FFTFormer**, p. PDF 9 (impresa 1429). RML22, ecuaciones (8), fundamenta el ruido complejo; su sección IV-B advierte sobre repetir semillas del canal. FFTFormer documenta variación de beta. La matriz de literatura registra las referencias exactas y una inconsistencia entre el orden dibujado en la figura 3 de RML22 y su sección IV-D; se conserva la cadena multipath → CFO/fase → AWGN validada en el proyecto.

La partición previa a derivados, jerarquía SeedSequence, 224 padres, parámetros del smoke y formato de almacenamiento son decisiones propias. No se atribuyen a los papers ni se afirma reproducir sus resultados, su canal ETU70 o sus modelos.

## Explicación y límites

Las dos ventanas de un padre representan mirar primero 256 muestras y después prolongar la misma observación hasta 512. Comparten sus primeras muestras y ruido: no son dos ejemplos independientes. Al separar padres antes de generar derivados se evita que una misma realización aparezca en entrenamiento y prueba.

Noise-only contiene gaussianas complejas reales del simulador, sin modulación ni señal escondida. Su SNR de señal es nula/N/A. La distribución de potencia aproxima la del ruido con señal, pero no coincide exactamente tras el FIR; se declara como limitación del smoke. La separación oracle es lógica, no un control de acceso del sistema operativo.

El conjunto es suficiente para probar el procedimiento, no para métricas finales, calibrar P_FA=0.01 ni entrenar clasificadores. No hay todavía evidencia de desempeño o generalización. El siguiente cambio mínimo propuesto es Fase 4: detector, espectro y estudio de SNR ciega con supuestos explícitos, después de ampliar realizaciones según el experimento. No se ejecutó automáticamente.
