# Fase 6: CNN 1D y comparación E3–E6

La CNN 1D de 256 muestras I/Q se entrenó y evaluó en la RTX 4060 Laptop de 8 GB. **No superó a los SVM**: obtuvo 40,94 % en test y 18,60 % bajo cambio de canal. Se conservan estos resultados sin ajustar el modelo mirando test.

| Modelo | Exactitud test | Exactitud test_shift |
|---|---:|---:|
| CNN1D, 256 muestras | 40,94 % | 18,60 % |
| SVM, 256 muestras | 43,69 % | 27,92 % |
| SVM, 512 muestras | 46,60 % | 28,67 % |

Cada split de evaluación contiene 4 800 señales presentes, con los mismos padres E2 para los tres modelos. Se comprobó separación de padres entre train, val, test y test_shift. La SNR nominal se utiliza solamente para agrupar resultados de evaluación. `metricas.json` conserva accuracy, precision, recall y F1 macro y por clase; `metrics_by_snr.csv`, `predictions.csv`, seis matrices JSON/PNG y dos curvas PNG completan E3–E6. El F1 macro test de la CNN es 0,40833, frente a 0,43765/0,4637 de los convencionales.

## Arquitectura y entrenamiento

Entrada `(batch,2,256)`: dos canales I y Q, convolución **unidimensional a lo largo del tiempo**. Se centra cada ventana y normaliza su potencia; no se usan etiquetas, SNR o canal en esta transformación. Cuatro Conv1d de 32/32/64/64 canales, ReLU, pooling temporal fijo y cabeza 512→128→6 con dropout 0,2. **94 726 parámetros**. No es CNN2D ni Transformer.

Semilla 2026110601; AdamW, LR inicial 0,001, weight decay 0,0001, batch128, float32; máximo100 épocas y paciencia15. Scheduler según pérdida val; selección según F1 macro val. Terminó en época61; mejor época46, F1 val0,405419. El bucle de entrenamiento/validación/checkpoints tardó16,34s; excluye instalación, importación, carga/preprocesamiento inicial y evaluación posterior. PyTorch2.10.0+cu128, CUDA12.8. La arquitectura es una propuesta moderada de ingeniería, no una reproducción de resultados bibliográficos.

## Costo observado

Mismo ensayo en i7-13620H, un hilo CPU, modelo cargado, batch1,16 calentamientos,128 ventanas,5 repeticiones. Incluye preprocesamiento y predicción; GPU incluye transferencia desde CPU y sincronización.

| Pipeline | Mediana ms | p95 ms |
|---|---:|---:|
| CNN256 CPU | 0,422 | 0,692 |
| CNN256 GPU | 0,962 | 1,718 |
| SVM256 CPU | 1,618 | 2,529 |
| SVM512 CPU | 1,995 | 2,577 |

Adquisición virtual aparte:32ms para256 y64ms para512 a Fs8000. La GPU acelera entrenamiento por lotes; su transferencia/lanzamiento pesa en inferencia de una ventana pequeña. Las latencias SVM difieren de Fase5 por variación de ejecución/carga del equipo: comparar las cifras del ensayo común. Un muestreo durante entrenamiento observó927MiB/8188MiB,40% utilización y48°C; no es un máximo de memoria. El campo `peak_cuda_allocated_bytes` de métricas corresponde exclusivamente al proceso de evaluación.

La CNN resultó **más barata en CPU** que los SVM implementados. Por tanto, estas mediciones no sustentan llamar a esta rama «más cómputo». Tampoco demuestran una política adaptativa: falta calibración y selección de acciones por riesgo/beneficio. A SNR−5/0dB la CNN ayuda respecto de SVM256, pero pierde a SNR alta y ante shift; esa observación es descriptiva del test, no una regla seleccionada para despliegue. ADC continúa pospuesto.

## Checkpoints y pruebas

- `results/fase6/last.pt`: final, época61, optimizador/scheduler/RNG y progreso completo.
- `results/fase6/best.pt`: seleccionado exclusivamente por val, época46.
- `results/fase6/selection_frozen.json`: hash del modelo antes de evaluar test/shift.
- `results/fase6_resume_verified/validation.json`: interrupción tras paso1 dentro de época, salida del proceso y reanudación en otro. Pesos y AdamW bit a bit iguales a corrida continua; historial idéntico.
- **235 pruebas pasaron en11,85s**, comando `python -m pytest -q -W error --basetemp tmp/pytest_fase6`.

La primera prueba de humo detectó pooling adaptativo sin backward determinista CUDA; se reemplazó por pooling fijo equivalente para longitud256 antes del entrenamiento final. Un test de reemplazo atómico falló en el directorio temporal del sandbox; pasó al usar el tmp del workspace. Se conserva la evidencia de humo inicial separada. Los checkpoints usan archivo temporal, fsync y reemplazo atómico; conservan RNG Python/NumPy/Torch CPU/CUDA, permutación/cursor de minibatch, época/paso, historial, configuración y hashes de todos los datos. No se utiliza scaler por trabajar en float32.

## Reproducción y continuación

Desde la raíz, con el entorno instalado:

```powershell
.\.venv\Scripts\python.exe -m experiments.exp03_advanced --output results/fase6_repeticion
.\.venv\Scripts\python.exe -m experiments.exp04_comparison --output results/fase6_repeticion
```

Para una corrida interrumpida añadir `--resume` al comando de entrenamiento con la misma configuración/carpeta. La corrida actual está finalizada; no necesita continuar entrenándose. Fuentes y protocolo fijado en [FASE6_PROTOCOLO.md](FASE6_PROTOCOLO.md). No se modificaron datos ni modelos E2. Siguiente fase de la guía: integración del monitor y tratamiento explícito de confianza/limitaciones; no se inició en este cierre.

No se detectó agotamiento de cuota ni se ejecutó apagado. No existe un detector fiable de cuota disponible al proceso; los checkpoints protegen la recuperación ante una interrupción.
