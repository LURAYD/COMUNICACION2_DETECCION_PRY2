# Fase 6: protocolo fijado antes de evaluar

La autorización actual de entrenamiento reemplaza la pausa anterior. Se usa un solo agente de trabajo, sin delegación adicional. ADC permanece pospuesto; la investigación sigue centrada en decidir entre finalizar con 256, observar 512 o procesar 256 con un modelo avanzado.

## Modelo y selección

Propuesta de ingeniería: CNN 1D pequeña sobre I/Q centradas y normalizadas con la potencia de cada ventana de 256. Cuatro convoluciones (32,32,64,64 canales), pooling y cabeza de 128 unidades. Las convoluciones aprenden patrones locales de muestras, sin imponer sincronización de símbolos; el pooling reduce tamaño. Esta elección no es reproducción de un paper ni garantía de superar SVM. No se introduce SNR verdadera ni canal como entrada.

Una arquitectura y una semilla prefijadas; AdamW, float32, batch 128, máximo 100 épocas, parada tras 15 épocas sin mejorar F1 macro de validación. Scheduler reduce LR usando exclusivamente pérdida de validación. Se conserva el mejor F1 val. Train y val son los de E2; no se incorporan datos de validación al ajuste. Test y test_shift se evalúan después de congelar el modelo; nunca seleccionan hiperparámetros.

E3: CNN256. E4/E5/E6: CNN256 y los SVM256/512 ya guardados. Misma población de señales presentes, clases conocidas, padres y condiciones del dataset E2. test_shift contiene cambio conjunto de CFO y multipath; no permite atribuir aisladamente la degradación a uno de ellos. Una semilla no estima variabilidad de entrenamiento.

## Recuperación

Checkpoint cada diez minibatches y fin de época; escritura temporal, flush/fsync y reemplazo atómico. last.pt guarda modelo, AdamW, scheduler, RNG Python/NumPy/Torch CPU y CUDA, permutación de train, siguiente minibatch, época/paso, acumuladores de pérdida, historial, configuración y SHA256 de observaciones/manifiestos. best.pt guarda el mismo contrato en cada mejora. No se usa AMP; scaler=None se registra explícitamente. La prueba corta usa procesos separados y verifica modelo/optimizador bit a bit e historial idéntico.

La continuidad exacta se pretende en el mismo entorno y GPU; no se promete reproducibilidad bit a bit entre plataformas/versiones. Un corte inesperado puede perder hasta diez minibatches, que se repiten desde last.pt. No se puede detectar desde Python la cuota restante de Codex ni garantizar un último llamado de herramienta al agotarse. No se programa apagado por tiempo: únicamente sería admisible tras guardado verificado y fin de procesos, sin forzar cierre de aplicaciones del usuario.

## Medición

Inferencia desde ventana IQ en CPU, batch 1, un hilo CPU, 16 calentamientos, 128 ventanas y cinco repeticiones. CNN CPU y GPU separadas, GPU incluye transferencias y sincronización. SVM se vuelve a medir en el mismo ensayo. Adquisición teórica N/fs (32/64 ms) separada de cómputo. No representa consumo energético ni hardware ASIC.

## Fuentes técnicas

- Guía maestra local, apartados 2, 5/Fase6 y 7.
- [Distribución oficial PyTorch 2.10 CUDA12.8](https://pytorch.org/get-started/previous-versions/), instalación solo de torch; no se necesitan torchvision/torchaudio.
- [Checkpoint general PyTorch](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
- [Reproducibilidad PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.html).
