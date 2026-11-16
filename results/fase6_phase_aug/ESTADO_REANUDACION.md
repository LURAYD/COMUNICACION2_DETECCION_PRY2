# Estado de entrenamiento

Época completada: 39; siguiente minibatch: 0; pasos: 1482; terminado: True.

Checkpoint last.pt atómico: modelo, optimizador, scheduler, RNG Python/NumPy/Torch CPU/CUDA, orden/paso, historial, configuración y hashes de datos. Float32; scaler no aplica.

Reanudar desde raíz: `.\.venv\Scripts\python.exe -m experiments.exp03_advanced --config results/fase6_phase_aug/config.json --output results/fase6_phase_aug --resume`

Entrenamiento terminado en época39; mejor época24. Diagnóstico train/val terminado en ../fase6_phase_diagnostico. Reporte: ../../docs/FASE6_AJUSTE_VALIDACION.md. No está pendiente ejecutar E3–E6 sobre el test antiguo: se dejó un holdout nuevo sellado para evaluación final posterior. No hay detector de cuota ni temporizador de apagado. Los checkpoints incluyen RNG de augmentation y su recuperación se probó en ../fase6_phase_resume/validation.json.
