# Estado de entrenamiento

Época completada: 2; siguiente minibatch: 0; pasos: 4; terminado: True.

Checkpoint last.pt atómico: modelo, optimizador, scheduler, RNG Python/NumPy/Torch CPU/CUDA, orden/paso, historial, configuración y hashes de datos. Float32; scaler no aplica.

Reanudar desde raíz: `.\.venv\Scripts\python.exe -m experiments.exp03_advanced --config results/fase6_phase_resume/interrupted/config.json --output results/fase6_phase_resume/interrupted --resume --limit 256`

No hay detector de cuota. Si el proceso se interrumpe abruptamente, se repiten como máximo checkpoint_steps minibatches desde el último guardado. Evaluación E3–E6 y reporte pendientes hasta existir metricas.json.
