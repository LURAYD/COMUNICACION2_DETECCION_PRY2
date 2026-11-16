# Fase 6 finalizada

Entrenamiento y E3–E6 terminados. last.pt: época61; best.pt: época46. Ambos se cargaron correctamente durante evaluación. Configuración: config.json; métricas: metricas.json; selección y SHA256: selection_frozen.json. Reporte: ../../docs/FASE6_VALIDACION.md.

No quedan pasos de entrenamiento pendientes. Para reanudar una interrupción de esta configuración desde raíz:

```powershell
.\.venv\Scripts\python.exe -m experiments.exp03_advanced --config results/fase6/config.json --output results/fase6 --resume
```

En este estado completed=True, el comando no añade épocas. Para una reproducción nueva utilizar otra carpeta. Los checkpoints contienen modelo, AdamW, scheduler, scaler=None (float32), época/paso, orden/cursor, RNG Python/NumPy/Torch CPU/CUDA, historial/configuración/hashes. Recuperación real probada en ../fase6_resume_verified/validation.json.

Pendiente fuera de Fase6: monitor integrado, confianza/calibración y eventual política por riesgo. ADC secundario pospuesto. No hay temporizador de apagado ni detector de cuota; no se apagó el equipo.
