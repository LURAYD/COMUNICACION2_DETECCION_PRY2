# Reporte de Fase 7 — Monitor Integrado y CLI de Señales Desconocidas

**Fecha:** 10 de noviembre de 2026  
**Requisitos cubiertos:** Fase 7 de `GUIA_MAESTRA_COMII_PARA_CODEX.md` (interfaz CLI para evaluación sobre archivos ciegos/desconocidos y consolidación de las 5 salidas obligatorias).

---

## 1. Resumen ejecutivo

Se completó e integró la herramienta de línea de comandos **Monitor Integrado (`src.monitor`)**, diseñada para que el docente o evaluador pueda alimentar cualquier archivo de señal desconocido y obtener de forma automática, reproducible y fundamentada las **cinco salidas obligatorias** del proyecto:

1. **Detección de presencia (Sí / No):** Estadística de energía con umbral calibrado bajo tasa de falsa alarma controlada ($P_{fa} = 1\%$).
2. **Espectro y ocupación espectral:** Estimación de densidad espectral de potencia (PSD con ventana periódica Hann) y ancho de banda ocupado al 99% de la potencia total.
3. **Estimación de SNR:** Estimación no-oracular si se suministra una referencia de ruido externo en la misma banda y ganancia; en su ausencia, el monitor documenta formalmente su indeterminación sin forzar aproximaciones arbitrarias.
4. **Tipo de modulación:** Clasificación automática entre las seis clases del curso (`BPSK`, `QPSK`, `8PSK`, `16QAM`, `64QAM`, `4PAM`) mediante el clasificador entrenado del proyecto (por defecto SVM256 con pipeline de características normalizadas).
5. **Confianza calibrada:** Estimación probabilística empírica (calibrada mediante regresión logística sobre el margen de decisión en validación independiente), evitando el uso de valores crudos o pseudo-probabilidades no calibradas.

---

## 2. Formatos de entrada soportados

El monitor detecta y procesa automáticamente los siguientes formatos:
- `.c64` / `.bin`: Archivos binarios de muestras complejas estándar (interleaved float32 Real + float32 Imag, formato nativo de GNU Radio y SDRs).
- `.npy`: Arrays de NumPy (1D complejo, o 2D con dos columnas correspondientes a componentes I y Q).
- `.npz`: Archivos comprimidos de NumPy (busca claves comunes `iq`, `samples`, `data` o toma el primer array).
- `.csv`: Archivos de texto delimitados por comas con dos columnas I/Q o una columna de números complejos.

---

## 3. Ejemplo de uso por línea de comandos (CLI)

### Comando básico (formato por defecto)
```powershell
.\.venv\Scripts\python.exe -m src.monitor --input ruta/a/archivo_desconocido.c64
```

### Comando con opciones avanzadas
```powershell
.\.venv\Scripts\python.exe -m src.monitor `
    --input data/unknown/X01.c64 `
    --fs-hz 8000 `
    --threshold 0.5 `
    --output results/monitor/X01.json `
    --plot results/monitor/X01_psd.png
```

### Parámetros CLI disponibles:
* `--input`, `-i`: Ruta obligatoria al archivo de señal.
* `--fs-hz`: Frecuencia de muestreo en Hz (por defecto 8000 Hz; si se omite o es None, el espectro se reporta en ciclos/muestra).
* `--dtype`: Tipo de dato binario para archivos raw (por defecto `complex64`).
* `--model`, `-m`: Ruta al pipeline clasificador `.joblib` (por defecto `results/fase5/conventional_256.joblib`).
* `--policy`: Ruta al calibrador logístico (por defecto `results/adaptive/policy.joblib`).
* `--output`, `-o`: Ruta destino del reporte JSON (por defecto `results/monitor/<signal_id>.json`).
* `--plot`, `-p`: Ruta destino de la figura PNG (por defecto `results/monitor/<signal_id>_psd.png`).
* `--threshold`: Umbral de energía para la decisión de presencia de señal.
* `--noise-ref`: Ruta opcional a un archivo de solo ruido para estimar la SNR numéricamente.

---

## 4. Estructura del reporte generado

### 1. Salida JSON estructurada (`results/monitor/<id>.json`)
```json
{
  "signal_id": "X01",
  "num_samples": 4096,
  "fs_hz": 8000.0,
  "detection": {
    "detected": true,
    "observed_energy": 0.9994,
    "threshold": 0.5,
    "status": "signal_present"
  },
  "spectrum": {
    "bandwidth_99": 1134.48,
    "bandwidth_unit": "Hz",
    "frequency_bounds": [-589.3, 584.9],
    "integrated_power": 1.0129,
    "window_method": "periodic_hann",
    "spectrum_file": "results/monitor/X01_psd.png"
  },
  "snr": {
    "snr_est_db": null,
    "status": "no_identificable_sin_supuestos",
    "reason": "falta_referencia_de_piso_de_ruido",
    "explanation": "La estimación rigurosa de SNR requiere una referencia independiente de ruido en la misma banda y ganancia."
  },
  "classification": {
    "modulation": "QPSK",
    "confidence": 0.8542,
    "confidence_calibrated": true,
    "model_used": "results/fase5/conventional_256.joblib",
    "scope": "signal_present_closed_set",
    "window_votes": {
      "QPSK": 14,
      "BPSK": 2
    }
  },
  "warnings": []
}
```

### 2. Figura gráfica PNG (`results/monitor/<id>_psd.png`)
Genera un panel doble de alta resolución:
* **Panel izquierdo:** Densidad espectral de potencia (PSD en dB/Hz) con la banda del 99% sombreada y límites marcados con líneas verticales discontinuas.
* **Panel derecho:** Diagrama de constelación I/Q de las muestras recibidas.

---

## 5. Pruebas automáticas

Las pruebas automáticas se ubican en `tests/test_monitor_cli.py`:
- `test_load_signal_various_formats`: Comprueba la lectura exacta y conversión en `.c64`, `.npy`, `.npz` y `.csv`.
- `test_monitor_complete_five_outputs`: Valida la entrega completa y coherente de las 5 salidas sobre una señal sintética.
- `test_monitor_with_noise_reference`: Valida el cálculo numérico de SNR cuando se proporciona una referencia de ruido.

Todas las pruebas pasan al 100% en conjunto con `tests/test_fase4_integration.py`.
