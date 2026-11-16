# Reporte de Fase 7 y Corrección Nivel 1 — Monitor Integrado y CLI de Señales Desconocidas

**Fecha de actualización:** 16 de noviembre de 2026  
**Requisitos cubiertos:** Fase 7 de `GUIA_MAESTRA_COMII_PARA_CODEX.md` y Nivel 1 (`N1.1`–`N1.6`) de `docs/PLAN_MODIFICACIONES_Y_CRITERIOS_DE_ACEPTACION.md`.

---

## 1. Contrato operativo del Monitor Integrado (`src/monitor.py`)

El monitor procesa una observación I/Q individual y entrega cinco salidas con estados explícitos:

1. **Detección de presencia (`N1.1`):**
   - Evalúa todas las ventanas completas de 256 muestras del archivo (`detect_energy_file`) con corrección de Šidák para controlar la falsa alarma **por archivo** (`pfa_file = 0.01`).
   - Calibración invariante a escala desde `--noise-ref` (preferido) o `detection_calibration`.
   - Si no se suministra referencia de ruido ni umbral manual explícito (`--threshold`), la detección devuelve `detected: null` y estado `indeterminate_uncalibrated` (nunca se presenta como «solo ruido» con un umbral silencioso unitario).
2. **Espectro y ancho de banda (`N1.2`, `N1.6`):**
   - Si no se especifica `--fs-hz` ni metadatos verificables (`fs_hz`), el espectro y el ancho de banda se expresan en `ciclos/muestra` sin inventar `8000 Hz`.
   - `bandwidth_99` se etiqueta explícitamente como `total_power_99_includes_noise` (incluye ruido y DC); cuando hay referencia de ruido y señal detectada se informa además `signal_bandwidth_est` tras restar el piso espectral plano estimado.
3. **Estimación de SNR (`N1.3`):**
   - Requiere señal detectada (`detected = True`) y referencia independiente compatible en frecuencia de muestreo y ganancia (`--noise-ref`).
   - En ausencia de referencia o ante resta no positiva / detección negativa o indeterminada, devuelve `snr_est_db: null` y motivo explícito (`status`, `reason`).
4. **Tipo de modulación (`N1.2`, `N1.5`, `N1.6`):**
   - Clasificador de alcance cerrado sobre 6 clases (`BPSK`, `QPSK`, `8PSK`, `16QAM`, `64QAM`, `4PAM`) identificando modelo, SHA-256 y longitud de ventana (`256` o `512`).
   - Distingue explícitamente `NO_SIGNAL`, `INDETERMINATE_DETECTION`, `OUT_OF_SCOPE_SHORT_SIGNAL` (no rellena señales de < 256 muestras), `OUT_OF_SCOPE_MISSING_FS` y `OUT_OF_SCOPE_INCOMPATIBLE_SAMPLING` (rechaza frecuencias o `sps` incompatibles con el modelo entrenado a `8000 Hz`, `sps=8`).
   - Evalúa únicamente las ventanas activas detectadas para no diluir transmisiones posteriores al primer prefijo.
5. **Confianza de la decisión final (`N1.4`):**
   - Sin recorte artificial a `[0.20, 0.99]`.
   - Marca `confidence_calibrated: true` únicamente cuando el paquete `results/nivel1_monitor/monitor_calibrators.joblib` coincide exactamente con el `model_sha256`, `feature_version`, `window_len`, modo de agregación (`single_window` o `majority_vote`) y número de ventanas evaluadas (`1` o `2` ventanas para `SVM256`, `1` ventana para `SVM512`). En cualquier otra combinación reporta la puntuación sin calibrar con `confidence_calibrated: false`.

---

## 2. Formatos de entrada soportados y validación (`N1.5`)

`load_signal` admite únicamente extensiones documentadas y rechaza entradas ambiguas o corruptas (código de salida `2` en CLI):
- `.c64` / `.bin`: muestras complejas intercaladas (`complex64` por defecto); rechaza archivos vacíos o cuyo tamaño en bytes no sea múltiplo exacto de `dtype.itemsize`.
- `.npy`: array 1D complejo (o 2D `(1, N)` / `(N, 1)`), o matriz real 2D `(N, 2)` / `(2, N)` ($N > 2$); rechaza lotes 2D complejos `(M, N)` y arrays 1D reales.
- `.npz`: exige exactamente una clave de señal entre `('iq', 'iq256', 'iq512', 'samples')` y admite metadatos escalares opcionales (`fs_hz`, `sps`, `symbol_rate_hz`, `gain`).
- `.csv`: exige 2 columnas `I,Q` numéricas (con cabecera opcional `I,Q`/`real,imag`) o 1 columna compleja explícita; rechaza filas inconsistentes o valores no finitos.
- Metadatos sidecar opcionales: `<archivo>.meta.json`.

---

## 3. Ejemplos de uso por línea de comandos (CLI)

### Con referencia de ruido independiente y frecuencia verificable
```powershell
.\.venv\Scripts\python.exe -m src.monitor `
    --input results\nivel1_monitor\demos\demo_py_qpsk_15db.c64 `
    --noise-ref results\nivel1_monitor\demos\demo_py_noise_ref.c64 `
    --fs-hz 8000 `
    --output results\nivel1_monitor\demos\demo_py_qpsk_15db.json `
    --plot results\nivel1_monitor\demos\demo_py_qpsk_15db_psd.png
```

### Sin frecuencia conocida (modo espectral seguro)
```powershell
.\.venv\Scripts\python.exe -m src.monitor --input senal.c64 --noise-ref ruido.c64
```
Reporta espectro en `ciclos/muestra` y marca clasificación como `OUT_OF_SCOPE_MISSING_FS`.

---

## 4. Validación y pruebas automáticas

- Calibración (`N1.4`) y SNR (`N1.3`): ejecutadas y guardadas en `results/nivel1_monitor/` mediante `experiments/eval_n1_monitor_validation.py`.
- Pruebas automáticas: `tests/test_monitor_cli.py`, `tests/test_nivel0_nivel1_acceptance.py` y `tests/test_fase4_integration.py`.
