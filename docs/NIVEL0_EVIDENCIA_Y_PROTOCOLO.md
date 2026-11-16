# Nivel 0 — Evidencia de referencia, integridad y protocolo congelado

**Fecha:** 16 de noviembre de 2026  
**Tareas cubiertas:** `N0.1`, `N0.2`  
**Script reproducible:** `experiments/verify_n0_protocol.py`  
**Configuración congelada:** `configs/validation_protocol_frozen.json`

## 1. Estado de referencia e integridad (`N0.1`)

- **Instantánea local en Git:** repositorio inicializado en la raíz del proyecto (`commit 7e81d0df252a52b69bed49d73467b4fa4618adc2`), conservando `.gitignore` con exclusión explícita de `data/`, `tmp/`, `.pytest_cache/` y artefactos binarios pesados (`*.pt`, `*.joblib`, `*.npz`, `*.c64`) para no publicar datasets ni pesos.
- **Manifiesto SHA-256:** `results/nivel0_evidencia/manifest_historical_state.json` registra **300 archivos** (`src/`, `experiments/`, `tests/`, `configs/`, `grc/`, modelos, métricas históricas y datasets).
- **Verificación de integridad y prueba de alteración:** `results/nivel0_evidencia/integrity_verification.json` confirma 300/300 hashes verificados (`live_integrity_check.passed = true`) y detección de 1 byte alterado sobre copia temporal de `results/paper_research/ablation_and_policy_metrics.json` sin modificar el original (`tamper_check_passed = true`).

### Cifras históricas de referencia (preservadas sin sobrescritura)

En el test adaptativo nominal (`data/adaptive_dataset`, partición `test`, $n=2.400$, fuente `results/paper_research/ablation_and_policy_metrics.json`):

| Modelo / Política | Exactitud nominal (`test`) | Fuente |
|---|---|---|
| SVM, 256 muestras (`always_a0_svm256`) | 43,75% (`0.4375`) | `results/paper_research/ablation_and_policy_metrics.json` |
| SVM, 512 muestras (`always_a1_svm512`) | 46,0417% (`0.4604166667`) | `results/paper_research/ablation_and_policy_metrics.json` |
| Red multidominio, 256 muestras (`always_a2_multidomain`) | 51,5833% (`0.5158333333`) | `results/paper_research/ablation_and_policy_metrics.json` |
| Fast LBAP (`lbap_full_a0_a1_a2`) | 51,625% (`0.51625`) | `results/paper_research/ablation_and_policy_metrics.json` |
| Ablación SNR ciega + planitud (`ablation_input_snr_flatness`) | 52,2917% (`0.5229166667`) | `results/paper_research/ablation_and_policy_metrics.json` |

### Correspondencia entre carpetas de resultados y procedencia

| Carpeta | Dataset y partición | Modelos / Configuración | Limitación de procedencia |
|---|---|---|---|
| `results/fase4` | Sintético al vuelo (`seed=20261105`, `configs/exp01_detection.json`) | Detector energía + SNR referencia | Umbral `1.15239` calibrado para ruido unitario; ID $-10\text{ dB}$ = `123/600 = 20,5%`. |
| `results/fase5` | `data/e2_dataset` (`train`/`val`/`test`, `test_shift` sellado) | `conventional_256.joblib`, `conventional_512.joblib` (`configs/exp02_conventional.json`) | Clasificación condicionada a señal presente; sin calibración de confianza. |
| `results/fase6_holdout_final` | `data/fase6_holdout_sellado` (`test` y `test_shift`, $n=4.800$) | CNN original, CNN phase_aug, SVM256, SVM512 | Evaluación histórica ya inspeccionada. |
| `results/adaptive` | `data/adaptive_dataset` (`train`, `val_cal`, `val_select`, `test`, `test_shift`) | `SVM256`, `SVM512`, `SmallCNN` (`fase6/best.pt`), `policy.joblib` | Calibrador ajustado sobre margen SVM256 de 1 ventana en `val_cal`. |
| `results/paper_research` | `data/adaptive_dataset` (+ `e2_dataset` + síntesis al vuelo para red) | `multidomain_*_best.pt`, SVM256, SVM512 | Selector entrenado con recuperabilidad in-sample sobre `train`; sin paquete serializado final. |
| `results/gnuradio`, `results/monitor` | `py_qpsk_15db.c64`, `grc_sim_output.c64` | `conventional_256.joblib` + `policy.joblib` | Los JSON de `results/monitor` usaron umbrales manuales (`0.1` y `0.5`) sin referencia de ruido guardada. |

## 2. Independencia de particiones y protocolo congelado (`N0.2`)

La auditoría `results/nivel0_evidencia/partition_independence_audit.json` verifica:

- **Cero padres compartidos internos** entre `train`, `val`, `test` y `test_shift` dentro de `e2_dataset` (19.456 padres), `adaptive_dataset` (9.856 padres), `fase6_holdout_sellado` (19.456 padres) y `fase3_smoke` (224 padres).
- **Cero padres, registros o señales `iq512` compartidos entre datasets** (`shared_parents_count = 0`).
- **Prefijos exactos** (`iq256 == iq512[:, :256]`) en todos los splits de los cuatro datasets (`all_prefixes_exact = true`).
- **Particiones de desarrollo adaptativo** (`results/adaptive/development_partitions.json`): `train` (2.400 padres), `val_cal` (1.200 padres) y `val_select` (1.200 padres) son disjuntas entre sí y con `test` / `test_shift`.

### Reglas congeladas (`configs/validation_protocol_frozen.json`)

- **Calibración de confianza del monitor (`N1.4`):** exclusivamente `adaptive_dataset/val_cal` (1.200 padres independientes del entrenamiento `e2_dataset` de `SVM256`/`SVM512`).
- **Evaluación de calibración y SNR (`N1.3`, `N1.4`):** `adaptive_dataset/test` (2.400 padres ID) y `test_shift` (2.400 padres cambio de canal), independientes de `e2_dataset` y `val_cal`.
- **Semillas nuevas reservadas antes de abrir evaluaciones finales:** `2026111601` (demos Nivel 1), `2026111602` (banco independiente E1 Nivel 2), `2026111603` (archivos desconocidos E2E N2.2), `2026111604` (test adaptativo final nuevo Nivel 3).
- **Distinción contractual:** la clasificación AMC condicionada a señal presente (E2–E6) se separa explícitamente de la evaluación conjunta del monitor (detección + espectro + SNR + clasificación + confianza) sobre entradas arbitrarias.
