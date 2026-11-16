# Informe Técnico Final — Proyecto 2: Comunicaciones II
## Sistema Integral de Clasificación Automática de Modulaciones (AMC) y Monitor de Señales con Software Definido por Radio

**Asignatura:** Comunicaciones II  
**Fecha:** Noviembre de 2026  
**Autores:** Equipo de Trabajo — Universidad  
**Tecnologías:** Python 3.11, PyTorch (CUDA / RTX 4060), Scikit-Learn, GNU Radio Companion 3.10.9.2, Pytest.

---

## 1. Resumen Ejecutivo

El presente proyecto implementa y valida un sistema completo, reproducible y de extremo a extremo para la generación sintética, degradación por canal inalámbrico, detección, estimación espectral, clasificación automática de modulaciones (AMC) y monitoreo de señales digitales por software.

El sistema cubre de forma rigurosa los requerimientos docentes fijados para la asignatura:
1. **Seis modulaciones digitales normalizadas:** BPSK, QPSK, 8PSK, 16QAM, 64QAM y 4PAM con mapeo Gray y conformación de pulso por filtro Coseno Alzado de Raíz Cuadrada (RRC truncado a `span=10` símbolos, con residual de ISI acotado en AWGN que aumenta en canal dispersivo).
2. **Modelo realista de canal:** Ruido blanco gaussiano aditivo (AWGN), desplazamiento de frecuencia de portadora (CFO), rotación de fase inicial y canal multitrayecto determinista por convolución directa.
3. **Monitor Integrado de Cinco Salidas:** Detección de energía calibrada por referencia de ruido con control de falsa alarma por archivo ($P_{fa} = 1\%$) y estado explícito indeterminado si falta calibración; densidad espectral de potencia (PSD) en Hz o ciclos/muestra distinguiendo potencia total de banda útil; estimación de SNR condicionada a referencia compatible; clasificación en alcance cerrado de 6 clases; y confianza calibrada verificada por hash y agregación.
4. **Evaluación comparativa integral (E1 a E6):** Contraste experimental entre modelo convencional (SVM con kernel RBF sobre características estadísticas y espectrales) y modelo avanzado (CNN 1D profunda entrenada en GPU con PyTorch), evaluados bajo idénticas particiones independientes y sometidos a condiciones de canal no vistas (*shift*).
5. **Interoperabilidad con GNU Radio Companion (GRC):** Flowgraphs reproducibles sin necesidad de hardware RF que permiten la demostración interactiva en tiempo real y el intercambio binario bidireccional en formato estándar `complex64`.
6. **Investigación adaptativa:** Formulación y prueba empírica de una política de selección dinámica de acciones guiada por riesgo y latencia de adquisición/cómputo.

---

## 2. Modelo de Señal y Transmisión

### 2.1 Modulaciones implementadas y constelaciones
Se generaron señales digitales para seis esquemas de modulación garantizando potencia media unitaria $E_s = \mathbb{E}[|s_k|^2] = 1$:
* **BPSK ($M=2$):** $s_k \in \{-1, +1\}$.
* **QPSK ($M=4$):** Mapeo con codificación Gray:
  $$s_k = \frac{(1 - 2b_{0,k}) + j(1 - 2b_{1,k})}{\sqrt{2}}$$
* **8PSK ($M=8$):** Puntos uniformemente distribuidos sobre la circunferencia unitaria con codificación Gray adyacente.
* **16QAM ($M=16$):** Rejilla rectangular $4 \times 4$ en $\{\pm 1, \pm 3\} \times \{\pm 1, \pm 3\}$ normalizada por $\sqrt{10}$.
* **64QAM ($M=64$):** Rejilla rectangular $8 \times 8$ en $\{\pm 1, \pm 3, \pm 5, \pm 7\}^2$ normalizada por $\sqrt{42}$.
* **4PAM ($M=4$):** Constelación unidimensional real en $\{\pm 1, \pm 3\}$ normalizada por $\sqrt{5}$.

### 2.2 Conformación de pulso (RRC)
La conformación de pulso utiliza un filtro RRC con factor de roll-off $\beta = 0.35$, sobremuestreo $sps = 8$ muestras/símbolo y longitud de $span = 10$ símbolos ($81$ coeficientes):
$$p(t) = \frac{\sin(\pi t / T_s (1 - \beta)) + 4\beta (t / T_s) \cos(\pi t / T_s (1 + \beta))}{\pi (t / T_s) (1 - (4\beta t / T_s)^2)}$$
A una tasa de $R_s = 1000\text{ baud}$ y frecuencia de muestreo $f_s = 8000\text{ Hz}$, el ancho de banda ocupado teórico bilateral es:
$$B = R_s(1 + \beta) = 1000(1 + 0.35) = 1350\text{ Hz}$$

---

## 3. Modelo de Canal Inalámbrico

La señal muestreada a la salida del transmisor $x[n]$ atraviesa cuatro etapas de degradación física:
1. **Desplazamiento de Portadora (CFO) y Fase:**
   $$x_{cfo}[n] = x[n] \cdot e^{j(2\pi \frac{\Delta f}{f_s} n + \theta_0)}$$
   donde $\Delta f \in [-100, 100]\text{ Hz}$ en condiciones normales (ID) y $|\Delta f| \in [200, 300]\text{ Hz}$ en condiciones fuera de distribución (*shift*).
2. **Canal Multitrayecto (FIR):**
   $$x_{mp}[n] = \sum_{l=0}^{L-1} h_l \cdot x_{cfo}[n - \tau_l]$$
   con retardos $\tau_l$ en muestras discretas y ganancias complejas deterministas por trama.
3. **Ruido Blanco Gaussiano Aditivo (AWGN):**
   $$y[n] = x_{mp}[n] + w[n], \quad w[n] \sim \mathcal{CN}(0, \sigma_w^2)$$
   donde $\sigma_w^2 = P_s \cdot 10^{-\text{SNR}_{\text{dB}}/10}$, calculada sobre la potencia media de la observación.

---

## 4. Resultados de los Experimentos Obligatorios (E1 a E6)

### Experimento E1: Detección de Energía y Caracterización Espectral
* **Calibración:** Sobre 5000 ventanas independientes de solo ruido blanco stationary ($N=256$, potencia $1.0$), se calculó el umbral empírico $\gamma = 1.1524$ para una tasa nominal de falsas alarmas $P_{fa} = 1.0\%$.
* **Desempeño ($P_D$ vs SNR) en E1 histórico (`results/fase4/pd_agregado.csv`, ID):**
  * $\text{SNR} \ge 0\text{ dB}$: $P_D = 100\%$ (`600/600` por nivel de SNR).
  * $\text{SNR} = -5\text{ dB}$: $P_D = 98.33\%$ (`590/600`).
  * $\text{SNR} = -10\text{ dB}$: $P_D = 20.5\%$ (`123/600` en ID; `122/600 = 20.33%` en `shift`).
  * $\text{SNR} \le -15\text{ dB}$: Límite de detectabilidad en ventanas cortas ($6.0\%$ a $-15\text{ dB}$ y $1.5\%$ a $-20\text{ dB}$ en ID).
* **Espectro (PSD):** La estimación mediante ventana Hann periódica mide el ancho de banda ocupado al 99% de la **potencia total observada** (incluye ruido de piso fuera de banda). En `results/gnuradio/summary.json`, la captura sintética Python a 15 dB (`py_qpsk_15db.c64`, ventana `200:456`) registró `5292.52 Hz` (y `5317.85 Hz` sobre las 4178 muestras en `results/monitor/py_qpsk_15db.json`) debido al piso de ruido AWGN, mientras que `grc_sim_output.c64` (SNR $\approx 34\text{ dB}$) registró `1134.48 Hz` (`1174.23 Hz` en `results/monitor/grc_sim_output.json`). Al restar el piso de ruido estimado con `--noise-ref` (`signal_bandwidth_est` en `results/nivel1_monitor/demos/demo_py_qpsk_15db.json`), la banda neta estimada (`1679.1 Hz`) se aproxima al ancho teórico de $1350\text{ Hz}$ para $R_s = 1000\text{ baud}$ y $\beta = 0.35$.

### Experimento E2: Clasificador Convencional (SVM RBF)
* **Pipeline:** Extracción de 65 características (estadística de orden superior: curtosis, asimetría, momentos; características espectrales: planitud espectral, energía espectral; y características de diferencia y fase).
* **Selección de modelo:** Se entrenó sobre el split de entrenamiento `train` ($N=4800$ por longitud) y se validó en `val`.
* **Resultados en Test:**
  * SVM 256 muestras: **$43.69\%$** de exactitud global (macro F1: 0.435).
  * SVM 512 muestras: **$46.60\%$** de exactitud global (macro F1: 0.464).
  * Ganancia por duplicar la ventana de observación: $+2.91\text{ pp}$.

### Experimentos E3, E4, E5 y E6: Modelo Avanzado (`SmallCNN` Fase 6) vs Convencional (SVM RBF)
* **Arquitectura `SmallCNN` (Fase 6):** Red convolucional de 4 bloques (`Conv1d`, `ReLU`, `AvgPool1d`) seguida de capas densas con `Dropout` (94 726 parámetros, distinta de la arquitectura `MultiDomainAMCNet` evaluada posteriormente en `experiments/exp12_paper_research.py` sobre `data/adaptive_dataset`). Entrenada en GPU RTX 4060 hasta época 61, seleccionando por `val` el checkpoint de la **época 46**.
* **Comparación pareada en Holdout Independiente (`data/fase6_holdout_sellado`, semilla `2026110609`, $n=4800$ por split):**
  | Modelo | Muestras ($N$) | Exactitud ID (`test`) | Exactitud Shift (`test_shift`) | Mediana Latencia CPU (`docs/FASE6_VALIDACION.md`) |
  |---|---|---|---|---|
  | **SVM RBF** | 256 | **$43.27\%$** | **$26.81\%$** | $1.618\text{ ms}$ (p95: $2.529\text{ ms}$) |
  | **SVM RBF** | 512 | **$47.21\%$** | **$29.06\%$** | $1.995\text{ ms}$ (p95: $2.577\text{ ms}$) |
  | **`SmallCNN` 1D** | 256 | $40.75\%$ | $19.02\%$ | **$0.422\text{ ms}$** (p95: $0.692\text{ ms}$) |
* **Nota de contexto con experimentos posteriores:** En el conjunto separado `data/adaptive_dataset` ($n=2400$), la red posterior `MultiDomainAMCNet` alcanzó $51.58\%$ nominal y $30.08\%$ bajo *shift* (frente a $43.75\%$ / $30.33\%$ de SVM256 y $46.04\%$ / $29.54\%$ de SVM512); dichos benchmarks no deben mezclarse en una única comparación.
* **Hallazgo fundamental:**
  1. En `fase6_holdout_sellado`, los clasificadores SVM superan a `SmallCNN` tanto en condiciones normales ($+2.52\text{ pp}$ SVM256; $+6.46\text{ pp}$ SVM512) como bajo canal desplazado ($+7.79\text{ pp}$ SVM256; $+10.04\text{ pp}$ SVM512).
  2. `SmallCNN` es más rápida en inferencia CPU batch=1 ($0.422\text{ ms}$ mediana frente a $1.618\text{ ms}$ de SVM256 en el mismo ensayo común de Fase 6).
  3. **Diagnóstico controlado de canal:** Se observó que el CFO elevado degrada fuertemente a `SmallCNN`, mientras que el multitrayecto severo afecta a ambos enfoques y constituye la principal limitación abierta de generalización OOD.

---

## 5. Fase 8: Demostración en GNU Radio Companion (GRC)

Para cumplir el requisito de demostración en radio definida por software sin hardware RF:
* Se desarrollaron tres flowgraphs en formato YAML GRC 3.10:
  1. `flowgraph_tx_channel.grc`: Demostración visual completa con interfaz Qt GUI. Dispone de deslizadores interactivos en tiempo real para modificar la SNR, el CFO y el multitrayecto, mostrando en tiempo real la constelación I/Q, la densidad espectral de potencia y la forma de onda temporal.
  2. `flowgraph_headless_sim.grc`: Simulación batch que genera 4096 muestras complejas para pipelines automáticos.
  3. `flowgraph_rx_from_file.grc`: Receptor interactivo capaz de cargar cualquier archivo binario `.c64` generado en Python y representarlo en los visualizadores gráficos de GRC.
* **Interoperabilidad:** Se comprobó la compatibilidad bit a bit del formato IEEE 754 float32 intercalado I/Q (`complex64`). El monitor y el clasificador en Python procesaron exitosamente las señales generadas en GRC con un ancho de banda al 99% de $1134.5\text{ Hz}$ y potencia unitaria normalizada.

---

## 6. Fase 7: Monitor Integrado y Procesamiento CLI

Se construyó y corrigió (Nivel 1) la herramienta ejecutable `src.monitor` para evaluar una señal desconocida por línea de comandos:
```powershell
python -m src.monitor --input ruta/a/archivo.c64 --noise-ref ruta/a/ruido.c64 --fs-hz 8000 --output results/reporte.json --plot results/reporte_psd.png
```
* **Soporte multiformato validado:** Carga estricta de `.c64`, `.bin`, `.npy`, `.npz` y `.csv`, rechazando archivos vacíos, binarios truncados, lotes 2D complejos y valores no finitos (código de salida `2`).
* **Entrega de las Cinco Salidas:**
  1. `detection`: Presencia de señal evaluada en todas las ventanas de 256 muestras con control de $P_{FA}$ por archivo (Šidák) e invariancia a ganancia al usar `--noise-ref`; devuelve `indeterminate_uncalibrated` si no existe calibración ni umbral explícito.
  2. `spectrum`: Ancho de banda al 99% de potencia total (con ruido) en `Hz` o `ciclos/muestra` si falta `--fs-hz`, más `signal_bandwidth_est` al restar el piso de ruido estimado con referencia.
  3. `snr`: Reporte condicional fundado bajo referencia compatible (devuelve `null` con motivo explícito sin referencia, ante señal no detectada o resta no positiva).
  4. `classification`: Modulación en alcance cerrado de 6 clases (SVM256/512) identificando `model_sha256` y declarando `OUT_OF_SCOPE_*` ante señales cortas (< 256 muestras) o muestreo incompatible.
  5. `confidence`: Probabilidad empírica calibrada sin recorte artificial cuando coincide exactamente con el calibrador validado (`results/nivel1_monitor/monitor_calibrators.joblib`); en otro caso indica `confidence_calibrated: false`.
* **Salida gráfica:** Genera automáticamente una imagen PNG con la PSD sombreada y el diagrama de constelación.

---

## 7. Conclusiones y Alcance Verificado

1. **Estado del sistema base (Niveles 0, 1 y 2):** Se implementaron, corrigieron y validaron de forma trazable las seis modulaciones, cuatro perturbaciones de canal, el monitor CLI de cinco salidas (`N1.1`–`N1.6`), el banco independiente de detección/SNR (`N2.1`, $9.600$ realizaciones), el desafío ciego extremo a extremo (`N2.2`, $164$ archivos), la evaluación unificada de clasificadores (`N2.3`) y la demostración reproducible con GNU Radio (`N2.4`).
2. **Rigor metodológico y limitaciones conocidas:** Todas las particiones entre `e2_dataset`, `adaptive_dataset`, `fase6_holdout_sellado` y los bancos de validación tienen cero padres compartidos (`results/nivel0_evidencia/`). Persisten como limitaciones documentadas: (a) caída severa de exactitud bajo cambio de canal en todos los clasificadores evaluados; (b) dependencia de una referencia estacionaria de ruido independiente para detección calibrada y SNR; y (c) el estudio adaptativo del artículo (`exp12_paper_research.py`, Nivel 3) requiere completar validación de paquete desplegable, medición integral de latencia y comparaciones estadísticas pareadas antes de sostener afirmaciones de superioridad.

---

## 8. Bibliografía y Referencias Clave

1. **RML22:** B. Bloemendal et al., *"Realistic Dataset Generation for Wireless Modulation Classification,"* IEEE Trans. Wireless Commun., 2023. DOI: `10.1109/TWC.2023.3254490`.
2. **BEACON:** *"Benefit-Aware Early-Exit for Automatic Modulation Classification via Recoverability Prediction,"* IEEE Internet of Things Journal, 2026. DOI: `10.1109/JIOT.2026.3710690`.
3. **Fewer-Sample:** *"Fast Automatic Modulation Recognition with Multimodal Deep Learning and Resized Signal Representation,"* IEEE ICCCS, 2025. DOI: `10.1109/ICCCS65393.2025.11069715`.
4. **Uncertainty Quantification:** *"An Uncertainty Quantification Framework for Deep Learning-Based AMC,"* IEEE Internet of Things Journal, vol. 13, no. 3, 2026. DOI: `10.1109/JIOT.2025.3640108`.
