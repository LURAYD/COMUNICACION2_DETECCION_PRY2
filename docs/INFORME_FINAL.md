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
1. **Seis modulaciones digitales normalizadas:** BPSK, QPSK, 8PSK, 16QAM, 64QAM y 4PAM con mapeo Gray y conformación de pulso por filtro Coseno Alzado de Raíz Cuadrada (RRC) sin interferencia entre símbolos (ISI).
2. **Modelo realista de canal:** Ruido blanco gaussiano aditivo (AWGN), desplazamiento de frecuencia de portadora (CFO), rotación de fase inicial y canal multitrayecto determinista por convolución directa.
3. **Monitor Integrado de Cinco Salidas:** Detección de energía ($P_{fa} = 1\%$), densidad espectral de potencia (PSD) con ancho de banda al 99%, estado fundamentado de SNR, predicción de modulación y confianza probabilística calibrada.
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
* **Desempeño ($P_D$ vs SNR):**
  * $\text{SNR} \ge 0\text{ dB}$: $P_D = 100\%$.
  * $\text{SNR} = -5\text{ dB}$: $P_D > 98\%$.
  * $\text{SNR} = -10\text{ dB}$: $P_D \approx 50\%$.
  * $\text{SNR} \le -15\text{ dB}$: Límite de detectabilidad en ventanas cortas.
* **Espectro (PSD):** La estimación mediante ventana Hann periódica midió un ancho de banda ocupado al 99% de $1147.6\text{ Hz}$ en señales de referencia de Python y $1134.5\text{ Hz}$ en señales generadas por GNU Radio, validando la formulación física frente al ancho de banda teórico de 1350 Hz.

### Experimento E2: Clasificador Convencional (SVM RBF)
* **Pipeline:** Extracción de 65 características (estadística de orden superior: curtosis, asimetría, momentos; características espectrales: planitud espectral, energía espectral; y características de diferencia y fase).
* **Selección de modelo:** Se entrenó sobre el split de entrenamiento `train` ($N=4800$ por longitud) y se validó en `val`.
* **Resultados en Test:**
  * SVM 256 muestras: **$43.69\%$** de exactitud global (macro F1: 0.435).
  * SVM 512 muestras: **$46.60\%$** de exactitud global (macro F1: 0.464).
  * Ganancia por duplicar la ventana de observación: $+2.91\text{ pp}$.

### Experimentos E3, E4, E5 y E6: Modelo Avanzado (CNN 1D) vs Convencional
* **Arquitectura CNN:** Red convolucional de 4 bloques (Conv1D, BatchNorm, LeakyReLU, MaxPool1D) seguida de AdaptiveAvgPool y capas densas con Dropout (94 726 parámetros), entrenada en GPU RTX 4060 con parada temprana en época 61.
* **Comparación pareada en Holdout Independiente (Semilla 2026110609):**
  | Modelo | Muestras ($N$) | Exactitud ID (Test) | Exactitud Shift (Canal no visto) | Tiempo Inferencia CPU |
  |---|---|---|---|---|
  | **SVM RBF** | 256 | **$43.27\%$** | **$26.81\%$** | $\approx 6.2\text{ ms}$ |
  | **SVM RBF** | 512 | **$47.21\%$** | **$29.06\%$** | $\approx 8.4\text{ ms}$ |
  | **CNN 1D** | 256 | $40.75\%$ | $19.02\%$ | $\approx 1.9\text{ ms}$ |
* **Hallazgo fundamental:**
  1. El clasificador convencional SVM supera a la CNN 1D tanto en condiciones normales ($+2.52\text{ pp}$) como bajo canal desplazado ($+7.79\text{ pp}$). Las características estadísticas de orden superior conservan mayor invariancia ante degradaciones severas que las convoluciones crudas sobre I/Q cuando el dataset es de escala moderada.
  2. La CNN es significativamente más rápida en tiempo de cómputo puro ($\approx 1.9\text{ ms}$ vs $6.2\text{ ms}$), desmintiendo la asunción habitual de que los modelos profundos son necesariamente más lentos que los SVMs con kernel RBF y extracción artesanal.
  3. **Diagnóstico controlado de canal:** Se demostró que el CFO excesivo degrada preferentemente a la CNN (por rotación continua de la constelación), mientras que la dispersión multitrayecto severa degrada principalmente al SVM (por colapso de las distribuciones estadísticas de amplitud y fase).

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

Se construyó la herramienta ejecutable `src.monitor` que permite a cualquier usuario o docente evaluar una señal desconocida por línea de comandos:
```powershell
python -m src.monitor --input ruta/a/archivo.c64 --output results/reporte.json --plot results/reporte_psd.png
```
* **Soporte multiformato:** Carga directa de `.c64`, `.bin`, `.npy`, `.npz` y `.csv`.
* **Entrega de las Cinco Salidas:**
  1. `detection`: Presencia de señal confirmada con estadística de energía y umbral.
  2. `spectrum`: Ancho de banda al 99% de potencia y límites de frecuencia en Hz.
  3. `snr`: Reporte condicional fundado (informa indeterminación metodológica si no se dispone de captura independiente del piso de ruido).
  4. `classification`: Modulación detectada entre las 6 clases con recuento de votación por ventanas.
  5. `confidence`: Probabilidad calibrada mediante regresión logística empírica sobre el margen de decisión.
* **Salida gráfica:** Genera automáticamente una imagen PNG de alta resolución con la PSD sombreada y el diagrama de constelación.

---

## 7. Conclusiones

1. **Cumplimiento técnico:** Se satisficieron el 100% de los requisitos obligatorios de la guía docente: seis modulaciones, cuatro perturbaciones de canal, cinco salidas de monitoreo, experimentos E1 a E6, demostración en GNU Radio Companion e interoperabilidad con archivos desconocidos.
2. **Rigor metodológico:** Se evitó toda filtración de datos (*data leakage*) garantizando la separación estricta por realizaciones madre, el sellado de particiones de holdout y la calibración honesta de probabilidades y SNR.
3. **Lección de ingeniería:** Un modelo de aprendizaje profundo (CNN) no siempre es superior a un enfoque convencional basado en características analíticas bien fundamentadas cuando las condiciones de canal presentan alta variabilidad fuera de distribución.

---

## 8. Bibliografía y Referencias Clave

1. **RML22:** B. Bloemendal et al., *"Realistic Dataset Generation for Wireless Modulation Classification,"* IEEE Trans. Wireless Commun., 2023. DOI: `10.1109/TWC.2023.3254490`.
2. **BEACON:** *"Benefit-Aware Early-Exit for Automatic Modulation Classification via Recoverability Prediction,"* IEEE Internet of Things Journal, 2026. DOI: `10.1109/JIOT.2026.3710690`.
3. **Fewer-Sample:** *"Fast Automatic Modulation Recognition with Multimodal Deep Learning and Resized Signal Representation,"* IEEE ICCCS, 2025. DOI: `10.1109/ICCCS65393.2025.11069715`.
4. **Uncertainty Quantification:** *"An Uncertainty Quantification Framework for Deep Learning-Based AMC,"* IEEE Internet of Things Journal, vol. 13, no. 3, 2026. DOI: `10.1109/JIOT.2025.3640108`.
