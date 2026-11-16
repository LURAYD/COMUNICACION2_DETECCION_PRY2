# Guion de Sustentación Oral y Diapositivas — Proyecto 2 de Comunicaciones II

**Título del Proyecto:** Sistema Integral de Clasificación Automática de Modulaciones (AMC) y Monitor de Señales con Software Definido por Radio  
**Duración sugerida:** 15 – 20 minutos + 5 minutos de preguntas.

---

## Diapositiva 1: Portada y Presentación
* **Título:** Sistema de Detección, Caracterización y Clasificación Automática de Modulaciones (AMC).
* **Asignatura:** Comunicaciones II — Proyecto 2.
* **Autores:** [Nombre del estudiante / equipo].
* **Mensaje clave de apertura:** *"Desarrollamos una plataforma completa y reproducible por software que abarca desde la síntesis física de señales moduladas y el modelado de canales degradados, hasta la detección de energía, estimación espectral, clasificación comparada convencional vs. avanzada en GPU, y demostración en GNU Radio Companion."*

---

## Diapositiva 2: Planteamiento del Problema y Objetivos
* **Reto en comunicaciones inalámbricas:** El receptor desconoce a priori la presencia de la señal, su nivel de potencia, su ancho de banda y el esquema de modulación empleado bajo canales ruidosos y distorsionantes.
* **Objetivos del proyecto:**
  1. Síntesis estricta de 6 modulaciones con filtro conformador RRC y normalización de energía.
  2. Implementación de perturbaciones realistas: AWGN, CFO, fase y canal multitrayecto determinista.
  3. Desarrollo de un monitor unificado con **cinco salidas obligatorias**.
  4. Evaluación experimental rigurosa (E1 a E6): Convencional (SVM) vs Avanzado (CNN 1D en GPU).
  5. Interoperabilidad con GNU Radio Companion sin hardware RF.

---

## Diapositiva 3: Cadena de Transmisión y Modulaciones
* **Esquemas implementados (6 clases):**
  * BPSK, QPSK (Gray), 8PSK (Gray), 16QAM, 64QAM y 4PAM.
  * Todas con potencia media unitaria: $E_s = \mathbb{E}[|s_k|^2] = 1$.
* **Conformación de pulso:**
  * Filtro RRC ($sps = 8$, $\beta = 0.35$, $span = 10$ símbolos, 81 coeficientes).
  * Ancho de banda bilateral teórico: $B = R_s(1 + \beta) = 1000 \times 1.35 = 1350\text{ Hz}$.
* **Evidencia a mostrar:** Gráficas de constelaciones y respuesta al impulso en `results/fase1/`.

---

## Diapositiva 4: Modelo de Canal Inalámbrico
* **Perturbaciones aplicadas:**
  1. **Oscilador local desfasado (CFO + Fase):** Rotación continua en el tiempo $e^{j(2\pi \frac{\Delta f}{f_s} n + \theta_0)}$.
  2. **Efecto multitrayecto (Multipath FIR):** Dispersión temporal por ecos con retardos discretos.
  3. **Ruido térmico AWGN:** Calculado sobre la potencia real de la trama observada.
* **División metodológica:**
  * Condiciones nominales (In-Distribution - ID): $|\Delta f| \le 100\text{ Hz}$.
  * Condiciones severas (Out-of-Distribution - Shift): $|\Delta f| \in [200, 300]\text{ Hz}$ y nuevos retardos.

---

## Diapositiva 5: Experimento E1 — Detección y Caracterización Espectral
* **Detección de energía:**
  * Umbral calibrado por referencia de ruido para $P_{fa}$ nominal $\le 1.0\%$; verificación en banco controlado (`N2.1`, $n=5000$): $P_{fa} = 0.96\%$ por ventana (`IC95% Wilson [0.72\%, 1.27\%]`) y $0.88\%$ por archivo $2\times 256$ con corrección de Šidák (`[0.66\%, 1.18\%]`), invariante a escalado de ganancia (`6427/6427` en factores $0.5\times, 1\times, 2\times$).
  * Resultados: $P_D = 100\%$ en celdas $\text{SNR} \ge 0\text{ dB}$ (`mín. IC95% inferior = 96.30%`), $P_D \approx 20.5\%$ a $-10\text{ dB}$, y colapso explícito de umbral estático ante deriva térmica $\times 2$ sin recalibración ($P_{fa}=100\%$).
* **Ancho de banda ocupado al 99% vs. señal útil:**
  * Se separa explícitamente el **ancho de banda de potencia total al 99%** (`total_power_99_includes_noise`: `5292–5668 Hz` cuando el ruido ocupa toda la banda Nyquist de `8000 Hz`) del **ancho de banda útil estimado tras sustracción de piso de ruido** (`signal_bandwidth_est`: `~1679 Hz` con referencia de ruido vs. `1350 Hz` teórico RRC y `1159 Hz` en captura GNU Radio de alta SNR).
* **Estimación de SNR:**
  * Enfoque riguroso: una estimación no-oracular exige una referencia de solo ruido en la misma banda y ganancia (`MAE = 0.3388 dB` y cobertura `100%` en $\text{SNR} \ge 0\text{ dB}$ en `N2.1`); sin referencia se reporta `no_identificable_sin_supuestos` y no se inventan cifras.

---

## Diapositiva 6: Experimento E2 a E6 — Clasificación Convencional vs. Avanzada (Sin Mezclar Benchmarks)
* **Modelos enfrentados (alcance cerrado de 6 clases):**
  * **Convencional (`a0` / `a1`):** SVM RBF en CPU sobre 65 características analíticas (HOC + espectrales) con ventanas $N \in \{256, 512\}$.
  * **Avanzado Fase 6 (`SmallCNN`, época 46):** CNN 1D de `94k` parámetros en GPU RTX 4060 (sin ajuste de fase vs. aumentación de fase).
  * **Experto Multi-Dominio (`MultiDomainAMCNet`, `a2`):** arquitectura multi-rama de `8.10M` parámetros con banco RRC adaptado y autoatención polifásica (`GroupNorm`/`LayerNorm`, sin fuga entre muestras).
* **Evaluación unificada en `fase6_holdout_sellado` ($n=4800$ por split, `N2.3`):**
  * **SVM 256 / SVM 512:** `43.27% / 47.21%` (ID) $\rightarrow$ `26.81% / 29.06%` (Shift)
  * **SmallCNN Orig / Phase-Aug:** `40.75% / 41.54%` (ID) $\rightarrow$ `19.02% / 19.48%` (Shift)
  * **MultiDomainAMCNet (`a2`):** `50.19%` (ID) $\rightarrow$ `27.77%` (Shift)
* **Evaluación en `adaptive_dataset` ($n=2400$ por split, `N2.3`):**
  * **SVM 256 / SVM 512:** `43.75% / 46.04%` (ID) $\rightarrow$ `30.33% / 29.54%` (Shift)
  * **SmallCNN Orig / Phase-Aug:** `42.46% / 41.17%` (ID) $\rightarrow$ `18.67% / 19.00%` (Shift)
  * **MultiDomainAMCNet (`a2`):** `51.58%` (ID, Macro-F1 `55.36%`) $\rightarrow$ `30.08%` (Shift)
* **Conclusión crítica para el docente:**
  * La `SmallCNN` 1D básica queda por debajo del SVM convencional en ID y colapsa ante CFO/multitrayecto OOD (`~19%`); `MultiDomainAMCNet` supera al SVM en ID (`+5.54 pp` sobre `SVM512` en `adaptive_dataset`), aunque bajo desplazamiento extremo (`test_shift`) todos los modelos caen a `~29.5–30.3%`.

---

## Diapositiva 7: Diagnóstico de Canal (`exp09` y `N5.1`)
* **¿Por qué fallan los modelos ante cambio de canal?** (`720` realizaciones pareadas):
  * **Efecto del CFO OOD ($200\text{--}300\text{ Hz}$):** Colapsa a `SmallCNN` (`-25.56 pp`), mientras que `MultiDomainAMCNet` (`a2`) solo cae `-3.61 pp` gracias a su seguidor espectral de CFO.
  * **Efecto del Multipath OOD (ecos $9\text{ y }13$ muestras):** Colapsa tanto a los SVM (`-22.92` / `-25.42 pp`) como a `MultiDomainAMCNet` (`-30.69 pp`), al superar retardos $> \text{sps}=8$ sin ecualizador ciego.
  * **Confusión `16QAM` $\leftrightarrow$ `64QAM` (`SNR >= 10 dB`, `N5.1`):** `MultiDomainAMCNet` sesga `64QAM -> 16QAM` en `N=256` incluso en AWGN limpio (`29.2%` en `64QAM`), mientras que el multipath OOD invierte al `SVM512` colapsando `16QAM` de `66.7%` a `10.8%` (`105/120` predichas como `64QAM`).

---

## Diapositiva 8: Fase 8 — Demostración en GNU Radio Companion (GRC)
* **Integración sin hardware:**
  * Flowgraphs en Qt GUI (`flowgraph_tx_channel.grc`, `flowgraph_rx_from_file.grc`) y simulación headless compilable con `grcc` (`flowgraph_headless_sim.grc`, `4096` muestras).
  * Interoperabilidad binaria `complex64`: `6/6` capturas GNU Radio QPSK detectadas por el monitor en Python (`SNR ~34 dB`, `BW ~1159 Hz`).
  * **Diagnóstico `N5.1` de `QPSK -> BPSK` en GNU Radio:** `SVM256` clasificó erróneamente como `BPSK` las tramas GRC por tener `CFO = 0 Hz` exacto (`c40_abs = 0.8231`, `phase_resultant_4 = 0.4982`); al inyectar `35 Hz` de CFO sin cambiar símbolos ni ruido, `SVM256` sube de `4/16` (`25%`) a `15/16` (`93.75%`) aciertos `QPSK`, y `MultiDomainAMCNet` alcanza `15/15` en régimen permanente ([n5_1_diagnostic_summary.png](file:///C:/PROYECTO_COM2_2026/COM2_Proyecto2/results/nivel5_diagnostico/n5_1_diagnostic_summary.png)).
* **Mostrar en vivo o con captura:** Gráfica comparativa [gnuradio_constellation_comparison.png](file:///C:/PROYECTO_COM2_2026/COM2_Proyecto2/results/gnuradio/gnuradio_constellation_comparison.png).

---

## Diapositiva 9: Fase 7 — Monitor CLI para Archivos Desconocidos
* **Herramienta de entrega para el docente:**
  ```powershell
  python -m src.monitor --input data/unknown/archivo.c64
  ```
* **Las Cinco Salidas en un solo reporte:**
  1. Presencia de señal (Detección de energía).
  2. Espectro PSD y Ancho de Banda al 99%.
  3. Estado de SNR formal.
  4. Modulación clasificada (con recuento de votación por ventanas).
  5. Confianza calibrada mediante regresión logística empírica.
* **Salidas generadas:** Archivo JSON estructurado y figura gráfica PNG de doble panel.

---

## Diapositiva 10: Conclusiones y Preguntas
* **Logros alcanzados y límites declarados:**
  * Entrega completa de los cinco bloques obligatorios, experimentos E1–E6, monitor CLI de cinco salidas, interoperabilidad con GNU Radio y paquete adaptativo desplegable, preservando la honestidad científica ante resultados mixtos o negativos (clasificación GRC QPSK por SVM256, diferencia inconclusa LBAP vs. proxy 2D en `test`, y entrenamiento mixto ADC a 4 bits).
  * Suite verificada de 278 pruebas automáticas aprobadas sin fallos ni omisiones (`pytest`).
  * Código modular, manifiestos SHA-256 e informe/artículo sincronizados.
* **Agradecimientos y ronda de preguntas con el docente.**
