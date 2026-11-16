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
  * Umbral calibrado sobre 5000 ventanas de solo ruido: $\gamma = 1.1524$ para $P_{fa} = 1.0\%$.
  * Resultados: Detección perfecta ($P_D = 100\%$) para $\text{SNR} \ge 0\text{ dB}$; caída progresiva por debajo de $-10\text{ dB}$.
* **Ancho de banda ocupado al 99%:**
  * Medido en señales sintetizadas: $\approx 1140 - 1170\text{ Hz}$, demostrando coherencia física con el pulso RRC.
* **Estimación de SNR:**
  * Enfoque riguroso: Se documenta que una estimación no-oracular exige una referencia de piso de ruido en la misma banda y ganancia; no se inventan valores ficticios.

---

## Diapositiva 6: Experimento E2 a E6 — Comparación Convencional (SVM) vs Avanzado (CNN)
* **Modelos enfrentados:**
  * **Convencional:** SVM RBF en CPU con 65 características analíticas (estadística de orden superior y espectrales).
  * **Avanzado:** CNN 1D profunda (94k parámetros) entrenada en GPU RTX 4060 con PyTorch.
* **Tabla de resultados pareados en Holdout Ciego (Semilla 2026110609):**
  * **SVM 256:** $43.27\%$ (ID) $\rightarrow$ $26.81\%$ (Shift)
  * **SVM 512:** $47.21\%$ (ID) $\rightarrow$ $29.06\%$ (Shift)
  * **CNN 256:** $40.75\%$ (ID) $\rightarrow$ $19.02\%$ (Shift)
* **Conclusión crítica para el docente:**
  * El SVM convencional superó a la CNN en precisión y robustez ante canal no visto.
  * La CNN demostró ser más rápida en inferencia pura ($\approx 1.9\text{ ms}$ vs $6.2\text{ ms}$).

---

## Diapositiva 7: Diagnóstico de Canal (CFO vs Multipath)
* **¿Por qué fallan los modelos?**
  * **Efecto del CFO:** Degrada agresivamente a la CNN 1D (la rotación rápida rompe los patrones convolucionales en el plano complejo).
  * **Efecto del Multipath:** Degrada principalmente al SVM (la interferencia intersímbolo colapsa las distribuciones de curtosis y momentos).
* **Existe complementariedad:** El diagnóstico demostró que un clasificador híbrido o adaptativo tiene fundamento físico.

---

## Diapositiva 8: Fase 8 — Demostración en GNU Radio Companion (GRC)
* **Integración sin hardware:**
  * Flowgraph interactivo en Qt GUI (`flowgraph_tx_channel.grc`): permite en vivo modificar SNR, CFO y multitrayecto mediante deslizadores, mostrando constelación, FFT y osciloscopio.
  * Interoperabilidad binaria `complex64`: las muestras sintetizadas en GNU Radio fueron analizadas y clasificadas con éxito por el monitor en Python.
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
* **Logros alcanzados:**
  * Cumplimiento del 100% de la guía académica y experimental.
  * Suite de 221 pruebas automáticas pasando en menos de 30 segundos (`pytest`).
  * Código limpio, modular y completamente documentado.
* **Agradecimientos y ronda de preguntas con el docente.**
