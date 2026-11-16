# Guía maestra para Codex — Proyecto 2 de Comunicaciones II

**Versión:** 1.0 — 03/11/2026  
**Idioma del proyecto y la documentación:** español.  
**Implementación principal:** Python, enteramente por software; demostración complementaria en GNU Radio Companion (GRC).  
**Entrega indicada por la guía docente:** semana del 1 de diciembre de 2026.  
**Finalidad de este documento:** servir como instrucciones reproducibles para un asistente de programación (**Codex**) y, al mismo tiempo, como guía de aprendizaje para el estudiante.

> **REGLA PRINCIPAL PARA CODEX:** no desarrolles todo el proyecto de una vez ni sustituyas la comprensión del estudiante. Primero inspecciona el repositorio y las referencias; después avanza por fases pequeñas, ejecutando pruebas y explicando cada decisión. Prioriza siempre los requisitos obligatorios del profesor. La investigación adaptativa y la cuantización ADC son extensiones controladas, no excusas para dejar incompleto el curso.

---

## 0. Instrucciones inmediatas para Codex

Cuando recibas este archivo:

1. **Inventaría** los archivos existentes, su estructura y el código previo. No sobreescribas archivos funcionales sin mostrar antes las diferencias.
2. **Localiza y lee el documento oficial** `references/Guia_proyectos_COMII 2026-3.pdf`, **páginas 4–7, Proyecto 2**. **No confundas** el Proyecto 2 con el Proyecto 1 de las páginas 1–3.
3. **Localiza y revisa los PDF académicos** del apartado 13. Los artículos físicos deben estar disponibles en `references/`; si no están, indícalo y no finjas haberlos leído. Crea la matriz de literatura con página o sección verificable.
4. Ejecuta las pruebas actuales, si existen, y crea `docs/AUDITORIA_INICIAL.md` con: código encontrado, funciones implementadas, errores, dependencias faltantes, pruebas reales ejecutadas y resultados.
5. Propón el siguiente cambio **mínimo y comprobable**. Como primera intervención, integra/valida moduladores, RRC y canales; no comiences por entrenar una red neuronal compleja.
6. Al terminar cada fase, entrega: archivos cambiados, pruebas ejecutadas, resultados reales, explicación física, decisiones pendientes del estudiante y siguiente paso.
7. Si un dato o fórmula no está en la documentación, indícalo expresamente como **propuesta de ingeniería** o **conocimiento externo**, no como cita del libro.

**Forma de trabajo:** implementa infraestructura repetitiva, pruebas, generación reproducible de datos, automatización experimental, gráficas y refactorizaciones. Reserva para el estudiante la interpretación de los resultados, la selección de hipótesis y modelos importantes, las justificaciones científicas y la aprobación de cada fase. Explica el código nuevo con claridad universitaria.

### Mensaje inicial que puedes dar a Codex

> Lee `GUIA_MAESTRA_COMII_PARA_CODEX.md` y los documentos en `references/`. Empieza **solo con la Fase 0 (auditoría) y la Fase 1 (validación del generador existente)**. No entrenes modelos todavía ni implementes la política adaptativa. Primero crea el inventario, una matriz inicial de papers, tests reproducibles y un reporte de problemas encontrados. Después explícame las decisiones y el código para aprobar el siguiente paso.

---

## 1. Fuente de autoridad y alcance

**Fuente normativa principal:** `Guia_proyectos_COMII 2026-3.pdf`, **Proyecto 2, pp. 4–7**: *Sistema inteligente de monitoreo y clasificación de señales digitales en un entorno de espectro virtual*.

**Referencias teóricas proporcionadas:**

- `Fundamentos de Ingeniería de Comunicación Carlos A. Medina Parte 2, capitulos del 9 al 15.pdf`, especialmente **capítulo 9**, sobre transmisión de datos en banda base, PAM, conformación de pulsos e interferencia intersimbólica. Es un PDF escaneado: verificar visualmente la numeración impresa y no inventar páginas ni fórmulas si la extracción falla.
- `Formulas_Comunicaciones 2 (5).pdf`, formulario complementario de modulaciones, probabilidad de error y variables aleatorias. Citar la página y el supuesto de cada fórmula que se use.
- `Investigación sobre Clasificación de Modulaciones.docx`, notas del proyecto: **documento de trabajo**, no fuente con igual autoridad que el enunciado ni prueba de originalidad.
- Artículos IEEE enumerados y priorizados en el apartado 13.

**Jerarquía:** (1) guía docente para obligaciones; (2) PDFs/libro para teoría y antecedentes; (3) experimentos verificables; (4) decisiones de diseño debidamente declaradas. No forzar una arquitectura porque aparezca en un paper: debe corresponder al alcance, datos y recursos del proyecto.

### 1.1 Cinco salidas obligatorias

El monitor recibe un segmento o archivo complejo `x[n] = I[n] + jQ[n]` y entrega:

1. **Detección de señal**: sí/no, preferiblemente con puntuación de detección.
2. **Contenido espectral**: PSD/FFT y mediciones asociadas; archivo/gráfica exportable.
3. **Estimación de SNR**: inferida desde la observación, sin utilizar la SNR real del simulador.
4. **Modulación estimada**: entre las clases entrenadas.
5. **Confianza de la clasificación**: probabilidad/puntuación con calibración y límites explicados.

**Condiciones especiales:** para una observación clasificada como «sin señal», no inventar una modulación ni una SNR numérica significativa: devolver `N/A` o un estado de estimación no válida. Si los metadatos necesarios (por ejemplo `fs`) faltan, señalizar esa limitación; no inventar valores para los archivos secretos del profesor.

### 1.2 Seis modulaciones

Cinco exigidas: **BPSK, QPSK, 8PSK, 16-QAM, 64-QAM**. La sexta elegida por el grupo: **4-PAM**. Esta elección debe quedar documentada.

### 1.3 Variaciones mínimas de la generación

- Diferentes valores de SNR, fases iniciales, secuencias de bits y amplitudes.
- Al menos **dos** de: CFO, phase offset, error de temporización, multipath, Rayleigh. **Nuestra selección principal: CFO + multipath**, más fase inicial. El ruido AWGN es obligatorio para el barrido de SNR, pero **no cuenta** como una de esas dos perturbaciones de la lista.
- Entrenamiento/prueba con **realizaciones independientes**. Prohibido dividir aleatoriamente ventanas, versiones cuantizadas o derivados de una misma realización después de generarlos.

### 1.4 Cinco bloques mínimos

Generación/adquisición virtual; detección; preprocesamiento; clasificador convencional; clasificador avanzado. La guía **no exige CNN 1D específicamente**; puede ser MLP, CNN u otro método razonado.

### 1.5 Seis experimentos obligatorios

| Nº | Experimento | Evidencia mínima |
|---|---|---|
| E1 | Detección | `P_D` o tasa de detección vs SNR; indicar umbral y `P_FA` medido |
| E2 | Clasificador convencional | Métricas sobre test independiente, arquitectura/características |
| E3 | Clasificador avanzado | Métricas sobre el **mismo** test, costo y configuración |
| E4 | Comparación | Accuracy, precision, recall, F1, matrices de confusión, tiempo de ejecución |
| E5 | Robustez | Curvas de exactitud vs SNR para ambos modelos |
| E6 | Generalización | Rendimiento con condiciones de canal distintas a entrenamiento |

### 1.6 Otros requisitos

- GRC sin hardware: generar/leer señales en un flowgraph `.grc` y visualizar tiempo/espectro/constelación o exportar a archivo; demostrar compatibilidad con Python.
- Desafío de archivos I/Q desconocidos del profesor con tabla de detección, SNR, modulación y confianza.
- Nueve entregables: dataset **o** procedimiento reproducible, código fuente, modelos entrenados, resultados, matrices de confusión, curvas vs SNR, informe, presentación y demostración.
- Evaluación de la guía (p. 7): diseño 15%, teoría 10%, simulación/validación 20%, implementación 10%, resultados/análisis 20%, informe 15%, presentación 10%.

---

## 2. Nuestra investigación: una sola pregunta, no varios proyectos

### 2.1 Tema de trabajo propuesto

**Risk-Aware Joint Observation and Computation Allocation for Automatic Modulation Classification**  
*Asignación conjunta de tiempo de observación y procesamiento guiada por riesgo para clasificación automática de modulaciones.*

**Pregunta experimental (hipótesis, no resultado demostrado):** ¿puede un monitor tomar decisiones sobre cuándo finalizar, adquirir más muestras o utilizar un clasificador avanzado, manteniendo buena exactitud y confianza con menor costo promedio, incluso bajo desajustes de canal y degradaciones de adquisición?

**NO afirmar que el tema sea novedoso o publicable** hasta completar comparación rigurosa con la literatura, principalmente BEACON, Fewer-Sample, early-exit y uncertainty quantification. El proyecto del curso debe funcionar aunque la investigación adaptativa no supere sus baselines.

### 2.2 Tres acciones planteadas

Siempre observar primero un prefijo de **256 muestras I/Q**:

- **a0 — Finalizar:** usar clasificador convencional con 256 muestras.
- **a1 — Observar más:** ampliar hasta **512 muestras** y ejecutar clasificador convencional compatible con esta longitud.
- **a2 — Procesar más:** conservar las **256 muestras** y utilizar el modelo avanzado.

La política solamente puede utilizar información disponible **en las primeras 256 muestras**: características, puntuación de detección, SNR estimada *a ciegas*, confianza calibrada y metadatos realmente conocidos. Nunca etiquetas reales, SNR real, canal real ni resultados que todavía no se hayan calculado.

La observación de 512 muestras es un **prefijo extendido de la misma realización**, no una señal independiente. Las decisiones tienen costo acumulado; medir qué cálculos se reutilizan y cuáles se vuelven a ejecutar.

### 2.3 ADC virtual como ensayo secundario de robustez

Conservar la cuantización de muestras I/Q como una **perturbación opcional del receptor**, no como contribución separada ni cuantización de pesos.

- Experimento A: entrenamiento de referencia, sin cuantización adicional simulada.
- Experimento B: entrenamiento con cuantización, por ejemplo resoluciones 6 y 8 bits.
- Experimento C: entrenamiento mixto: versiones de referencia y cuantizadas.
- Evaluar en resoluciones vistas y **no vistas** (por ejemplo 5 bits, si es viable), sin prometer mejora.
- Definir un rango fijo `full_scale`, paso de cuantización, saturación y métricas de recorte; **no ajustar el rango usando etiquetas o estadísticas del test**. Registrar también ganancia de entrada. Aplicar el cuantizador después de definir el canal y antes de que el monitor reciba la señal.
- **Distinguir** este ADC uniforme simple de los problemas de *time-interleaved ADC mismatch* del artículo de Gros et al.; no son fenómenos equivalentes.
- Todas las versiones de una misma forma de onda deben permanecer en el mismo split y compartir un `parent_waveform_id`.

**Prioridad:** si no hay tiempo, mantener el cuantizador funcional y una comparación básica en el anexo; jamás sacrificar E1–E6 ni el monitor final.

---

## 3. Estado conocido del trabajo y auditoría necesaria

En nuestra conversación previa se han redactado **ejemplos de referencia** de:

- `bpsk_modulation(bits)` (el estudiante mostró una implementación propia validando bits; convención `0→+1`, `1→−1`).
- `qpsk_modulation(bits)` (mapeo Gray propuesto, pero **no hay código del estudiante verificado**).
- `psk8_modulation(bits)` con 3 bits/símbolo, ocho puntos, energía unitaria.
- `qam16_modulation(bits)` con niveles `+3,+1,−3,−1`, normalización `√10`.
- `qam64_modulation(bits)` con niveles binarios `[7,5,1,3,−7,−5,−1,−3]`, normalización `√42`.
- `pam4_modulation(bits)` con `[3,1,−3,−1]`, normalización `√5`.
- `rrc_filter(beta,sps,span)` y `pulse_shape(symbols,...)` usando inserción de ceros y convolución completa.
- `add_awgn(iq,snr_db,rng)`; `apply_carrier_impairments(iq,fs_hz,phase_rad,cfo_hz)`; `apply_multipath(iq,taps)`.

**Importante:** estos códigos fueron compartidos como propuestas de aprendizaje, **no se debe asumir que el repositorio los contiene ni que sus pruebas han sido ejecutadas de forma integrada**. Codex debe descubrir el estado verdadero. Detectar nombres e imports inconsistentes (`from signals` vs `from src.signals`), errores, comentarios contradictorios, malas unidades y supuestos.

### Revisión matemática obligatoria de la integración

- Energía **promedio de la constelación equiprobable** ≈ 1 para BPSK/QPSK/8PSK/16QAM/64QAM/4PAM; **una trama aleatoria finita de QAM/PAM no tiene por qué dar exactamente 1**.
- Bits por símbolo: 1/2/3/4/6/2; validar forma de entrada y bits inválidos.
- Número de muestras de `np.convolve(..., mode='full')` y compensación de retardo.
- RRC: casos límite `t=0` y `|t|=1/(4β)`, energía discreta, simetría, estabilidad y respuesta de filtro adaptado; verificar la fórmula contra el libro o fuente técnica antes de adoptarla.
- `fs = sps × Rs` con unidades físicas explícitas. No confundir `sps` con símbolos por segundo.
- SNR del simulador basada en potencia de **muestras recibidas** antes de AWGN; para ruido complejo la varianza de cada componente es `P_noise/2`.
- CFO: `exp(j(phase0 + 2π Δf n/fs))` conserva potencia; muestras posteriores deben mantener continuidad de fase, no reiniciarse si se dividen ventanas.
- Multipath: retardo en **muestras**, no en símbolos; `mode='full'` agrega transitorios. No llamar físicamente «reflejos de habitación» a retardos de cientos de microsegundos sin justificar geometría y frecuencia de muestreo.
- ADC: si se implementa, debe cuantizar I y Q con una convención definida, manejar clipping, no introducir información futura ni ajustar `full_scale` a cada trama por defecto.
- **No alterar silenciosamente** el orden o definiciones de las seis modulaciones; el mapping debe estar escrito y versionado.

---

## 4. Estructura del repositorio sugerida

Respetar el código existente si ya existe una estructura razonable. Si el proyecto está empezando, usar:

```text
COM2_Proyecto2/
├── GUIA_MAESTRA_COMII_PARA_CODEX.md
├── README.md
├── pyproject.toml                    # o requirements.txt; no duplicar sin razón
├── configs/
│   ├── base.yaml
│   ├── in_distribution.yaml
│   └── channel_shift.yaml
├── references/                       # PDFs y DOCX proporcionados
├── docs/
│   ├── AUDITORIA_INICIAL.md
│   ├── MATRIZ_LITERATURA.md
│   ├── DECISIONES.md
│   ├── MODELO_DE_SENAL.md
│   ├── PROTOCOLO_EXPERIMENTAL.md
│   └── LIMITACIONES.md
├── src/
│   ├── __init__.py
│   ├── signals.py                     # seis mapeadores
│   ├── waveform.py                    # sps, RRC, filtrado
│   ├── channels.py                    # AWGN, CFO, fase, multipath
│   ├── adc.py                         # opcional, cuantizador de IQ
│   ├── dataset.py                     # creación, split y manifiestos
│   ├── detection.py                   # detector y P_FA
│   ├── spectrum.py                    # FFT, Welch PSD, métricas
│   ├── snr_estimation.py              # estimación sin oracle
│   ├── features.py                    # ingeniería de características
│   ├── conventional.py                # clasificador clásico
│   ├── advanced.py                    # modelo ML/DL avanzado
│   ├── calibration.py                 # calibración y fiabilidad
│   ├── adaptive.py                    # opcional: 3 acciones
│   └── monitor.py                     # API y CLI final
├── tests/
│   ├── test_signals.py
│   ├── test_waveform.py
│   ├── test_channels.py
│   ├── test_adc.py
│   ├── test_dataset.py
│   ├── test_detector.py
│   ├── test_snr.py
│   ├── test_monitor.py
│   └── test_adaptive.py
├── experiments/
│   ├── exp01_detection.py
│   ├── exp02_conventional.py
│   ├── exp03_advanced.py
│   ├── exp04_comparison.py
│   ├── exp05_snr_robustness.py
│   ├── exp06_channel_generalization.py
│   ├── exp07_adaptive_optional.py
│   └── exp08_adc_optional.py
├── grc/                              # flowgraph(s) .grc
├── models/                           # modelos guardados, no datos personales
├── data/                             # preferir manifiestos; datos grandes ignorados por Git
├── results/                          # csv, png, json, tablas
└── reports/                           # informe y material de presentación
```

Agregar `.gitignore` para `.venv/`, cachés, datos y pesos grandes; **no ignorar configuraciones, código ni resultados resumidos**. Anotar sistema operativo, versiones, CPU/GPU y dependencias. Evitar requerir GPU para el mínimo obligatorio.

### Contrato sencillo de funciones

```python
# Moduladores: bits -> símbolos complejos
symbols = qam16_modulation(bits)

# Forma de onda: símbolos -> muestras complejas + filtro
clean_iq, rrc_taps = pulse_shape(symbols, sps=8, beta=0.35, span=10)

# Canal: IQ -> IQ
faded_iq = apply_multipath(clean_iq, channel_taps)
rotated_iq = apply_carrier_impairments(faded_iq, fs_hz=8000,
                                      phase_rad=phase0, cfo_hz=cfo)
received_iq, oracle_noise = add_awgn(rotated_iq, snr_db=snr_db, rng=rng)

# Monitor: únicamente recibe datos observables y metadatos conocidos
result = monitor.analyze(received_iq, fs_hz=8000)
# result: detected, spectrum, snr_est_db, modulation, confidence, warnings
```

**`oracle_noise`, etiquetas, canal real y `snr_db` nominal** pueden usarse para diagnosticar el simulador y evaluar después, **nunca** como entrada a `monitor.analyze()`.

---

## 5. Plan de implementación por fases, con criterios de aceptación

### Fase 0 — Auditoría del repositorio y de las fuentes

**Codex realiza:** leer la guía docente, inventariar papers, inspeccionar scripts existentes, crear entorno reproducible y tests iniciales, `docs/AUDITORIA_INICIAL.md`, `docs/MATRIZ_LITERATURA.md` inicial. Revisar `Investigación sobre Clasificación de Modulaciones.docx` como notas, contrastando sus citas con PDF originales.

**Entregar:** árbol real de archivos, problemas concretos, comandos ejecutados, matriz priorizada, dependencias y primera decisión solicitada. **No continuar a Fase 2 automáticamente.**

### Fase 1 — Validar modulaciones y generador RRC

**Codex realiza:** integrar seis moduladores, documentación de mapping, `waveform.py`, tests de energía, símbolo/bit, Gray, validación, filtro RRC, retardo, longitud, reproducibilidad. Crear gráficas diagnósticas de constelación y pulso.

**Aceptación:** 6/6 modulaciones producen puntos correctos; energía media teórica comprobada con todas las combinaciones; pruebas pasan; señal filtrada con I y Q; retardo y respuesta de filtro adaptado documentados. Nada de usar solo gráficos «bonitos» como evidencia.

**Estudiante revisa:** por qué las normalizaciones son `1`, `√2`, `1`, `√10`, `√42`, `√5`; qué distingue un símbolo de una muestra I/Q.

### Fase 2 — Validar canales, perturbaciones y ADC opcional

**Codex realiza:** AWGN, fase, CFO, multipath FIR con parámetros por trama, tests y rangos. Asegurar SNR medida ≈ SNR nominal solo dentro de tolerancia estadística razonable. Separar verdaderos parámetros del canal del acceso del monitor. Añadir `adc.py` pequeño y opcional después del canal, si no compromete el avance.

**Aceptación:** CFO conserva potencia, identidad multipath, AWGN complejo con varianzas correctas, retardo reproducible, efectos separados y combinados. `adc.py` verifica cuantización y clipping, sin mezclarla con cuantización de pesos neuronales.

**Estudiante revisa:** interpretación de fase fija vs fase acumulativa, qué es ISI, qué significa SNR medida sobre las muestras.

### Fase 3 — Dataset reproducible, ventanas 256/512 y ausencia de fuga de datos

**Codex realiza:** `dataset.py`, identificadores de trama, configuraciones YAML, manifiesto CSV/Parquet y tests. Generar observaciones **con señal** y **solo ruido**; cada señal clase/condición debe tener varias tramas independientes.

**Decisiones de diseño que documentar:**

- `sample_rate_hz`, `symbol_rate_hz`, `sps`, `beta`, longitud de trama, política de transitorios y `full_scale` (si ADC).
- Clases balanceadas en cada split; SNR suficiente para curva (`-10`, `-5`, `0`, `5`, `10`, `15`, `20` dB como ejemplo, no norma de la guía).
- Semillas **separadas** por split; secuencias binarias nuevas, fases, CFO, canales y ruido independientes entre tramas.
- Split por **trama madre** antes de generar derivados. Agrupar prefijos 256/512, representaciones FFT, variantes ADC, múltiples ruidos/canales de la misma fuente y cualquier augmentation ligada.
- Escenario **in-distribution** y escenario **channel-shift** reservado para E6. Variar parámetros reales de canal (p. ej. retardos/taps no vistos, distribución CFO, roll-off si procede), no solamente cambiar otra semilla del mismo régimen.
- Prefijo corto de 256 y largo de 512 exactamente de una misma señal y con idéntica fase/CFO/ruido sobre el prefijo compartido.
- Prohibido entrenar usando nombres de archivo, índices, SNR/canal reales, semillas o rutas que identifiquen indirectamente la clase.
- Guardar `train`, `val` y `test` sin solapamiento; si se usa calibración y política, reservar subconjuntos separados dentro de validación cuando sea viable.

**Aceptación:** auditoría automática de `parent_waveform_id` sin intersección, conteos y distribución de clases/SNR, pruebas de longitud, consistencia de prefijos, manifiesto documentado y un pequeño dataset de humo para tests rápidos.

**Nota:** mantener una distinción conceptual entre 256 muestras (no 256 símbolos), y los segundos de adquisición virtual `N/fs`; los transitorios RRC no se cuentan como símbolos perfectamente sincronizados.

### Fase 4 — Detector, contenido espectral y SNR estimada a ciegas

**Codex realiza:**

1. Detector inicial de energía: `T = mean(abs(iq)**2)`; calibrar umbral `gamma` con señales **solo ruido** de validación para una tasa nominal `P_FA` (por ejemplo 0.01), no eligiéndolo mirando el test.
2. E1: curva `P_D` vs SNR a `P_FA` fija; medir también `P_FA` sobre noise-only independiente y mostrar intervalos de confianza si la cantidad de ensayos lo permite.
3. Contenido espectral: FFT/PSD con ventanas, normalización de unidades, frecuencia y estimación razonable de ancho de banda; exportar PNG/CSV.
4. **SNR sin conocimiento del simulador:** investigar si se puede usar un segmento solo ruido, bandas libres espectrales o un estimador bajo hipótesis explícitas. Documentar **identificabilidad**: desde una sola mezcla I/Q arbitraria, SNR no siempre es recuperable sin supuestos adicionales. Si el método falla bajo CFO o multipath, reportarlo honestamente.

**Aceptación:** `monitor.analyze()` nunca importa los campos `true_snr_db` o `noise` usados para generación. Figuras etiquetadas y error de SNR por régimen con incertidumbre.

### Fase 5 — Clasificador convencional (obligatorio)

**Codex realiza:** comparar un método simple y justificable (p. ej. SVM, Random Forest o regresión logística), normalización ajustada **solo sobre train**, características de estadística/spectral y documentación de ingeniería.

**Características candidatas** (evaluar, no usar todas sin análisis): momentos de amplitud, potencia relativa, magnitud de circularidad, características de fase, estadísticos robustos, densidad espectral normalizada, ancho de banda, cumulantes de orden superior cuidadosamente definidos.

**Dos longitudes:** usar modelos/pipelines compatibles con 256 y 512; no introducir padding o leakage sin justificarlo.

**Aceptación:** E2 completo, matriz de confusión, F1 macro, exactitud por SNR, modelo reproducible guardado, inferencia con observaciones nuevas y sin etiquetas.

### Fase 6 — Clasificador avanzado (obligatorio)

**Codex realiza:** implementar un modelo avanzado **moderado** elegido por evidencia (MLP o CNN 1D pequeña; no está impuesta una CNN). Evaluar contra el convencional con los **mismos splits y condiciones**. Registrar entrenamiento, semilla, epochs, tiempo, parámetros y latencia; usar early stopping basado en validación.

**Aceptación:** E3 y E4, comparación transparente de accuracy, precision/recall/F1, confusiones y tiempo de inferencia. Si el avanzado no mejora, no ocultarlo. Ejecutar E5 y E6 para **ambos** modelos.

### Fase 7 — Monitor integrado y archivos desconocidos (obligatorio)

**Codex realiza:** interfaz de módulo y CLI reproducible. Por ejemplo:

```bash
python -m src.monitor --input data/unknown/X01.c64 --dtype complex64 --fs-hz 8000 --output results/X01.json
```

El formato mostrado es una **propuesta**; el programa debe soportar o documentar el formato real que entregue el profesor (`.npy`, `.npz`, `.csv`, `.bin`/`complex64`, etc.). Si la frecuencia de muestreo no se conoce, no imprimir un eje de frecuencia en Hz ficticio.

**Salida sugerida:**

```json
{
  "signal_id": "X01",
  "detected": true,
  "spectrum_file": "results/X01_psd.png",
  "snr_est_db": null,
  "snr_status": "no_identificable_sin_supuestos",
  "modulation": "16QAM",
  "confidence": 0.81,
  "warnings": []
}
```

El JSON es solo el **contrato de estructura**: los valores mostrados **son ilustrativos**, no predicciones reales. La confianza debe estar calibrada antes de interpretarse como probabilidad. Validar lectura del archivo, detección, gráficas y respuesta sobre una trama verdaderamente nueva.

### Fase 8 — GNU Radio Companion (obligatorio)

Crear al menos un flowgraph `.grc` **ejecutable sin hardware**, por ejemplo fuente aleatoria / modulación / RRC / canal / File Sink y una ruta de lectura/visualización. Confirmar versión GNU Radio y compatibilidad del formato complejo con Python. Registrar capturas y comandos. Si GNU Radio no está disponible, **no afirmar que se ejecutó**: guardar la propuesta y documentar el bloqueo para resolverlo.

### Fase 9 — Investigación adaptativa: tres acciones (opcional hasta completar E1–E6)

**Codex realiza:** desarrollar baselines primero:

- Siempre a0 (convencional, 256).
- Siempre a1 (convencional, 512).
- Siempre a2 (avanzado, 256).
- Umbral sobre confianza **calibrada**: decidir a0 o una acción adicional.
- Política basada solo en estimación ciega de SNR.
- Comparación conceptual y, si viable, implementación sencilla tipo early-exit/benefit-aware inspirada por BEACON, sin copiar resultados ni atribuir falsamente la autoría.

**Después**, probar una política de costo/riesgo de tres acciones. Se permite obtener las etiquetas de utilidad de las acciones en **datos de entrenamiento** evaluando contrafactualmente las tres alternativas. Pero durante **inferencia** solo puede observar las primeras 256 muestras y las salidas de modelos efectivamente ejecutados. No seleccionar la acción con el error real en test.

Propuesta de función de costo (declarar pesos elegidos):

```text
J = tasa_de_error + lambda_t * tiempo_promedio
                    + lambda_c * costo_computacional_medio
```

Medir **tiempo de observación virtual** `N/fs` separado del **tiempo de CPU de inferencia**, incluir extracción de características, PSD/detección si se amortiza, inferencia adicional y recálculos. Los MACs/parámetros son proxies computacionales, **no mediciones de energía física ASIC/FPGA**.

**Aceptación:** curvas de exactitud vs tiempo/costo, distribución de elecciones a0/a1/a2, riesgo por SNR/canal, comparación con baselines, protocolo sin leakage y prueba con canales no vistos. Si la política no mejora, dejarlo explícito.

### Fase 10 — ADC virtual y entrenamiento mixto (extensión secundaria)

Ensayo de 6/8 bits y al menos una condición no vista, según recursos. Definir recorte, escala por componente, cuantización, SQNR **medida con referencia solo para evaluación**, y guardar versiones relacionadas bajo un mismo `parent_waveform_id`. Comparar entrenamiento original, cuantizado y mixto; evaluar efecto también sobre confianza y decisiones de la política, no solo accuracy. No asumir que añadir cuantización mejora modelos de forma automática.

### Fase 11 — Informe, gráficas, reproducibilidad y presentación

Producir nueve entregables del profesor. Incluir métodos, ecuaciones con referencias, configuración de hardware/software, tablas, seis experimentos, figuras legibles, errores/limitaciones, protocolo sin leakage, demo y guion de defensa. Separar **resultado observado**, **hipótesis** e **interpretación**. No fabricar ejecución de tests, gráficas, lecturas, fuentes, fechas ni resultados.

---

## 6. Diseño mínimo del dataset

**Unidad de independencia:** `parent_waveform_id` (identifica una generación madre). Cada muestra de entrenamiento puede contener:

```text
record_id
parent_waveform_id
split                         # train / val / test / test_shift
modulation                     # etiqueta de la simulación
has_signal                     # 0/1
snr_true_db                    # SOLO evaluación / simulador
fs_hz, symbol_rate_hz, sps
beta, phase_rad, cfo_hz
channel_id, channel_taps       # SOLO simulación / evaluación
adc_bits, adc_full_scale       # si aplica
samples_512_path
short_prefix_len = 256
long_prefix_len = 512
seed_or_rng_id                 # SOLO reproducibilidad
```

**El monitor no recibe el manifiesto de verdad:** solo recibe I/Q y datos auxiliares explícitamente disponibles en despliegue. Para detección, crear también registros noise-only. El dataset puede ser generado sobre la marcha con manifiestos y semillas, sin guardar archivos enormes innecesariamente.

**Orden sugerido de operaciones:** símbolos → RRC → multipath → fase/CFO → AWGN → ADC opcional → segmentación en ventanas. Documentar la localización del filtro receptor *solo* para fines de diagnóstico; los modelos AMC utilizarán observaciones I/Q recibidas, no constelaciones ecualizadas con conocimiento del canal verdadero.

**Cuidado con escalado:** variar la amplitud recibida por trama de acuerdo con la guía; si se normaliza cada ventana antes de clasificación, documentar qué información se pierde y mantener el detector de energía en la ruta adecuada para no borrar la diferencia señal/ruido.

---

## 7. Protocolo estadístico y métricas

Para todas las corridas finales guardar `config`, semilla, hash/versión del código, división del dataset, conteos y fecha de ejecución.

| Aspecto | Métricas sugeridas |
|---|---|
| Detección | `P_D(SNR)`, `P_FA`, umbral, n de tramas |
| Clasificación | Accuracy, precision/recall/F1 **macro** y por clase, matriz de confusión |
| SNR | MAE/RMSE en dB, sesgo y fallos por condición; siempre estimación ciega |
| Confianza | ECE con bins declarados, Brier, NLL, diagrama de confiabilidad |
| Generalización | métricas por tipo y grado de canal no visto; caída frente a ID |
| Eficiencia | tiempo CPU median/p95, parámetros, FLOPs/MACs estimados, `N/fs` |
| Adaptación | riesgo-cobertura/costo, porcentaje a0/a1/a2, Pareto exactitud-costo |
| ADC | tasa de clipping, error de cuantización, exactitud y calibración por bits |

**Protocolo de tiempos:** medición con warm-up, varias repeticiones, mismo equipo y batch size declarado (preferiblemente 1 para la demo), distinguir entrenamiento, carga de modelos, preprocesamiento e inferencia. No mezclar unidades segundos y milisegundos. Reportar hardware/software; no convertir tiempos Python en estimaciones de ASIC sin metodología adicional.

**Protocolo de confianza:** max-softmax puede ser una puntuación, no una probabilidad fiable. Evaluar y, si procede, calibrar en validación independiente (p. ej. temperature scaling para probabilidades de red o calibración apropiada del modelo). Una señal OOD puede producir alta confianza errónea; registrar la limitación.

**Protocolo de comparación:** usar exactamente el mismo split y las mismas realizaciones para modelos y políticas, semillas replicadas cuando el presupuesto lo permita, y evitar seleccionar hiperparámetros mirando el test. Añadir márgenes de error si se dispone de suficientes repeticiones.

---

## 8. Pruebas automáticas obligatorias

Codex debe utilizar `pytest` y **registrar los resultados reales**. Pruebas sugeridas:

```bash
python -m pytest -q
python -m experiments.exp01_detection
python -m experiments.exp02_conventional
python -m experiments.exp03_advanced
python -m experiments.exp04_comparison
python -m experiments.exp05_snr_robustness
python -m experiments.exp06_channel_generalization
```

Los comandos de `experiments` se vuelven válidos cuando los módulos se hayan implementado; **no afirmar que existen antes de crearlos**.

**Checks por módulo:**

- `signals`: combinación de **todos** los puntos, energía teórica, longitud, Gray, bits inválidos, tipos complejos, reproducibilidad.
- `waveform`: longitud `len(symbols)*sps + len(h)-1`, respuesta de RRC, convolución de dos filtros, retardo, espectro y casos límite.
- `channels`: varianza compleja AWGN, SNR tolerancia estadística, fase/CFO conservación de magnitud, FIR identidad y eco conocido, canal coherente por trama.
- `adc`: cuantización uniforme y clipping a escala fija, bits no válidos, caso saturado, salida compleja.
- `dataset`: cero intersección de padres entre splits, prefijo 256 idéntico al inicio del de 512, clases balanceadas, ruido independiente entre realizaciones, ningún campo oracle llega al monitor.
- `detector`: `P_FA` sobre ruido independiente no muy lejos del objetivo dentro de margen binomial; `P_D` razonablemente aumenta con SNR en condiciones controladas.
- `monitor`: procesar señal presente y ruido-only; respuestas con `N/A` cuando corresponda, carga real de formatos declarados.
- `adaptive`: solo 256 muestras a la decisión, sin etiquetas/SNR verdaderos; costo y elección verificables.

**No usar la misma semilla como única fuente de diversidad ni repetir una sola trama 10.000 veces bajo diferentes etiquetas.**

---

## 9. Cronograma sugerido (ajustable, con prioridad al curso)

La guía docente indica entrega durante la **semana del 1 de diciembre de 2026**. Este calendario es de trabajo **propuesto** y ajustable:

| Fechas | Prioridad y resultado |
|---|---|
| 3–7 nov. | Fases 0–2: auditar, integrar seis moduladores, RRC, canales, tests |
| 8–13 nov. | Fases 3–4: dataset independiente, detección, PSD, estimación SNR |
| 14–20 nov. | Fases 5–6: convencional y avanzado, E1–E5 |
| 21–26 nov. | Fases 7–8 y E6: monitor completo, generalización y GRC |
| 27–30 nov. | Integración, informe, demo; fase 9 adaptativa si E1–E6 ya están completos |

**Fase 10 (ADC):** implementar su bloque liviano tan pronto como esté validado el canal si es fácil; reservar la comparación amplia para cuando el mínimo docente esté cerrado. No prometernos completar todas las variantes de paper si el rendimiento, cronograma o capacidad de cómputo no lo permiten.

---

## 10. Política de desarrollo para Codex: evitar sobreingeniería

1. Módulos cortos, nombres consistentes, `numpy` vectorizado cuando aporte claridad, configuración separada.
2. Generar primero dataset diminuto de humo y tests rápidos; luego corridas finales configurables.
3. No descargar datasets masivos ni instalar librerías pesadas sin necesidad/explicación. Preferir `numpy`, `scipy`, `matplotlib`, `pandas`, `scikit-learn`, `pytest`; añadir `torch` si se elige un modelo que lo requiera.
4. Evitar dependencias ocultas y rutas absolutas del computador de otra persona.
5. No inventar señales sin ruido para clases «sin señal» ni devolver modulación obligatoriamente cuando se detecte ausencia.
6. No usar el mismo archivo de test para seleccionar arquitectura y reportar el resultado final.
7. No sustituir entrenamiento/validación por gráficos de una constelación ideal.
8. No construir FPGA ni afirmar ahorro energético medido; la conexión microelectrónica es por requerimientos de buffer, memoria, MACs, latencia y futura arquitectura digital.
9. Si hay una contradicción entre papers o con la guía, registrarla en `docs/DECISIONES.md` con citas precisas y la fuente priorizada.
10. **Entregar una fase cada vez**. Preguntar al estudiante solo las decisiones importantes no cubiertas por el contexto; no solicitarle información repetida.

---

## 11. Lista de entregables verificables

- [ ] Repositorio funcional, dependencias y README de ejecución.
- [ ] Seis moduladores normalizados, RRC, canales y tests.
- [ ] Dataset reproducible con scripts y separación estricta por realizaciones.
- [ ] Detector y curva `P_D` vs SNR con `P_FA` documentada.
- [ ] FFT/PSD del monitor y estimación de SNR no-oracular.
- [ ] Modelo convencional y reporte E2.
- [ ] Modelo avanzado y reporte E3.
- [ ] E4: comparación con métricas, matrices y tiempos.
- [ ] E5: curvas clasificación vs SNR.
- [ ] E6: condiciones de canal no vistas.
- [ ] Flowgraph GRC sin hardware y prueba/documentación de interoperabilidad.
- [ ] CLI para I/Q desconocidas y tabla de cinco salidas.
- [ ] Informe técnico, presentación y demostración.
- [ ] Matriz de artículos y apartado de limitaciones/validez experimental.
- [ ] **Opcional:** baselines y política adaptativa de tres acciones.
- [ ] **Opcional:** evaluación cuantización ADC y entrenamiento mixto.

---

## 12. Cómo estudiar los papers sin convertir el proyecto en una copia

Codex creará `docs/MATRIZ_LITERATURA.md`, una fila y una ficha corta por artículo, **leyendo el PDF original**. Para cada paper extraer:

1. Referencia IEEE exacta: título, autores, año, venue, DOI; **no inventar ni corregir en silencio**.
2. Qué problema resuelve y bajo qué supuestos.
3. Modelo matemático y entrada (I/Q, espectro, SNR conocida/estimada, longitud de ventana).
4. Dataset y split; clases, SNR, canal, tamaño de muestra, variaciones.
5. Arquitectura o algoritmo y costo reportado con unidades/contexto.
6. Experimentos, baselines, métricas y principales resultados **comprobados en tablas/figuras**.
7. Limitaciones y amenazas a validez.
8. Qué se reutiliza como **idea metodológica** en el proyecto, qué **no** se reproducirá y por qué.
9. Citas con **nombre de PDF + número de página del PDF / sección / ecuación / figura**; distinguir página impresa de página PDF cuando difieren.
10. Relación con nuestra hipótesis: complementa, compite, contradice o no aplica.

**Revisión inicial orientativa:** el mapa del apartado siguiente identifica temas observables en los artículos, **no sustituye una lectura metodológica completa ni la verificación de cada resultado numérico**. Antes de citar resultados de un paper, Codex debe comprobarlos en el documento real.

---

## 13. Artículos proporcionados: inventario, prioridad y lectura dirigida

Se han identificado **15 artículos diferentes** en los PDF adjuntos. Hay **dos copias** de los PDFs sobre cuantificación de incertidumbre y adaptación de dominio: **no contarlos dos veces**. Todos los títulos que siguen corresponden a archivos presentes en los materiales aportados.

### Prioridad A — Lectura necesaria para diseño de datos y pregunta científica

| Artículo / archivo exacto | DOI | Qué estudiar y utilidad concreta |
|---|---|---|
| **RML22** — `RML22_Realistic_Dataset_Generation_for_Wireless_Modulation_Classification.pdf` | `10.1109/TWC.2023.3254490` | Generación realista de datasets, cadena de transmisión y canal, GNU Radio, defectos de RML16, parámetros no ideales. **Principal antecedente del generador**. Estudiar introducción y secciones de modelo de señal, generación y evaluación. |
| **BEACON** — `BEACON_Benefit-Aware_Early-Exit_for_Automatic_Modulation_Classification_via_Recoverability_Prediction.pdf` | `10.1109/JIOT.2026.3710690` | *Early-exit* que estima si profundizar recupera errores de la salida temprana. **Competidor directo** para cualquier afirmación de «decidir si procesar más». Analizar qué observa el predictor, costos y casos de beneficio/no beneficio. |
| **Fewer-Sample** — `Fewer-Sample_Fast_Automatic_Modulation_Recognition_with_Multimodal_Deep_Learning_and_Resized_Signal_Representation.pdf` | `10.1109/ICCCS65393.2025.11069715` | Reducción de muestras observadas, tiempo de adquisición más tiempo de cómputo; comparación de longitudes. **Competidor en «adquirir menos»**. Distinguir decisión fija de adaptación por muestra y revisar los supuestos de velocidad de muestreo. |
| **Uncertainty Quantification** — `An_Uncertainty_Quantification_Framework_for_Deep_Learning-Based_Automatic_Modulation_Classification.pdf` | `10.1109/JIOT.2025.3640108` | Incertidumbre y confianza de clasificación mediante ensemble, evaluación de calidad probabilística. **Base para confianza calibrada**; comparar con max-softmax y revisar métricas de incertidumbre. |
| **Shahriar, Domain-Adaptive** — `Lightweight_Domain-Adaptive_Modulation_Classification_Under_SNR_and_Channel_Mismatch.pdf` | `10.1109/LWC.2026.3669041` | Adaptación ante desajuste de SNR y canal, alineación de dominio, eficiencia. **Base para E6**. Distinguir pipelines según dataset/representación; no asumir que la SNR verdadera está disponible en despliegue. |

### Prioridad B — Modelos, adquisición y costo

| Artículo / archivo exacto | DOI | Qué estudiar y utilidad concreta |
|---|---|---|
| **MLEA-AMR** — `MLEA-AMR_A_Multi-Domain_Lightweight_Expert_Framework_With_SNR-Conditioned_Attention_for_Robust_Automatic_Modulation_Recognition.pdf` | `10.1109/TCCN.2026.3725098` | Vistas temporal, espectral y descriptores polares; agregación condicionada por SNR. Estudiar la **procedencia de la SNR**: no alimentar a nuestro modelo la SNR real del simulador sin declararlo. |
| **OF-ACF** — `Optimal-Feature-Driven_Adversarial_Adaptive_Cascade_Forest_Method_for_Automatic_Modulation_Recognition.pdf` | `10.1109/JIOT.2026.3685655` | Selección de características, bosques en cascada y profundidad adaptable. Referencia de modelos no-CNN y costo variable; explorar ideas de baseline, no reproducir innecesariamente el sistema completo. |
| **Joint Early Exit and Structured Pruning** — `Joint_Early_Exit_and_Structured_Pruning_for_Automatic_Modulation_Classification_in_Vehicular_Networks.pdf` | `10.1109/VTC2025-Fall65116.2025.11310312` | Interacción de poda estructurada y salida temprana. **Advertencia:** menos parámetros no siempre implica menor costo promedio cuando cambia la frecuencia de salida temprana. |
| **FFTFormer** — `FFTFormer_A_Frequency-Aware_Transformer_With_Multi-Task_Learning_for_Robust_Modulation_Recognition.pdf` | `10.1109/JMW.2026.3706228` | Variabilidad del roll-off RRC, representaciones en frecuencia, robustez a SNR y mediciones. **Motivación para variar β** en el dataset; no es necesario implementar Transformer. |
| **ADC mismatch** — `Mitigating_Interleaved_ADC_Mismatches_in_Automatic_Modulation_Classification_via_Sampling_Schemes_and_Layer_Fusion.pdf` | `10.1109/MILCOM64451.2025.11310579` | Ganancia, offset y skew de canales TI-ADC, muestreo y fusión. Pertinente a microelectrónica, **pero es distinto de cuantización uniforme de B bits**. No usarlo como demostración de que augmentation 6/8 bits mejora nuestra política. |
| **FPGA accelerator** — `FPGA-based_Deep_Learning_Accelerator_for_RF_Applications.pdf` | `10.1109/MILCOM52596.2021.9652891` | Arquitecturas/recursos FPGA para inferencia RF; lectura conceptual para buffers, latencia, throughput y memoria. No afirmar resultados de hardware propio. |

### Prioridad C — Contexto adicional y límites del alcance

| Artículo / archivo exacto | DOI | Utilidad y límite |
|---|---|---|
| **Survey of Deep Transfer Learning in AMC** — `A_Survey_of_Deep_Transfer_Learning_in_Automatic_Modulation_Classification.pdf` | `10.1109/TCCN.2025.3558027` | Marco general de transfer learning, shift y generalización. Útil para definiciones y related work. |
| **FSOS-AMC** — `FSOS-AMC_Few-Shot_Open-Set_Learning_for_Automatic_Modulation_Classification_Over_Multipath_Fading_Channels.pdf` | `10.1109/JIOT.2025.3557973` | Few-shot y clases no vistas con multipath. Nuestro curso no exige necesariamente clasificación open-set; citar como límite o futura extensión. |
| **Open-Set Domain Adaptation** — `Open-Set_Domain_Adaptation_for_Automatic_Modulation_Classification_via_Domain-Adaptive_Mixup_and_Prototype-Based_Representative_Learning.pdf` | `10.1109/JIOT.2026.3718553` | Desajuste de dominio y modulación desconocida; no convertir el proyecto en open-set sin necesidad. |
| **Single-shot Detector** — `Single-shot_Detector_for_Joint_Signal_Detection_and_Modulation_Classification_with_Open-set_Recognition.pdf` | `10.1109/RadarConf2663773.2026.11675698` | Detección y clasificación conjunta con reconocimiento open-set; útil para situar la integración de nuestro monitor, no imprescindible reproducir. |

### 13.1 Preguntas críticas de revisión cruzada

Después de leer los trabajos prioritarios, Codex debe responder con evidencias y referencias precisas:

1. ¿En qué se diferencia una acción que **adquiere más muestras** de una que **ejecuta una rama más profunda**? Comparar Fewer-Sample y BEACON.
2. ¿En BEACON se aprende la **recuperabilidad del error**, o simplemente la confianza de softmax? ¿Qué baseline correcto debemos incluir?
3. ¿Qué fuentes de información usan los papers para condicionar el modelo por SNR? ¿Puede nuestro monitor obtenerlas sin un oracle?
4. ¿Cómo afecta el dataset a la comparación? ¿Estamos usando las mismas clases, `sps`, canales y longitudes que el paper? Si no, no copiar porcentajes como si fueran comparables.
5. ¿La cuantización de **muestras I/Q** de nuestro experimento tiene antecedentes directos en estos PDFs, o el trabajo TI-ADC estudia otro tipo de distorsión? Declararlo con rigor.
6. ¿Qué evidencia sostendría novedad de la **elección conjunta entre más observación y más cómputo condicionada por riesgo**, frente a literatura de adquisición adaptativa, early-exit y aprendizaje secuencial? Si faltan antecedentes, investigar fuera del corpus **solo después** de registrar lo que dicen los PDFs proporcionados, citando por separado las fuentes externas.
7. ¿Qué pruebas de costo y calibración serían suficientes para defender la propuesta sin hardware RF?

**Producto:** `docs/MATRIZ_LITERATURA.md` + `docs/RESEARCH_GAP.md` (afirmaciones, evidencias, posibles objeciones y baseline que cada paper obliga a considerar). No escribir «primera propuesta de la historia» sin revisión específica y resultados.

---

## 14. Criterios de revisión antes de redactar un eventual paper IEEE

Esta parte es **aspiracional**. No forma parte del mínimo de la guía del curso y depende de resultados:

- Pregunta de investigación precisa y falsable; contribuciones separables pero integradas.
- Revisión ampliada de clasificación secuencial, adquisición adaptativa, early-exit, predicción de beneficio y modelos para edge.
- Baselines robustos, mismo presupuesto de información y mismos datos.
- Ablación de entrada de política: solo confianza, solo SNR estimada, confianza+estadísticas, beneficio predicho.
- Ablación de acciones: solo a0/a1, solo a0/a2, las tres juntas.
- Validación ID/OOD, variación de SNR, CFO, multipath, roll-off y cuantización opcional.
- Costos separados de adquisición (`N/fs`) e inferencia medidos; comportamiento al cambiar hardware y tamaño de muestra.
- Calibración, incertidumbre, intervalos de confianza, análisis de errores y reproducibilidad.
- Limitaciones explícitas: sin RF real, simulador de canal discreto simplificado, estimación SNR bajo supuestos, medidas de CPU y no consumo de ASIC.
- Si el método no supera un umbral simple o always-long, reportarlo y concentrar el informe del curso en los resultados honestos y obligatorios.

---

## 15. Plantilla del reporte de cada fase que debe mostrar Codex

```markdown
## Fase ejecutada

### Objetivo
[Qué se intentó]

### Archivos modificados
[Rutas, propósito, compatibilidad]

### Fuentes consultadas
[Nombre del PDF, página/sección, dato realmente utilizado]

### Decisiones de ingeniería
[Qué asumimos y por qué; qué NO afirma el paper]

### Pruebas realmente ejecutadas
[Comandos, n pruebas, éxitos/fallos y logs relevantes]

### Resultados obtenidos
[Tablas/figuras/ubicación; no inventar]

### Explicación para el estudiante
[Qué hace el código y qué significa físicamente]

### Riesgos y limitaciones
[Pendientes técnicos/científicos]

### Siguiente cambio mínimo propuesto
[Una fase o subfase, no todo el backlog]
```

---

## 16. Condiciones de finalización

**Mínimo para aprobar técnicamente el curso:** sistema reproducible con seis modulaciones, perturbaciones, cinco salidas del monitor, modelos convencional y avanzado, E1–E6, GRC, archivo desconocido, informe y presentación.

**Meta adicional de investigación:** política de tres acciones comparada limpiamente con baselines, sin fugas de datos, con confianza evaluada y costos medidos.

**Meta secundaria de microelectrónica:** demostrar qué ocurre con la robustez al limitar la resolución de adquisición, con un ADC virtual bien definido; no afirmar que se diseñó/fabricó un ADC ni un acelerador.

**Última instrucción para Codex:** no confundas una idea plausible con un resultado validado. Tu papel es ayudar a construir software sencillo, probado y bien documentado para que el estudiante pueda explicar, defender y aprender cada componente.
