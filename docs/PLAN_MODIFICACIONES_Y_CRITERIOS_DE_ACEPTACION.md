# Plan de modificaciones y criterios de aceptación

**Proyecto:** COM2 — Proyecto 2, Comunicaciones II  
**Fecha:** 16 de noviembre de 2026  
**Estado:** plan de trabajo; las tareas descritas no se consideran ejecutadas por la creación de este documento.  
**Base:** revisión independiente documentada en `docs/REVISION_RESULTADOS_2026-11-15.md`.

## 1. Objetivo y reglas de cumplimiento

El objetivo es obtener un monitor confiable para la entrega del curso, resultados reproducibles y conclusiones científicas proporcionales a la evidencia. Primero se corrigen los problemas de funcionamiento y validación. Después se evalúa si la política adaptativa aporta una ventaja suficiente para sostener la contribución del artículo.

Este plan distingue tres cosas:

1. **Corrección funcional:** el código hace lo que declara, maneja sus límites y pasa pruebas pertinentes.
2. **Cumplimiento experimental:** el procedimiento se ejecuta correctamente y deja evidencia verificable, incluso si el resultado es negativo.
3. **Demostración de una mejora:** los resultados y su incertidumbre satisfacen una regla de comparación fijada antes de la evaluación final.

Un experimento sin mejora puede quedar **cumplido con resultado negativo**. Eso no autoriza a escribir que el método es superior. Un fallo funcional, como rechazar una entrada válida por un error del programa, sí impide cerrar la tarea correspondiente.

### 1.1 Origen de los mínimos numéricos

Los tamaños de muestra, tolerancias y objetivos adicionales de este documento son **propuestas internas de ingeniería y validación**. No se presentan como requisitos numéricos del profesor. La guía docente exige experimentos, salidas y análisis; no establece una exactitud universal del 90% ni un límite de ECE determinado.

Antes de ejecutar una evaluación final, estos criterios deben quedar fijados en una configuración versionada. Pueden revisarse durante el desarrollo con justificación técnica, pero nunca relajarse después de ver el test para declarar éxito. Un cambio posterior requiere dejar registro y utilizar una evaluación final nueva.

### 1.2 Estados permitidos

| Estado | Significado |
|---|---|
| Pendiente | Aún no hay evidencia suficiente de ejecución. |
| En curso | Hay cambios o resultados parciales, pero faltan criterios. |
| Cumplido | Se satisfacen los criterios funcionales y experimentales aplicables. |
| Cumplido con resultado negativo | La evaluación es válida, pero la hipótesis de mejora no se confirma. |
| No cumple | Hay errores funcionales, criterios incumplidos o evidencia inválida. |
| No aplica, justificado | La tarea queda fuera del alcance declarado; no puede usarse para sostener una afirmación asociada. |

No se sustituye la evidencia por expresiones como «parece funcionar», «se ve mejor» o «la IA lo confirmó».

### 1.3 Evidencia obligatoria para cerrar una tarea

Cada tarea cerrada debe registrar: identificador, archivos modificados, comando o procedimiento, configuración, particiones usadas, resultados observados, comparación con el criterio, limitaciones y ruta a los artefactos. Cuando corresponda, incluir semillas, hashes, versiones y hardware.

Conservar los resultados históricos. Crear carpetas de ejecución nuevas y un manifiesto que identifique qué resultados son históricos, de desarrollo o de evaluación final.

## 2. Niveles y orden de ejecución

| Nivel | Propósito | Prioridad | Condición de salida |
|---|---|---|---|
| 0 | Preservar evidencia y fijar el protocolo | Inmediata | Resultados históricos identificados y criterios congelados. |
| 1 | Corregir el monitor | Crítica para el curso | Entradas, detección, SNR, espectro, clasificación y confianza funcionan según un contrato explícito. |
| 2 | Validar el sistema completo | Crítica para el curso | Pruebas independientes y evaluación E1–E6 trazables. |
| 3 | Validar la investigación adaptativa | Necesaria para las afirmaciones del artículo | Política desplegable, costes completos y comparaciones justas. |
| 4 | Corregir documentación y preparar entrega | Necesaria para cerrar el alcance elegido | Documentos, cifras, comandos y artefactos coinciden. |
| 5 | Mejoras adicionales de rendimiento | Opcional | Beneficio confirmado en evaluación nueva o resultado negativo documentado. |

**Ruta del curso:** nivel 0 → nivel 1 → nivel 2 → nivel 4.  
**Ruta del artículo:** nivel 0 → niveles 1 y 2 → nivel 3 → nivel 4.  
**Mejoras posteriores:** nivel 5, después de estabilizar la base.

Las correcciones de texto conocidas pueden adelantarse. No hay que esperar a nuevos experimentos para retirar afirmaciones ya identificadas como incorrectas. Si se conserva una afirmación adaptativa en el informe del curso, se necesita la validación correspondiente del nivel 3.

## 3. Nivel 0 — Evidencia y protocolo

### N0.1 — Identificar y preservar el estado de referencia

**Modificar:** organización de resultados y trazabilidad del código. Añadir un manifiesto del estado actual con hashes SHA-256 de fuentes relevantes, configuraciones, modelos, métricas y datasets; registrar dependencias y comandos. Establecer control de versiones local o, como mínimo, una instantánea identificable del código. No publicar datos ni modelos para cumplir esta tarea.

**Resultado mínimo para cumplir:**

- Cada conjunto de resultados utilizado tiene identificados dataset, partición, modelo, configuración y código asociado, o una limitación de procedencia explícita si no puede reconstruirse retrospectivamente.
- Las cifras históricas principales quedan registradas sin sobrescribirlas: SVM256 43,75%, SVM512 46,04%, red multidominio 51,5833% y LBAP 51,625% en el test adaptativo nominal.
- La comprobación de integridad detecta una copia de prueba alterada; nunca alterar el original para ensayar este control.

**Evidencia:** manifiesto, inventario de versiones y tabla de correspondencia entre experimentos.  
**Dependencias:** ninguna.

### N0.2 — Fijar alcance, particiones y reglas de aceptación

**Modificar:** protocolo experimental y configuraciones de validación. Separar entrenamiento de expertos, desarrollo de política, calibración, selección y evaluación final según el experimento. Si una partición cumple más de una función, documentar la justificación y sus límites; el test final nunca participa en ajuste.

**Resultado mínimo para cumplir:**

- Cero padres compartidos entre particiones que deben ser independientes, comprobados sobre identificadores y procedencia de generación.
- Prefijos de 256 y 512 muestras y variantes ADC de un padre permanecen en su partición.
- Los tests ya inspeccionados se etiquetan como evaluación histórica; para cambios guiados por esta revisión se reserva un test final nuevo.
- Se fijan antes de abrirlo: modelos candidatos, criterio de selección, comparación primaria, tolerancias, semillas y tratamiento de resultados negativos.
- Se distingue el test de clasificación condicionada a señal presente del test completo de detección y clasificación.

**Evidencia:** protocolo y reporte de independencia.  
**Dependencias:** N0.1.

## 4. Nivel 1 — Corrección funcional del monitor

### N1.1 — Calibración de detección y decisión por archivo

**Archivos principales:** `src/monitor.py`, `src/detection.py`, pruebas del monitor y documentación de uso.

**Modificar:** eliminar el uso silencioso de un umbral calibrado con ruido unitario para cualquier archivo. La referencia de ruido debe servir para calibrar la detección, además de la SNR, cuando sus condiciones sean compatibles. Validar banda, ganancia, longitud de ventana y método de calibración.

Definir también qué significa detectar una transmisión en un archivo de varias ventanas. Analizar únicamente el primer prefijo puede omitir una transmisión posterior. Si se combinan detecciones de varias ventanas, medir la falsa alarma **por archivo**: un objetivo del 1% por ventana no equivale al 1% por archivo.

**Resultado mínimo para cumplir:**

- Una entrada sin calibración suficiente recibe un estado explícito de detección no validada o indeterminada; no se presenta como «solo ruido» con falsa certeza.
- Con calibración válida, escalar señal y referencia por la misma ganancia conserva la decisión y la estadística normalizada, dentro de la precisión numérica declarada.
- Las dos demos históricas se procesan mediante un procedimiento de calibración documentado, sin elegir umbrales usando sus etiquetas. Si sus referencias originales no existen, se documenta esa limitación y se añaden demos nuevas con referencias independientes.
- Un archivo de prueba con transmisión posterior al primer prefijo se evalúa de acuerdo con la regla por archivo; la transmisión no queda fuera del análisis por error de segmentación.
- La meta cuantitativa de falsa alarma y detección se valida en N2.1; aprobar pruebas funcionales por sí solo no autoriza a anunciar `P_FA = 1%`.

**Evidencia:** pruebas de escala, segmentación y calibración; JSON por demo.  
**Dependencias:** N0.2.

### N1.2 — Frecuencia de muestreo y compatibilidad con el modelo

**Archivos principales:** `src/monitor.py`, `src/spectrum.py`, metadatos de modelos.

**Modificar:** retirar la suposición silenciosa de 8 kHz cuando el archivo no aporta ese dato. Admitir metadatos verificables o argumento explícito y registrar su origen. Separar la posibilidad de calcular un espectro normalizado de la compatibilidad de la señal con el clasificador entrenado.

**Resultado mínimo para cumplir:**

- Sin frecuencia conocida, el espectro usa ciclos/muestra; no se inventan valores en Hz.
- Con frecuencia conocida, los ejes y anchos de banda se convierten correctamente; duplicar únicamente el metadato duplica las frecuencias reportadas sin cambiar la potencia integrada.
- Frecuencias no positivas o no finitas se rechazan con un mensaje útil.
- La inferencia con frecuencia o sobremuestreo incompatible devuelve un estado de fuera de alcance, o utiliza un remuestreo previamente validado. Cambiar `fs` en el JSON no basta para hacer compatible una observación.

**Evidencia:** pruebas de unidades y matriz de compatibilidad.  
**Dependencias:** N0.2.

### N1.3 — Estimación SNR con supuestos explícitos

**Archivos principales:** `src/snr_estimation.py`, `src/monitor.py`.

**Modificar:** conservar un modo con referencia independiente y documentar banda, estacionariedad y ganancia. Si se necesita un modo ciego, implementarlo como método separado, con supuestos de ocupación espectral y validación propia; no transferir automáticamente la heurística del selector a un estimador universal.

**Resultado mínimo para cumplir funcionalmente:**

- Con referencia válida se calcula SNR y se indica el método; sin información suficiente se devuelve `null` y motivo.
- Ruido puro, estimaciones no físicas y detector negativo/indeterminado no producen una SNR significativa sin justificar su estado.
- No se usan SNR, canal ni potencia verdadera del simulador como entrada del estimador desplegado.
- En N2.1 se presenta error, sesgo y cobertura, incluyendo el número de casos sin estimación; no se ocultan esos casos al promediar.

**Meta interna de calidad:** en ID, SNR ≥ 0 dB, 256 muestras y referencia compatible, cobertura válida ≥ 90% y error absoluto medio ≤ 3 dB. Evaluar cada SNR y el agregado. Fuera de ese dominio se reporta el rendimiento sin prometer la misma precisión.

Si la meta no se alcanza, la implementación puede quedar evaluada, pero la capacidad SNR queda **no conforme con la meta**. Devolver siempre `null` no resuelve el requisito operativo de estimación.

**Evidencia:** errores por clase/SNR/canal y tabla de cobertura.  
**Dependencias:** N1.1 y N1.2.

### N1.4 — Confianza de la decisión final

**Archivos principales:** `src/monitor.py`, calibradores y artefactos de inferencia.

**Modificar:** definir si se informa probabilidad de acierto de una ventana, del voto de un archivo o probabilidades multiclase. Retirar la etiqueta de calibración de la fórmula actual de margen promedio, proporción de votos y recorte, salvo que se calibre y valide esa salida final.

**Resultado mínimo para cumplir:**

- El calibrador corresponde exactamente al modelo, características, longitud y agregación usados. Una combinación incompatible se rechaza o se identifica como no calibrada.
- Se usa un conjunto de calibración independiente por padre/archivo, con al menos 1.200 unidades del tipo de decisión que se pretende calibrar.
- No se recorta artificialmente la probabilidad a 20–99% para mejorar su apariencia.
- Se evalúan Brier, NLL y ECE, con definición explícita de intervalos, sobre al menos 2.400 decisiones finales independientes; se informa ID y cambio de canal por separado.
- Si no hay calibrador validado, la salida indica puntuación no calibrada y no se anuncia como probabilidad validada.

**Meta interna de calidad nominal:** ECE de diez intervalos de igual anchura ≤ 0,05 y Brier no peor que la puntuación base definida antes del test. Publicar también incertidumbre y ocupación de intervalos; ECE aislado no certifica calibración. Si falla, corregir la afirmación de calidad o desarrollar otro calibrador sin ajustar sobre ese test.

**Evidencia:** curvas de confiabilidad, métricas antes/después y metadatos del calibrador.  
**Dependencias:** N1.1 y definición de la agregación por archivo.

### N1.5 — Entradas, errores y contrato de salida

**Archivos principales:** `src/monitor.py`, `tests/test_monitor_cli.py`.

**Modificar:** validar formatos declarados, formas de arrays y metadatos; evitar aplanar silenciosamente un lote de observaciones independientes como si fuera una sola señal. Definir clave de `.npz`, representación I/Q, cabecera CSV y longitud mínima por función. Manejar modelos ausentes, rutas de salida, valores `null` y errores de CLI.

**Resultado mínimo para cumplir:**

- `.c64`, `.bin`, `.npy`, `.npz` y `.csv` tienen al menos un caso válido y uno inválido comprobados según su contrato documentado.
- Arrays ambiguos, archivos vacíos, valores no finitos y binarios incompletos se rechazan sin predicciones inventadas.
- Señales de 255, 256 y 512 muestras tienen comportamiento explícito; no se rellena una ventana para clasificar sin declarar y validar ese procedimiento.
- Un modelo ausente no provoca una excepción secundaria al imprimir una confianza `null`.
- Cada entrada válida produce un JSON consistente con las cinco salidas y sus estados. Cada error de entrada produce mensaje y código de salida documentados.

**Evidencia:** pruebas CLI reales y ejemplos de salida válida, indeterminada y errónea.  
**Dependencias:** N1.1–N1.4 para el contrato final.

### N1.6 — Espectro y alcance de clasificación

**Archivos principales:** `src/spectrum.py`, `src/monitor.py`, documentación de modelos.

**Modificar:** distinguir ancho de banda del 99% de potencia total, que puede incluir ruido, de ancho de banda de la señal útil. Identificar claramente el modelo ejecutado y su alcance cerrado de seis clases. No presentar una clasificación forzada como detección de modulación desconocida o rechazo OOD.

**Resultado mínimo para cumplir:**

- Un tono de prueba tiene su pico dentro de un bin de la frecuencia esperada, para la configuración fijada.
- La integral de la PSD coincide con la potencia de referencia correspondiente a la normalización de ventana implementada, con tolerancia relativa ≤ 10⁻⁶ en una prueba numérica de doble precisión. No confundir esa identidad con el error estadístico de estimación de potencia.
- El ancho de banda con ruido se etiqueta como potencia total; cualquier estimación de señal útil tiene método y evaluación propios.
- La salida identifica versión/hash del modelo y no afirma haber ejecutado la red o LBAP cuando solo se ejecutó el SVM.
- Ausencia de señal, detección indeterminada y clasificación fuera de alcance tienen estados distintos de una clase válida.

**Evidencia:** pruebas espectrales, ejemplos y esquema final de salida.  
**Dependencias:** N1.1 y N1.2.

## 5. Nivel 2 — Validación independiente y demostración

### N2.1 — Banco independiente de detección y SNR

**Modificar:** añadir un experimento reproducible de validación del monitor, separado de sus datos de calibración.

**Diseño mínimo propuesto:**

- 5.000 unidades de ruido para calibración y otras 5.000 para evaluación, independientes, por condición principal de ruido/ganancia que se declare compatible.
- Al menos 100 padres por combinación de seis clases, ocho SNR y dos regímenes de canal: 9.600 realizaciones de señal. Si se prueban varias ganancias de una misma realización, conservar la agrupación por padre al calcular incertidumbre.
- Tres ganancias globales representativas: 0,5; 1; 2. Aplicarlas a señal y referencia de manera físicamente coherente; documentar cualquier clipping.
- Ensayo adicional de deriva de ruido ×2 sin actualizar la referencia, como prueba de límite, y ensayo con recalibración. No exigir robustez a una deriva desconocida que el método no promete resolver.

**Resultado mínimo para cumplir:**

- Configuración fijada antes de evaluar, cero etiquetas verdaderas en inferencia y reporte de `P_D`, `P_FA` e intervalos Wilson del 95%.
- Informar por separado falsa alarma por ventana y por archivo cuando ambas decisiones existan.
- **Meta nominal propuesta:** objetivo `P_FA = 1%`, límite superior Wilson ≤ 2% con referencia compatible; límite inferior Wilson de `P_D` ≥ 95% en cada celda ID de clase/SNR ≥ 0 dB, con al menos 100 padres por celda.
- No se exige detección perfecta a SNR negativa; se exige curva completa, conteos y explicación de límites.
- La SNR cumple o declara incumplida la meta N1.3; la deriva no calibrada no se oculta del informe.

Un método que detecta todo no cumple por violar falsa alarma. Un método que declara todo indeterminado tampoco cumple la meta operativa.

**Evidencia:** configuración, CSV por observación, métricas agregadas y curvas.  
**Dependencias:** nivel 1 y N0.2.

### N2.2 — Prueba de extremo a extremo con archivos desconocidos

**Modificar:** construir un conjunto de archivos con etiquetas guardadas fuera de la entrada del monitor. Incluir señales Python y GNU Radio, ruido puro, diferentes potencias y archivos largos con inicio de transmisión variable. Documentar el supuesto de una sola modulación por archivo; si hay múltiples transmisiones, segmentarlas o declararlas fuera de alcance.

**Resultado mínimo para cumplir:**

- Al menos 120 archivos nuevos con señal —20 por clase— más 40 de ruido, con cobertura de dos regímenes de canal, tres ganancias y SNR bajas, medias y altas.
- Se ejecutan todos con una configuración común y metadatos/referencias obtenidos sin conocer las etiquetas. No se ajusta un umbral por archivo después de revelar la verdad.
- 100% de entradas válidas procesadas sin fallo no controlado y 100% con las cinco salidas y estados requeridos.
- La tabla final incluye verdaderos positivos, falsas alarmas, señales omitidas, indeterminados, clases y confianza. Reportar cobertura y exactitud condicionada, además de la tasa de señales correctamente detectadas **y** clasificadas sobre todas las señales presentes.
- Guardar matrices de confusión con estados de omisión/abstención cuando existan; no eliminar los fallos de detección para presentar únicamente el mejor subconjunto.

Este conjunto es una demostración de integración, no sustituye las muestras mayores de N2.1 ni las de clasificación N2.3. La tarea puede revelar rendimiento insuficiente; eso debe impedir anunciar una demostración robusta hasta corregirlo o limitar el alcance explícitamente.

**Evidencia:** tabla del desafío, comandos y salidas de cada archivo.  
**Dependencias:** N2.1.

### N2.3 — Clasificación E2–E6 y control de regresión

**Modificar:** unificar evaluación de SVM256, SVM512 y modelos avanzados sobre las mismas realizaciones, distinguiendo CNN histórica y red multidominio. Mantener E1 enlazado a N2.1.

**Resultado mínimo para cumplir:**

- Al menos 2.400 padres de señal por test nominal y por test con cambio de canal, con las seis clases y ocho SNR representados de forma balanceada.
- Accuracy, precision, recall, F1 macro, resultados por clase/SNR y matrices de confusión calculados desde predicciones identificables por padre.
- Recargar un modelo sin modificarlo reproduce las clases de referencia; para logits usar una tolerancia explícita apropiada al hardware. Si hay discrepancias de clase, investigarlas y registrar su efecto.
- Cambios exclusivos de interfaz o reporte no alteran las predicciones del clasificador sobre el mismo vector de entrada y preprocesamiento. Una modificación intencional del modelo/preprocesamiento se evalúa como nueva versión.
- Presentar rendimiento nominal y de cambio de canal sin mezclar datasets históricos distintos en una sola comparación.

**No se exige superar 51,58% para cumplir la tarea.** Se exige evaluar correctamente. Para anunciar una mejora de modelo se aplica la regla estadística N3.4 sobre una evaluación final nueva.

**Evidencia:** métricas, predicciones, confusiones, curvas y registro de modelos.  
**Dependencias:** N0.2; para rendimiento integral, nivel 1.

### N2.4 — Suite, GNU Radio y ejecución reproducible

**Modificar:** completar pruebas que cubran los fallos detectados y documentar requisitos de Radioconda. Evitar pruebas que solo repitan la implementación sin comprobar el comportamiento esperado.

**Resultado mínimo para cumplir:**

- Cero fallos no explicados en las pruebas aplicables. Los tests omitidos tienen motivo y no cuentan como aprobados.
- Los tres flowgraphs compilan y la simulación sin interfaz genera un archivo utilizable por el monitor.
- Una ejecución documentada desde un directorio de resultados limpio reproduce la demostración sin editar el código ni ajustar manualmente cada archivo.
- Se informa si se necesitaron ejecuciones separadas por restricciones del entorno. No afirmar que una suite completa pasó en una sola ejecución si no ocurrió.

**Evidencia:** logs de pytest, compilación, simulación y comando de demostración.  
**Dependencias:** N1.5 y N2.2.

## 6. Nivel 3 — Investigación adaptativa y validez científica

### N3.1 — Política nueva serializable y desplegable

**Archivos principales:** `src/research_models.py`, `src/adaptive.py`, `experiments/exp12_paper_research.py`, integración opcional en `src/monitor.py`.

**Modificar:** guardar selector, calibradores, normalización, orden de características y clases, modelos asociados, parámetros de coste, restricciones de acciones y hashes como un paquete de inferencia. Separar claramente la política nueva del artefacto adaptativo histórico.

**Resultado mínimo para cumplir:**

- Recargar en un proceso nuevo reproduce el 100% de acciones y etiquetas de un lote de validación de al menos 100 padres, con las rutas disponibles representadas.
- La decisión inicial solo recibe las primeras 256 muestras y metadatos permitidos. Alterar las muestras 257–512 manteniendo el prefijo fijo no cambia la acción inicial.
- Elegir `a0` o `a2` no ejecuta previamente SVM512 ni utiliza sus predicciones; elegir `a0` no ejecuta previamente la red. Verificar llamadas, no solo la forma de la matriz.
- Modelo, calibrador o caché incompatibles se rechazan por metadatos/hashes.
- Existe un comando real de inferencia del paquete. Si se añade al monitor, se declara expresamente el modo seleccionado.

**Evidencia:** paquete guardado, pruebas de recarga/causalidad y demo.  
**Dependencias:** N0.2 y N1.4.

### N3.2 — Datos independientes para aprender recuperabilidad

**Modificar:** separar el entrenamiento del experto del conjunto usado para estimar sus aciertos contrafactuales al entrenar el selector, o generar predicciones fuera de pliegue. Mantener calibración y selección fuera del test final. Fortalecer invalidación de cachés y manifiestos.

**Resultado mínimo para cumplir:**

- Cada objetivo de recuperabilidad empleado para desarrollar la política procede de un padre que el experto correspondiente no utilizó para ajustar sus pesos, o de una predicción fuera de pliegue identificable.
- Cero cruces prohibidos de padres y ninguna selección por resultados del test final.
- Calibradores, pesos, hiperparámetros y baseline primario se congelan antes de evaluar.
- Las cachés identifican datos, orden de filas, código, modelo y preprocesamiento. Una incompatibilidad fuerza regeneración o rechazo.

No afirmar retrospectivamente que la evaluación anterior estaba contaminada solo por compartir datos de entrenamiento entre experto y selector: el problema identificado es riesgo de optimismo en los objetivos de la política. Esta tarea corrige ese riesgo y permite una nueva evaluación más sólida.

**Evidencia:** auditoría de particiones, manifiesto congelado e informe de cachés.  
**Dependencias:** N0.2 y N3.1.

### N3.3 — Medición de latencia completa

**Modificar:** medir cada ruta real incluyendo características, predicción inicial, scores adicionales, calibración, construcción de entradas, selector, transferencias CPU/GPU y rama elegida. Sincronizar GPU de forma adecuada. Informar aparte carga inicial, entrada/salida y gráficas; distinguir latencia de clasificación de la del monitor completo.

**Resultado mínimo para cumplir:**

- Al menos tres sesiones de medición y 300 observaciones por política en cada sesión, batch 1, con calentamiento y orden de políticas aleatorizado/reproducible.
- Guardar tiempos individuales, hardware, versiones, hilos y modo de ejecución; reportar media, mediana y percentil 95 por sesión.
- Separar cómputo medido de adquisición virtual `1000 × N/fs` ms. Documentar si se modela procesamiento secuencial o solapado con adquisición; el barrido de `fs` es un análisis de coste, no una validación automática en nuevas bandas físicas.
- El coste del selector incluye construir lo que necesita; no se sustituye por el tiempo de `choose()` sobre entradas precalculadas.
- El resultado permite aceptar o rechazar el ahorro previo de 3,59 ms. No se exige reproducir esa cifra para cerrar correctamente el experimento.

**Para anunciar menor latencia:** el intervalo del 95% de la diferencia media, definido como política menos baseline, debe estar completamente por debajo de cero bajo el protocolo fijado. Incorporar variación entre sesiones o limitar explícitamente la conclusión a la sesión medida.

**Evidencia:** tiempos por observación y sesión, análisis y coste de adquisición separado.  
**Dependencias:** N3.1.

### N3.4 — Comparación adaptativa contra los baselines pertinentes

**Modificar:** comparar LBAP con SVM256, SVM512, red sola, política de confianza y política de SNR ciega/estadísticas espectrales. Añadir comparación contra la red sola y contra el mejor baseline fijo seleccionado en validación, no solo contra SVM.

**Resultado mínimo para cumplir el experimento:**

- Evaluación pareada sobre los mismos padres, con al menos 2.400 en ID y 2.400 con cambio de canal.
- Reportar diferencias de accuracy, F1 macro, tiempo y utilidad; intervalos con al menos 10.000 réplicas bootstrap estratificadas por clase/SNR y agrupadas por padre cuando existan derivados.
- Recalcular F1 macro dentro de cada réplica; no tratarlo como una media de aciertos individuales.
- Fijar comparación y métrica primarias antes del test; marcar el resto como exploratorio o aplicar corrección por comparaciones múltiples.
- Para afirmaciones sobre estabilidad del método, al menos tres semillas de entrenamiento/selección. Tres semillas son un mínimo exploratorio: reportar su dispersión y no confundirla con incertidumbre por muestreo del test.

**Reglas para afirmar éxito científico:**

| Afirmación | Evidencia mínima |
|---|---|
| Mayor exactitud | Límite inferior del IC95% de Δaccuracy > 0 frente al comparador declarado. |
| Mayor utilidad | Límite inferior del IC95% de Δutilidad > 0, con λ y coste fijados previamente. |
| Menor latencia con calidad preservada | IC95% de Δtiempo < 0 y límites inferiores de Δaccuracy y ΔF1 macro > −0,01, margen interno propuesto de un punto porcentual fijado antes del test. |
| Superior bajo cambio de canal | Regla correspondiente satisfecha específicamente en ese test; el resultado nominal no la sustituye. |
| Superior a todos los baselines | Contrastes pertinentes para todos, con control de multiplicidad; no basta elegir el baseline más débil. |

Conservar el resultado como negativo o inconcluso si el intervalo cruza el umbral. No elegir otra variante mirando el test y volver a llamarlo «ciego».

**Evidencia:** comparaciones pareadas, intervalos, protocolo y decisión por afirmación.  
**Dependencias:** N3.2 y N3.3.

### N3.5 — Separar mejora de arquitectura y mayor entrenamiento

**Modificar:** documentar presupuesto de cada modelo: padres, datos sintéticos, distribución de SNR/canal, épocas, selección y tiempo. Para atribuir una mejora a la arquitectura, añadir comparación controlada con datos y protocolo de selección equivalentes; si no se ejecuta, limitar la conclusión a los sistemas entrenados completos.

**Resultado mínimo para cumplir:**

- Tabla de presupuestos para todos los modelos comparados.
- Cada afirmación causal sobre arquitectura tiene una comparación que controla datos y selección, o se retira/reformula.
- Si se realiza la comparación controlada, usar al menos tres semillas y N3.4 para afirmar mejora.

**Evidencia:** tabla, configuración y resultados controlados, o reformulación explícita del alcance.  
**Dependencias:** N0.1; N3.2 si se realizan nuevos entrenamientos.

### N3.6 — ADC y calibración: métricas e hipótesis correctas

**Modificar:** distinguir cuantización uniforme de saturación, conservar rango fijo y ganancia declarada. Comparar entrenamiento limpio, 8 bits, 6 bits y mixto en float, 8, 6, 5 y 4 bits. Definir si Brier/NLL son binarios de acierto o multiclase y si ECE usa igual anchura o igual frecuencia.

**Resultado mínimo para cumplir:**

- Mismos padres entre resoluciones y ninguna variante derivada cruzando particiones.
- Rango del ADC fijado en desarrollo, con clipping, saturación y SQNR por condición; no adaptar el rango al test usando etiquetas.
- Las fórmulas del texto coinciden exactamente con las funciones y métricas guardadas.
- La comparación informa que, en los resultados históricos a 4 bits, el mixto obtuvo 44,96% frente a 46,29% del limpio; no se anuncia mejora mixta con esos números.
- Para una nueva afirmación de mejora del entrenamiento mixto, repetir con al menos tres semillas y contraste pareado previamente fijado, manteniendo el resultado negativo si no supera el criterio.
- Si se pretende atribuir la pérdida exclusivamente a cuantización, añadir un control que separe clipping; de lo contrario, describirla como degradación conjunta del ADC simulado.

**Evidencia:** CSV por resolución, calibración, clipping y conclusión de cada hipótesis.  
**Dependencias:** N0.2 y N3.5 para afirmaciones nuevas.

## 7. Nivel 4 — Documentación coherente y entrega

### N4.1 — Corregir cifras y afirmaciones científicas

**Archivos principales:** `README.md`, `docs/INFORME_FINAL.md`, `docs/FASE7_MONITOR.md`, `docs/RESEARCH_GAP.md`, presentación y secciones del artículo.

**Modificar:**

- Corregir detección a −10 dB: el resultado histórico ID es 123/600 = 20,5%, no aproximadamente 50%.
- Resolver la discrepancia del ancho de banda del ejemplo Python y explicar las diferencias de ventana/longitud/ruido entre reportes.
- Retirar el «100% cumplido» hasta tener la matriz de requisitos y evidencias cerrada.
- Distinguir CNN histórica, red multidominio, monitor SVM y política adaptativa; asociar cada cifra a su dataset y ejecución.
- Sustituir «superioridad» por resultado puntual, inconcluso o negativo cuando corresponda; reconocer que una entrada más simple rindió mejor nominalmente.
- Corregir definiciones de calibración, coste modelado frente a tiempo medido y supuestos de SNR.
- Revisar afirmaciones teóricas absolutas: BatchNorm no implica por sí sola fuga en evaluación; distinguir símbolos de baud; no prometer ausencia exacta de ISI con un filtro RRC truncado y canal dispersivo; no prometer latencias futuras concretas sin medición.
- Retirar atribuciones causales sobre CFO, recuperabilidad o clipping que no tengan un control específico, o identificarlas como hipótesis.

**Resultado mínimo para cumplir:** cada cifra central tiene fuente rastreable y coincide con ella al redondeo mostrado; cada afirmación principal se clasifica como medida, inferida, hipotética o futura; no quedan contradicciones conocidas entre código y documentos.

**Evidencia:** matriz «afirmación → artefacto → limitación» y documentos actualizados.  
**Dependencias:** resultados de los niveles aplicables; las correcciones conocidas pueden empezar inmediatamente.

### N4.2 — Verificar antecedentes y moderar la novedad

**Modificar:** revisar referencias empleadas para comparar el proyecto con otros métodos. Verificar autores, título, DOI y pasaje que respalda cada afirmación sobre SNR oracular, CFO, horizonte de observación, cómputo o ADC. No equiparar una regla genérica de confianza con una reproducción de BEACON.

**Resultado mínimo para cumplir:**

- Todas las comparaciones bibliográficas centrales tienen fuente primaria y página/sección verificable.
- Una comparación conceptual se distingue de una reproducción experimental.
- Expresiones como «primera formulación» se respaldan con una búsqueda documentada o se sustituyen por una contribución más acotada. Una búsqueda limitada no demuestra prioridad universal.

**Evidencia:** matriz bibliográfica revisada y registro de búsqueda cuando se use para sostener novedad.  
**Dependencias:** N4.1 para el alcance de las afirmaciones.

### N4.3 — Paquete de entrega y revisión visual

**Modificar:** reunir instrucciones ejecutables, dependencias, modelos usados, resultados seleccionados, informe, presentación y demostración. Compilar y revisar visualmente el PDF y las figuras finales.

**Resultado mínimo para cumplir:**

- Un comando de demostración y uno de validación funcionan desde una carpeta de resultados nueva sin modificar fuentes.
- Los modelos y datasets incluidos o regenerables coinciden con sus manifiestos; no se requiere adivinar cuál versión usar.
- PDF sin referencias rotas, tablas cortadas ni figuras ilegibles; presentación consistente con las cifras finales.
- Matriz de los nueve entregables de la guía con ruta y estado. Los requisitos pendientes no aparecen como completos.
- Registro de dependencias y de la última ejecución de pruebas, con limitaciones del entorno.

**Evidencia:** lista de entrega, logs y artefactos revisados.  
**Dependencias:** N2.4, N4.1 y N4.2 si se entrega el artículo.

## 8. Nivel 5 — Mejoras opcionales, después de cerrar la base

### N5.1 — Diagnóstico de generalización y clases difíciles

**Modificar:** ampliar diagnósticos de CFO, multitrayecto y confusión entre 16QAM/64QAM, con cambios controlados de un factor y realizaciones pareadas. Investigar la caída bajo cambio de canal antes de añadir complejidad.

**Resultado mínimo para cumplir:** al menos un experimento controlado por factor investigado, métricas por clase/SNR, comparación pareada e hipótesis aceptada, rechazada o inconclusa. Una gráfica visual sin conteos ni control no basta.

**Evidencia:** diagnóstico y propuesta de intervención sustentada.  
**Dependencias:** N2.3 y N3.5.

### N5.2 — Reentrenamiento o nueva arquitectura justificados

**Modificar:** únicamente el componente que el diagnóstico señale, manteniendo baselines y protocolo. Posibles cambios: distribución de entrenamiento, sincronización, características o arquitectura; no introducir todos simultáneamente si se quiere conocer su efecto.

**Resultado mínimo para cumplir el experimento:** al menos tres semillas, presupuesto registrado, validación para selección y test final independiente. Aplicar N3.4 para afirmar mejora; conservar el modelo de referencia si el candidato no satisface el objetivo fijado.

No es requisito del curso incorporar otra arquitectura ni conseguir una mejora positiva en esta fase.

**Evidencia:** comparación antes/después y decisión de conservar o descartar.  
**Dependencias:** N5.1.

## 9. Criterios globales para declarar el trabajo cerrado

### 9.1 Monitor corregido

Se puede declarar corregido cuando N1.1–N1.6 cumplen sus contratos y N2.1–N2.4 aportan validación. Cualquier meta operativa no alcanzada se declara como limitación; no se sustituye una detección calibrada por umbrales escogidos después de conocer cada señal.

### 9.2 Entrega del curso preparada

Requiere niveles 0, 1, 2 y 4 aplicables, con cinco salidas implementadas, E1–E6 evaluados, demostración reproducible y nueve entregables identificados. La preparación del paquete no garantiza la calificación del profesor. Si la SNR o la detección dependen de referencias adicionales, el alcance y esos requisitos deben estar visibles en la demostración y el informe.

### 9.3 Contribución adaptativa validada

Requiere nivel 3 completo para las afirmaciones conservadas. Si LBAP no supera los criterios, se presenta como estudio experimental con resultado negativo o inconcluso, sin bloquear por ello una entrega de curso válida.

### 9.4 Artículo preparado para revisión

Requiere trazabilidad, metodología reproducible, resultados completos, comparación pertinente y bibliografía verificada. «Preparado para revisión» no significa aceptación ni novedad certificada. Una ventaja de un acierto, un tiempo incompleto o una comparación únicamente contra el baseline más débil no bastan para sostener superioridad general.

## 10. Registro de seguimiento

Todas las tareas comienzan pendientes. Cambiar su estado únicamente al adjuntar evidencia.

| ID | Tarea | Estado | Evidencia de cierre |
|---|---|---|---|
| N0.1 | Referencia e integridad | Cumplido | `results/nivel0_evidencia/manifest_historical_state.json`, `integrity_verification.json`, commit `7e81d0df` |
| N0.2 | Protocolo y particiones | Cumplido | `results/nivel0_evidencia/partition_independence_audit.json`, `configs/validation_protocol_frozen.json`, `docs/NIVEL0_EVIDENCIA_Y_PROTOCOLO.md` |
| N1.1 | Detección calibrada por ventana/archivo | Cumplido | `src/detection.py`, `src/monitor.py`, `results/nivel1_monitor/historical_demos_audit.json`, `results/nivel1_monitor/demos/` |
| N1.2 | Frecuencia y compatibilidad | Cumplido | `src/spectrum.py`, `src/monitor.py`, `tests/test_nivel0_nivel1_acceptance.py` |
| N1.3 | SNR y límites | Cumplido | `src/snr_estimation.py`, `results/nivel1_monitor/n1_3_snr_evaluation.json`, `n1_3_snr_by_class_snr.csv` |
| N1.4 | Confianza final | Cumplido | `results/nivel1_monitor/monitor_calibrators.joblib`, `n1_4_calibration_metrics.json`, `n1_4_reliability_bins.csv`, `n1_4_reliability_curves.png` |
| N1.5 | Entradas y CLI | Cumplido | `src/monitor.py`, `tests/test_monitor_cli.py`, `tests/test_nivel0_nivel1_acceptance.py` |
| N1.6 | Espectro y alcance | Cumplido | `src/spectrum.py`, `src/monitor.py`, `tests/test_nivel0_nivel1_acceptance.py` |
| N2.1 | Validación de detección/SNR | Cumplido | `results/nivel2_validacion/n2_1_bank/metrics.json`, `pfa_by_gain_and_mode.csv`, `drift_results.csv`, `pd_and_snr_curves.png` |
| N2.2 | Archivos desconocidos | Cumplido | `results/nivel2_validacion/n2_2_unknown_challenge/challenge_summary.json`, `challenge_results_table.csv`, `confusion_with_omissions.png` |
| N2.3 | Clasificación E2–E6 | Cumplido | `results/nivel2_validacion/n2_3_classification/unified_metrics.json`, `regression_reload_check.json`, `unified_benchmark_comparison.png` |
| N2.4 | Suite y GNU Radio | Cumplido | `experiments/run_n2_4_reproducible_demo.py`, `results/nivel2_validacion/n2_4_clean_demo/`, `n2_4_pytest_summary.json`, `tests/test_nivel2_validation.py` |
| N3.1 | Política desplegable | Cumplido | `src/adaptive.py`, `results/nivel3_adaptativo/lbap_deployable_package.joblib`, `n3_1_package_verification.json`, `tests/test_nivel3_adaptive.py` |
| N3.2 | Recuperabilidad y cachés | Cumplido | `results/nivel3_adaptativo/caches/`, `n3_2_cache_and_partition_audit.json`, `tests/test_nivel3_adaptive.py` |
| N3.3 | Latencia completa | Cumplido | `results/nivel3_adaptativo/n3_3_latency_sessions.json`, `n3_3_latency_individual_calls.csv` |
| N3.4 | Comparaciones estadísticas | Cumplido (resultado mixto/inconcluso) | `results/nivel3_adaptativo/n3_4_paired_bootstrap.json` (supera `always_a2` en utilidad ID; cruza cero en Acc ID vs `a2` y proxy I2; cae vs `a0` en utilidad OOD) |
| N3.5 | Presupuesto y arquitectura | Cumplido | `results/nivel3_adaptativo/n3_5_training_budgets.json` |
| N3.6 | ADC y calibración | Cumplido (resultado negativo 4 bits) | `results/nivel3_adaptativo/n3_6_adc_and_calibration_audit.json` (`Exp C` mixto `44.96%` < `Exp A` limpio `46.29%`) |
| N4.1 | Cifras y afirmaciones | Cumplido | `docs/MATRIZ_AFIRMACIONES_Y_ENTREGABLES.md`, `README.md`, `docs/INFORME_FINAL.md`, `docs/RESEARCH_GAP.md`, `docs/paper/sections/*.tex` |
| N4.2 | Bibliografía y novedad | Cumplido | `docs/MATRIZ_LITERATURA.md`, `docs/paper/sections/01_intro_related.tex`, `docs/paper/references.bib` |
| N4.3 | Entrega y revisión visual | Cumplido | `docs/MATRIZ_AFIRMACIONES_Y_ENTREGABLES.md`, `docs/PRESENTACION_DIAPOSITIVAS.md`, `docs/paper/paper_ieee.pdf` |
| N5.1 | Diagnóstico adicional | Cumplido | `experiments/exp_n5_1_generalization_diagnostic.py`, `results/nivel5_diagnostico/{n5_1_diagnostic_report.json,n5_1_diagnostic_summary.png}`, `tests/test_nivel5_diagnostic.py` |
| N5.2 | Mejora de modelos | Diferido (opcional no requerido) | Propuestas `P1` (`derotated_*` GNU Radio) y `P2` (ecualización ciega CMA/margen `64QAM`) documentadas en `N5.1` para iteración futura |

### Registros de cierre (16 de noviembre de 2026)

```text
ID: N0.1
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: .gitignore, experiments/verify_n0_protocol.py
Versión o hashes: git HEAD inicial 7e81d0df252a52b69bed49d73467b4fa4618adc2; verify_n0_protocol.py sha256=aa90de47a4a30e1d
Comando o procedimiento: .\.venv\Scripts\python.exe -m experiments.verify_n0_protocol
Configuración y semillas: Sin muestreo aleatorio; inventario determinista SHA-256
Datos/particiones utilizados: src/, experiments/, tests/, configs/, grc/, results/, data/{e2_dataset,adaptive_dataset,fase6_holdout_sellado,fase3_smoke}
Resultado mínimo exigido: Manifiesto SHA-256 completo, cifras históricas preservadas (SVM256 43,75%, SVM512 46,04%, red multidominio 51,5833%, LBAP 51,625%) y detección de alteración en copia temporal sin tocar originales.
Resultado observado: 315 archivos inventariados, integridad 315/315 aprobada, tamper_check_passed=true sobre copia temporal, cifras exactas registradas.
Intervalos o tolerancias aplicables: Comparación exacta de hash SHA-256.
Ruta de evidencias: results/nivel0_evidencia/manifest_historical_state.json, results/nivel0_evidencia/integrity_verification.json
Limitaciones: Repositorio git local sin remoto; demos históricas de results/monitor carecen de archivo de referencia de ruido original guardado y las fases 1-3 tempranas carecen de manifiesto unificado retroactivo.
Conclusión autorizada por la evidencia: El estado histórico de referencia queda trazado, inmutable y protegido contra alteraciones silenciosas.
Siguiente dependencia habilitada: N0.2, N3.5
```

```text
ID: N0.2
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: configs/validation_protocol_frozen.json, docs/NIVEL0_EVIDENCIA_Y_PROTOCOLO.md, experiments/verify_n0_protocol.py
Versión o hashes: verify_n0_protocol.py sha256=aa90de47a4a30e1d
Comando o procedimiento: .\.venv\Scripts\python.exe -m experiments.verify_n0_protocol
Configuración y semillas: Semillas reservadas congeladas 2026111601..2026111604 en configs/validation_protocol_frozen.json
Datos/particiones utilizados: oracle/manifest.jsonl, index.csv, observations/*.npz y supervision/*.csv de los 4 datasets + results/adaptive/development_partitions.json
Resultado mínimo exigido: Cero padres compartidos entre particiones independientes, prefijos 256/512 exactos, criterios congelados antes de abrir evaluaciones nuevas y distinción explícita señal presente vs monitor conjunto.
Resultado observado: shared_parents_count=0 (interno y entre datasets), all_prefixes_exact=true, val_cal (1200) y val_select (1200) disjuntas con train/test/test_shift.
Intervalos o tolerancias aplicables: Igualdad exacta de prefijos e intersección nula de padres y hashes iq512.
Ruta de evidencias: results/nivel0_evidencia/partition_independence_audit.json, configs/validation_protocol_frozen.json, docs/NIVEL0_EVIDENCIA_Y_PROTOCOLO.md
Limitaciones: En el experimento histórico exp12_paper_research, el selector LBAP usó recuperabilidad in-sample de train (pendiente de corrección en N3.2).
Conclusión autorizada por la evidencia: Las particiones base y los criterios de aceptación quedan formalmente auditados y congelados.
Siguiente dependencia habilitada: N1.1, N1.2, N2.1, N2.3, N3.1, N3.2, N3.6
```

```text
ID: N1.1
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: src/detection.py, src/monitor.py, experiments/eval_n1_monitor_validation.py, tests/test_nivel0_nivel1_acceptance.py
Versión o hashes: src/detection.py sha256=3f973d4f55007aa7, src/monitor.py sha256=252d3e08a6ef0c46
Comando o procedimiento: .\.venv\Scripts\python.exe -m experiments.eval_n1_monitor_validation y .\.venv\Scripts\python.exe -m src.monitor --input results/nivel1_monitor/demos/demo_late_tx_16qam_12db.c64 --noise-ref results/nivel1_monitor/demos/demo_late_tx_noise_ref.c64 --fs-hz 8000
Configuración y semillas: pfa_file=0.01 con corrección Šidák multi-ventana; semilla reservada 2026111601
Datos/particiones utilizados: Demos históricas (py_qpsk_15db.c64, grc_sim_output.c64) y 2 demos nuevas con referencia independiente (demo_py_qpsk_15db.c64, demo_late_tx_16qam_12db.c64)
Resultado mínimo exigido: Estado indeterminado sin calibración, invarianza a ganancia entre señal y referencia, detección multi-ventana por archivo y demos documentadas sin ajustar umbrales con etiquetas.
Resultado observado: Sin calibración devuelve indeterminate_uncalibrated; escalar por 0.25/1.0/4.0 conserva normalized_statistic=10.874858438219048 (rtol<1e-12); demo tardía detecta ventanas activas [3, 4] omitiendo 0..2 de ruido.
Intervalos o tolerancias aplicables: Invarianza de estadística normalizada dentro de rtol=1e-9.
Ruta de evidencias: results/nivel1_monitor/historical_demos_audit.json, results/nivel1_monitor/demos/demo_py_qpsk_15db.json, results/nivel1_monitor/demos/demo_late_tx_16qam_12db.json
Limitaciones: Las dos demos históricas no tienen archivo de referencia de ruido original guardado, por lo que sin umbral manual quedan correctamente en estado indeterminate_uncalibrated. La meta cuantitativa de P_FA/P_D con banco independiente completo corresponde a N2.1.
Conclusión autorizada por la evidencia: El detector del monitor calibra desde --noise-ref, es invariante a escala, controla P_FA por archivo y evita falsos NO_SIGNAL sin calibración.
Siguiente dependencia habilitada: N1.3, N1.4, N1.6, N2.1
```

```text
ID: N1.2
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: src/spectrum.py, src/monitor.py, tests/test_nivel0_nivel1_acceptance.py
Versión o hashes: src/spectrum.py sha256=725d5df6027f0f59, src/monitor.py sha256=252d3e08a6ef0c46
Comando o procedimiento: .\.venv\Scripts\python.exe -m pytest tests/test_nivel0_nivel1_acceptance.py -k n1_2 -q
Configuración y semillas: Verificación determinista de unidades, duplicación de fs_hz y compatibilidad (fs_hz=8000 Hz, sps=8, symbol_rate_hz=1000 Hz)
Datos/particiones utilizados: Señales sintéticas de prueba y metadatos sidecar/.npz
Resultado mínimo exigido: Ciclos/muestra sin fs conocida, escalado exacto al duplicar fs sin cambiar potencia integrada, rechazo de fs no positiva/no finita y estado OUT_OF_SCOPE ante muestreo incompatible.
Resultado observado: Todas las aserciones aprobadas; sin fs devuelve bandwidth_unit="ciclos/muestra" y OUT_OF_SCOPE_MISSING_FS; con fs=16000 Hz o sps=4 devuelve OUT_OF_SCOPE_INCOMPATIBLE_SAMPLING.
Intervalos o tolerancias aplicables: Tolerancia relativa 1e-12 en escalado de ancho de banda e invariancia de potencia integrada.
Ruta de evidencias: tests/test_nivel0_nivel1_acceptance.py
Limitaciones: No incluye remuestreo automático en inferencia; señales fuera de fs=8000 Hz / sps=8 se marcan explícitamente fuera de alcance del clasificador.
Conclusión autorizada por la evidencia: El manejo de frecuencia de muestreo es explícito, verificable y separa espectro normalizado de compatibilidad del clasificador.
Siguiente dependencia habilitada: N1.3, N1.6
```

```text
ID: N1.3
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: src/snr_estimation.py, src/monitor.py, experiments/eval_n1_monitor_validation.py, tests/test_nivel0_nivel1_acceptance.py
Versión o hashes: src/snr_estimation.py sha256=eb2d3c4de4f43b6e, eval_n1_monitor_validation.py sha256=1bd8ee02f7c11fca
Comando o procedimiento: .\.venv\Scripts\python.exe -m experiments.eval_n1_monitor_validation
Configuración y semillas: configs/exp01_detection.json con semilla reservada 2026111601; 5000 ventanas de referencia, 9600 realizaciones (2400 en ID SNR >= 0 dB)
Datos/particiones utilizados: Banco controlado de 256 muestras (6 clases x 8 SNR x 2 regímenes x 100 realizaciones)
Resultado mínimo exigido: Estados explícitos ante ausencia/incompatibilidad/negatividad; meta interna ID SNR >= 0 dB: cobertura >= 90% y MAE <= 3 dB sin ocultar faltantes.
Resultado observado: En ID SNR >= 0 dB (n=2400): n_valid=2400, cobertura=1.000 (>=0.90), MAE=0.3319 dB (<=3.0 dB), sesgo=+0.0087 dB. A baja SNR (-20 a -10 dB) reporta cobertura (1% a 22.2%) y conteo de omisiones sin ocultarlas.
Intervalos o tolerancias aplicables: Meta interna cobertura >= 90% y MAE <= 3.0 dB en ID SNR >= 0 dB.
Ruta de evidencias: results/nivel1_monitor/n1_3_snr_evaluation.json, results/nivel1_monitor/n1_3_snr_by_class_snr.csv
Limitaciones: Requiere estacionariedad de ruido e igualdad de banda/ganancia entre señal y referencia; deriva no recalibrada sesga la resta de potencias.
Conclusión autorizada por la evidencia: El estimador de SNR cumple la meta interna bajo referencia compatible y comunica explícitamente sus límites y casos no estimables.
Siguiente dependencia habilitada: N2.1
```

```text
ID: N1.4
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: src/monitor.py, experiments/eval_n1_monitor_validation.py, tests/test_nivel0_nivel1_acceptance.py
Versión o hashes: monitor_calibrators.joblib sha256=2dae5d4dd8050835, src/monitor.py sha256=252d3e08a6ef0c46
Comando o procedimiento: .\.venv\Scripts\python.exe -m experiments.eval_n1_monitor_validation
Configuración y semillas: Ajuste en val_cal (seed 2026111601); bootstrap 2000 réplicas para IC95%
Datos/particiones utilizados: data/adaptive_dataset: val_cal (1200 padres), test (2400 ID) y test_shift (2400 cambio de canal)
Resultado mínimo exigido: Correspondencia exacta modelo/características/longitud/agregación por SHA-256, sin recorte 20-99%, >=1200 unidades cal y >=2400 eval ID/shift, ECE10 <= 0.05 y Brier no peor que base en ID.
Resultado observado: En test_id (n=2400): single_window_256 ECE10=0.0083 (Brier 0.2461 vs 0.3142 base, NLL 0.6853); majority_vote_2x256 ECE10=0.0348 (Brier 0.2011 vs 0.3103 base, NLL 0.5922); single_window_512 ECE10=0.0155 (Brier 0.2483 vs 0.3044 base, NLL 0.6897). En test_shift (n=2400): ECE10=0.1259, 0.1477 y 0.1798 (Brier mejorado respecto a base, pero calibración degradada bajo cambio de canal).
Intervalos o tolerancias aplicables: Meta nominal ID ECE10 <= 0.05 cumplida en las tres configuraciones calibradas (IC95% superior en majority_vote_2x256: 0.0534).
Ruta de evidencias: results/nivel1_monitor/monitor_calibrators.joblib, n1_4_calibration_metrics.json, n1_4_reliability_bins.csv, n1_4_reliability_curves.png
Limitaciones: La calibración solo marca confidence_calibrated=true para 1 ventana (256/512) o voto 2x256 sobre los modelos verificados por SHA-256; >2 ventanas o modelos externos se marcan honestamente como no calibrados. Bajo test_shift la calibración nominal pierde exactitud de probabilidades.
Conclusión autorizada por la evidencia: La salida de confianza elimina el recorte artificial, valida correspondencia exacta de modelo/agregación y satisface ECE10 <= 0.05 en ID.
Siguiente dependencia habilitada: N3.1
```

```text
ID: N1.5
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: src/monitor.py, tests/test_nivel0_nivel1_acceptance.py
Versión o hashes: src/monitor.py sha256=252d3e08a6ef0c46
Comando o procedimiento: .\.venv\Scripts\python.exe -m pytest tests/test_monitor_cli.py tests/test_nivel0_nivel1_acceptance.py -q
Configuración y semillas: Pruebas deterministas de formatos y CLI
Datos/particiones utilizados: Archivos válidos e inválidos .c64, .bin, .npy, .npz, .csv, .txt y señales de 255, 256 y 512 muestras
Resultado mínimo exigido: Caso válido e inválido para los 5 formatos, rechazo de lotes 2D/truncados/NaN, comportamiento explícito 255/256/512 sin zero-padding, modelo ausente sin excepción secundaria y códigos CLI documentados.
Resultado observado: Todos los contratos verificados; errores de entrada retornan código de salida 2 por stderr; modelo ausente retorna status="model_not_found" y confidence=null con código 0.
Intervalos o tolerancias aplicables: Tolerancia absoluta cero para excepciones no controladas o predicciones inventadas.
Ruta de evidencias: tests/test_monitor_cli.py, tests/test_nivel0_nivel1_acceptance.py
Limitaciones: El cargador acepta únicamente una observación por archivo (lotes 2D complejos se rechazan explícitamente).
Conclusión autorizada por la evidencia: La CLI y el cargador de señales cumplen un contrato estricto de entrada, longitud y salida JSON.
Siguiente dependencia habilitada: N2.4
```

```text
ID: N1.6
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: src/spectrum.py, src/monitor.py, tests/test_nivel0_nivel1_acceptance.py
Versión o hashes: src/spectrum.py sha256=725d5df6027f0f59, src/monitor.py sha256=252d3e08a6ef0c46
Comando o procedimiento: .\.venv\Scripts\python.exe -m pytest tests/test_detection_spectrum.py tests/test_nivel0_nivel1_acceptance.py -q
Configuración y semillas: Verificación en doble precisión sobre tono complejo y ventana Hann periódica
Datos/particiones utilizados: Tono de prueba f0=1250 Hz a fs=8000 Hz, demos y pruebas unitarias
Resultado mínimo exigido: Pico de tono dentro de 1 bin, Parseval en doble precisión rtol <= 1e-6, separación de ancho de banda de potencia total vs señal útil, hash de modelo en salida y estados diferenciados de alcance cerrado.
Resultado observado: Error de pico = 0 bins (<=1), identidad de Parseval verificada dentro de rtol <= 1e-6, bandwidth_type="total_power_99_includes_noise" separado de signal_bandwidth_est, model_sha256 y closed_set_classes incluidos en JSON.
Intervalos o tolerancias aplicables: Parseval rtol <= 1e-6; pico espectral <= 1 bin.
Ruta de evidencias: tests/test_nivel0_nivel1_acceptance.py, results/nivel1_monitor/demos/demo_py_qpsk_15db.json
Limitaciones: El estimador de ancho de banda útil asume piso de ruido blanco plano estimado desde --noise-ref; sin referencia informa requires_noise_reference. El clasificador opera en alcance cerrado de 6 clases (sin rechazo OOD de modulaciones desconocidas).
Conclusión autorizada por la evidencia: Las salidas espectrales y de clasificación distinguen explícitamente ruido, potencia total, señal útil y alcance cerrado del modelo.
Siguiente dependencia habilitada: N2.1, N2.2
```

```text
ID: N2.1
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: configs/exp_n2_1_detection_snr_bank.json, experiments/exp_n2_1_detection_snr_bank.py, tests/test_nivel2_validation.py
Versión o hashes: exp_n2_1_detection_snr_bank.py sha256=391d0c5aea5dc155
Comando o procedimiento: .\.venv\Scripts\python.exe -m experiments.exp_n2_1_detection_snr_bank
Configuración y semillas: seed=2026111602; 5000 ruido cal + 5000 ruido eval en ganancias [0.5, 1.0, 2.0]; 9600 padres de señal; escenario deriva térmica x2
Datos/particiones utilizados: Banco independiente de 9600 realizaciones (6 clases x 8 SNR x 2 regímenes ID/shift x 100) + 10000 ruido + deriva x2
Resultado mínimo exigido: P_FA nominal con IC95% Wilson superior <= 0.02; P_D en ID SNR >= 0 dB con mínimo IC95% Wilson inferior >= 0.95; invarianza exacta a escala; SNR ID >= 0 dB cobertura >= 90% y MAE <= 3 dB; registro de deriva sin recalibrar vs recalibrada.
Resultado observado: P_FA ventana 256 = 48/5000 (0.0096, Wilson [0.0072, 0.0127]); P_FA archivo 2x256 Šidák = 44/5000 (0.0088, Wilson [0.0066, 0.0118]) en las 3 ganancias; mínimo IC95% inferior de P_D en las 24 celdas ID SNR >= 0 dB = 0.9630 (>= 0.95); invarianza exacta a escala (6427/6427/6427); SNR ID>=0 dB cobertura = 1.000, MAE = 0.3388 dB; deriva x2 no recalibrada eleva P_FA a 1.0000 y sesga SNR en +3.01 a +4.76 dB, recuperándose al recalibrar con nueva referencia.
Intervalos o tolerancias aplicables: Wilson IC95%; P_FA sup <= 0.02; P_D inf >= 0.95; SNR MAE <= 3.0 dB.
Ruta de evidencias: results/nivel2_validacion/n2_1_bank/{metrics.json,pfa_by_gain_and_mode.csv,drift_results.csv,pd_and_snr_curves.png}
Limitaciones: Válido bajo ruido gaussiano blanco estacionario y ganancia/banda compartida entre señal y --noise-ref; deriva térmica no recalibrada invalida umbral y resta de potencia.
Conclusión autorizada por la evidencia: La calibración de energía con corrección de Šidák e invarianza de escala y la estimación formal de SNR cumplen estadísticamente las tolerancias en banco controlado.
Siguiente dependencia habilitada: N2.2, N4.1
```

```text
ID: N2.2
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: experiments/exp_n2_2_unknown_files_e2e.py, tests/test_nivel2_validation.py
Versión o hashes: exp_n2_2_unknown_files_e2e.py sha256=7c8dd8ef0918b19e
Comando o procedimiento: .\.venv\Scripts\python.exe -m experiments.exp_n2_2_unknown_files_e2e
Configuración y semillas: seed=2026111603; 164 archivos ciegos .c64/.npy/.npz/.csv (120 señal = 20/clase incl. 6 GNU Radio headless QPSK + 40 largas inicio variable + 74 estándar 512, y 44 solo ruido) con referencia compatible por ganancia/origen
Datos/particiones utilizados: Desafío ciego generado ad hoc separado de entrenamiento y calibración
Resultado mínimo exigido: 164 archivos procesados sin error entregando cinco salidas, tabla de resultados completa y matriz de confusión con omisiones (NO_SIGNAL) y falsas alarmas sin ocultar fallos.
Resultado observado: 164/164 procesados sin error con 5 salidas; TP=107, FP=0, FN=13 (baja SNR), TN=44, indeterminados=0; cobertura detección=89.17%, tasa falsa alarma=0.0%; exactitud condicional en detectadas=42.99% y tasa conjunta detectar+acertar=38.33%; en 6 archivos GNU Radio QPSK detección=6/6 y SNR ~33.85–34.06 dB, pero SVM256 predijo BPSK (0/6 aciertos de clase).
Intervalos o tolerancias aplicables: Conteo exacto sobre 164 archivos.
Ruta de evidencias: results/nivel2_validacion/n2_2_unknown_challenge/{challenge_summary.json,challenge_results_table.csv,confusion_with_omissions.png,confusion_with_omissions.csv}
Limitaciones: Supuesto explícito de una única modulación por archivo; las capturas GNU Radio sin ajuste de ganancia/temporización al dominio de SVM256 fueron detectadas correctamente pero confundidas con BPSK.
Conclusión autorizada por la evidencia: El monitor CLI opera de extremo a extremo sin errores sobre formatos heterogéneos y longitudes variables, registrando con transparencia tanto omisiones a baja SNR como la brecha de clasificación sobre capturas GNU Radio.
Siguiente dependencia habilitada: N4.1, N4.3
```

```text
ID: N2.3
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: experiments/exp_n2_3_unified_classification_regression.py, tests/test_nivel2_validation.py
Versión o hashes: exp_n2_3_unified_classification_regression.py sha256=c73b886b55561f0f
Comando o procedimiento: .\.venv\Scripts\python.exe -m experiments.exp_n2_3_unified_classification_regression
Configuración y semillas: SVM256, SVM512, SmallCNN (original y phase_aug en FP32 'highest') y MultiDomainAMCNet (TF32 'high') evaluados por separado sobre ambos benchmarks
Datos/particiones utilizados: data/adaptive_dataset (n=2400 test y test_shift) y data/fase6_holdout_sellado (n=4800 test y test_shift); data/e2_dataset/test
Resultado mínimo exigido: Comparación E2–E6 separada por benchmark sin mezclar datasets, más recarga exacta frente a predicciones históricas y equivalencia monitor vs predictor directo.
Resultado observado: adaptive_dataset (test/shift Acc%): SVM256 43.75/30.33, SVM512 46.04/29.54, SmallCNN orig 42.46/18.67, SmallCNN phase 41.17/19.00, MultiDomainAMCNet 51.58/30.08. fase6_holdout_sellado (test/shift Acc%): SVM256 43.27/26.81, SVM512 47.21/29.06, SmallCNN orig 40.75/19.02, SmallCNN phase 41.54/19.48, MultiDomainAMCNet 50.19/27.77. Regresión de recarga: fase5 0/9600 discrepancias, fase6_holdout 0/38400 en FP32 'highest' (4/38400 bajo TF32 'high'), adaptive_paper_research 0/4800 cambios de clase y diff máx logits = 0.00097 (<=0.002), interfaz monitor vs SVM256 0/200 discrepancias.
Intervalos o tolerancias aplicables: Tolerancia de logits <= 0.002; discrepancias de clase = 0.
Ruta de evidencias: results/nivel2_validacion/n2_3_classification/{unified_metrics.json,regression_reload_check.json,by_class_and_snr.csv,unified_benchmark_comparison.png}
Limitaciones: SmallCNN mostró sensibilidad numérica marginal de 4/38400 casos al cambiar precisión TensorCore TF32 vs FP32; MultiDomainAMCNet usó mayor presupuesto de entrenamiento que SVM y SmallCNN (N3.5).
Conclusión autorizada por la evidencia: Quedan reproducidas exactamente todas las métricas de E2–E6 y separados ambos benchmarks sin contradicciones.
Siguiente dependencia habilitada: N3.5, N4.1
```

```text
ID: N2.4
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: experiments/run_n2_4_reproducible_demo.py, tests/test_nivel2_validation.py, results/nivel2_validacion/n2_4_pytest_summary.json
Versión o hashes: run_n2_4_reproducible_demo.py sha256=067f6e7fbce1afa7, tests/test_nivel2_validation.py sha256=742b05d594a5eb3a
Comando o procedimiento: .\.venv\Scripts\python.exe -m experiments.run_n2_4_reproducible_demo --overwrite && .\.venv\Scripts\python.exe -m pytest -q
Configuración y semillas: Radioconda GNU Radio 3.10.9.2 (grcc compilación desde cero) + flujo headless de 4096 muestras + señal Python QPSK 15 dB con --noise-ref
Datos/particiones utilizados: Flowgraphs grc/*.grc y carpeta nueva limpia results/nivel2_validacion/n2_4_clean_demo/
Resultado mínimo exigido: Suite completa de pruebas en verde con resumen exacto y demostración limpia GRC + monitor reproducible desde cero.
Resultado observado: Los 3 flowgraphs compilados sin error; simulación headless generó 4096 muestras (potencia 0.9991); monitor entregó 5 salidas válidas para Python QPSK (QPSK, conf=0.7085 calibrada, SNR=15.10 dB, BW total=5668.2 Hz) y GNU Radio (predicho BPSK, conf=0.6935 confidence_calibrated=false por >2 ventanas, SNR=34.08 dB, BW total=1158.8 Hz). Suite completa pytest verificada con 278 pruebas aprobadas (0 fallos, 0 omisiones) registradas en n2_4_pytest_summary.json.
Intervalos o tolerancias aplicables: 0 fallos, 0 saltos en suite global; retorno CLI = 0.
Ruta de evidencias: results/nivel2_validacion/n2_4_clean_demo/demo_execution_summary.json, results/nivel2_validacion/n2_4_pytest_summary.json
Limitaciones: Los flowgraphs GUI requieren sesión gráfica QT en ejecución interactiva; en prueba automatizada se ejecutó el flujo headless y la compilación grcc de los tres esquemas.
Conclusión autorizada por la evidencia: La interoperabilidad GNU Radio–Python y la suite completa de pruebas quedan verificadas de punta a punta desde directorio limpio.
Siguiente dependencia habilitada: N4.3
```

```text
ID: N3.1
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: src/adaptive.py, experiments/exp_n3_adaptive_validation.py, tests/test_nivel3_adaptive.py
Versión o hashes: src/adaptive.py sha256=9f15629de637dd7a, lbap_deployable_package.joblib sha256=10adf2b6685f1111
Comando o procedimiento: .\.venv\Scripts\python.exe -m pytest tests/test_nivel3_adaptive.py -k "n3_1" -q
Configuración y semillas: Paquetes serializables completos con estados NumPy del selector, calibrador a0, expectativa fs=8000 Hz y hashes SHA-256 de SVM256, SVM512 y MultiDomainAMCNet
Datos/particiones utilizados: 300 señales de adaptive_dataset val_select (recarga/perezoso) + 200 señales mutadas en muestras 256:512 (causalidad)
Resultado mínimo exigido: 100% igualdad al recargar en proceso nuevo y CLI, 100% invarianza causal sobre [:256], 0 ejecuciones perezosas indebidas y rechazo ante hash incompatible.
Resultado observado: new_process_reload_matches=300/300 (acciones 132 a0, 37 a1, 131 a2); causal_prefix_matches=200/200; llamadas perezosas exactas svm512=37, a2=131 (lazy_execution_violations=0); incompatible_hash_rejected=true; CLI python -m src.adaptive verificado.
Intervalos o tolerancias aplicables: Coincidencia exacta 100% y 0 violaciones perezosas.
Ruta de evidencias: results/nivel3_adaptativo/{lbap_deployable_package.joblib,n3_1_package_verification.json}, tests/test_nivel3_adaptive.py
Limitaciones: El paquete está acoplado a fs=8000 Hz y a ventanas mínimas de 256 muestras (512 cuando selecciona a1).
Conclusión autorizada por la evidencia: La política adaptativa deja de ser una matriz en memoria y pasa a ser un artefacto desplegable verificable, causal y perezoso.
Siguiente dependencia habilitada: N3.3, N3.4
```

```text
ID: N3.2
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: experiments/exp_n3_adaptive_validation.py, tests/test_nivel3_adaptive.py
Versión o hashes: exp_n3_adaptive_validation.py sha256=c32b9a9944876a2b, test_nivel3_adaptive.py sha256=7f1cf9013b007a38
Comando o procedimiento: .\.venv\Scripts\python.exe -m pytest tests/test_nivel3_adaptive.py -k "n3_2" -q
Configuración y semillas: Selector reentrenado exclusivamente sobre data/fase6_holdout_sellado/train (4800 padres) con 0 solapamiento respecto al entrenamiento de expertos (e2_dataset y adaptive_dataset)
Datos/particiones utilizados: Caches verificadas: policy_train_holdout (4800), adaptive_val (2400), adaptive_test (2400), adaptive_test_shift (2400)
Resultado mínimo exigido: 0 solapamiento con entrenamiento de expertos, regeneración determinista desde cachés NPZ con manifiestos SHA-256 (config, CSV, NPZ, orden de filas, hashes de modelos, feature_version y código).
Resultado observado: overlap_with_e2_or_adaptive_experts=0; 4 cachés reconstruidas y verificadas con sus 4 manifiestos JSON SHA-256 completos en results/nivel3_adaptativo/caches/.
Intervalos o tolerancias aplicables: Comparación exacta de hashes SHA-256 e intersección vacía.
Ruta de evidencias: results/nivel3_adaptativo/n3_2_cache_and_partition_audit.json, results/nivel3_adaptativo/caches/*.manifest.json
Limitaciones: Al estar los modelos congelados en evaluación, la caché almacena las salidas observadas bajo sus hashes exactos y falla explícitamente si algún archivo o código cambia.
Conclusión autorizada por la evidencia: La evaluación del selector es recuperable, libre de solapamiento de padres y protegida por manifiestos criptográficos.
Siguiente dependencia habilitada: N3.4
```

```text
ID: N3.3
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: experiments/exp_n3_adaptive_validation.py, docs/paper/sections/03_adaptive_policy.tex
Versión o hashes: exp_n3_adaptive_validation.py sha256=c32b9a9944876a2b
Comando o procedimiento: .\.venv\Scripts\python.exe -m experiments.exp_n3_adaptive_validation
Configuración y semillas: batch=1, torch.cuda.synchronize(), torch.set_num_threads(1), threadpool_limits(1), 3 sesiones aleatorizadas x 300 observaciones (900 llamadas por política) en i7-13620H + RTX 4060 Laptop GPU
Datos/particiones utilizados: 300 realizaciones de validación (adaptive_dataset/val)
Resultado mínimo exigido: Medición de ruta completa (extracción + gate + rama ejecutada) con media, mediana y P95 en 3 sesiones independientes para baselines y políticas adaptativas.
Resultado observado: Medias agrupadas de cómputo (mediana / P95 ms): always_a0=2.221 (1.827 / 3.957), always_a1=3.218 (2.806 / 5.262), always_a2=32.545 (28.289 / 58.251), policy_confidence=35.916 (33.373 / 62.465), policy_blind_snr_flatness=16.813 (5.015 / 41.559), lbap_full=18.688 (6.555 / 46.639). Diferencia LBAP vs always_a2 = -13.857 ms (IC95% [-15.348, -12.317] ms). Costo efectivo en ruta LBAP por rama elegida: a0=4.133 ms, a1=6.621 ms, a2=36.763 ms.
Intervalos o tolerancias aplicables: IC95% bootstrap de diferencia de latencias.
Ruta de evidencias: results/nivel3_adaptativo/n3_3_latency_sessions.json, n3_3_latency_individual_calls.csv
Limitaciones: T_acq = N/fs (32 ms / 64 ms a 8 kHz) es adquisición virtual calculada por frecuencia de muestreo; el cómputo depende de la carga y sincronización CUDA en equipo de laboratorio Windows.
Conclusión autorizada por la evidencia: La medición sincronizada de ruta completa sustituye estimaciones parciales y cuantifica el ahorro neto de cómputo real al evitar la rama profunda cuando no aporta beneficio.
Siguiente dependencia habilitada: N3.4, N4.1
```

```text
ID: N3.4
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido (resultado mixto/inconcluso respecto a superioridad global)
Archivos modificados: experiments/exp_n3_adaptive_validation.py, docs/paper/sections/05_results_discussion.tex, docs/paper/sections/06_conclusion.tex, docs/paper/paper_ieee.tex
Versión o hashes: exp_n3_adaptive_validation.py sha256=c32b9a9944876a2b
Comando o procedimiento: .\.venv\Scripts\python.exe -m experiments.exp_n3_adaptive_validation
Configuración y semillas: Bootstrap pareado estratificado (clase x SNR) con 10 000 réplicas recalculando macro-F1 en cada réplica (seed=2026111611) + 3 semillas del selector (2026111611..13) con selector entrenado en 4800 padres independientes y latencias de ruta N3.3
Datos/particiones utilizados: policy_train_holdout (4800 train), adaptive_dataset val_select (1200), test (2400 ID), test_shift (2400 OOD)
Resultado mínimo exigido: Reglas explícitas e intervalos de confianza al 95% (exactitud, macro-F1, tiempo total, utilidad) contra always_a0, always_a1, always_a2, policy_confidence y policy_blind_snr_flatness, más dispersión en 3 semillas.
Resultado observado: En ID test (n=2400): LBAP (alpha*=0.0 Ridge determinista, 52.29% +/- 0.00% en 3 semillas; 53.00% Macro-F1; 56.66 ms; U=0.4379) supera significativamente en utilidad U a always_a0 (+0.0518 [0.0352, 0.0688]), always_a1 (+0.0783 [0.0601, 0.0966]), policy_confidence (+0.0299 [0.0165, 0.0431]) y always_a2 (+0.0189 [0.0058, 0.0318], -7.89 ms). PERO frente a always_a2 la exactitud ID cruza cero (+0.71 pp [-0.63, +2.00]) y Macro-F1 cae (-2.37 pp [-3.65, -1.11]); frente al proxy de 2 variables policy_blind_snr_flatness (51.63%, U=0.4369; 3 semillas 51.79% +/- 0.18%) la diferencia es inconclusa (Delta Acc +0.67 pp [-0.50, +1.83], Delta U +0.0010 [-0.0105, +0.0128]). En test_shift (OOD), LBAP (30.88%, U=0.2235) supera en utilidad a always_a2 (+0.0194) y always_a1 (+0.0289), pero queda por debajo de always_a0 en utilidad (-0.0285 [-0.0462, -0.0110]) e inconcluso frente a blind_snr_flatness (31.46%).
Intervalos o tolerancias aplicables: Bootstrap pareado 10 000 réplicas IC95%; superioridad requiere límite inferior > 0 sin degradar macro-F1 > 1 pp.
Ruta de evidencias: results/nivel3_adaptativo/n3_4_paired_bootstrap.json
Limitaciones: Con el protocolo más riguroso (selector en holdout independiente + costos reales de ruta N3.3), LBAP demuestra mejora estadísticamente significativa de utilidad frente a modelos fijos en ID, pero NO logra superioridad simultánea en exactitud/Macro-F1 ni sobre el proxy compacto de SNR ciega + flatness ni en utilidad OOD frente a exit rápido a0.
Conclusión autorizada por la evidencia: Se valida ganancia de utilidad en ID respecto a always_a2 y umbral de confianza, y se declara honestamente resultado mixto/inconcluso respecto a superioridad global y desplazamiento de canal.
Siguiente dependencia habilitada: N4.1, N4.3
```

```text
ID: N3.5
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: experiments/exp_n3_adaptive_validation.py, docs/paper/sections/05_results_discussion.tex
Versión o hashes: exp_n3_adaptive_validation.py sha256=c32b9a9944876a2b
Comando o procedimiento: .\.venv\Scripts\python.exe -m experiments.exp_n3_adaptive_validation
Configuración y semillas: Auditoría documental y cuantitativa de presupuestos reales de entrenamiento
Datos/particiones utilizados: Configuraciones y logs históricos de exp02, exp06 y exp12
Resultado mínimo exigido: Tabla explícita con padres reales, lotes sintéticos, épocas, presentaciones totales, distribuciones SNR/canal, criterio de selección y duración; regla de interpretación sin atribuir diferencia solo a arquitectura.
Resultado observado: Registrado explícitamente: SVM256/512 (4800 padres reales, 4800 presentaciones), SmallCNN (4800 padres, 61 épocas máx / 46 elegida, 292 800 presentaciones, 16.34 s) vs MultiDomainAMCNet (7200 padres reales + 24x1024 síntesis GPU/época x 90 épocas = 2 859 840 presentaciones, 989.55 s).
Intervalos o tolerancias aplicables: Conteo exacto de presentaciones y tiempos medidos.
Ruta de evidencias: results/nivel3_adaptativo/n3_5_training_budgets.json
Limitaciones: Comparación entre sistemas finales con presupuestos de datos desiguales; la igualación por presupuesto en reentrenamiento corresponde a N5.2 opcional.
Conclusión autorizada por la evidencia: Queda explícitamente acotada la comparación arquitectónica respecto al volumen de datos y síntesis empleados.
Siguiente dependencia habilitada: N4.1
```

```text
ID: N3.6
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido (con resultado negativo de entrenamiento mixto a 4 bits)
Archivos modificados: experiments/exp_n3_adaptive_validation.py, docs/paper/sections/03_adaptive_policy.tex, docs/paper/sections/05_results_discussion.tex
Versión o hashes: exp_n3_adaptive_validation.py sha256=c32b9a9944876a2b
Comando o procedimiento: .\.venv\Scripts\python.exe -m experiments.exp_n3_adaptive_validation
Configuración y semillas: Rango fijo V_FS=2.5, cuantizador mid-rise uniforme sobre los mismos padres en Inf/8/6/5/4 bits; auditoría de fórmula de Brier/NLL binario y bins uniformes de ECE10
Datos/particiones utilizados: results/paper_research/adc_quantization_experiment.csv y calibration_comparison.json
Resultado mínimo exigido: Evaluación con misma población padre, reporte conjunto de cuantización y clipping, aclaración explícita de Brier/NLL binario vs multiclase y cierre negativo honesto si entrenamiento mixto no supera al limpio.
Resultado observado: A 4 bits en test ID (SQNR=7.87 dB, clipping_rate_iq=5.12%), ExpC_mixed_inf_8_6 logra 44.96% (46.62% F1) frente a ExpA_clean_ref 46.29% (48.80% F1), diferencia -1.33 pp: hipótesis de mejora por entrenamiento mixto RECHAZADA en 4 bits (se conserva no obstante ventaja clara sobre SVM256 24.08%). Se corrige en 03_adaptive_policy.tex y 05_results_discussion.tex que Brier/NLL de calibración son binarios de acierto/error sobre max-probabilidad con ECE10 en 10 bins de igual anchura.
Intervalos o tolerancias aplicables: Diferencia exacta sobre 2400 realizaciones pareadas de test ID.
Ruta de evidencias: results/nivel3_adaptativo/n3_6_adc_and_calibration_audit.json
Limitaciones: Con escala fija V_FS=2.5 coexisten error de cuantización uniforme y recorte del 5.12% en I/Q; el enrutador LBAP cae a 40.88% por depender de márgenes SVM256 degradados por cuantización.
Conclusión autorizada por la evidencia: La cuantización uniforme no valida mejora con entrenamiento mixto a 4 bits y se presenta como hallazgo empírico negativo junto con las métricas de calibración bien definidas.
Siguiente dependencia habilitada: N4.1, N4.3
```

```text
ID: N4.1
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: README.md, docs/INFORME_FINAL.md, docs/FASE7_MONITOR.md, docs/RESEARCH_GAP.md, docs/PRESENTACION_DIAPOSITIVAS.md, docs/paper/sections/{03_adaptive_policy.tex,05_results_discussion.tex,06_conclusion.tex}, docs/paper/paper_ieee.tex, docs/MATRIZ_AFIRMACIONES_Y_ENTREGABLES.md
Versión o hashes: MATRIZ_AFIRMACIONES_Y_ENTREGABLES.md sha256=3eef0a4791743eb7
Comando o procedimiento: Verificación cruzada frente a JSONs/CSVs generados en results/nivel0..nivel3
Configuración y semillas: Consolidación documental integral
Datos/particiones utilizados: Artefactos N0, N1, N2, N3 y fases históricas
Resultado mínimo exigido: Matriz «afirmación -> artefacto -> limitación», clasificación (medida/inferida/hipotética/futura) y eliminación de contradicciones e hipérboles ("100%", SNR "-10 dB", ancho de banda total vs útil, Brier binario, época 46 SmallCNN).
Resultado observado: Todos los documentos principales, diapositivas y secciones LaTeX sincronizados con cifras observadas verificables; creada matriz de 11 afirmaciones centrales con limitaciones explícitas.
Intervalos o tolerancias aplicables: Trazabilidad directa exacta a rutas de archivo verificadas.
Ruta de evidencias: docs/MATRIZ_AFIRMACIONES_Y_ENTREGABLES.md
Limitaciones: Documentos históricos intermedios de fases tempranas (docs/FASE1..FASE6) conservan su registro histórico de fase sin alterarse, prevaleciendo docs/INFORME_FINAL.md y MATRIZ_AFIRMACIONES_Y_ENTREGABLES.md como consolidación final.
Conclusión autorizada por la evidencia: Cero contradicciones conocidas entre código, métricas guardadas, informe, diapositivas y artículo.
Siguiente dependencia habilitada: N4.3
```

```text
ID: N4.2
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: docs/MATRIZ_LITERATURA.md, docs/paper/sections/01_intro_related.tex
Versión o hashes: MATRIZ_LITERATURA.md sha256=c71cbfc2694786a6
Comando o procedimiento: Revisión documental de citas, DOIs y frases de novedad
Configuración y semillas: Auditadas las 15 fichas con PDF local en references/ + 5 referencias externas con DOI
Datos/particiones utilizados: docs/MATRIZ_LITERATURA.md, docs/paper/references.bib, docs/paper/sections/01_intro_related.tex
Resultado mínimo exigido: Fuente primaria verificable por comparación, distinción explícita entre comparación conceptual y reproducción experimental, y moderación de expresiones de novedad absoluta.
Resultado observado: Añadida nota explícita de distinción conceptual vs reproducción en MATRIZ_LITERATURA.md; moderada frase "establishing a new state of the art" en 01_intro_related.tex por evaluación acotada al benchmark estudiado con documentación de casos inconclusos/negativos.
Intervalos o tolerancias aplicables: Verificación exacta de DOIs y páginas/tablas de los 15 PDFs.
Ruta de evidencias: docs/MATRIZ_LITERATURA.md, docs/paper/sections/01_intro_related.tex
Limitaciones: Los 5 artículos externos al directorio references/ no cuentan con copia PDF local almacenada y solo se utilizan como contraste conceptual verificado por DOI/referencia bibliográfica.
Conclusión autorizada por la evidencia: Las comparaciones bibliográficas cumplen rigor académico sin exagerar prioridad ni mezclar métricas de datasets externos como reproducciones propias.
Siguiente dependencia habilitada: N4.3
```

```text
ID: N4.3
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: docs/MATRIZ_AFIRMACIONES_Y_ENTREGABLES.md, docs/PRESENTACION_DIAPOSITIVAS.md, docs/paper/paper_ieee.pdf
Versión o hashes: paper_ieee.pdf sha256=81f1ab3361bdfe52, PRESENTACION_DIAPOSITIVAS.md sha256=53db4d69fd103137
Comando o procedimiento: pdflatex/bibtex/pdflatex x2 + inspección visual de figuras + pytest -q + run_n2_4_reproducible_demo
Configuración y semillas: MiKTeX + Radioconda 3.10.9.2 + Python 3.11.0 + Torch 2.10.0+cu128
Datos/particiones utilizados: Paquete completo de entrega (9 entregables de GUIA_MAESTRA_COMII_PARA_CODEX.md)
Resultado mínimo exigido: PDF compilado sin referencias rotas, revisión visual de figuras, matriz de 9 entregables con rutas/estados reales, comandos reproducibles en carpeta limpia y registro de dependencias/pruebas.
Resultado observado: paper_ieee.pdf compilado (23 páginas, 1 699 379 bytes, 0 referencias/citas indefinidas); 10 figuras PNG clave inspeccionadas visualmente; matriz de 9 entregables documentada y verificada en MATRIZ_AFIRMACIONES_Y_ENTREGABLES.md; N5.1 y N5.2 preservados honestamente como pendientes opcionales.
Intervalos o tolerancias aplicables: 0 errores LaTeX y 0 fallos en suite pytest.
Ruta de evidencias: docs/MATRIZ_AFIRMACIONES_Y_ENTREGABLES.md, docs/paper/paper_ieee.pdf, results/nivel2_validacion/n2_4_pytest_summary.json
Limitaciones: No se ha ejecutado en hardware RF físico ni en FPGA; la demostración interactiva QT GUI requiere entorno gráfico del usuario (la compilación e interoperabilidad headless sí están automatizadas).
Conclusión autorizada por la evidencia: El paquete integral de curso e informe técnico queda ensamblado, compilado, visualmente revisado y validado antes del corte de cierre.
Siguiente dependencia habilitada: N5.1 (diagnóstico opcional completado)
```

```text
ID: N5.1
Fecha: 2026-11-16
Responsable: Ejecución automatizada (COM2)
Estado: Cumplido
Archivos modificados: experiments/exp_n5_1_generalization_diagnostic.py, tests/test_nivel5_diagnostic.py, docs/PLAN_MODIFICACIONES_Y_CRITERIOS_DE_ACEPTACION.md
Versión o hashes: exp_n5_1_generalization_diagnostic.py sha256=0658aedc164483a4, test_nivel5_diagnostic.py sha256=26722f840fb931ed
Comando o procedimiento: .\.venv\Scripts\python.exe -m experiments.exp_n5_1_generalization_diagnostic && .\.venv\Scripts\python.exe -m pytest tests/test_nivel5_diagnostic.py -q
Configuración y semillas: H1 sobre 720 padres pareados de results/channel_diagnostic (seed=2026110611) ampliado con MultiDomainAMCNet y 10 000 réplicas bootstrap; H2 sobre 240 padres pareados 16QAM/64QAM SNR >= 10 dB (seed=2026111652); H3 ensayo factorial de un factor sobre grc_sim_output.c64 (16 ventanas x 256)
Datos/particiones utilizados: Realizaciones pareadas de channel_diagnostic, padres sintéticos H2 y captura headless grc_sim_output.c64
Resultado mínimo exigido: Experimentos controlados de un factor por hipótesis, conteos exactos por clase/SNR, comparación pareada con IC95%, hipótesis aceptada/rechazada/inconclusa y propuesta sustentada de intervención sin entrenar aún.
Resultado observado: (H1 Aceptada) SmallCNN cae -25.56 pp por CFO shift mientras MultiDomainAMCNet solo cae -3.61 pp [-7.36, +0.14] en CFO shift pero colapsa -30.69 pp [-34.58, -26.81] en multipath_shift (vs -22.92 pp SVM256 y -25.42 pp SVM512). (H2 Aceptada) Aplica dos sesgos opuestos: en AWGN/ID limpio MultiDomainAMCNet sesga 64QAM->16QAM (83-85/120 fallos, solo 26.7%-29.2% Acc en 64QAM); bajo multipath_shift, SVM512 colapsa 16QAM a 10.8% (105/120 predichas como 64QAM). (H3 Aceptada) En grc_sim_output.c64 la causa dominante de QPSK->BPSK en SVM256 es CFO=0 Hz frente a descriptores no desrotados (c40_abs=0.8231, phase_resultant_4=0.4982): inyectar 35 Hz CFO sin cambiar símbolos eleva aciertos QPSK en SVM256 de 4/16 (25%) a 15/16 (93.75%); MultiDomainAMCNet alcanza 15/15 QPSK en régimen permanente (ventanas 1..15) fallando solo en ventana 0 (16QAM) por transitorio inicial FIR.
Intervalos o tolerancias aplicables: IC95% bootstrap estratificado de 10 000 réplicas en H1; conteos exactos pareados en H2 (n=240) y H3 (n=16 ventanas).
Ruta de evidencias: results/nivel5_diagnostico/{n5_1_diagnostic_report.json,n5_1_factorial_metrics_by_class_snr.csv,n5_1_qam_confusion_breakdown.csv,n5_1_diagnostic_summary.png}
Limitaciones: Diagnóstico de alcance exploratorio sobre modelos congelados; no entrena ni modifica ningún modelo (las intervenciones P1 y P2 quedan formuladas para N5.2 opcional).
Conclusión autorizada por la evidencia: Quedan aisladas y cuantificadas las causas físicas y estadísticas de las caídas OOD, la confusión 16QAM/64QAM y el fallo QPSK->BPSK en GNU Radio.
Siguiente dependencia habilitada: N5.2 (diferido como trabajo futuro opcional)
```

```text
ID: N5.2
Fecha: 2026-11-16
Responsable: Cierre de alcance (COM2)
Estado: Diferido (opcional no requerido)
Archivos modificados: docs/PLAN_MODIFICACIONES_Y_CRITERIOS_DE_ACEPTACION.md
Versión o hashes: Verificado contra results/nivel5_diagnostico/n5_1_diagnostic_report.json
Comando o procedimiento: N/A (fase de reentrenamiento con >=3 semillas explícitamente marcada como opcional y no requerida para aprobar el curso)
Configuración y semillas: Pendiente opcional para trabajo futuro según líneas base N3.4/N3.5 y propuestas P1/P2 de N5.1
Datos/particiones utilizados: N/A
Resultado mínimo exigido: Dejar constancia explícita del alcance cerrado (N0–N4 + N5.1) y de las intervenciones técnicas justificadas antes de modificar modelos.
Resultado observado: Propuestas registradas en N5.1: (P1) restringir descriptores SVM a variantes desrotadas derotated_* y descartar transitorio inicial FIR L_f-1 para eliminar la anomalía CFO=0 Hz en GNU Radio; (P2) incorporar ecualización ciega lineal / CMA corto para ecos > sps=8 o ajuste geométrico 64QAM en ventana extendida, evaluándolo con 3 semillas e IC95% bootstrap de N3.4 si se ejecuta en el futuro.
Intervalos o tolerancias aplicables: N/A (diferido; modelos vigentes sin alterar y 278 pruebas en verde).
Ruta de evidencias: results/nivel5_diagnostico/n5_1_diagnostic_report.json
Limitaciones: No se realizó reentrenamiento en este corte de las 16:00 PM para preservar congelados los modelos validados en N2 y N3.
Conclusión autorizada por la evidencia: El sistema obligatorio del curso, el artículo, la validación estadística N3 y el diagnóstico N5.1 quedan cerrados y trazables; cualquier reentrenamiento futuro debe seguir P1/P2 y reglas N3.4.
Siguiente dependencia habilitada: Publicación/respaldo en repositorio Git
```

### 10.1 Acta de cierre del corte (16 de noviembre de 2026 — 16:00 PM)

- **Estado global consolidado:** Completadas y verificadas las **22 tareas ejecutadas (`N0.1`, `N0.2`, `N1.1`–`N1.6`, `N2.1`–`N2.4`, `N3.1`–`N3.6`, `N4.1`–`N4.3`, `N5.1`)**, con constancia formal de diferimiento opcional para `N5.2` (no requerido por la guía del curso).
- **Verificación automatizada vigente:** `.\.venv\Scripts\python.exe -m pytest -q` → **`278 passed in 39.59s`** (`0` fallos, `0` omisiones), incluyendo compilación GNU Radio `3.10.9.2` con `grcc`, contratos CLI, validación adaptativa desplegable y diagnóstico `N5.1`.
- **Resultados negativos o inconclusos declarados sin omisiones:** (1) En `N2.2`, las 6 capturas GNU Radio QPSK con `CFO=0 Hz` fueron detectadas (`6/6`) pero clasificadas como `BPSK` por `SVM256` (`N5.1` demostró que inyectar `35 Hz` eleva aciertos de `4/16` a `15/16`); (2) en `N3.4`, `FastLBAPSelector` supera significativamente en utilidad ID a los baselines fijos (`+0.0189` vs `always_a2`), pero cruza cero en exactitud ID frente a `always_a2`, resulta inconcluso frente al proxy de 2 variables `SNR ciega + flatness`, y pierde utilidad en `test_shift` frente a `always_a0`; (3) en `N3.6`, el entrenamiento mixto ADC `Exp C` (`44.96%`) no mejoró al entrenamiento limpio `Exp A` (`46.29%`) a 4 bits (`5.12%` *clipping*).
- **Control de versiones e historial:** Estado histórico inicial congelado en commit `7e81d0df`; estado final listo para commit y empuje a repositorio remoto (`data/`, `*.pt`, `*.joblib`, `*.npz`, `*.c64` excluidos por `.gitignore` para mantener el repositorio liviano, manteniendo scripts deterministas y manifiestos SHA-256).

### Plantilla de cierre por tarea

```text
ID:
Fecha:
Responsable:
Estado:
Archivos modificados:
Versión o hashes:
Comando o procedimiento:
Configuración y semillas:
Datos/particiones utilizados:
Resultado mínimo exigido:
Resultado observado:
Intervalos o tolerancias aplicables:
Ruta de evidencias:
Limitaciones:
Conclusión autorizada por la evidencia:
Siguiente dependencia habilitada:
```

**Primera ejecución recomendada:** N0.1 → N0.2 → N1.1 y N1.2. No comenzar reentrenando redes; la primera meta es que el monitor trate correctamente una señal real de entrada y comunique sus condiciones de validez.
