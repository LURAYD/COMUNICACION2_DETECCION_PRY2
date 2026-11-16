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

| ID | Tarea | Estado inicial | Evidencia de cierre |
|---|---|---|---|
| N0.1 | Referencia e integridad | Pendiente | Por completar |
| N0.2 | Protocolo y particiones | Pendiente | Por completar |
| N1.1 | Detección calibrada por ventana/archivo | Pendiente | Por completar |
| N1.2 | Frecuencia y compatibilidad | Pendiente | Por completar |
| N1.3 | SNR y límites | Pendiente | Por completar |
| N1.4 | Confianza final | Pendiente | Por completar |
| N1.5 | Entradas y CLI | Pendiente | Por completar |
| N1.6 | Espectro y alcance | Pendiente | Por completar |
| N2.1 | Validación de detección/SNR | Pendiente | Por completar |
| N2.2 | Archivos desconocidos | Pendiente | Por completar |
| N2.3 | Clasificación E2–E6 | Pendiente | Por completar |
| N2.4 | Suite y GNU Radio | Pendiente | Por completar |
| N3.1 | Política desplegable | Pendiente | Por completar |
| N3.2 | Recuperabilidad y cachés | Pendiente | Por completar |
| N3.3 | Latencia completa | Pendiente | Por completar |
| N3.4 | Comparaciones estadísticas | Pendiente | Por completar |
| N3.5 | Presupuesto y arquitectura | Pendiente | Por completar |
| N3.6 | ADC y calibración | Pendiente | Por completar |
| N4.1 | Cifras y afirmaciones | Pendiente | Por completar |
| N4.2 | Bibliografía y novedad | Pendiente | Por completar |
| N4.3 | Entrega y revisión visual | Pendiente | Por completar |
| N5.1 | Diagnóstico adicional | Pendiente, opcional | Por completar |
| N5.2 | Mejora de modelos | Pendiente, opcional | Por completar |

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
