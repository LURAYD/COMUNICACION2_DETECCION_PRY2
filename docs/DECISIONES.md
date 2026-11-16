# Decisiones y pendientes

## Decisiones ya fijadas por la guía maestra

- Idioma español, Python principal, GNU Radio complementario sin RF real.
- Sexta modulación: 4-PAM. Perturbaciones principales: CFO y multipath, además de fase inicial y AWGN.
- Prioridad: obligaciones docentes E1–E6 antes de investigación adaptativa y ADC.
- Separación futura de datos por realización madre antes de crear derivados.

## Auditoría del 3 de noviembre de 2026

Se conserva todo src sin cambios. Se añaden pruebas diagnósticas; los defectos conocidos se marcan xfail estricto para identificarlos explícitamente. No se adopta ninguna ecuación nueva. Se instala únicamente pytest y se registran dependencias directas; no hace falta SciPy, PyTorch o GPU para los diagnósticos actuales.

La guía maestra menciona dos copias duplicadas de artículos; el paquete real contiene 15 artículos sin duplicados, consistente con LEER_PRIMERO.txt y los hashes del inventario. Se usa el estado real.

El DOI de Uncertainty Quantification contiene 2025, pero el encabezado del PDF lo sitúa en IEEE IoT Journal, vol. 13, n.º 3, febrero de 2026; su publicación inicial fue en diciembre de 2025 (p. PDF 1). No deducir el año bibliográfico únicamente del DOI.

## Propuesta histórica de Fase 1 en la auditoría

Preservar BPSK 0→+1 y 1→−1. Para QPSK, proponer `((1−2*b0) + j*(1−2*b1))/sqrt(2)`: 00→(+1,+1), 01→(+1,−1), 10→(−1,+1), 11→(−1,−1), normalizados. Es una decisión de ingeniería, todavía no implementada.

Aceptar vectores binarios 0/1, incluidos valores float exactamente binarios, convertir después de validarlos y rechazar vacíos/matrices. Preservar las tablas existentes de los otros cinco moduladores. Mostrar diferencias antes de modificar código funcional.

Libro de Medina pendiente de incorporación a references. No bloquea corregir aritmética y validación; sí impide afirmar una verificación contra ese libro. Para cerrar RRC se necesita una referencia técnica comprobable si el libro sigue ausente.

## Fase 1 autorizada el 3 de noviembre de 2026

El usuario autorizó corregir los defectos de la auditoría. La ejecución se distribuyó entre dos agentes de trabajo solicitados: Astra extra high para orquestación/revisión y Astra medium para código. No son modelos de clasificación del proyecto ni se entrenó ML.

- Se adopta QPSK según la propuesta anterior y se conservan las cinco tablas originales. Contrato completo versionado en `MAPPING_MODULACIONES.md`.
- La entrada binaria se valida antes de convertir a entero con signo; esto corrige uint8 y floats sin aceptar truncamientos de valores como 0.5.
- RRC y conformación pasan a `src.waveform`. `src.signals` reexporta ambas funciones para conservar la API usada por la auditoría. La ejecución documentada es desde la raíz con `python -m ...` y los imports oficiales son `src.*`.
- Se conserva la longitud histórica `N*sps + len(h)-1` y los ceros finales del tren de impulsos. No se recorta retardo automáticamente.
- Se conserva el dominio RRC `0<beta<=1`, `sps>=2` entero y `span` positivo par; los enteros NumPy se convierten a int de Python tras validación. Se excluyen booleanos como parámetros de filtro.
- La fórmula RRC y límites se verifican con documentación de OptiCommPy; MathWorks respalda energía discreta, longitud y cascada truncada. La diferencia de factor observada en el PDF de Purdue queda registrada en `MODELO_DE_SENAL.md`, sin atribuir fórmulas a Medina ni afirmar ejecución de MATLAB/OptiCommPy.
- `pytest.ini` limita la recolección a `tests`, evitando que demos, propuestas y copias originales se ejecuten como pruebas. Se verifica expresamente que importar los módulos no imprima ni cargue pyplot.
- Al no existir Git, las copias previas y el diff de Fase 1 se guardan como evidencia local. `AUDITORIA_INICIAL.md` conserva el estado histórico, incluidos los seis xfail originales; el reporte nuevo documenta su resolución.
- Fase 2, modelos, dataset y política adaptativa requieren el siguiente paso del proyecto; esta intervención se limita a cerrar Fase 1.

## Fase 2 autorizada después de completar Fase 1

El usuario indicó continuar. Se mantienen los mismos dos agentes de trabajo y se implementa solo la siguiente fase.

- Se conserva AWGN por potencia media de muestras, con salida `(observacion, ruido)`; el ruido es oracle de diagnóstico. No se fuerza potencia exacta por realización ni se confunde esta SNR con Eb/N0.
- Fase fija y CFO se aplican mediante un oscilador complejo de módulo uno. Se añade `sample_offset` entero no negativo para conservar el índice absoluto al procesar porciones. El parámetro de fase se expresa en radianes y CFO en Hz.
- Multipath es un FIR determinista, constante por trama, con retardos enteros en muestras y convolución completa. No se normalizan taps ni salida implícitamente. Se admite canal nulo, aunque AWGN rechaza una entrada de potencia cero.
- El diagnóstico usa Rs=1000 baud, sps=8, fs=8000 Hz; fase=0.6 rad, CFO=75 Hz, SNR=10 dB y trayectos en 0/2/5 muestras. Son parámetros ilustrativos de ingeniería, no rangos prescritos por el docente ni un canal físico identificado.
- Para revisar SNR se barren −10/0/10/20 dB y amplitudes 0.5/1/1.5. Los tests usan semillas independientes; el diagnóstico varía realizaciones de ruido sobre una misma forma de onda con fines de validación y no constituye un dataset independiente.
- La aplicación recomendada es por trama completa antes de extraer ventanas: FIR requiere memoria entre bloques y AWGN vuelve a calcular potencia y consumir RNG en cada llamada. Solo portadora dispone aquí de continuidad explícita por índice.
- ADC sigue pospuesto por ser opcional. No se crea Fase 3 ni se entrena ML. Detalles de fórmulas, acceso a la verdad y limitaciones en `CANALES.md`; resultados reales en `FASE2_VALIDACION.md`.

## Fase 3 autorizada y completada el 4 de noviembre de 2026

- Se retomaron las candidatas conservadas tras una interrupción por límite de uso. El diff se mostró y luego se aplicó; se mantuvieron los mismos dos agentes de trabajo.
- Smoke de 224 padres: cada split tiene dos realizaciones por cada una de seis clases y cuatro SNR, más ocho noise-only. Se aumentó respecto del diseño preliminar de 98 para cubrir cada clase/SNR también en validación y pruebas.
- Split y semillas se asignan antes de generar; ambas longitudes permanecen juntas. Se separan streams de parámetros/bits/ruido y se verifican padres, semillas, prefijos y ausencia de I/Q idéntica entre padres.
- JSON reemplaza al YAML sugerido para evitar una dependencia adicional. CSV administrativo, NPZ solo I/Q y manifiesto oracle JSONL están separados; la API pública funciona sin oracle/configuración.
- `test_shift` usa CFO y retardos fuera del soporte ID. Noise-only es ruido complejo con potencia explícita y sin SNR de señal; su escala aproximada y el carácter reducido del smoke quedan documentados en `PROTOCOLO_EXPERIMENTAL.md`.
- RML22 y FFTFormer se consultaron en los PDF originales; páginas, evidencia y discrepancias se registran en `MATRIZ_LITERATURA.md`. No se copiaran sus particiones o parámetros como garantía de independencia.
- Se detiene al cerrar Fase 3. Fase 4, entrenamiento y ADC no se implementaron. Resultados reales en `FASE3_VALIDACION.md`.

## Fase 4 · 5 de noviembre de 2026

- El estudiante cambió el agente de código a Sol high para la reanudación de esta fase. Astra extra high se ocupó de la revisión y la orquestación.
- Se eligió una ventana de 256 muestras y una tasa nominal de falsas alarmas del 1 %. E1 reserva conjuntos independientes con piso de ruido estacionario para calibrar y medir esa tasa. El ensayo de deriva verifica qué ocurre si deja de cumplirse esa condición con el umbral ya fijado.
- La SNR necesita una referencia independiente de ruido de fondo adquirida en la misma banda y con la misma ganancia. La decisión de detección controla cuándo procede estimarla. La disponibilidad de una referencia así en una futura captura del profesor sigue abierta. Sin ella se informa que la SNR es indeterminada en una detección.
- El ancho de banda mostrado corresponde al 99 % de la potencia recibida, que incluye ruido. Se expresa en Hz si existe frecuencia de muestreo conocida o en ciclos/muestra cuando solo está disponible I/Q. Las fórmulas y los antecedentes se detallan en [Fundamentos](FUNDAMENTOS_FASE4.md).

## Fase 5 · Clasificador convencional en CPU

- El estudiante autorizó E2 y preguntó por el uso de GPU. Se eligió CPU para los candidatos de bosque aleatorio y SVM de esta fase.
- Se amplió el procedimiento de datos de Fase 3 hasta 4 800 señales por split con otra semilla. Se mantienen los mismos padres para comparar 256 y 512 muestras y para la futura comparación convencional/avanzado. La prueba con cambio de canal permanece reservada para E6.
- Se comparan cinco candidatos prefijados por longitud y dos grupos de características. Selección por F1 macro en validación, entrenamiento limitado a train y ambos modelos guardados antes de medir test.
- E2 se define sobre señales presentes de seis clases conocidas. Su evaluación sirve para el bloque clasificador. La evaluación conjunta de detector, estimador y clasificación requerirá el monitor integrado.
- Cada pipeline calcula características desde la ventana disponible y conserva la escala ajustada en train cuando corresponde. Se entrega una etiqueta con alcance de clasificación cerrada. Una estimación de confianza calibrada necesita su propio procedimiento posterior.

## Preferencia de hardware para Fase 6 · 6 de noviembre de 2026

- El estudiante indicó usar su NVIDIA RTX 4060 de 8 GB para entrenar el modelo avanzado. `nvidia-smi` detectó la GPU con 8188 MiB y controlador 592.00.
- PyTorch todavía no está instalado en `.venv`. Antes del entrenamiento se instalará una distribución compatible con CUDA y se comprobará una operación real en GPU. La detección por el controlador aún no verifica el entorno de entrenamiento.
- Se conserva Astra extra alto para orquestación y GPT-6.1 Sol alto para código.

## Revisión de investigación previa al entrenamiento avanzado

- El estudiante indicó expresamente no entrenar todavía. La Fase 6 queda pendiente hasta una nueva instrucción de entrenamiento. Antes se revisará la contribución científica frente a los antecedentes.
- ADC uniforme de muestras I/Q sigue sin implementar ni evaluar. Las cifras de E1/E2 corresponden al simulador sin esta degradación adicional. Su aplazamiento se debió a la prioridad por fases de la guía.
- Aumentar de 256 a 512 muestras, usar una CNN o añadir cuantización no bastan por sí solos para afirmar novedad. La mejora de E2 compara pipelines con distintos grupos de características y requiere una ablación controlada antes de atribuirla exclusivamente a la longitud.
- Antes de futuras corridas largas, implementar y probar reanudación desde checkpoints con modelo, optimizador, scheduler si aplica, época/paso, estado de RNG, configuración y referencias del dataset. Guardar archivos de forma atómica. Esta capacidad aún no existe para el entrenamiento avanzado.
- La hipótesis de selección de acciones según beneficio bajo degradaciones de adquisición sigue siendo candidata. Requiere revisión de antecedentes específicos, baselines y una prueba falsable antes de presentarse como contribución.

## Corrección de enfoque solicitada por el estudiante

El estudiante reiteró que el ADC debe permanecer como una extensión secundaria de robustez, conforme al apartado 2.3 de la guía maestra. La propuesta conversacional de centrar la hipótesis en resolución y saturación le dio un protagonismo indebido y queda descartada como eje del trabajo.

El foco de investigación permanece en la asignación conjunta de tiempo de observación y procesamiento guiada por riesgo. La revisión debe precisar qué limitación de los métodos existentes resuelve la decisión entre finalizar, observar más o procesar más, y si existe una contribución defendible. La mera combinación de acciones requiere contraste bibliográfico y experimental. ADC será opcional y no condicionará la hipótesis principal ni la selección del método.

Sigue vigente la instrucción de no iniciar el entrenamiento avanzado mientras se revisa el planteamiento científico.

## Fase 6 cerrada · 6 de noviembre de 2026

La nueva autorización permitió entrenar con un único agente Astra. CNN1D 256 IQ, 94726 parámetros, RTX4060; selección val época46 y parada época61. E3–E6 completos: CNN test40,94%/shift18,60%, inferiores a SVM. Checkpoints atómicos y recuperación entre procesos comprobada bit a bit. 235 tests pasan. La CNN es más rápida en CPU que los SVM de este pipeline; no se afirma mayor cómputo ni beneficio de política adaptativa. ADC sigue secundario. Reporte: FASE6_VALIDACION.md. No se ejecutó apagado ni se detectó agotamiento de cuota.

## Diagnóstico posterior y variante de fase · 6 de noviembre de 2026

Se autorizó continuar revisando la CNN. Train/val evidenció sobreajuste; se ejecutó una sola variante con rotación global aleatoria solo en train, misma CNN/config restante. F1 val0,40542→0,41396 y brecha accuracy train-val15,94→3,25puntos. No se afirma mejora final. Antes del ajuste se generó holdout semilla2026110609 y sigue sellado; test original no reevaluado. 236 tests y recuperación GPU incluyendo augmentation pasan. Artefactos originales preservados. Reporte FASE6_AJUSTE_VALIDACION.md; sin búsqueda adicional, ADC ni apagado.

## Holdout final abierto y evaluado · 6 de noviembre de 2026

Autorizada evaluación independiente semilla2026110609 con cuatro modelos congelados. ID accuracy CNN original40,75%, fase41,54%, SVM25643,27%, SVM51247,21%; shift19,02/19,48/26,81/29,06%. Ganancia fase-original +0,79pp IC95%[-0,60;2,17] ID y +0,46pp[-0,71;1,67] shift. No se demuestra superioridad de variante; F1 shift empeora. Se preservan ambos artefactos y SVM como referencia convencional. Holdout ya consumido para futuros ajustes. Reporte FASE6_HOLDOUT_VALIDACION.md. Monitor/calibración próximos, no ejecutados.

## Diagnóstico controlado de canal · 6 de noviembre de 2026

Sin reentrenamiento ni GNU Radio:720 padres nuevos,9 versiones pareadas,4 modelos congelados. CFOshift domina deterioro CNN; multipathshift domina SVM. Beta no cambió en shift histórico y la prueba aislada no muestra caída dominante. Hay complementariedad, pero oracle usa verdad y no demuestra selector viable ni ahorro. Resultados/protocolo en DIAGNOSTICO_CANAL_RESULTADOS.md y DIAGNOSTICO_CANAL_PROTOCOLO.md. Modelos preservados, dos tests y auditoría de tablas/prefijos pasan. ADC secundario; GNU Radio expresamente aplazado.

## Prueba adaptativa cerrada (Exp 10) · 6 de noviembre de 2026

Evaluación de política adaptativa multi-salida (Random Forest) condicionada por riesgo y tiempo de observación. Bases congeladas (SVM256, SVM512, CNN), partición independiente semilla 2026110613. El selector aportó +4,12pp de accuracy pero con penalización de latencia (+22,74 ms), resultando en delta de utilidad no significativo (-0,0042). Test y shift sellados en results/adaptive/ESTADO_REANUDACION.md.

## Fase 8 — GNU Radio Companion e Interoperabilidad · 10 de noviembre de 2026

Implementación y validación de la integración complementaria con GNU Radio (GRC 3.10.9.2 bajo Radioconda):
- Se estructuró el directorio `grc/` con tres flowgraphs reproducibles sin hardware: `flowgraph_tx_channel.grc` (interfaz gráfica Qt GUI con sliders de SNR/CFO/multipath y visualizadores de constelación, PSD y osciloscopio), `flowgraph_headless_sim.grc` (simulación por lotes con generación prefijada de 4096 muestras complex64), y `flowgraph_rx_from_file.grc` (receptor visual para archivos binarios de señal generados en Python).
- Módulo de interoperabilidad `src/gnuradio_interop.py` con serialización binaria nativa float32 I/Q (8 bytes/muestra) y orquestación automática vía `grcc.exe`.
- Experimento y validación cruzada en `experiments/exp11_gnuradio_demo.py` y reporte formal en `docs/FASE8_GNURADIO.md`.
- El monitor integrado midió un ancho de banda ocupado al 99% de 1134,5 Hz sobre la señal de GRC (acorde a Rs=1000 baud, beta=0.35) y el clasificador SVM256 reconoció las ventanas correctamente.
- 5 tests unitarios pasan en `tests/test_gnuradio_interop.py`. Entregable de GRC completado según los lineamientos de la guía maestra.

## Fase 7 — Monitor Integrado y CLI de Señales Desconocidas · 10 de noviembre de 2026

Cierre de la herramienta CLI unificada en `src/monitor.py` para la evaluación docente sobre archivos desconocidos:
- Soporte para entrada directa en formatos `.c64`, `.bin`, `.npy`, `.npz` y `.csv`.
- Implementación completa de las cinco salidas obligatorias del proyecto: detección de energía con umbral calibrado ($P_{fa}=1\%$), espectro PSD con ancho de banda al 99%, estado formal de SNR (con o sin referencia de piso de ruido), predicción de modulación (seis clases) y confianza probabilística calibrada mediante el modelo logístico de la fase adaptativa.
- Generación automática de archivo JSON de resultados y figura PNG de doble panel (PSD + constelación).
- 6 tests unitarios pasan en `tests/test_monitor_cli.py` y `tests/test_fase4_integration.py`. Reporte formal en `docs/FASE7_MONITOR.md`.
