# Revisión independiente de resultados — 15 de noviembre de 2026

## Veredicto

Hay una implementación sustancial y resultados numéricos reproducibles. El nuevo experto multidominio sí mejora la exactitud respecto de los SVM en el test nominal. Sin embargo, el monitor tiene defectos prácticos para señales desconocidas y la documentación presenta como demostradas varias conclusiones que los experimentos no sostienen. No considero cerrado el proyecto ni listo el artículo para envío en su estado actual.

Esta revisión no modifica modelos, código de producción ni resultados originales. No se reentrenaron redes. No existe un repositorio Git en la raíz que permita atribuir cambios concretos a una IA o comparar con una versión anterior.

## Qué comprobé realmente

- Leí código del monitor, del experimento 12, del selector y del sintetizador, pruebas, métricas JSON/CSV, informe final y secciones metodológicas y de resultados del artículo.
- Consulté las páginas 4–7 de la guía docente y revisé la figura de exactitud y acciones por SNR.
- Ejecuté `.venv/Scripts/python.exe -m pytest -q`: 257 aprobadas y 2 fallidas por `WinError 623` al arrancar `grcc.exe` dentro del entorno aislado.
- Repetí únicamente esas dos pruebas fuera del aislamiento: **2 aprobadas en 20,05 s**. Las 259 pruebas aprobaron entre ambas ejecuciones; no hubo una ejecución única de la suite completa fuera del aislamiento.
- Cargué el checkpoint `multidomain_clean_a2_best.pt` y volví a inferir las 2.400 señales de `test` y las 2.400 de `test_shift`. Hubo **cero diferencias de clase** respecto de la caché, y las etiquetas coincidieron con el dataset actual.
- Recalculé las exactitudes de las tres acciones a partir de las predicciones almacenadas.
- Ejecuté `analyze_complete(load_signal(...))` sobre las dos señales de demostración con los valores predeterminados, sin generar nuevos gráficos ni sobrescribir sus resultados.

No reproduje todo el entrenamiento, no reconstruí todas las políticas adaptativas ni repetí el estudio ADC. Sus cifras siguientes provienen de los artefactos guardados, contrastados con la implementación. Tampoco hice una búsqueda bibliográfica exhaustiva ni una auditoría visual completa del PDF del artículo.

## Resultados principales

Mismo dataset adaptativo, 2.400 observaciones por partición. Las métricas de clasificación excluyen ruido puro y presuponen señal presente de una de las seis clases.

| Modelo o política | Exactitud nominal | Exactitud con cambio de canal | F1 macro nominal | Tiempo total nominal reportado |
|---|---:|---:|---:|---:|
| SVM, 256 muestras | 43,75% | 30,33% | 43,73% | 34,03 ms |
| SVM, 512 muestras | 46,04% | 29,54% | 45,93% | 66,27 ms |
| Red multidominio, 256 muestras | 51,58% | 30,08% | 55,36% | 56,15 ms |
| Política completa Fast LBAP | 51,625% | 31,625% | 52,78% | 52,56 ms |
| Política con SNR ciega y estadísticas espectrales | 52,29% | 31,21% | 53,49% | 53,99 ms |

Fuente: `results/paper_research/ablation_and_policy_metrics.json`. Los tiempos suman adquisición virtual y costes estimados a partir de mediciones parciales; no son una medición integral de cada ruta del monitor.

El promedio nominal incorpora ocho SNR entre −20 y +20 dB. Por tanto, 51,58% no significa que el modelo falle igual a alta SNR: a +5 dB la red alcanza 84,67%, según `policy_by_snr.csv`. La curva bajo cambio de canal sí muestra una pérdida importante incluso con SNR alta.

## Hallazgos prioritarios

### 1. El detector predeterminado rechaza las propias demostraciones [alta]

En `src/monitor.py:152` se usa por defecto el umbral absoluto 1,15239, heredado de una calibración con potencia de ruido nominal unitaria. No se adapta automáticamente a la ganancia o al piso de ruido de otro archivo.

Reproducción observada:

| Archivo | Energía del prefijo | Umbral predeterminado | Resultado |
|---|---:|---:|---|
| `results/gnuradio/py_qpsk_15db.c64` | 0,11677 | 1,15239 | NO_SIGNAL |
| `results/gnuradio/sim_run/grc_sim_output.c64` | 0,87543 | 1,15239 | NO_SIGNAL |

Los JSON de las demostraciones en `results/monitor` usaron umbrales manuales de 0,1 y 0,5. Por tanto, esas demostraciones no validan el comando predeterminado para archivos desconocidos. Además, `--noise-ref` alimenta el estimador SNR, pero no recalibra el detector.

Corrección necesaria: hacer explícita la calibración de ruido/ganancia y verificar la detección en archivos independientes con potencias distintas. No basta con bajar el umbral hasta que cada ejemplo pase.

### 2. La confianza final se declara calibrada sin validar esa transformación [alta]

En `src/monitor.py:269` se aplica el calibrador al margen promedio de varias ventanas, se multiplica por la proporción del voto ganador y se recorta a [0,20; 0,99]. Después se marca `confidence_calibrated=True`.

El calibrador disponible estima la probabilidad de acierto del SVM para una ventana individual. Su calibración no se transfiere automáticamente a esta nueva fórmula ni al voto mayoritario del archivo. Se debe validar/calibrar la salida final sobre archivos independientes, o identificarla como puntuación heurística. También falta comprobar compatibilidad entre un modelo alternativo `--model` y el calibrador cargado.

### 3. La ventaja adaptativa sobre la red sola no está demostrada estadísticamente [alta para el artículo]

La red acierta 1.238 de 2.400 observaciones y LBAP 1.239: **un acierto adicional**, +0,0417 puntos porcentuales. El F1 macro baja de 55,36% a 52,78%. El ahorro nominal de tiempo es 3,59 ms, aproximadamente 6,4%.

El experimento produce bootstrap frente a SVM256 y SVM512, pero no frente al experto multidominio, que es el mejor baseline fijo por utilidad en validación. La frase de superioridad frente a la red debe sustituirse por una descripción de resultados puntuales hasta contar con la comparación pareada pertinente.

Bajo cambio de canal, LBAP supera puntualmente al SVM256 en 1,29 puntos, pero su intervalo de confianza del 95% es aproximadamente **[−0,46; +3,00] puntos**: incluye cero. La utilidad de LBAP es 0,2400 frente a 0,2523 del SVM256. Existe evidencia de mejora de exactitud frente a SVM512, pero eso no demuestra superioridad frente a todos los baselines ni mejor utilidad OOD.

Fuentes: `results/paper_research/bootstrap_vs_a0.json`, `bootstrap_vs_a1.json` y `docs/paper/sections/06_conclusion.tex`.

### 4. La contabilidad de latencia omite trabajo de la ruta real [alta para el argumento de eficiencia]

En `experiments/exp12_paper_research.py:526`, el coste del selector se mide llamando a `choose()` con el vector de entrada ya construido. El tiempo de SVM256 se mide con `predict()`, mientras `observe_prefix()` necesita también `decision_function()`. La ruta requiere además calibración, entropía y construcción de características adicionales, según la política.

Posteriormente se suman costes medios constantes mediante `action_cost_ms()`. Esto no equivale a medir cada ruta completa con su coste efectivo. Tampoco se debe llamar a la suma de adquisición virtual y tiempos parciales una medición completa de pared.

Se debe cronometrar la ruta desplegable, incluyendo observación del prefijo, scores, calibración, selector y rama seleccionada, y separar la adquisición virtual. El ahorro reportado puede existir, pero todavía debe validarse con esa medición.

### 5. Una política de entradas más simples obtiene mejores resultados nominales [media]

La ablación con SNR ciega, planitud y entropía espectral llega a 52,29% y utilidad 0,4419; el vector completo llega a 51,625% y 0,4374. Por tanto, la hipótesis de que el vector completo supera esa alternativa no queda respaldada por estos resultados. La diferencia es puntual; hace falta un contraste pareado para afirmar superioridad estadística.

Esta es información científica útil y debe formar parte de las conclusiones. No se debe elegir ahora otra variante usando el test; cualquier nueva selección requiere validación y evaluación final independientes.

### 6. La comparación mezcla arquitectura y presupuesto de entrenamiento [media]

El experto nuevo combina el entrenamiento de `adaptive_dataset`, el de `e2_dataset` y síntesis de lotes al vuelo, mientras los SVM permanecen congelados. Es válido comparar estos sistemas finales, pero no atribuir toda la mejora exclusivamente a la arquitectura multidominio sin controlar el volumen y la distribución del entrenamiento.

Además, el entrenamiento del selector usa aciertos del experto sobre `adaptive_dataset/train`, que también se empleó para entrenar ese experto (`exp12_paper_research.py:359` y `:541`). Esto no demuestra fuga del test; sí introduce riesgo de que los objetivos de recuperabilidad aprendan un rendimiento de entrenamiento optimista. Conviene usar realizaciones independientes para desarrollar la política o predicciones fuera de pliegue.

### 7. El informe final contiene discrepancias verificables [alta para la entrega]

- `docs/INFORME_FINAL.md:67` afirma aproximadamente 50% de detección a −10 dB. `results/fase4/pd_agregado.csv` registra **20,5%**, 123/600, en ID.
- El informe cita 1.147,6 Hz para el ejemplo Python de interoperabilidad. El resumen guardado de GNU Radio reporta **5.292,52 Hz**, y el JSON del monitor **5.317,85 Hz**. No encontré respaldo en esos artefactos para el valor que se presenta como esa comparación. Un ancho de banda del 99% que incluye ruido debe distinguirse del ancho de banda de la señal útil.
- Se afirma cumplimiento del 100%, pero el comando predeterminado falla en detección con las demos y la SNR queda indeterminada sin referencia externa. Declarar honestamente la indeterminación es correcto; eso no demuestra que el desafío de señales desconocidas esté resuelto.
- El informe y README describen principalmente la CNN anterior, mientras el artículo usa otra red y otro conjunto de evaluación. Son experimentos distintos: deben identificarse con dataset, modelo y versión para que sus números no parezcan contradictorios.

### 8. Las métricas de calibración del artículo no corresponden a las fórmulas declaradas [media]

`experiments/exp10_adaptive.py:31` calcula Brier y NLL **binarios de acierto/error**, a partir de una confianza escalar. ECE utiliza diez intervalos de igual anchura. El experimento 12 reutiliza esa función sobre la máxima probabilidad.

Sin embargo, `docs/paper/sections/03_adaptive_policy.tex` declara Brier multiclase y bins de igual frecuencia. Ambas alternativas pueden ser útiles, pero representan métricas distintas. Hay que corregir las definiciones del artículo o recalcular las métricas correspondientes; las cifras actuales no prueban calibración multiclase completa.

### 9. ADC: resultado interesante, hipótesis mixta no confirmada [media]

A 4 bits, en el test nominal, los artefactos registran:

- SVM256: 24,08%.
- Red entrenada sin ADC adicional: 46,29%.
- Entrenamiento a 8 bits: 46,50%.
- Entrenamiento a 6 bits: 45,33%.
- Entrenamiento mixto: 44,96%.
- LBAP con experto limpio: 40,88%.

Esto respalda una mayor resistencia de las redes evaluadas respecto de los SVM en este ensayo. No respalda que el entrenamiento mixto mejore al limpio: aquí obtiene menos exactitud. La diferencia entre 46,50% y 46,29% necesita incertidumbre estadística antes de interpretarla como mejora. El clipping reportado para esta evaluación es 5,12% de componentes I/Q; la degradación combina cuantización y saturación.

Fuente: `results/paper_research/adc_quantization_experiment.csv`. No volví a ejecutar las variantes ADC.

### 10. Faltan garantías para reproducir y desplegar la nueva política [media]

El experimento 12 conserva checkpoints, cachés y métricas, pero no serializa el selector Fast LBAP final y su calibrador como un paquete de inferencia. El monitor continúa cargando el SVM de fase 5 y el artefacto adaptativo anterior para obtener confianza; no ejecuta la política nueva del artículo.

Las cachés se reutilizan por existencia del archivo sin comprobar hashes de datos, checkpoint o implementación. La inferencia nueva confirmó que la caché del experto actual sí coincide con el checkpoint y los tests actuales, pero el mecanismo puede quedar obsoleto tras cambios futuros. Se recomienda guardar configuración, hashes y política seleccionada antes de abrir una evaluación final nueva.

## Interpretación y prioridades

Como proyecto de curso, existe mucho trabajo aprovechable: generación, clasificación convencional y avanzada, validación por SNR, cambio de canal, artefactos y GNU Radio. La suite y la inferencia repetida aportan confianza en esa base.

Como investigación, la afirmación defendible es más limitada: el sistema multidominio mejora el test nominal, mientras la selección adaptativa presenta un posible compromiso entre exactitud y tiempo que aún requiere medición integral y comparación contra el mejor baseline. La robustez bajo cambio de canal sigue siendo insuficiente y la novedad bibliográfica no queda certificada por esta revisión.

Orden recomendado de corrección:

1. Resolver detector, metadatos de frecuencia de muestreo y confianza del monitor; validar las cinco salidas en archivos independientes con distintos niveles de potencia. La CLI asume actualmente 8 kHz si no recibe `--fs-hz`.
2. Corregir cifras y lenguaje del informe/artículo; distinguir resultados medidos, costes modelados e hipótesis.
3. Guardar una política nueva desplegable y medir sus rutas completas. Comparar con la red sola y con la alternativa de entradas simples, incluyendo F1 macro.
4. Si se modifica selección o entrenamiento, reservar una evaluación final nueva y repetir con varias semillas para cuantificar variación del entrenamiento.

No recomiendo añadir más arquitecturas antes de cerrar estos puntos.
