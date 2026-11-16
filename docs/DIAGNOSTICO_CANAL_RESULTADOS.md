# Diagnóstico exploratorio: canal y complementariedad

El ensayo controlado apunta a **CFO fuera de rango como principal debilidad de las CNN y multipath como principal debilidad de los SVM**, entre los factores ensayados. Beta no cambió entre ID y shift históricos y no explica aquella caída. Los modelos tienen errores complementarios, pero todavía no se ha demostrado que un selector observable pueda aprovecharlos.

No se entrenó, ajustó ni sustituyó ningún modelo. No se implementaron GNU Radio, ADC ni política adaptativa.

## Escenarios controlados

720 padres nuevos, semilla 2026110611: seis clases × SNR −5/0/10/20 dB × 30. Cada padre tiene nueve versiones pareadas. Las cifras agregadas **no son comparables directamente con el holdout anterior**, que tenía otra mezcla de SNR y recorte. Son diagnósticos exploratorios, no resultados finales nuevos.

| Escenario | CNN original | CNN fase | SVM256 | SVM512 |
|---|---:|---:|---:|---:|
| ID | 53,33% | 54,44% | 55,69% | 62,08% |
| CFO = 0, resto ID | 55,00% | 55,97% | 55,69% | 59,72% |
| Solo CFO fuera de rango | 27,78% | 28,75% | 55,97% | 59,31% |
| Solo multipath distinto | 35,56% | 35,00% | 32,78% | 36,67% |
| CFO y multipath distintos | 22,36% | 21,94% | 33,06% | 32,92% |
| Beta fija 0,25, resto ID | 53,47% | 54,03% | 54,31% | 62,36% |
| Beta fija 0,35, resto ID | 55,42% | 54,72% | 55,28% | 62,08% |
| Beta fija 0,50, resto ID | 54,86% | 54,31% | 56,11% | 63,33% |
| Beta fija 0,65, resto ID | 55,69% | 54,86% | 56,67% | 65,14% |

Solo CFOshift frente a ID: CNN original cae **25,56 puntos** (IC95% descriptivo [−29,58; −21,53]); CNN fase cae 25,69. SVM256 cambia +0,28 [−3,33; +3,89] y SVM512 −2,78 [−6,39; +0,69]. Esto es compatible con que la rotación global de augmentation no enseña invariancia frente a una fase que crece muestra a muestra por CFO. No demuestra el mecanismo interno exacto de la red.

Solo multipathshift: CNN original cae **17,78 puntos** [−21,67; −14,03], SVM256 **22,92** [−26,81; −19,03] y SVM512 **25,42** [−29,03; −21,81]. Por tanto, la aparente robustez de las características frente a CFO no implica robustez frente a mezcla retardada de símbolos. Cambiar el banco multipath modifica retardos y ganancias conjuntamente; no se puede atribuir todo al retardo aislado.

Beta0,65 frente a0,35 cambia CNN original +0,28 puntos [−1,94; +2,36]. No aparece deterioro dominante de roll-off aquí; no equivale a demostrar invariancia para otros valores/canales. El cambio histórico tampoco variaba beta. Las pérdidas CFO y multipath no deben sumarse: la interacción factorial de accuracy es +12,36 puntos para CNN original, +12,64 para CNN fase, 0 para SVM256 y −0,97 para SVM512, descriptivamente. Esto refleja no aditividad de errores y límites de la métrica, no un beneficio físico de combinar perturbaciones.

Los IC usan 10 000 réplicas de padres pareados, estratificados por clase/SNR. Son exploratorios, sin corrección por múltiples comparaciones ni variabilidad entre semillas de entrenamiento.

## Complementariedad y techo oracle

La tabla siguiente muestra decisiones sobre los mismos 720 padres. «Solo A/B» equivale a correcciones potenciales de errores del otro modelo, pero elegir siempre uno también rompe aciertos del otro.

| Régimen; A / B | Ambos aciertan | Solo A | Solo B | Ambos fallan |
|---|---:|---:|---:|---:|
| ID; CNN original / SVM256 | 256 | 128 | 145 | 191 |
| ID; SVM256 / SVM512 | 330 | 71 | 117 | 202 |
| Ambos shift; CNN original / SVM256 | 71 | 90 | 167 | 392 |
| Ambos shift; SVM256 / SVM512 | 147 | 91 | 90 | 392 |

Así, ampliar de256 a512 en ID corrige117 errores y rompe71 aciertos; con ambos shift corrige90 y rompe91. Ejecutar CNN original después de SVM256 ofrece128 correcciones potenciales en ID, pero tiene145 errores donde SVM acertaba. No basta con que otra rama pueda corregir: hay que predecir cuándo beneficiará.

| Régimen | Mejor modelo fijo, descriptivo | Oracle SVM256/SVM512/CNN original | Oracle cuatro modelos |
|---|---:|---:|---:|
| ID | 62,08% | 82,64% | 86,53% |
| Solo CFOshift | 59,31% | 78,06% | 81,11% |
| Solo MPshift | 36,67% | 59,03% | 63,47% |
| Ambos shift | 33,06% | 55,56% | 59,17% |

**El oracle utiliza etiquetas verdaderas y elige una respuesta correcta si existe. No es una política implementable ni un ahorro medido.** El techo grande puede coexistir con un selector imposible de aprender desde los primeros256 puntos. Las seis parejas, cuatro SNR y nueve escenarios están en `complementarity.csv` (270 filas); `oracle_upper_bounds.csv` incluye las ternas con cada CNN. No se ajustó selector usando estos datos.

## Tiempo de observación y cómputo

Fs=8000 muestras/s: adquirir256 cuesta32ms y ampliar a512 requiere32ms adicionales. Las mediciones previas batch1, un hilo CPU y preprocesamiento incluido fueron SVM256≈1,618ms, SVM512≈1,995ms y CNN≈0,422ms CPU (≈0,962ms GPU). Son referencias de Fase6, no mediciones nuevas de esta corrida.

Si primero se ejecuta SVM256, la acción a1 pagaría además SVM512 y la observación adicional; a2 pagaría además la CNN sobre las mismas256 muestras. Sin reutilización, sumar medianas da solo una **aproximación contable**, no la mediana medida del recorrido completo: a0≈33,62ms, a1≈67,61ms, a2 CPU≈34,04ms, sin costo de selector/detección. Ejecutar CNN directamente sería≈32,42ms: la CNN aislada es más barata que el SVM de este pipeline. Por eso «modelo avanzado» no significa automáticamente «más cómputo», aunque añadirlo después sí tenga costo incremental. El oracle no recibe una cifra de ahorro porque no existe un selector real.

## Validez física y evidencia

Se usan los bloques validados de modulación, RRC, FIR, CFO y AWGN. Bits, amplitud, fase, timing y ruido normalizado se comparten entre versiones. FIR con ceros hasta14 taps y recorte común después de13 muestras de retardo máximo evitan mover la ventana al cambiar canal. Este recorte controlado difiere del generador histórico, explícitamente. El prefijo256 coincide exactamente con512.

La SNR se define por potencia de muestras de trama antes de AWGN, no Eb/N0; el ruido cambia de escala si cambia esa potencia. La SNR medida en ventana supera la nominal en promedio ≈0,366dB en ID y≈0,370dB con multipathshift, por recorte de regiones de menor potencia y fluctuaciones finitas. No se ajustó esa diferencia artificialmente ni se entregó SNR verdadera a los clasificadores. Las versiones no son realizaciones independientes: la unidad experimental es el padre.

Dos tests pasaron en0,51s: reproducibilidad, independencia entre padres, bits/ruido pareados, aislamiento CFO/beta y conservación de magnitud. La auditoría final confirmó720 padres,6480 versiones,25920 predicciones, prefijos exactos, parámetros compartidos y270 tablas consistentes. Los hashes de modelos permanecen iguales. Se inspeccionaron ambas figuras. [Protocolo detallado](DIAGNOSTICO_CANAL_PROTOCOLO.md).

## Contraste bibliográfico y siguiente paso

- **BEACON**, p.PDF12/impresa40362, secciónVII-B y conclusión: distingue beneficio de profundizar de confianza genérica y usa el vector de probabilidades temprano para predecir recuperabilidad. Nuestros conteos de correcciones/regresiones justifican estudiar ese objetivo, pero no prueban que nuestra elección conjunta sea novedosa ni que un predictor funcione.
- **Fewer-Sample**, p.PDF2/impresa465, secciónII y figura1: separa adquisición y cálculo; además presupone recuperación de portadora/temporización. La sensibilidad observada a CFO impide comparar directamente sus resultados con nuestras ventanas sin sincronización corregida.
- **FFTFormer**, p.PDF9/impresa1429, secciónIV-A: incluye beta0,20/0,35/0,50/0,65 en su adaptación de RML22. Motiva una prueba separada de roll-off; no implica que beta haya causado nuestra caída histórica. Se releen las páginas originales; no se reproducen sus porcentajes ni arquitecturas.

La hipótesis ahora razonable es que el beneficio de más observación o de otra representación depende del tipo de degradación; la dificultad es inferirlo desde observables de256 muestras. Siguiente paso propuesto: definir una evaluación de confianza/beneficio con datos de desarrollo separados y baselines sencillos, además de la integración pendiente del monitor. No se ejecuta esa fase aquí. Cualquier corrección de CFO o nuevo entrenamiento debe tener protocolo propio y no reutilizar el holdout consumido como prueba independiente. ADC sigue secundario; GNU Radio permanece expresamente aplazado.

## Archivos y recuperación

Resultados: `results/channel_diagnostic/`; `metrics.csv`, `paired_scenario_effects.json`, `complementarity.csv`, `oracle_upper_bounds.csv`, `verification.json`, observaciones y metadatos separados por escenario. Figuras: `controlled_effects.png`, `oracle_complementarity.png`. Modelos originales intactos.

Reproducción/reanudación: `.\.venv\Scripts\python.exe -m experiments.exp09_channel_diagnostic`. Reutiliza predicciones atómicas ya completas; para una carpeta nueva añadir `--output results/diagnostico_repeticion`. Una repetición con igual semilla no es evidencia independiente. No quedan cálculos pendientes ni apagado programado.
