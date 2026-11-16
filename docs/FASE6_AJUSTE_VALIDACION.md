# Diagnóstico y ajuste mínimo de la CNN1D

La rotación aleatoria de fase durante entrenamiento **redujo la brecha train–val**, pero produjo una mejora pequeña de F1 macro val: **0,40542→0,41396**. No se ha comprobado mejora en un test independiente. Se mantiene la CNN original como referencia E3–E6 y la variante como candidata de desarrollo.

| Medida en checkpoint elegido por val | Original | Rotación train |
|---|---:|---:|
| Exactitud train, modo eval | 56,50% | 44,63% |
| Exactitud val, modo eval | 40,56% | 41,38% |
| Brecha train−val | 15,94 puntos | 3,25 puntos |
| F1 macro val | 0,405419 | 0,413958 |
| Entropía cruzada train, modo eval | 1,003966 | 1,177722 |
| Entropía cruzada val | 1,276134 | 1,206023 |
| Época elegida / final | 46 / 61 | 24 / 39 |

El diagnóstico original mostró pérdida train descendente y val estancada; la mínima pérdida val se alcanzó en época25, aunque el criterio prefijado F1 eligió época46. No cambiamos retrospectivamente ese criterio. A baja SNR, train podía memorizar realizaciones de ruido sin generalizar. La distribución está balanceada:600 señales por SNR y800 por clase en cada split;37,5% de las ventanas están entre−20 y−10dB. Por ello, una exactitud agregada moderada no basta para diagnosticar capacidad de la red. A SNR alta también había brecha train–val.

## Qué se cambió y por qué

Se conservó la CNN1D de94 726 parámetros, entrada2×256, centrado y normalización por ventana, train/val, semilla, AdamW, batch128, float32, scheduler y parada. Único cambio configurable: en cada minibatch train se aplica a cada ventana una fase uniforme independiente entre0 y2π, constante en sus256 muestras. La rotación conserva potencia y clase bajo el modelo de canal; también rota ruido complejo circular. No crea nuevas realizaciones ni agrega CFO, ni cambia val. La pérdida train durante entrenamiento incluye dropout y augmentation; para la tabla se recalculó sin ambos, haciendo comparables las dos columnas.

La prueba de sensibilidad original cambiaba34,38%,47,81%,49,46% de etiquetas val al rotar45°,90°,180°. En la candidata cambian18,04%,20,42%,18,58%. Esto respalda mayor estabilidad frente a fase global, pero no invariancia exacta ni robustez general a canal. La mejora no es uniforme por clase: F1 val16QAM cae de0,2967 a0,1695. Las métricas completas por clase y SNR permiten ver este costo; no se atribuye a la variante superioridad global demostrada.

La corrida final duró36,86s de bucle, con parada en época39 por15 épocas sin superar el F1 de época24. Su duración no se compara como benchmark de aceleración con la corrida anterior porque la carga del equipo varía. Un muestreo de GPU mostró868MiB,39% utilización y43°C, no un pico certificado. No se prolongó entrenamiento por el mero hecho de ser corto.

## Validez y protección del holdout

Antes de ajustar se generó un dataset nuevo con semilla2026110609 y los mismos rangos físicos; se reservan sus4800 señales test y4800 test_shift. No se usaron sus train/val. Se guardaron SHA256 y la verificación final confirma que permanecen iguales. No se calcularon predicciones en ese holdout ni se volvió a evaluar el test original. El sellado es administrativo, no un control criptográfico de acceso.

Solo hay una semilla y una corrida nueva. Val se reutilizó para diagnosticar y seleccionar: la diferencia0,00854 de F1 no es una estimación imparcial de mejora final y podría depender del azar. Augmentation consume RNG adicional, incluyendo cambios posteriores de secuencia de dropout; no se identifica un efecto causal promedio con esta réplica única. El nuevo holdout queda reservado para una decisión final posterior y debe evaluarse una sola vez cuando el pipeline completo quede congelado.

## Evidencia y reproducción

- [Protocolo previo y antecedentes](FASE6_AJUSTE_PROTOCOLO.md): lectura directa de Shahriar p.PDF2, secciónIII-A/figura2; la rotación es decisión propia, no método atribuido a ese paper.
- Diagnóstico original: `results/fase6_diagnostico/train_val.json`, `by_snr.csv`, `learning_diagnostics.png`.
- Variante: `results/fase6_phase_aug/best.pt`, `last.pt`, `history.json`, `config.json`, `integrity.json` y `ESTADO_REANUDACION.md`.
- Diagnóstico variante: `results/fase6_phase_diagnostico/train_val.json`, `by_snr.csv`, `learning_diagnostics.png`.
- **236 tests pasaron en17,26s**. Energía y producto temporal se conservan bajo augmentation. La recuperación GPU en procesos distintos conservó pesos/AdamW bit a bit e historial idéntico, incluyendo RNG de augmentation: `results/fase6_phase_resume/validation.json`.
- Hash de modelo original preservado y holdout intacto: `results/fase6_phase_aug/integrity.json`.

```powershell
.\.venv\Scripts\python.exe -m experiments.exp03_advanced --config configs/exp03_phase_aug.json --output results/phase_repeticion
.\.venv\Scripts\python.exe -m experiments.diagnose_advanced --model-root results/phase_repeticion --output results/phase_repeticion_diagnostico
```

No ejecutar `exp04_comparison` sobre esta variante como si el test antiguo fuera una evaluación nueva sin sesgo. La próxima decisión razonable es mantener acotado el desarrollo: integrar el monitor y calibrar confianza con partición de desarrollo apropiada, antes de abrir holdout final. Si posteriormente se cambia arquitectura o se amplían datos, fijar un protocolo nuevo; los resultados presentes no justifican una búsqueda masiva ni una afirmación de novedad. ADC sigue secundario y no se implementó. No se apagó el equipo.
