# Evaluación final independiente autorizada

Se evalúan una vez, sin entrenamiento ni ajustes posteriores: CNN original best época46, CNN con rotación train best época24, SVM256 y SVM512 congelados de Fase5. El script registra sus SHA256 y configuración antes de inferir.

Dataset reservado semilla2026110609: test y test_shift,4800 señales presentes cada uno,100 por celda clase/SNR. Se verifican hashes contra reserva, configuración física idéntica salvo semilla, padres separados y ausencia de ventanas256 idénticas a cualquier split E2. No se usan los train/val del nuevo dataset. Los modelos comparten exactamente realizaciones; SVM512 observa el prefijo extendido. La población excluye noise-only y clases desconocidas.

Comparación primaria: CNN con rotación menos CNN original. Diferencia pareada de accuracy e IC percentil95% con10000 réplicas, semilla2026110610, remuestreo de padres dentro de cada celda clase/SNR. El muestreo multinomial de diferencias−1/0/+1 es equivalente para esa estadística. Comparaciones con SVM son descriptivas y no corrigen multiplicidad. Se reportan además F1 macro, precision/recall, métricas por clase/SNR y matrices. Los intervalos condicionan a modelos fijos y simulador; no incorporan variabilidad de entrenamiento (una semilla), incertidumbre del modelo de canal ni datos RF reales.

test_shift cambia conjuntamente CFO y multipath; no identifica su contribución causal separada. Ningún resultado se usará para volver a ajustar ni elegir una nueva variante en esta intervención. El resultado final es del clasificador, no del monitor completo ni de confianza calibrada. Al abrirse esta reserva deja de ser un holdout sellado para futuros desarrollos.
