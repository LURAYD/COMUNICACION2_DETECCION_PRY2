# Fase 1: modulaciones y generador RRC

Fecha: 3 de noviembre de 2026. Estado: implementación y validación técnica completadas. Se ejecutó la corrección autorizada por el usuario después de la auditoría. No se inició Fase 2 ni se entrenaron modelos.

## Objetivo y cambios

Se corrigieron los seis defectos reproducidos por los xfail de Fase 0: QPSK ausente, BPSK con uint8, BPSK vacío, BPSK 2D y entradas float en 16-QAM/64-QAM. Los seis xfail se retiraron y ahora pasan como pruebas ordinarias. Se eliminó la impresión al importar `signals` y la ejecución gráfica al importar la demostración.

| Archivo | Cambio |
|---|---|
| `src/signals.py` | Validador binario compartido, seis moduladores, tablas previas conservadas, reexportación RRC. |
| `src/waveform.py` | RRC y conformación extraídos; símbolos finitos, parámetros y escalares NumPy validados. |
| `src/__init__.py` | Paquete explícito. |
| `src/test_signal.py` | Demostración bajo main, import consistente y mensaje de energía corregido. |
| `tests/test_auditoria.py` | Regresiones originales sin xfail; QPSK incluida. |
| `tests/test_fase1.py` | Tablas exactas, tipos/entradas, imports, límites RRC, retardo, recuperación y espectro. |
| `pytest.ini` | Recolección exclusiva desde tests. |
| `experiments/__init__.py`, `experiments/diagnostico_fase1.py` | Diagnóstico determinista con tres figuras y métricas. |
| `README.md`, `docs/MAPPING_MODULACIONES.md`, `docs/MODELO_DE_SENAL.md`, `docs/DECISIONES.md` | Ejecución, convenciones y explicación física. |
| `docs/AUDITORIA_INICIAL.md` | Aviso de que describe el estado histórico previo; se mantienen sus resultados originales. |

El diff se mostró antes de editar: `docs/PROPUESTA_FASE1.diff`. El diff funcional aplicado está en `results/fase1/cambios.diff`; los tres archivos originales reemplazados se conservan en `results/fase1/antes/`. No existe Git. No se modificó `src/channels.py` ni el notebook.

## Fuentes realmente consultadas

Medina sigue ausente. Se consultó [OptiCommPy, rrcFilterTaps, ecuaciones (1)–(3)](https://opticommpy.readthedocs.io/en/stable/_modules/optic/dsp/core.html#rrcFilterTaps) para la fórmula y límites; [MathWorks, rcosdesign](https://www.mathworks.com/help/signal/ref/rcosdesign.html) para normalización, longitud y efecto del truncamiento. No se ejecutaron esas bibliotecas.

Se descargó y revisó visualmente `references/externas/SquareRootRaisedCosine.pdf` de [Purdue](https://engineering.purdue.edu/~ee538/SquareRootRaisedCosine.pdf), páginas PDF/impresas 1, 3 y 5. El espectro RC de la ecuación (1) sirve como contraste independiente. Se detectó y documentó una diferencia de factor global en su ecuación (5); no se usó como única autoridad de la fórmula RRC. Detalles y enlace de origen en `MODELO_DE_SENAL.md` y `references/externas/README.md`.

## Pruebas realmente ejecutadas

Desde la raíz, con el entorno existente Python 3.11.0, NumPy 2.4.6, Matplotlib 3.11.2 y pytest 9.2.0:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# 136 passed in 1.43s

.\.venv\Scripts\python.exe -m pytest -q -W error
# Revisión independiente del orquestador: 136 passed in 1.34s

.\.venv\Scripts\python.exe -m experiments.diagnostico_fase1
# Tres PNG y metricas.json generados correctamente.

.\.venv\Scripts\python.exe -m experiments.diagnostico_fase1 --output tmp/fase1_repeticion
# Los tres PNG y el JSON son idénticos por SHA-256 a la primera ejecución.
```

La suite cubre todas las palabras binarias y vecinos Gray de las seis constelaciones; también verifica tablas exactas para detectar permutaciones que conservarían energía. RRC se contrasta por límites singulares, simetría, energía, respuesta impulsional, recuperación de símbolos y espectro RC ideal. Las pruebas AWGN existentes pasan como regresión, sin ampliar el canal.

Evidencia: `results/fase1/pytest.txt`, `diagnostico.txt`, `reproducibilidad.json` y `metricas.json`. Este último registra semilla 2026, versiones y hashes del código generador. No se instalaron dependencias adicionales en Fase 1. La compilación de candidatas también pasó antes de la aplicación.

## Resultados medidos

| Comprobación | Resultado |
|---|---:|
| Modulaciones con puntos/energía/Gray correctos | 6/6 |
| Pruebas aprobadas / xfail | 136 / 0 |
| Energías equiprobables | 1; 64-QAM: 1.0000000000000002 por redondeo |
| RRC beta / sps / span | 0.35 / 8 / 10 |
| Coeficientes / energía discreta | 81 / 1 |
| Retardo TX / TX+RX, muestras | 40 / 80 |
| Máximo residuo de respuesta adaptada en instantes de otros símbolos | 0.0058165683 |
| Máxima diferencia absoluta entre espectro `abs(H)^2/sps` y RC ideal | 0.0123060905 |
| Trama diagnóstica QPSK | 32 símbolos, 336 muestras |
| Error máximo / RMS después de filtro adaptado y compensar retardo | 0.0140774008 / 0.0080965360 |

El error espectral se evaluó con FFT de 65536 puntos sobre la banda digital completa. Las tolerancias 0.03 para ISI/recuperación y 0.04 para diferencia espectral son criterios de ingeniería para las configuraciones probadas, no cotas del libro ni garantías para todo beta/span. La recuperación automática de la suite utiliza otra trama de 64 símbolos; las métricas de esta tabla corresponden al diagnóstico de 32.

Figuras inspeccionadas por ambos agentes, sin texto cortado ni ejes físicos inventados:

- `results/fase1/constelaciones.png`: seis constelaciones completas; bits rotulados salvo 64-QAM para mantener legibilidad.
- `results/fase1/rrc_diagnostico.png`: pulso, respuesta adaptada y espectro frente a RC ideal.
- `results/fase1/iq_qpsk.png`: I/Q de QPSK, transitorios y retardo TX.

## Explicación para el estudiante

Un símbolo representa un grupo de bits; una muestra I/Q es un punto temporal de la señal conformada. QPSK agrupa dos bits y divide las componentes ±1 entre √2 para que la energía del símbolo sea 1. QAM y PAM se normalizan usando la energía media de todos los puntos; una trama corta no tiene por qué dar exactamente esa media.

El filtro RRC reparte la energía de cada símbolo entre muestras y limita el espectro. Por eso la amplitud de una muestra de la gráfica no tiene que ser 1. El retardo de 40 muestras proviene del centro de los 81 coeficientes; otro filtro adaptado añade 40. Al observar cada ocho muestras desde el retardo total de 80 se recuperan símbolos cercanos a los transmitidos.

El error residual es esperado: el pulso ideal dura infinitamente y aquí se conserva solo un tramo de diez símbolos. La respuesta combinada no tiene ceros perfectos en todos los instantes vecinos. Tampoco la energía total de esta trama filtrada es exactamente 32: se midió 31.9474180 debido a términos cruzados de pulsos truncados; no se forzó una renormalización por trama.

## Limitaciones y siguiente paso

Esta validación corresponde a banda base con sincronización ideal. Las figuras no prueban reconocimiento de modulaciones, robustez a canales ni desempeño de un clasificador. Beta=0 sigue fuera de la API. No se verificó una instalación limpia en otro equipo y no se ejecutó GNU Radio. Falta incorporar Medina si se desea contrastar específicamente con ese libro.

La decisión de QPSK está documentada y aplicada conforme a la propuesta autorizada. El estudiante debe revisar las normalizaciones y la diferencia símbolo/muestra antes de la siguiente fase. Siguiente cambio mínimo propuesto: Fase 2, validar AWGN y añadir fase/CFO/multipath con unidades y pruebas. No se ejecutó automáticamente.
