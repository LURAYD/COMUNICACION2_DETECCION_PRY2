# Auditoría inicial — Fase 0

> Registro histórico previo a las correcciones. La Fase 1 autorizada después de esta auditoría se documenta en `FASE1_VALIDACION.md`; los resultados de abajo describen el código original, no el estado corregido.

Fecha: 3 de noviembre de 2026. Alcance: inventario, revisión inicial de fuentes y pruebas diagnósticas del código existente. No se ha completado la Fase 1 ni se han entrenado modelos. Las implementaciones originales permanecen intactas.

## Inventario real previo a la auditoría

```text
COM2_Proyecto2/
  .gitignore
  .venv/                         entorno Python existente
  GUIA_MAESTRA_COMII_PARA_CODEX.md
  README.md
  notes/requisitos.md            plantilla todavía sin requisitos desarrollados
  src/
    .gitkeep
    signals.py                  cinco moduladores + RRC + pulse_shape
    channels.py                 solamente add_awgn
    test_signal.py              demostración gráfica BPSK, assert a nivel de módulo
    PRUEBA.ipynb                ejemplo de cuatro puntos I/Q; no modulador QPSK
    __pycache__/signals.cpython-311.pyc
  tests/.gitkeep
  results/.gitkeep
  references/                   17 PDF, un DOCX y LEER_PRIMERO.txt
```

El inventario individual de PDF, páginas y SHA-256 está en `results/auditoria/referencias.json`. Hay 15 artículos distintos, más guía y formulario. No hay `.git`: `git status --short` devolvió «not a git repository». No existe historial local para mostrar diffs o identificar commits. No se encontraron archivos AGENTS.md dentro del proyecto.

## Fuentes y obligaciones verificadas

Se leyó el texto y se inspeccionaron las imágenes de las páginas PDF 4–7 de `references/Guia_proyectos_COMII 2026-3.pdf` (Proyecto 2).

| Página PDF | Requisito comprobado |
|---|---|
| 4 | Cinco salidas del monitor; BPSK, QPSK, 8PSK, 16-QAM, 64-QAM y una clase adicional; variar SNR, fase, datos y amplitud; al menos dos perturbaciones de la lista. |
| 5 | Clasificador convencional y avanzado; una CNN específica no es obligatoria; uso complementario de GRC; experimentos E1–E3. |
| 6 | E4 comparación de métricas y tiempo; E5 robustez vs SNR; E6 canal distinto; independencia entre realizaciones de entrenamiento y prueba; nueve entregables. |
| 7 | Pesos de evaluación y énfasis en generalización. |

La guía maestra elige 4-PAM y CFO + multipath: son decisiones del proyecto, no elecciones impuestas por el docente. El RRC es parte del generador elegido; no trasladar automáticamente al Proyecto 2 todas las obligaciones del Proyecto 1.

El libro escaneado de Medina no está presente, como advierte `references/LEER_PRIMERO.txt`. No se le atribuye ninguna fórmula. El formulario está presente; la extracción de fórmulas es desordenada y no se utilizó para validar ecuaciones. La comparación bibliográfica inicial se encuentra en `MATRIZ_LITERATURA.md`; no equivale a una lectura completa de los 15 artículos.

Las notas DOCX se inspeccionaron mediante extracción de párrafos y tablas. No se validó su diseño visual ni sus ecuaciones: la extracción omite contenido matemático. Su tabla sobre el survey coincide en título, autores abreviados y DOI con el PDF original de Wang, Zhao y Huang, p. PDF 1. Muchos otros artículos citados en las notas no forman parte de los 15 PDF: sus afirmaciones quedan sin verificar. Las predicciones de superioridad de CNN en sus tablas son expectativas, no resultados de este proyecto.

## Funciones encontradas y problemas

| Componente | Evidencia y estado |
|---|---|
| BPSK | Convención efectiva 0 → +1, 1 → −1. El comentario inicial dice lo contrario. Acepta vacío y matrices; `uint8` produce 255 para el bit 1 por aritmética sin signo. |
| QPSK | No existe `qpsk_modulation`. Los cuatro puntos del notebook no constituyen una función de mapeo. |
| 8PSK | Mapeo Gray circular, energía unitaria, validación de vector y grupos de tres bits. |
| 16-QAM / 64-QAM | Constelaciones completas normalizadas y vecinos Gray comprobados con bits enteros. Aceptan 0.0/1.0 en la validación, pero fallan al indexar tablas con floats. |
| 4-PAM | Normalización por raíz de 5 y mapeo Gray comprobados. Convierte bits a enteros antes de indexar. |
| RRC | Existe dentro de `signals.py`; no hay `waveform.py`. Longitud 81 para sps=8, span=10, energía discreta 1 y simetría comprobadas en cuatro beta. No se certificó todavía la fórmula contra una fuente teórica. |
| pulse_shape | Inserta ceros y convoluciona en modo full; longitud `N*sps + len(h)-1`. No rechaza símbolos no finitos explícitamente. |
| AWGN | Valida vector finito y potencia no nula; ruido complejo con varianza objetivo por componente. Prueba estadística y repetición con semilla aprobadas. No es un generador de observaciones solo ruido. |
| CFO, fase, multipath | Ausentes. |
| Importaciones | `signals.py` imprime una demostración al importarse. `src/test_signal.py` usa `from signals`; depende de cómo se ejecuta y del sys.path. No se ha normalizado esta interfaz. |
| Pruebas antiguas | `src/test_signal.py` ejecuta assert y plt.show al importarse, sin funciones test. Su mensaje de assert dice «close to 0» aunque comprueba 1. |
| Resto | No hay dataset, detector, PSD del monitor, estimador SNR, clasificadores, monitor, flowgraphs ni experimentos E1–E6. |

## Entorno y ejecución real

Python local 3.11.0, Windows reportado por `platform` como `Windows-10-10.0.26200-SP0` (no se infiere edición comercial). NumPy 2.4.6 y Matplotlib 3.11.2 ya instalados. SciPy y pytest estaban ausentes. Solo se instaló pytest 9.2.0 y sus dependencias faltantes iniconfig 2.3.1 y pluggy 1.7.0. SciPy no es necesaria para estas pruebas y no se instaló. No se investigó GPU; no hace falta para esta fase. La consulta de CPU por CIM recibió acceso denegado; no se inventa el modelo del procesador.

La primera instalación por pip falló por resolución de red; el reintento autorizado fuera del sandbox terminó correctamente. Se añadió `requirements.txt` con las tres dependencias directas observadas. No se probó todavía una instalación desde cero en otro equipo.

Para lectura de fuentes se usaron pypdf, python-docx y Poppler del runtime de documentos, separados del entorno del proyecto. Los textos e imágenes de lectura están en `tmp/auditoria/`; no son resultados científicos.

Comandos ejecutados desde la raíz:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# Antes de instalar: No module named pytest.
.\.venv\Scripts\python.exe -m pip install pytest
$env:MPLBACKEND='Agg'
.\.venv\Scripts\python.exe -m pytest -q
# Código original: ninguna función de prueba recolectada; un warning de plt.show.
.\.venv\Scripts\python.exe -m pytest -q -rx
# Después de añadir diagnósticos: 20 passed, 6 xfailed, 1 warning in 2.05s.
```

Logs: `results/auditoria/pytest_inicial.txt` y `results/auditoria/pytest_diagnostico.txt`.

Los seis `xfail(strict=True)` reproducen problemas conocidos; **no son pruebas aprobadas ni correcciones**. Son QPSK ausente, BPSK vacío, BPSK matriz, BPSK uint8 y floats en dos QAM. Al corregir cada defecto debe retirarse su xfail; strict hace que una reparación sin actualizar el test produzca XPASS y falle la suite.

Las 20 pruebas aprobadas cubren energía de todas las combinaciones, número de puntos, tipo complejo y adyacencia Gray de los cinco moduladores presentes; bits inválidos; cuatro configuraciones RRC; longitud y componentes de pulse_shape; AWGN estadístico reproducible y cuatro entradas inválidas. Tolerancias de 0.08 dB para SNR, 2% para varianza y 0.03 para lóbulos del filtro adaptado son propuestas diagnósticas de ingeniería, no cifras atribuidas al libro. No se ejecutaron E1–E6, que todavía no existen.

## Explicación física

Un modulador convierte grupos de bits en símbolos. La energía media unitaria se verifica recorriendo todos los puntos equiprobables; una trama QAM aleatoria corta puede tener otra energía media. El RRC convierte símbolos en muestras de una señal continua aproximada: sps representa muestras por símbolo, no símbolos por segundo; `fs = sps * Rs`.

Con 81 coeficientes simétricos, el retardo del RRC es 40 muestras. Dos filtros adaptados acumulan 80 muestras; se comprobó un máximo unitario y residuos pequeños a intervalos de un símbolo. El truncamiento deja ISI residual: las pruebas no prueban ISI exactamente cero. AWGN reparte la potencia de ruido entre I y Q, por lo que cada componente tiene la mitad de la varianza compleja total. Esta SNR es por potencia de muestras y no se identifica automáticamente con Eb/N0.

## Archivos añadidos y siguiente fase

- `tests/test_auditoria.py`: diagnóstico reproducible sin modificar src.
- `requirements.txt`: dependencias directas mínimas observadas.
- `docs/AUDITORIA_INICIAL.md`, `MATRIZ_LITERATURA.md`, `DECISIONES.md`, `RESEARCH_GAP.md`: hallazgos, alcance y pendientes.
- `results/auditoria/`: logs e inventario verificable de referencias.

Siguiente cambio mínimo propuesto, Fase 1: mostrar el diff de `signals.py`, corregir entradas y efectos laterales, incorporar QPSK con tabla explícita y validar el RRC contra una fuente técnica. Mantener compatibilidad de imports al separar waveform si se decide hacerlo. Después ampliar tests y generar constelaciones/pulso. No añadir modelos ni política adaptativa.

Decisión del estudiante: aprobar esa Fase 1 y revisar la convención QPSK propuesta en DECISIONES.md. La guía maestra, apartados 0 y 10, reserva al estudiante la aprobación de cada fase y pide entregar una fase cada vez. Se termina esta auditoría en ese punto.
