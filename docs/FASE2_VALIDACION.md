# Fase 2: canales y perturbaciones

Fecha: 3 de noviembre de 2026. Estado: completada tras la indicación del usuario de continuar. Se conservaron los dos agentes de trabajo: Astra extra high para revisión/orquestación y Astra medium para código. No se inició Fase 3 ni se entrenaron modelos. ADC, opcional, se pospuso.

## Objetivo y archivos

Se integraron fase/CFO y multipath, y se reforzó la validación de AWGN sin cambiar su definición de SNR por potencia de muestras.

| Archivo | Cambio |
|---|---|
| `src/channels.py` | AWGN con validación numérica; fase/CFO con continuidad por índice absoluto; FIR completo sin normalización implícita. |
| `tests/test_channels.py` | 55 casos nuevos de validación física, entradas y reproducibilidad. |
| `experiments/diagnostico_fase2.py` | Cadena QPSK/RRC/canal, barrido pequeño de SNR y amplitud, PNG y JSON. |
| `docs/CANALES.md` | Ecuaciones, unidades, significado de SNR y coherencia de trama. |
| `docs/DECISIONES.md`, `README.md` | Alcance autorizado, convenciones y comandos. |

La propuesta `docs/PROPUESTA_FASE2.diff` se mostró antes de editar. El diff final está en `results/fase2/cambios.diff`; el archivo anterior en `results/fase2/antes/src/channels.py`. Se revisó el código aplicado; no se cambiaron los moduladores ni el RRC de Fase 1.

## Fuentes y decisiones

Se consultaron [MathWorks, PhaseFrequencyOffset, Algorithms](https://www.mathworks.com/help/comm/ref/comm.phasefrequencyoffset-system-object.html) y [awgn, opción measured](https://www.mathworks.com/help/comm/ref/awgn.html) como referencias externas; no se ejecutó MATLAB. Las fórmulas y la distinción respecto de Eb/N0 están en `CANALES.md`. No se atribuye teoría a Medina, que continúa ausente.

El FIR es constante por trama, con retardos enteros en muestras; no es Rayleigh ni un canal físico medido. La cadena se aplica completa antes de segmentar. `sample_offset` conserva continuidad específicamente de fase/CFO; no añade estado streaming a FIR ni AWGN. Los parámetros verdaderos y el ruido se usan solo para validar el simulador; todavía no existe un monitor sobre el cual verificar aislamiento de esos campos.

## Configuración y rangos comprobados

Diagnóstico con semilla 2026, 4096 símbolos QPSK, beta=0.35, span=10, Rs=1000 baud y sps=8; por tanto fs=8000 Hz. Fase inicial=0.6 rad, CFO=75 Hz y SNR=10 dB. Taps:

```python
[1, 0, 0.35j, 0, 0, 0.15*exp(-1j*pi/3)]
```

Sus retardos no nulos son 0/2/5 muestras: 0/250/625 microsegundos. Son elecciones ilustrativas de ingeniería; no se atribuyen a reflejos de una habitación ni a rangos impuestos por el docente.

Se prueban SNR −10/0/10/20 dB con tres semillas independientes (2026, 73 y 180), 200000 muestras por caso y tolerancia de 0.08 dB. Las varianzas I/Q se verifican al 2%, con media y correlación cruzada cercanas a cero. Son márgenes estadísticos de ingeniería, no cotas universales. El diagnóstico adicional usa amplitudes 0.5/1/1.5 con doce realizaciones de ruido distintas sobre la misma señal: no son doce tramas independientes para entrenamiento.

CFO se comprueba con desplazamientos −250/0/350 Hz, un caso analítico de cuatro muestras y una rotación fija de pi/2. FIR se verifica por identidad, ganancia, retardo puro, eco conocido, canal nulo y rechazo de desbordamiento.

## Ejecución real

```powershell
.\.venv\Scripts\python.exe -m pytest -q -W error
# Agente de código: 191 passed in 3.75s
# Revisión independiente: 191 passed in 1.48s

.\.venv\Scripts\python.exe -m experiments.diagnostico_fase2
.\.venv\Scripts\python.exe -m experiments.diagnostico_fase2 --output tmp/fase2_repeticion
```

No hubo fallos, xfail ni advertencias. La reejecución produjo PNG y JSON idénticos por SHA-256. Evidencia en `results/fase2/pytest.txt`, `revision_orquestador.txt`, `diagnostico.txt` y `reproducibilidad.json`. `metricas.json` guarda parámetros, semilla, versiones y hashes de fuentes. Se usó el entorno existente; no se instalaron dependencias.

## Resultados

| Magnitud | Valor observado |
|---|---:|
| Símbolos / muestras RRC / muestras con canal | 4096 / 32848 / 32853 |
| Potencia limpia | 0.1246828045 |
| Potencia después de multipath | 0.1422662321 |
| Potencia después de fase/CFO | 0.1422662321 |
| Error máximo de continuidad de portadora | 0 |
| SNR nominal / medida | 10 / 10.01434595 dB |
| Varianza de ruido objetivo por componente | 0.0071133116 |
| Varianza I / Q observada | 0.0070684895 / 0.0071111009 |
| Máximo error absoluto en el barrido de 12 casos | 0.03854971 dB |

`results/fase2/canal_diagnostico.png` fue inspeccionado visualmente por ambos agentes: cuatro paneles legibles, sin recortes ni solapamientos. Presenta taps, fase impuesta, observación con ruido y fluctuación de SNR. La fase dibujada es la del oscilador configurado; su aplicación a la señal se verifica mediante las pruebas analíticas y de continuidad.

## Explicación y límites

La fase fija rota toda la señal el mismo ángulo; CFO hace que ese ángulo crezca con el tiempo. Ambos conservan amplitud. Multipath suma copias retrasadas y puede cambiar potencia y producir ISI; por eso aquí la potencia aumentó antes de añadir ruido. AWGN reparte su potencia entre I y Q, y su SNR medida fluctúa porque cada realización es finita.

Esta SNR incluye transitorios y usa potencia de señal conocida por el simulador. No es un estimador ciego ni un piso térmico fijo: al variar la amplitud, se ajusta también el ruido para conservar la SNR nominal. No se implementaron retardos fraccionales, canal variable en el tiempo, ADC ni GRC. La aritmética de portadora usa float64, con precisión limitada para índices/fases extremos, según su docstring.

Siguiente cambio mínimo propuesto: Fase 3, generación reproducible de tramas independientes, observaciones solo ruido y ventanas 256/512 agrupadas por realización madre. Esta fase queda sin ejecutar hasta la siguiente autorización.
