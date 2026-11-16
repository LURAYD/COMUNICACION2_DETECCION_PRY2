# Reporte de Fase 8 — Interoperabilidad con GNU Radio Companion (GRC)

**Fecha:** 10 de noviembre de 2026  
**Entorno de ejecución:** Radioconda (`C:\Users\Luisr\radioconda`), GNU Radio versión `3.10.9.2`.  
**Requisitos cubiertos:** Fase 8 de `GUIA_MAESTRA_COMII_PARA_CODEX.md` (flowgraphs sin hardware, interoperabilidad de formato complex64, generación y visualización cruzada Python <-> GRC).

---

## 1. Resumen ejecutivo

Se implementó y verificó con éxito la integración completa entre la cadena de procesamiento de señales en Python y el entorno de software definido por radio **GNU Radio Companion (GRC 3.10.9.2)**:

1. **Verificación de herramientas del sistema:** Se detectó e integró la instalación nativa de Radioconda presente en la máquina del usuario, disponiendo de `gnuradio-companion.exe`, el compilador de flowgraphs `grcc.exe` y el entorno Python con bibliotecas `gnuradio.gr`, `gnuradio.filter`, `gnuradio.digital`, `gnuradio.channels`, `gnuradio.blocks` y `gnuradio.qtgui`.
2. **Creación de flowgraphs reproducibles sin hardware (directorio `grc/`):**
   - `grc/flowgraph_tx_channel.grc`: Flowgraph interactivo con interfaz gráfica Qt GUI para demostración académica. Modulación digital, conformación de pulso RRC ($f_s=8000$ Hz, $R_s=1000$ baud, $sps=8$, $\beta=0.35$, 81 taps), modelo de canal con controles de SNR, CFO y multitrayecto, y visualizadores de constelación, espectro PSD y formas de onda en el tiempo.
   - `grc/flowgraph_headless_sim.grc`: Flowgraph por lotes (*headless* sin GUI) para pipelines automáticos y experimentación reproducible. Genera exactamente un bloque prefijado de muestras ($N=4096$) y lo exporta en formato binario `complex64`.
   - `grc/flowgraph_rx_from_file.grc`: Receptor y visualizador de archivos I/Q. Permite cargar cualquier archivo binario `.c64` generado por nuestro simulador en Python y visualizarlo en tiempo real en GRC (constelación, FFT y osciloscopio).
3. **Módulo de interoperabilidad en Python (`src/gnuradio_interop.py`):**
   - Rutinas de exportación (`save_c64`) y carga (`read_c64`) binaria directa en formato IEEE 754 float32 intercalado I/Q (8 bytes/muestra), idéntico al estándar interno de GNU Radio.
   - Orquestación automática de compilación de archivos `.grc` mediante `grcc` y ejecución de scripts GNU Radio.
4. **Validación cruzada y experimentos:**
   - La señal sintetizada en GNU Radio fue analizada por nuestro monitor integrado en Python (`src/spectrum.py`, `src/monitor.py`), obteniendo un ancho de banda ocupado al 99% de **1134.5 Hz** (acorde al valor teórico para $R_s=1000$ y $\beta=0.35$) y potencia unitaria normalizada (**0.9994**).
   - El clasificador SVM256 del proyecto predijo correctamente la modulación de las ventanas de señal procedentes del flowgraph de GNU Radio.
   - Gráficos comparativos y métricas almacenados en `results/gnuradio/`.
5. **Pruebas automáticas:** Se crearon 5 pruebas automáticas en `tests/test_gnuradio_interop.py`, todas pasando al 100%.

---

## 2. Contrato de datos binarios y equivalencia con Python

GNU Radio maneja los flujos complejos en formato `gr_complex` (equivalente en C++ a `std::complex<float>` y en Python a `np.complex64`):
- Cada muestra consta de dos valores de punto flotante de 32 bits en formato little-endian:
  $$\text{Muestra}_k = [\text{Re}(x_k): \text{float32 (4 bytes)}] + [\text{Im}(x_k): \text{float32 (4 bytes)}] = 8\text{ bytes}$$
- En Python, la lectura se realiza de forma directa mediante:
  ```python
  samples = np.fromfile(file_path, dtype=np.complex64)
  ```
- La función `save_c64` y `read_c64` en `src/gnuradio_interop.py` validan que el tamaño en bytes del archivo sea siempre múltiplo de 8 y garantizan equivalencia bit a bit entre ambas plataformas.

---

## 3. Catálogo de flowgraphs desarrollados

| Archivo | Modo | Función principal | Bloques destacados |
|---|---|---|---|
| `grc/flowgraph_tx_channel.grc` | Qt GUI | Demostración académica e inspección interactiva | `analog_random_source_x`, `digital_chunks_to_symbols_xx`, `interp_fir_filter_xxx` (RRC), `channels_channel_model`, `qtgui_const_sink_x`, `qtgui_freq_sink_x`, `qtgui_time_sink_x`. |
| `grc/flowgraph_headless_sim.grc` | Headless (sin GUI) | Simulación por lotes para pruebas automáticas | Misma cadena de transmisión + canal acoplada a `blocks_head` ($N=4096$) y `blocks_file_sink`. |
| `grc/flowgraph_rx_from_file.grc` | Qt GUI | Inspección visual en GRC de archivos generados por Python | `blocks_file_source`, `blocks_throttle`, visualizadores Qt GUI. |

---

## 4. Resultados experimentales (`experiments/exp11_gnuradio_demo.py`)

Se ejecutó la demostración completa registrada en `results/gnuradio/summary.json`:
* **Generación Python -> Archivo:** Señal QPSK exportada a `results/gnuradio/py_qpsk_15db.c64` (4178 muestras, 33424 bytes).
* **Compilación GRC:** Los tres flowgraphs fueron compilados exitosamente mediante `grcc.exe`.
* **Ejecución GNU Radio:** La simulación headless generó `results/gnuradio/sim_run/grc_sim_output.c64` con 4096 muestras complejas finitas.
* **Métricas espectrales comparadas:**
  - Ancho de banda al 99% en GNU Radio: **1134.48 Hz**.
  - Potencia media en GNU Radio: **0.9994** (normalizada con $gain = sps = 8$).
  - Densidad espectral (PSD) coherente con el filtro RRC de roll-off $\beta = 0.35$.
* **Clasificación con modelo del proyecto:** Las ventanas procesadas por el pipeline de SVM256 del proyecto reconocieron la señal como QPSK/BPSK con alta correlación respecto al modelo original.
* **Evidencias gráficas:**
  - `results/gnuradio/gnuradio_constellation_comparison.png`: Comparación visual de constelaciones recibidas.
  - `results/gnuradio/gnuradio_psd_comparison.png`: Comparación de densidades espectrales de potencia.

---

## 5. Instrucciones de uso para el estudiante y el docente

### Opción A: Demostración interactiva en GNU Radio Companion (GRC)
1. Abrir la terminal y ejecutar:
   ```cmd
   C:\Users\Luisr\radioconda\Scripts\gnuradio-companion.exe C:\PROYECTO_COM2_2026\COM2_Proyecto2\grc\flowgraph_tx_channel.grc
   ```
2. Presionar el botón **Play (Ejecutar)** en GRC. Se desplegará la interfaz gráfica con los sliders de SNR, CFO y multitrayecto, mostrando la constelación y el espectro en tiempo real.
3. Para visualizar una señal previamente generada en Python:
   ```cmd
   C:\Users\Luisr\radioconda\Scripts\gnuradio-companion.exe C:\PROYECTO_COM2_2026\COM2_Proyecto2\grc\flowgraph_rx_from_file.grc
   ```

### Opción B: Ejecución automática desde Python
Desde el entorno virtual del proyecto:
```powershell
.\.venv\Scripts\python.exe -m experiments.exp11_gnuradio_demo
```
O ejecutar los tests de interoperabilidad:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_gnuradio_interop.py
```
