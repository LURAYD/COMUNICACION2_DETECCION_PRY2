"""Experimento 11: Demostración de interoperabilidad con GNU Radio (Fase 8).

Orquesta:
1. Exportación de señales sintetizadas en Python a archivos binarios complex64 (.c64).
2. Compilación de flowgraphs .grc con grcc (GNU Radio Companion Compiler).
3. Ejecución de la simulación headless de GNU Radio sin hardware.
4. Ingesta y validación en Python mediante el monitor integrado y el clasificador.
5. Generación de artefactos de evidencia (JSON de métricas y gráficos comparativos).
"""
from pathlib import Path
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.gnuradio_interop import (
    save_c64,
    read_c64,
    compile_grc,
    run_headless_simulation,
    is_radioconda_available,
    DEFAULT_RADIOCONDA_PATH,
)
from src.signals import qpsk_modulation, rrc_filter
from src.waveform import pulse_shape
from src.channels import add_awgn, apply_carrier_impairments, apply_multipath
from src.spectrum import estimate_spectrum
from src.monitor import analyze


def run_gnuradio_demonstration(output_dir="results/gnuradio"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    grc_dir = Path("grc")

    print("=== Fase 8: Demostración e Interoperabilidad con GNU Radio ===")
    radioconda_ok = is_radioconda_available()
    print(f"Entorno Radioconda disponible: {radioconda_ok} ({DEFAULT_RADIOCONDA_PATH})")

    results = {
        "radioconda_path": str(DEFAULT_RADIOCONDA_PATH),
        "radioconda_available": radioconda_ok,
        "gnuradio_version": "3.10.9.2" if radioconda_ok else None,
        "flowgraphs": [],
        "python_to_grc": {},
        "grc_to_python": {},
    }

    # 1. Generar señal representativa en Python y exportar a formato complex64
    print("\n[1/4] Generando señal QPSK en Python y exportando a .c64...")
    rng = np.random.default_rng(20261110)
    bits = rng.integers(0, 2, size=1024)
    symbols = qpsk_modulation(bits)
    # Filtro RRC sps=8, span=10, beta=0.35
    shaped, _ = pulse_shape(symbols, sps=8, beta=0.35, span=10)
    # Canal: CFO=50 Hz, multipath suave y AWGN a 15 dB
    carr = apply_carrier_impairments(shaped, fs_hz=8000.0, cfo_hz=50.0, phase_rad=0.3)
    mp = apply_multipath(carr, taps=[1.0, 0.0, 0.15j])
    py_rx, _ = add_awgn(mp, snr_db=15.0, rng=rng)
    py_c64_file = out_path / "py_qpsk_15db.c64"
    save_c64(py_c64_file, py_rx)
    print(f"  Guardado: {py_c64_file} ({py_c64_file.stat().st_size} bytes, {len(py_rx)} muestras)")

    py_spec = estimate_spectrum(py_rx[200:456], fs_hz=8000)
    results["python_to_grc"] = {
        "file": str(py_c64_file),
        "num_samples": len(py_rx),
        "mean_power": float(np.mean(np.abs(py_rx) ** 2)),
        "bandwidth_99_hz": float(py_spec["total_power_bandwidth"]),
    }

    # 2. Compilar flowgraphs .grc
    print("\n[2/4] Compilando flowgraphs .grc con grcc...")
    flowgraph_files = [
        grc_dir / "flowgraph_headless_sim.grc",
        grc_dir / "flowgraph_tx_channel.grc",
        grc_dir / "flowgraph_rx_from_file.grc",
    ]
    compiled_scripts = {}
    for fg in flowgraph_files:
        if fg.is_file():
            py_script = compile_grc(fg, output_dir=grc_dir)
            compiled_scripts[fg.stem] = py_script
            results["flowgraphs"].append({
                "grc_file": str(fg),
                "compiled_script": str(py_script),
                "status": "compiled_successfully",
            })
            print(f"  Compilado: {fg.name} -> {py_script.name}")

    # 3. Ejecutar simulación headless de GNU Radio
    print("\n[3/4] Ejecutando simulación headless en GNU Radio...")
    headless_script = compiled_scripts["flowgraph_headless_sim"]
    sim_dir = out_path / "sim_run"
    sim_dir.mkdir(parents=True, exist_ok=True)
    run_headless_simulation(headless_script, work_dir=sim_dir, timeout_sec=25)

    grc_c64_file = sim_dir / "grc_sim_output.c64"
    if not grc_c64_file.is_file():
        raise FileNotFoundError(f"GNU Radio no generó el archivo de salida: {grc_c64_file}")

    grc_samples = read_c64(grc_c64_file)
    print(f"  Simulación completada con éxito. Muestras leídas: {len(grc_samples)}")

    # 4. Ingesta y validación con monitor y clasificadores
    print("\n[4/4] Validando señal de GNU Radio en Python con Monitor y Clasificador...")
    grc_spec = estimate_spectrum(grc_samples[200:456], fs_hz=8000)
    monitor_res = analyze(grc_samples[200:456], fs_hz=8000)

    # Clasificador SVM256
    svm_path = Path("results/fase5/conventional_256.joblib")
    svm_preds = []
    if svm_path.is_file():
        svm_bundle = joblib.load(svm_path)
        svm_pipe = svm_bundle["pipeline"]
        classes = svm_bundle["classes"]
        # Evaluar ventanas de 256 muestras
        windows = np.array([grc_samples[200 + i * 256 : 200 + (i + 1) * 256] for i in range(12)])
        preds_idx = svm_pipe.predict(windows)
        svm_preds = [classes[idx] for idx in preds_idx]
        print(f"  Predicciones SVM256 sobre 12 ventanas GNU Radio: {svm_preds}")

    results["grc_to_python"] = {
        "output_file": str(grc_c64_file),
        "num_samples": len(grc_samples),
        "mean_power": float(np.mean(np.abs(grc_samples) ** 2)),
        "bandwidth_99_hz": float(grc_spec["total_power_bandwidth"]),
        "integrated_power": float(grc_spec["integrated_power"]),
        "svm256_predictions": svm_preds,
    }

    # 5. Generar gráficos comparativos
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Constelación Python vs GNU Radio
    axes[0].scatter(
        np.real(py_rx[200:1000]),
        np.imag(py_rx[200:1000]),
        alpha=0.4,
        color="royalblue",
        s=12,
        label="Python (QPSK + RRC + Canal)",
    )
    axes[0].set_title("Constelación Sintetizada en Python")
    axes[0].set_xlabel("En-Fase (I)")
    axes[0].set_ylabel("Cuadratura (Q)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].set_xlim([-2, 2])
    axes[0].set_ylim([-2, 2])
    axes[0].legend()

    axes[1].scatter(
        np.real(grc_samples[200:1000]),
        np.imag(grc_samples[200:1000]),
        alpha=0.4,
        color="darkorange",
        s=12,
        label="GNU Radio 3.10 (QPSK + RRC + Canal)",
    )
    axes[1].set_title("Constelación Generada por GNU Radio")
    axes[1].set_xlabel("En-Fase (I)")
    axes[1].set_ylabel("Cuadratura (Q)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].set_xlim([-2, 2])
    axes[1].set_ylim([-2, 2])
    axes[1].legend()

    plt.tight_layout()
    const_png = out_path / "gnuradio_constellation_comparison.png"
    plt.savefig(const_png, dpi=180)
    plt.close()
    print(f"  Gráfico guardado: {const_png}")

    # Gráfico de PSD
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        py_spec["frequency"],
        10 * np.log10(np.maximum(py_spec["psd"], 1e-12)),
        label=f"Python PSD (BW99={py_spec['total_power_bandwidth']:.1f} Hz)",
        color="royalblue",
        alpha=0.85,
    )
    ax.plot(
        grc_spec["frequency"],
        10 * np.log10(np.maximum(grc_spec["psd"], 1e-12)),
        label=f"GNU Radio PSD (BW99={grc_spec['total_power_bandwidth']:.1f} Hz)",
        color="darkorange",
        linestyle="--",
        alpha=0.85,
    )
    ax.set_title("Densidad Espectral de Potencia (PSD): Python vs GNU Radio")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_ylim([-60, 10])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    spec_png = out_path / "gnuradio_psd_comparison.png"
    plt.savefig(spec_png, dpi=180)
    plt.close()
    print(f"  Gráfico guardado: {spec_png}")

    # Guardar resumen JSON
    summary_file = out_path / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Resumen guardado: {summary_file}")
    print("\n=== Demostración GNU Radio completada exitosamente ===")
    return results


if __name__ == "__main__":
    run_gnuradio_demonstration()
