"""Nivel 2 (N2.4): Demostración reproducible desde directorio limpio (compilación GRC + simulación + monitor CLI)."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.channels import apply_carrier_impairments, apply_multipath
from src.detection import energy_statistic
from src.gnuradio_interop import (
    DEFAULT_RADIOCONDA_PATH,
    compile_grc,
    is_radioconda_available,
    read_c64,
    run_headless_simulation,
    save_c64,
)
from src.signals import qpsk_modulation
from src.waveform import pulse_shape


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "results" / "nivel2_validacion" / "n2_4_clean_demo"


def run_clean_demo(output_dir: Path = DEFAULT_OUT, seed: int = 2026111603, overwrite: bool = False) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"El directorio {output_dir} no está vacío; use una ruta limpia o --overwrite.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not is_radioconda_available():
        raise RuntimeError(f"Radioconda no encontrado en {DEFAULT_RADIOCONDA_PATH}")

    grc_src_dir = PROJECT_ROOT / "grc"
    grc_compiled_dir = output_dir / "grc_compiled"
    sim_dir = output_dir / "sim_run"
    monitor_dir = output_dir / "monitor_outputs"
    grc_compiled_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir.mkdir(parents=True, exist_ok=True)

    # 1. Compilar los 3 flowgraphs .grc
    flowgraph_names = [
        "flowgraph_headless_sim.grc",
        "flowgraph_tx_channel.grc",
        "flowgraph_rx_from_file.grc",
    ]
    compiled_results = {}
    for fg_name in flowgraph_names:
        fg_path = grc_src_dir / fg_name
        py_out = compile_grc(fg_path, output_dir=grc_compiled_dir)
        compiled_results[fg_name] = {
            "compiled_script": str(py_out),
            "size_bytes": int(py_out.stat().st_size),
            "status": "compiled_successfully",
        }

    # 2. Ejecutar simulación headless en GNU Radio
    headless_py = Path(compiled_results["flowgraph_headless_sim.grc"]["compiled_script"])
    run_headless_simulation(headless_py, work_dir=sim_dir, timeout_sec=25)
    grc_c64 = sim_dir / "grc_sim_output.c64"
    grc_samples = read_c64(grc_c64)

    # 3. Generar referencia de ruido compatible con GNU Radio (noise_volt=0.02 -> potencia=0.0004) y demo Python
    rng = np.random.default_rng(seed)
    grc_noise_pow = 0.02**2
    grc_noise_ref = np.sqrt(grc_noise_pow / 2.0) * (rng.standard_normal(4096) + 1j * rng.standard_normal(4096))
    grc_ref_c64 = output_dir / "grc_noise_ref.c64"
    save_c64(grc_ref_c64, grc_noise_ref)

    py_noise_pow = 0.05
    bits = rng.integers(0, 2, size=512)
    syms = qpsk_modulation(bits)
    shaped, h = pulse_shape(syms, sps=8, beta=0.35, span=10)
    carr = apply_carrier_impairments(shaped, fs_hz=8000.0, phase_rad=0.2, cfo_hz=30.0)
    mp = apply_multipath(carr, taps=[1.0, 0.0, 0.15j])
    clean_seg = mp[len(h) + 2 : len(h) + 2 + 512]
    clean_seg = clean_seg * np.sqrt((py_noise_pow * 10 ** (15.0 / 10.0)) / energy_statistic(clean_seg))
    py_iq = clean_seg + np.sqrt(py_noise_pow / 2.0) * (rng.standard_normal(512) + 1j * rng.standard_normal(512))
    py_ref = np.sqrt(py_noise_pow / 2.0) * (rng.standard_normal(4096) + 1j * rng.standard_normal(4096))

    py_c64 = output_dir / "py_qpsk_15db.c64"
    py_ref_c64 = output_dir / "py_noise_ref.c64"
    save_c64(py_c64, py_iq)
    save_c64(py_ref_c64, py_ref)

    # 4. Ejecutar Monitor CLI sin umbrales manuales ad-hoc
    monitor_summaries = {}
    for label, sig_path, ref_path in (
        ("gnuradio_headless", grc_c64, grc_ref_c64),
        ("python_qpsk_15db", py_c64, py_ref_c64),
    ):
        out_json = monitor_dir / f"{label}.json"
        out_png = monitor_dir / f"{label}_psd.png"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.monitor",
                "--input",
                str(sig_path),
                "--noise-ref",
                str(ref_path),
                "--fs-hz",
                "8000",
                "--output",
                str(out_json),
                "--plot",
                str(out_png),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        monitor_summaries[label] = {
            "returncode": proc.returncode,
            "json_path": str(out_json),
            "png_path": str(out_png),
            "detected": payload["detection"]["detected"],
            "calibration_status": payload["detection"]["calibration_status"],
            "bandwidth_99_hz": payload["spectrum"]["bandwidth_99"],
            "snr_est_db": payload["snr"]["snr_est_db"],
            "modulation": payload["classification"]["modulation"],
            "confidence": payload["classification"]["confidence"],
            "confidence_calibrated": payload["classification"]["confidence_calibrated"],
        }

    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "radioconda_path": str(DEFAULT_RADIOCONDA_PATH),
        "flowgraphs_compiled": compiled_results,
        "gnuradio_simulation": {
            "output_c64": str(grc_c64),
            "num_samples": int(len(grc_samples)),
            "mean_power": float(np.mean(np.abs(grc_samples) ** 2)),
        },
        "monitor_runs": monitor_summaries,
    }
    (output_dir / "demo_execution_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    res = run_clean_demo(args.output_dir, overwrite=args.overwrite)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
