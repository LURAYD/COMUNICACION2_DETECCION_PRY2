"""Nivel 2 (N2.2): Prueba ciega extremo a extremo con 164 archivos desconocidos (Python + GNU Radio)."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from experiments.exp_n2_1_detection_snr_bank import _clean_parent_waveform
from src.conventional import CLASSES
from src.detection import energy_statistic
from src.gnuradio_interop import (
    compile_grc,
    is_radioconda_available,
    read_c64,
    run_headless_simulation,
    save_c64,
)
from src.monitor import analyze_complete, load_signal
from src.snr_estimation import calibrate_noise_reference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_N2_1 = PROJECT_ROOT / "configs" / "exp_n2_1_detection_snr_bank.json"
CHALLENGE_DIR = PROJECT_ROOT / "results" / "nivel2_validacion" / "n2_2_unknown_challenge"
RESERVED_SEED_N2_2 = 2026111603


def _build_and_run_n2_2(output_dir: Path = CHALLENGE_DIR) -> dict:
    base_cfg = json.loads(CONFIG_N2_1.read_text(encoding="utf-8"))
    inputs_dir = output_dir / "inputs"
    json_out_dir = output_dir / "outputs_json"
    grc_work_dir = output_dir / "_grc_tmp"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    json_out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RESERVED_SEED_N2_2)
    fs_hz = float(base_cfg["fs_hz"])
    gains = [0.5, 1.0, 2.0]
    base_noise_power = 0.1

    # 1. Referencias independientes de ruido por ganancia (Python y GNU Radio)
    noise_refs_files = {}
    for g in gains:
        npow = base_noise_power * (g**2)
        n_iq = np.sqrt(npow / 2.0) * (rng.standard_normal(4096) + 1j * rng.standard_normal(4096))
        ref_name = f"noise_ref_g{str(g).replace('.', '_')}.c64"
        save_c64(inputs_dir / ref_name, n_iq)
        (inputs_dir / f"{ref_name}.meta.json").write_text(
            json.dumps({"fs_hz": fs_hz, "gain": g}, indent=2) + "\n", encoding="utf-8"
        )
        noise_refs_files[("python", g)] = ref_name

    # Referencia compatible con GNU Radio (noise_volt=0.02 -> potencia compleja 0.0004 por ganancia^2)
    grc_base_noise_power = 0.02**2
    for g in gains:
        npow = grc_base_noise_power * (g**2)
        n_iq = np.sqrt(npow / 2.0) * (rng.standard_normal(4096) + 1j * rng.standard_normal(4096))
        ref_name = f"noise_ref_grc_g{str(g).replace('.', '_')}.c64"
        save_c64(inputs_dir / ref_name, n_iq)
        (inputs_dir / f"{ref_name}.meta.json").write_text(
            json.dumps({"fs_hz": fs_hz, "gain": g}, indent=2) + "\n", encoding="utf-8"
        )
        noise_refs_files[("gnuradio", g)] = ref_name

    # Generar señal real de GNU Radio mediante compilación y simulación headless
    grc_segments = []
    if is_radioconda_available():
        grc_work_dir.mkdir(parents=True, exist_ok=True)
        py_script = compile_grc(PROJECT_ROOT / "grc" / "flowgraph_headless_sim.grc", output_dir=grc_work_dir)
        run_headless_simulation(py_script, work_dir=grc_work_dir, timeout_sec=25)
        grc_raw = read_c64(grc_work_dir / "grc_sim_output.c64")
        # Tomar 6 segmentos no solapados de 512 muestras tras transitorio inicial (muestras 256..3328)
        for idx_seg in range(6):
            seg = grc_raw[256 + idx_seg * 512 : 256 + (idx_seg + 1) * 512].astype(np.complex128)
            grc_segments.append(seg)
    else:
        raise RuntimeError("Radioconda requerido para generar señales GNU Radio de N2.2.")

    # 2. Especificar los 164 archivos (120 señales = 20 por clase + 44 ruido puro)
    specs = []
    snr_pool = [-10, -5, 0, 5, 10, 20]  # 2 bajas, 2 medias, 2 altas
    for cls_idx, mod in enumerate(CLASSES):
        n_py = 14 if mod == "QPSK" else 20
        for k in range(n_py):
            regime = "id" if (k % 2 == 0) else "shift"
            gain = gains[(k + cls_idx) % len(gains)]
            snr_db = snr_pool[k % len(snr_pool)]
            snr_band = "low" if snr_db < 0 else ("medium" if snr_db <= 5 else "high")
            is_long = (k % 3 == 1)
            specs.append(
                {
                    "source": "python",
                    "has_signal": True,
                    "modulation": mod,
                    "regime": regime,
                    "gain": gain,
                    "snr_nominal_db": snr_db,
                    "snr_band": snr_band,
                    "file_type": "python_long_variable_start" if is_long else "python_standard_512",
                }
            )
        if mod == "QPSK":
            for k_grc in range(6):
                gain = gains[k_grc % len(gains)]
                specs.append(
                    {
                        "source": "gnuradio",
                        "grc_seg_idx": k_grc,
                        "has_signal": True,
                        "modulation": "QPSK",
                        "regime": "id",
                        "gain": gain,
                        "snr_nominal_db": 34.0,
                        "snr_band": "high",
                        "file_type": "gnuradio_headless_qpsk",
                    }
                )

    for n_idx in range(44):
        gain = gains[n_idx % len(gains)]
        is_long = (n_idx % 4 == 0)
        specs.append(
            {
                "source": "python",
                "has_signal": False,
                "modulation": "NO_SIGNAL",
                "regime": "noise_only",
                "gain": gain,
                "snr_nominal_db": None,
                "snr_band": "noise_only",
                "file_type": "noise_long" if is_long else "noise_standard_512",
            }
        )

    # Barajar orden de creación para que los IDs opacos U0001..U0164 no revelen clase ni orden
    perm = rng.permutation(len(specs))
    oracle_rows = []
    input_manifest = []

    for out_idx, spec_idx in enumerate(perm, start=1):
        sp = specs[int(spec_idx)]
        uid = f"U{out_idx:04d}"
        filename = f"{uid}.c64"
        g = float(sp["gain"])

        if sp["source"] == "gnuradio":
            iq = g * grc_segments[sp["grc_seg_idx"]]
            ref_file = noise_refs_files[("gnuradio", g)]
            tx_start = 0
            total_len = len(iq)
        elif not sp["has_signal"]:
            total_len = int(rng.choice([1280, 1536, 2048])) if sp["file_type"] == "noise_long" else 512
            npow = base_noise_power * (g**2)
            iq = np.sqrt(npow / 2.0) * (rng.standard_normal(total_len) + 1j * rng.standard_normal(total_len))
            ref_file = noise_refs_files[("python", g)]
            tx_start = None
        else:
            clean = _clean_parent_waveform(base_cfg, sp["modulation"], sp["regime"], rng, 512)
            npow = base_noise_power * (g**2)
            sig_pow = npow * (10.0 ** (sp["snr_nominal_db"] / 10.0))
            scaled = clean * np.sqrt(sig_pow / energy_statistic(clean))
            if sp["file_type"] == "python_long_variable_start":
                n_windows = int(rng.choice([5, 6, 8]))
                total_len = n_windows * 256
                start_win = int(rng.integers(1, n_windows - 1))
                tx_start = start_win * 256
                bg = np.sqrt(npow / 2.0) * (rng.standard_normal(total_len) + 1j * rng.standard_normal(total_len))
                bg[tx_start : tx_start + 512] += scaled
                iq = bg
            else:
                total_len = 512
                tx_start = 0
                noise = np.sqrt(npow / 2.0) * (rng.standard_normal(512) + 1j * rng.standard_normal(512))
                iq = scaled + noise
            ref_file = noise_refs_files[("python", g)]

        save_c64(inputs_dir / filename, iq)
        # Metadatos públicos estrictamente sin etiquetas de verdad ni SNR
        (inputs_dir / f"{filename}.meta.json").write_text(
            json.dumps(
                {
                    "fs_hz": fs_hz,
                    "sps": 8,
                    "symbol_rate_hz": 1000.0,
                    "gain": g,
                    "single_modulation_assumption": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        input_manifest.append({"uid": uid, "file": filename, "noise_ref_file": ref_file, "gain": g, "fs_hz": fs_hz})
        oracle_rows.append(
            {
                "uid": uid,
                "file": filename,
                "has_signal": int(sp["has_signal"]),
                "true_modulation": sp["modulation"],
                "regime": sp["regime"],
                "gain": g,
                "snr_nominal_db": sp["snr_nominal_db"] if sp["snr_nominal_db"] is not None else "",
                "snr_band": sp["snr_band"],
                "file_type": sp["file_type"],
                "num_samples": total_len,
                "tx_start_sample": tx_start if tx_start is not None else "",
            }
        )

    # Guardar oracle fuera de inputs/
    with (output_dir / "oracle_truth.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(oracle_rows[0]))
        writer.writeheader()
        writer.writerows(oracle_rows)

    # 3. Ejecución ciega común del monitor sobre los 164 archivos (cero lectura de oracle_truth.csv)
    cached_refs = {}
    monitor_outputs = {}
    processing_failures = 0
    contract_complete_count = 0

    for item in input_manifest:
        ref_key = item["noise_ref_file"]
        if ref_key not in cached_refs:
            ref_iq, ref_meta = load_signal(inputs_dir / ref_key, return_metadata=True)
            cached_refs[ref_key] = calibrate_noise_reference(
                ref_iq, fs_hz=ref_meta.get("fs_hz"), gain=ref_meta.get("gain", 1.0)
            )
        try:
            iq, sig_meta = load_signal(inputs_dir / item["file"], return_metadata=True)
            res = analyze_complete(
                iq=iq,
                signal_id=item["uid"],
                noise_reference=cached_refs[ref_key],
                gain=sig_meta.get("gain", 1.0),
                signal_metadata=sig_meta,
            )
            (json_out_dir / f"{item['uid']}.json").write_text(
                json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            required_keys = {"detection", "spectrum", "snr", "classification"}
            if required_keys.issubset(res.keys()) and res["classification"]["modulation"] is not None:
                contract_complete_count += 1
            monitor_outputs[item["uid"]] = res
        except Exception:
            processing_failures += 1

    # 4. Cruce posterior contra oracle_truth.csv para reporte
    results_table = []
    tp = fp = fn = tn = indet = 0
    cond_correct = 0
    joint_correct = 0

    labels_7 = list(CLASSES) + ["NO_SIGNAL"]
    y_true_7 = []
    y_pred_7 = []

    for row in oracle_rows:
        uid = row["uid"]
        res = monitor_outputs[uid]
        has_sig = bool(row["has_signal"])
        det = res["detection"]["detected"]
        pred_mod = res["classification"]["modulation"]
        conf = res["classification"]["confidence"]

        if det is None:
            indet += 1
            det_outcome = "INDETERMINATE"
        elif has_sig and det is True:
            tp += 1
            det_outcome = "TP"
            if pred_mod == row["true_modulation"]:
                cond_correct += 1
                joint_correct += 1
        elif has_sig and det is False:
            fn += 1
            det_outcome = "FN_OMISSION"
        elif (not has_sig) and det is True:
            fp += 1
            det_outcome = "FP"
        else:
            tn += 1
            det_outcome = "TN"

        y_true_7.append(row["true_modulation"])
        y_pred_7.append(pred_mod if pred_mod in labels_7 else "NO_SIGNAL")

        results_table.append(
            {
                **row,
                "detected": det,
                "detection_outcome": det_outcome,
                "pred_modulation": pred_mod,
                "confidence": conf if conf is not None else "",
                "confidence_calibrated": res["classification"]["confidence_calibrated"],
                "snr_est_db": round(res["snr"]["snr_est_db"], 3) if res["snr"]["snr_est_db"] is not None else "",
                "bandwidth_99_hz": round(res["spectrum"]["bandwidth_99"], 2) if res["spectrum"]["bandwidth_99"] is not None else "",
                "joint_correct": int(has_sig and det is True and pred_mod == row["true_modulation"]),
            }
        )

    with (output_dir / "challenge_results_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_table[0]))
        writer.writeheader()
        writer.writerows(results_table)

    cm7 = confusion_matrix(y_true_7, y_pred_7, labels=labels_7)
    with (output_dir / "confusion_with_omissions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true \\ pred"] + labels_7)
        for lbl, r_vals in zip(labels_7, cm7.tolist()):
            writer.writerow([lbl] + r_vals)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    norm_cm = cm7 / np.maximum(cm7.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(norm_cm, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(labels_7)):
        for j in range(len(labels_7)):
            ax.text(
                j,
                i,
                f"{cm7[i, j]}",
                ha="center",
                va="center",
                color="white" if norm_cm[i, j] > 0.5 else "black",
                fontsize=9,
            )
    ax.set(
        xticks=range(len(labels_7)),
        yticks=range(len(labels_7)),
        xticklabels=labels_7,
        yticklabels=labels_7,
        xlabel="Predicción del monitor (incluye omisión NO_SIGNAL)",
        ylabel="Verdad (oracle)",
        title="N2.2 Matriz de confusión con omisiones y falsas alarmas (n=164)",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.colorbar(im, ax=ax, label="Fracción por fila")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_with_omissions.png", dpi=160)
    plt.close(fig)

    # Desglose por categoría
    def _breakdown(field):
        groups = {}
        for r in results_table:
            if not r["has_signal"]:
                continue
            key = str(r[field])
            g_stats = groups.setdefault(key, {"n_signals": 0, "detected": 0, "joint_correct": 0})
            g_stats["n_signals"] += 1
            if r["detected"] is True:
                g_stats["detected"] += 1
            g_stats["joint_correct"] += r["joint_correct"]
        for g_stats in groups.values():
            g_stats["detection_coverage"] = g_stats["detected"] / g_stats["n_signals"]
            g_stats["conditional_accuracy"] = (
                g_stats["joint_correct"] / g_stats["detected"] if g_stats["detected"] > 0 else 0.0
            )
            g_stats["joint_detect_and_classify_rate"] = g_stats["joint_correct"] / g_stats["n_signals"]
        return groups

    n_sig = tp + fn
    n_noise = tn + fp
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": RESERVED_SEED_N2_2,
        "assumption": "Una sola modulación por archivo; múltiples transmisiones heterogéneas quedan fuera de alcance sin segmentación previa.",
        "total_files": len(oracle_rows),
        "signal_files": n_sig,
        "noise_files": n_noise,
        "processed_without_error": len(oracle_rows) - processing_failures,
        "complete_five_outputs_count": contract_complete_count,
        "detection_counts": {"TP": tp, "FP": fp, "FN_omissions": fn, "TN": tn, "indeterminate": indet},
        "detection_coverage_on_signals": tp / n_sig,
        "noise_false_alarm_rate": fp / n_noise,
        "conditional_accuracy_on_detected": cond_correct / tp if tp > 0 else 0.0,
        "joint_detect_and_classify_rate": joint_correct / n_sig,
        "breakdown_by_regime": _breakdown("regime"),
        "breakdown_by_snr_band": _breakdown("snr_band"),
        "breakdown_by_file_type": _breakdown("file_type"),
    }
    (output_dir / "challenge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    out = _build_and_run_n2_2()
    print(json.dumps(out, indent=2, ensure_ascii=False))
