"""Nivel 0 (N0.1 y N0.2): Manifiesto de integridad SHA-256 y auditoría de independencia de particiones."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import scipy
import sklearn
import torch

from src.dataset import SPLITS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "nivel0_evidencia"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _git_head(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _collect_tracked_files(root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {
        "src": sorted((root / "src").glob("*.py")),
        "experiments": sorted((root / "experiments").glob("*.py")),
        "tests": sorted((root / "tests").glob("*.py")),
        "configs": sorted((root / "configs").glob("*.json")),
        "grc": sorted(list((root / "grc").glob("*.grc")) + list((root / "grc").glob("*.py"))),
        "models": sorted(
            [
                p
                for p in (root / "results").rglob("*")
                if p.is_file() and p.suffix in (".joblib", ".pt")
            ]
        ),
        "metrics_and_artifacts": sorted(
            [
                p
                for p in (root / "results").rglob("*")
                if p.is_file()
                and "nivel0_evidencia" not in p.parts
                and p.suffix in (".json", ".csv", ".c64", ".npz")
            ]
        ),
        "datasets": sorted(
            [
                p
                for p in (root / "data").rglob("*")
                if p.is_file() and p.suffix in (".json", ".csv", ".jsonl", ".npz")
            ]
        ),
    }
    return groups


def _extract_historical_figures(root: Path) -> dict:
    paper_metrics = json.loads(
        (root / "results/paper_research/ablation_and_policy_metrics.json").read_text(encoding="utf-8")
    )
    fase4_metrics = json.loads((root / "results/fase4/metricas.json").read_text(encoding="utf-8"))
    with (root / "results/fase4/pd_agregado.csv").open(encoding="utf-8", newline="") as f:
        fase4_pd = list(csv.DictReader(f))
    fase5_metrics = json.loads((root / "results/fase5/metricas.json").read_text(encoding="utf-8"))
    holdout_metrics = json.loads(
        (root / "results/fase6_holdout_final/metrics.json").read_text(encoding="utf-8")
    )
    adaptive_metrics = json.loads((root / "results/adaptive/metrics.json").read_text(encoding="utf-8"))
    gnuradio_summary = json.loads((root / "results/gnuradio/summary.json").read_text(encoding="utf-8"))
    mon_py = json.loads((root / "results/monitor/py_qpsk_15db.json").read_text(encoding="utf-8"))
    mon_grc = json.loads((root / "results/monitor/grc_sim_output.json").read_text(encoding="utf-8"))

    test_paper = paper_metrics["test"]
    shift_paper = paper_metrics["test_shift"]
    pd_id_m10 = next(r for r in fase4_pd if r["regime"] == "id" and r["snr_nominal_db"] == "-10")

    return {
        "adaptive_test_nominal_paper_research": {
            "source": "results/paper_research/ablation_and_policy_metrics.json",
            "dataset": "data/adaptive_dataset",
            "split": "test",
            "n_observations": int(test_paper["always_a0_svm256"]["n"]),
            "svm256_accuracy": float(test_paper["always_a0_svm256"]["accuracy"]),
            "svm256_pct": round(test_paper["always_a0_svm256"]["accuracy"] * 100.0, 4),
            "svm512_accuracy": float(test_paper["always_a1_svm512"]["accuracy"]),
            "svm512_pct": round(test_paper["always_a1_svm512"]["accuracy"] * 100.0, 4),
            "multidomain_net_accuracy": float(test_paper["always_a2_multidomain"]["accuracy"]),
            "multidomain_net_pct": round(test_paper["always_a2_multidomain"]["accuracy"] * 100.0, 4),
            "lbap_full_accuracy": float(test_paper["lbap_full_a0_a1_a2"]["accuracy"]),
            "lbap_full_pct": round(test_paper["lbap_full_a0_a1_a2"]["accuracy"] * 100.0, 4),
            "ablation_snr_flatness_accuracy": float(test_paper["ablation_input_snr_flatness"]["accuracy"]),
            "ablation_snr_flatness_pct": round(test_paper["ablation_input_snr_flatness"]["accuracy"] * 100.0, 4),
        },
        "adaptive_test_shift_paper_research": {
            "source": "results/paper_research/ablation_and_policy_metrics.json",
            "split": "test_shift",
            "svm256_accuracy": float(shift_paper["always_a0_svm256"]["accuracy"]),
            "svm512_accuracy": float(shift_paper["always_a1_svm512"]["accuracy"]),
            "multidomain_net_accuracy": float(shift_paper["always_a2_multidomain"]["accuracy"]),
            "lbap_full_accuracy": float(shift_paper["lbap_full_a0_a1_a2"]["accuracy"]),
        },
        "fase4_e1_detection": {
            "source": "results/fase4/metricas.json + results/fase4/pd_agregado.csv",
            "calibration_threshold_unit_noise": float(fase4_metrics["calibration"]["threshold"]),
            "pfa_measured": float(fase4_metrics["pfa_measured"]),
            "id_minus10db_detections": f"{pd_id_m10['detections']}/{pd_id_m10['trials']}",
            "id_minus10db_pd": float(pd_id_m10["pd"]),
        },
        "fase5_e2_conventional": {
            "source": "results/fase5/metricas.json",
            "dataset": "data/e2_dataset",
            "svm256_test_accuracy": float(fase5_metrics["metrics"]["256"]["test"]["accuracy"]),
            "svm512_test_accuracy": float(fase5_metrics["metrics"]["512"]["test"]["accuracy"]),
        },
        "fase6_holdout_final": {
            "source": "results/fase6_holdout_final/metrics.json",
            "dataset": "data/fase6_holdout_sellado",
            "test_cnn_original": float(holdout_metrics["test"]["cnn_original"]["accuracy"]),
            "test_cnn_phase": float(holdout_metrics["test"]["cnn_phase"]["accuracy"]),
            "test_svm256": float(holdout_metrics["test"]["svm256"]["accuracy"]),
            "test_svm512": float(holdout_metrics["test"]["svm512"]["accuracy"]),
        },
        "adaptive_fase7_initial": {
            "source": "results/adaptive/metrics.json",
            "dataset": "data/adaptive_dataset",
            "test_selector_accuracy": float(adaptive_metrics["test"]["selector"]["accuracy"]),
        },
        "gnuradio_and_monitor_demos": {
            "gnuradio_summary_py_bw99_hz": float(gnuradio_summary["python_to_grc"]["bandwidth_99_hz"]),
            "gnuradio_summary_grc_bw99_hz": float(gnuradio_summary["grc_to_python"]["bandwidth_99_hz"]),
            "monitor_py_demo_threshold_manual": float(mon_py["detection"]["threshold"]),
            "monitor_py_demo_bw99_hz": float(mon_py["spectrum"]["bandwidth_99"]),
            "monitor_grc_demo_threshold_manual": float(mon_grc["detection"]["threshold"]),
            "monitor_grc_demo_bw99_hz": float(mon_grc["spectrum"]["bandwidth_99"]),
        },
    }


def _experiment_correspondence() -> list[dict]:
    return [
        {
            "artifact_dir": "results/fase1, results/fase2, results/fase3",
            "dataset": "data/fase3_smoke (para fase3)",
            "partition": "train/val/test/test_shift (smoke)",
            "models": "ninguno",
            "config": "configs/dataset_smoke.json",
            "scripts": "experiments/diagnostico_fase1.py, diagnostico_fase2.py, src/dataset.py",
            "provenance_notes": "Validaciones iniciales de moduladores, canales y dataset smoke.",
        },
        {
            "artifact_dir": "results/fase4",
            "dataset": "Sintético al vuelo (seed 20261105)",
            "partition": "calibración (5000) + eval ruido (5000) + señales (9600)",
            "models": "Detector de energía y SNR por referencia",
            "config": "configs/exp01_detection.json",
            "scripts": "experiments/exp01_detection.py",
            "provenance_notes": "Calibrado con potencia de ruido unitaria (umbral=1.15239245). P_D a -10 dB ID = 123/600 = 20.5%.",
        },
        {
            "artifact_dir": "results/fase5",
            "dataset": "data/e2_dataset",
            "partition": "train (fit), val (selección), test (evaluación)",
            "models": "conventional_256.joblib (svm_rbf_c10_core), conventional_512.joblib (svm_rbf_c10_full)",
            "config": "configs/exp02_conventional.json",
            "scripts": "experiments/exp02_conventional.py",
            "provenance_notes": "test_shift permaneció sellado en E2.",
        },
        {
            "artifact_dir": "results/fase6_holdout_final",
            "dataset": "data/fase6_holdout_sellado",
            "partition": "test (4800 por longitud/split) y test_shift (4800)",
            "models": "CNN original, CNN phase_aug, SVM256, SVM512",
            "config": "configs/exp03_advanced.json, configs/exp03_phase_aug.json",
            "scripts": "experiments/exp06_final_holdout.py",
            "provenance_notes": "Dataset independiente de e2_dataset; evaluación histórica.",
        },
        {
            "artifact_dir": "results/adaptive",
            "dataset": "data/adaptive_dataset",
            "partition": "train (2400), val_cal (1200), val_select (1200), test (2400), test_shift (2400)",
            "models": "SVM256, SVM512, SmallCNN (fase6/best.pt), policy.joblib",
            "config": "configs/adaptive_experiment.json",
            "scripts": "experiments/exp10_adaptive.py",
            "provenance_notes": "Calibrador de policy.joblib ajustado sobre margen de SVM256 en 1 ventana en val_cal.",
        },
        {
            "artifact_dir": "results/paper_research",
            "dataset": "data/adaptive_dataset (+ e2_dataset + síntesis al vuelo para red multidominio)",
            "partition": "train, val (val_cal/val_select), test, test_shift",
            "models": "multidomain_clean_a2_best.pt, multidomain_adc{8,6,_mixed}_best.pt, SVM256, SVM512",
            "config": "Hardcoded en experiments/exp12_paper_research.py",
            "scripts": "experiments/exp12_paper_research.py",
            "provenance_notes": "Limitación histórica: selector LBAP entrenado sobre recuperabilidad de train compartido con experto y sin paquete de inferencia serializado.",
        },
        {
            "artifact_dir": "results/gnuradio y results/monitor",
            "dataset": "py_qpsk_15db.c64 (seed 20261110) y grc_sim_output.c64 (GNU Radio 3.10.9.2)",
            "partition": "Demostración única por archivo",
            "models": "results/fase5/conventional_256.joblib + results/adaptive/policy.joblib",
            "config": "CLI ad-hoc",
            "scripts": "experiments/exp11_gnuradio_demo.py, src/monitor.py",
            "provenance_notes": "Limitación de procedencia histórica: los JSON de results/monitor usaron umbrales manuales (0.1 y 0.5) sin archivo de referencia de ruido guardado.",
        },
    ]


def build_historical_manifest(root: Path = PROJECT_ROOT) -> dict:
    groups = _collect_tracked_files(root)
    files_record = {}
    for group_name, paths in groups.items():
        entries = {}
        for p in paths:
            entries[_rel(p, root)] = {
                "sha256": _sha256_file(p),
                "bytes": int(p.stat().st_size),
            }
        files_record[group_name] = entries

    total_files = sum(len(v) for v in files_record.values())
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "historical_baseline_reference",
        "git_head_commit": _git_head(root),
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "total_hashed_files": total_files,
        "historical_key_figures": _extract_historical_figures(root),
        "experiment_correspondence": _experiment_correspondence(),
        "files": files_record,
    }


def verify_manifest_integrity(manifest: dict, root: Path = PROJECT_ROOT) -> dict:
    checked = 0
    mismatches = []
    missing = []
    for group_name, entries in manifest["files"].items():
        for rel_path, meta in entries.items():
            target = root / rel_path
            checked += 1
            if not target.is_file():
                missing.append({"group": group_name, "path": rel_path})
                continue
            actual_hash = _sha256_file(target)
            if actual_hash != meta["sha256"]:
                mismatches.append(
                    {
                        "group": group_name,
                        "path": rel_path,
                        "expected_sha256": meta["sha256"],
                        "actual_sha256": actual_hash,
                    }
                )
    return {
        "passed": len(mismatches) == 0 and len(missing) == 0,
        "checked_files": checked,
        "mismatches": mismatches,
        "missing": missing,
    }


def check_tamper_detection_on_copy(manifest: dict, root: Path = PROJECT_ROOT) -> dict:
    sample_rel = "results/paper_research/ablation_and_policy_metrics.json"
    original_path = root / sample_rel
    original_hash_before = _sha256_file(original_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        tampered_path = tmp_root / sample_rel
        tampered_path.parent.mkdir(parents=True, exist_ok=True)
        raw = bytearray(original_path.read_bytes())
        raw[-1] = (raw[-1] ^ 0x01)
        tampered_path.write_bytes(bytes(raw))

        # Mini manifiesto sobre la copia temporal para verificar detección sin tocar originales
        mini_manifest = {
            "files": {
                "metrics_and_artifacts": {
                    sample_rel: manifest["files"]["metrics_and_artifacts"][sample_rel]
                }
            }
        }
        tamper_res = verify_manifest_integrity(mini_manifest, tmp_root)

    original_hash_after = _sha256_file(original_path)
    return {
        "tested_sample": sample_rel,
        "original_unmodified": original_hash_before == original_hash_after == manifest["files"]["metrics_and_artifacts"][sample_rel]["sha256"],
        "tamper_detected_on_copy": (not tamper_res["passed"]) and len(tamper_res["mismatches"]) == 1,
        "tamper_check_details": tamper_res,
    }


def _inspect_dataset(dataset_root: Path) -> dict:
    with (dataset_root / "index.csv").open(encoding="utf-8", newline="") as f:
        index_rows = list(csv.DictReader(f))
    with (dataset_root / "oracle/manifest.jsonl").open(encoding="utf-8") as f:
        oracle_rows = [json.loads(line) for line in f if line.strip()]
    config = json.loads((dataset_root / "config.json").read_text(encoding="utf-8"))

    by_split_parents: dict[str, set[str]] = {s: set() for s in SPLITS}
    by_split_records: dict[str, set[str]] = {s: set() for s in SPLITS}
    by_split_seeds: dict[str, set[tuple[int, ...]]] = {s: set() for s in SPLITS}
    signal_parents_by_split: dict[str, set[str]] = {s: set() for s in SPLITS}

    for row in oracle_rows:
        s = row["split"]
        by_split_parents[s].add(row["parent_waveform_id"])
        by_split_records[s].add(row["record_id"])
        by_split_seeds[s].add(tuple(row["parent_spawn_key"]))
        if row["has_signal"]:
            signal_parents_by_split[s].add(row["parent_waveform_id"])

    # Verificar disjunción entre splits internos
    internal_parent_overlaps = {}
    for i, s1 in enumerate(SPLITS):
        for s2 in SPLITS[i + 1 :]:
            overlap = by_split_parents[s1] & by_split_parents[s2]
            internal_parent_overlaps[f"{s1}__{s2}"] = len(overlap)

    # Verificar exactitud de prefijos iq256 == iq512[:, :256] y hashes de señales
    prefixes_exact = {}
    iq512_hashes_by_split: dict[str, set[str]] = {}
    for s in SPLITS:
        with np.load(dataset_root / f"observations/{s}.npz", allow_pickle=False) as data:
            iq256 = data["iq256"]
            iq512 = data["iq512"]
            prefixes_exact[s] = bool(np.array_equal(iq256, iq512[:, :256]))
            iq512_hashes_by_split[s] = {hashlib.sha256(w.tobytes()).hexdigest() for w in iq512}

    all_hashes = set().union(*iq512_hashes_by_split.values())
    total_rows = sum(len(v) for v in iq512_hashes_by_split.values())

    # Verificar coincidencia con supervision/*.csv si existe
    supervision_matches = {}
    sup_dir = dataset_root / "supervision"
    if sup_dir.is_dir():
        for s in SPLITS:
            with (sup_dir / f"{s}.csv").open(encoding="utf-8", newline="") as f:
                sup_rows = list(csv.DictReader(f))
            sup_parents = {r["parent_waveform_id"] for r in sup_rows}
            supervision_matches[s] = sup_parents == by_split_parents[s]

    return {
        "dataset": dataset_root.name,
        "seed": config["seed"],
        "total_parents": len(oracle_rows),
        "index_count": len(index_rows),
        "split_parent_counts": {s: len(v) for s, v in by_split_parents.items()},
        "internal_parent_overlaps": internal_parent_overlaps,
        "all_internal_splits_disjoint": all(v == 0 for v in internal_parent_overlaps.values()),
        "prefixes_exact_iq256_in_iq512": all(prefixes_exact.values()),
        "prefixes_by_split": prefixes_exact,
        "unique_iq512_waveforms": len(all_hashes) == total_rows,
        "supervision_matches_oracle": supervision_matches,
        "_sets": {
            "all_parents": set().union(*by_split_parents.values()),
            "all_records": set().union(*by_split_records.values()),
            "all_iq_hashes": all_hashes,
            "signal_parents_by_split": signal_parents_by_split,
        },
    }


def audit_partition_independence(root: Path = PROJECT_ROOT) -> dict:
    dataset_names = ["e2_dataset", "adaptive_dataset", "fase6_holdout_sellado", "fase3_smoke"]
    inspected = {name: _inspect_dataset(root / "data" / name) for name in dataset_names}

    cross_dataset_overlaps = {}
    for i, d1 in enumerate(dataset_names):
        for d2 in dataset_names[i + 1 :]:
            s1, s2 = inspected[d1]["_sets"], inspected[d2]["_sets"]
            cross_dataset_overlaps[f"{d1}__{d2}"] = {
                "shared_parents": len(s1["all_parents"] & s2["all_parents"]),
                "shared_records": len(s1["all_records"] & s2["all_records"]),
                "shared_iq512_hashes": len(s1["all_iq_hashes"] & s2["all_iq_hashes"]),
            }

    # Verificar val_cal vs val_select en results/adaptive/development_partitions.json
    dev_part_path = root / "results/adaptive/development_partitions.json"
    dev_parts = json.loads(dev_part_path.read_text(encoding="utf-8"))
    train_p = set(dev_parts["train_parents"])
    cal_p = set(dev_parts["cal_parents"])
    sel_p = set(dev_parts["selection_parents"])
    ad_sig = inspected["adaptive_dataset"]["_sets"]["signal_parents_by_split"]

    adaptive_dev_audit = {
        "train_count": len(train_p),
        "val_cal_count": len(cal_p),
        "val_select_count": len(sel_p),
        "overlap_train_val_cal": len(train_p & cal_p),
        "overlap_train_val_select": len(train_p & sel_p),
        "overlap_val_cal_val_select": len(cal_p & sel_p),
        "overlap_val_cal_test": len(cal_p & ad_sig["test"]),
        "overlap_val_cal_test_shift": len(cal_p & ad_sig["test_shift"]),
        "val_cal_plus_select_equals_adaptive_val_signals": (cal_p | sel_p) == ad_sig["val"],
        "train_equals_adaptive_train_signals": train_p == ad_sig["train"],
    }

    # Verificar cachés de paper_research (tamaños y ausencia de mezcla con test)
    cache_audit = {}
    for split in SPLITS:
        cpath = root / f"results/paper_research/{split}_research_cache.npz"
        if cpath.is_file():
            with np.load(cpath, allow_pickle=False) as z:
                cache_audit[split] = {
                    "files": z.files,
                    "n_rows": int(len(z["truth"])) if "truth" in z.files else None,
                    "expected_signal_parents": len(ad_sig[split]),
                }

    # Limpiar conjuntos internos antes de serializar
    datasets_clean = {}
    for name, info in inspected.items():
        datasets_clean[name] = {k: v for k, v in info.items() if k != "_sets"}

    shared_parents_total = sum(
        sum(info["internal_parent_overlaps"].values()) for info in datasets_clean.values()
    ) + sum(v["shared_parents"] + v["shared_iq512_hashes"] for v in cross_dataset_overlaps.values())

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "shared_parents_count": int(shared_parents_total),
        "all_prefixes_exact": all(d["prefixes_exact_iq256_in_iq512"] for d in datasets_clean.values()),
        "all_internal_splits_disjoint": all(d["all_internal_splits_disjoint"] for d in datasets_clean.values()),
        "all_cross_datasets_disjoint": all(
            v["shared_parents"] == 0 and v["shared_records"] == 0 and v["shared_iq512_hashes"] == 0
            for v in cross_dataset_overlaps.values()
        ),
        "datasets": datasets_clean,
        "cross_dataset_overlaps": cross_dataset_overlaps,
        "adaptive_development_partitions_audit": adaptive_dev_audit,
        "paper_research_cache_audit": cache_audit,
        "known_limitation_n3_2_deferred": (
            "En el experimento histórico exp12_paper_research, el selector LBAP se entrenó sobre predicciones in-sample "
            "de train (riesgo de optimismo en recuperabilidad); ese ajuste queda aislado como histórico para corregirse en N3.2."
        ),
    }


def run_n0_verification(output_dir: Path = RESULTS_DIR, root: Path = PROJECT_ROOT) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_historical_manifest(root)
    manifest_path = output_dir / "manifest_historical_state.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    integrity_live = verify_manifest_integrity(manifest, root)
    tamper_test = check_tamper_detection_on_copy(manifest, root)
    integrity_report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "live_integrity_check": integrity_live,
        "tamper_check_passed": bool(tamper_test["tamper_detected_on_copy"] and tamper_test["original_unmodified"]),
        "tamper_test": tamper_test,
    }
    (output_dir / "integrity_verification.json").write_text(
        json.dumps(integrity_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    partition_audit = audit_partition_independence(root)
    (output_dir / "partition_independence_audit.json").write_text(
        json.dumps(partition_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "manifest_path": str(manifest_path),
        "total_hashed_files": manifest["total_hashed_files"],
        "integrity_passed": integrity_live["passed"],
        "tamper_check_passed": integrity_report["tamper_check_passed"],
        "shared_parents_count": partition_audit["shared_parents_count"],
        "all_prefixes_exact": partition_audit["all_prefixes_exact"],
    }


if __name__ == "__main__":
    summary = run_n0_verification()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
