"""Pruebas de aceptación para Nivel 2 (N2.1-N2.4)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from src.conventional import CLASSES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_n2_1_independent_detection_and_snr_bank():
    bank_dir = PROJECT_ROOT / "results/nivel2_validacion/n2_1_bank"
    metrics = json.loads((bank_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_noise_calibration"] == 5000
    assert metrics["n_noise_evaluation"] == 5000
    assert metrics["total_signal_parents"] == 9600
    assert metrics["pfa_all_nominal_wilson_upper_le_0_02"] is True
    assert metrics["all_id_ge_0db_cells_meet_wilson_low_ge_0_95"] is True
    assert metrics["id_snr_ge_0db_24_cells_min_wilson95_low_window"] >= 0.95
    assert metrics["id_snr_ge_0db_24_cells_min_wilson95_low_file"] >= 0.95
    assert metrics["gain_scale_invariance_exact_across_all_9600"] is True
    assert metrics["snr_meets_n1_3_targets"] is True

    with (bank_dir / "drift_results.csv").open(encoding="utf-8", newline="") as f:
        drift = list(csv.DictReader(f))
    stale_pfa = [r for r in drift if r["scenario"] == "drift_x2_uncalibrated_stale_ref" and "pfa" in r["metric_type"]]
    recal_pfa = [r for r in drift if r["scenario"] == "drift_x2_recalibrated_ref" and "pfa" in r["metric_type"]]
    assert all(float(r["rate"]) > 0.9 for r in stale_pfa)
    assert all(float(r["wilson95_high"]) <= 0.02 for r in recal_pfa)


def test_n2_2_unknown_files_challenge():
    chal_dir = PROJECT_ROOT / "results/nivel2_validacion/n2_2_unknown_challenge"
    summary = json.loads((chal_dir / "challenge_summary.json").read_text(encoding="utf-8"))
    assert summary["total_files"] >= 160
    assert summary["signal_files"] == 120
    assert summary["noise_files"] >= 40
    assert summary["processed_without_error"] == summary["total_files"]
    assert summary["complete_five_outputs_count"] == summary["total_files"]
    assert summary["detection_counts"]["indeterminate"] == 0

    with (chal_dir / "oracle_truth.csv").open(encoding="utf-8", newline="") as f:
        truth_rows = list(csv.DictReader(f))
    sig_counts = {c: sum(1 for r in truth_rows if r["true_modulation"] == c) for c in CLASSES}
    assert all(cnt == 20 for cnt in sig_counts.values())

    # Verificar que los archivos de entrada y sus sidecars no filtran etiquetas oracle
    for meta_path in (chal_dir / "inputs").glob("U*.meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for forbidden in ("modulation", "true_modulation", "snr_nominal_db", "has_signal"):
            assert forbidden not in meta

    assert (chal_dir / "confusion_with_omissions.csv").is_file()
    assert (chal_dir / "confusion_with_omissions.png").is_file()


def test_n2_3_classification_and_regression_checks():
    n2_3_dir = PROJECT_ROOT / "results/nivel2_validacion/n2_3_classification"
    unified = json.loads((n2_3_dir / "unified_metrics.json").read_text(encoding="utf-8"))
    reg = json.loads((n2_3_dir / "regression_reload_check.json").read_text(encoding="utf-8"))

    assert set(unified["benchmarks"].keys()) == {"adaptive_dataset", "fase6_holdout_sellado"}
    assert unified["benchmarks"]["adaptive_dataset"]["test"]["svm256"]["n"] == 2400
    assert unified["benchmarks"]["fase6_holdout_sellado"]["test"]["svm256"]["n"] == 4800

    assert reg["fase5_e2_test_reload"]["zero_mismatches"] is True
    assert reg["fase6_holdout_reload"]["zero_mismatches"] is True
    for split in ("test", "test_shift"):
        sp = reg["adaptive_paper_research_reload"][split]
        assert sp["svm256_class_mismatches"] == 0
        assert sp["svm512_class_mismatches"] == 0
        assert sp["multidomain_a2_class_mismatches"] == 0
        assert sp["within_logit_tolerance"] is True
    assert reg["monitor_interface_vs_svm256_direct"]["zero_mismatches"] is True


def test_n2_4_clean_reproducible_demo():
    demo_dir = PROJECT_ROOT / "results/nivel2_validacion/n2_4_clean_demo"
    summary = json.loads((demo_dir / "demo_execution_summary.json").read_text(encoding="utf-8"))
    assert len(summary["flowgraphs_compiled"]) == 3
    assert all(v["status"] == "compiled_successfully" for v in summary["flowgraphs_compiled"].values())
    assert summary["gnuradio_simulation"]["num_samples"] == 4096
    for run_key in ("gnuradio_headless", "python_qpsk_15db"):
        run = summary["monitor_runs"][run_key]
        assert run["returncode"] == 0
        assert run["detected"] is True
        assert run["calibration_status"] == "calibrated_from_noise_reference"
        assert Path(run["json_path"]).is_file() or (PROJECT_ROOT / run["json_path"]).is_file()
        assert Path(run["png_path"]).is_file() or (PROJECT_ROOT / run["png_path"]).is_file()
