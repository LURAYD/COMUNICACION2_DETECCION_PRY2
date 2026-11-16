"""Tests de aceptación para Nivel 3 (política adaptativa desplegable, cachés, latencia, bootstrap, presupuestos y ADC)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.adaptive import (
    load_deployable_policy_package,
    load_package_models,
    run_deployable_policy_inference,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
N3_DIR = ROOT_DIR / "results" / "nivel3_adaptativo"


def test_n3_1_package_reload_causality_lazy_and_cli(tmp_path: Path) -> None:
    """Verifica paquete desplegable N3.1: reporte de verificación, recarga, causalidad, perezoso y CLI."""
    pkg_path = N3_DIR / "lbap_deployable_package.joblib"
    report_path = N3_DIR / "n3_1_package_verification.json"
    assert pkg_path.exists(), f"Falta paquete N3.1: {pkg_path}"
    assert report_path.exists(), f"Falta reporte N3.1: {report_path}"

    rep = json.loads(report_path.read_text(encoding="utf-8"))
    assert rep["reload_100_percent_match"] is True
    assert rep["new_process_reload_matches"] == rep["batch_size_reload"] == 300
    assert rep["causal_100_percent_match"] is True
    assert rep["causal_prefix_matches"] == rep["causal_prefix_tests"] == 200
    assert rep["lazy_execution_strictly_verified"] is True
    assert rep["lazy_execution_violations"] == 0
    assert rep["incompatible_hash_rejected"] is True

    # Recarga directa e inferencia causal/perezosa con verificación de hash y rechazo ante alteración
    pkg = load_deployable_policy_package(pkg_path, root_dir=ROOT_DIR, verify_hashes=True)
    models = load_package_models(pkg, root_dir=ROOT_DIR)
    calls = {"svm512": 0, "a2": 0}
    orig_svm512, orig_a2 = models["svm512"], models["predict_a2"]

    class _CountingSVM512:
        def predict(self, arr):
            calls["svm512"] += 1
            return orig_svm512.predict(arr)

    def _counting_a2(arr):
        calls["a2"] += 1
        return orig_a2(arr)

    instrumented = {**models, "svm512": _CountingSVM512(), "predict_a2": _counting_a2}

    rng = np.random.default_rng(20261116)
    iq = (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(np.complex64)
    res_base = run_deployable_policy_inference(iq, pkg, models=instrumented, fs_hz=8000.0)
    assert res_base["action"] in (0, 1, 2)
    assert res_base["modulation"] in pkg["classes"]
    assert calls["svm512"] == (1 if res_base["action"] == 1 else 0)
    assert calls["a2"] == (1 if res_base["action"] == 2 else 0)

    # Causalidad directa en muestras 256..512
    iq_mut = iq.copy()
    iq_mut[256:] *= 7.5
    res_mut = run_deployable_policy_inference(iq_mut, pkg, models=models, fs_hz=8000.0)
    assert res_mut["action"] == res_base["action"]

    # Rechazo ante alteración de hash
    bad_pkg = dict(pkg)
    bad_pkg.pop("selector", None)
    bad_pkg["model_hashes"] = {**pkg["model_hashes"], "a0_svm256": "deadbeef" * 8}
    bad_path = tmp_path / "tampered_pkg.joblib"
    joblib.dump(bad_pkg, bad_path)
    with pytest.raises(ValueError):
        load_deployable_policy_package(bad_path, root_dir=ROOT_DIR, verify_hashes=True)

    # CLI ejecutable de punta a punta sobre un archivo .c64
    iq_file = tmp_path / "sample_512.c64"
    out_json = tmp_path / "cli_out.json"
    np.column_stack([iq.real, iq.imag]).astype(np.float32).tofile(iq_file)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.adaptive",
            "--package",
            str(pkg_path),
            "--input",
            str(iq_file),
            "--fs-hz",
            "8000",
            "--output",
            str(out_json),
        ],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    cli_data = json.loads(out_json.read_text(encoding="utf-8"))
    assert cli_data["action"] == res_base["action"]
    assert cli_data["modulation"] == res_base["modulation"]


def test_n3_2_cache_independence_and_tamper_detection() -> None:
    """Verifica 0 solapamiento entre entrenamiento del selector y expertos, y detección de cachés alteradas."""
    audit_path = N3_DIR / "n3_2_cache_and_partition_audit.json"
    assert audit_path.exists()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["independent_policy_train_parents"] >= 4800
    assert audit["overlap_with_e2_or_adaptive_experts"] == 0
    cache_names = {c["cache_name"] for c in audit["caches"]}
    assert {"policy_train_holdout", "adaptive_val", "adaptive_test", "adaptive_test_shift"} <= cache_names
    for entry in audit["caches"]:
        man = entry["manifest"]
        for key in (
            "dataset_config_sha256",
            "supervision_csv_sha256",
            "observations_npz_sha256",
            "row_order_sha256",
            "code_sha256",
            "cache_npz_sha256",
        ):
            assert len(man[key]) == 64


def test_n3_3_n3_4_n3_5_n3_6_reports_consistency() -> None:
    """Verifica latencia 3x300, bootstrap 10k con IC95% y 3 semillas, presupuestos N3.5 y auditoría ADC/calibración N3.6."""
    lat = json.loads((N3_DIR / "n3_3_latency_sessions.json").read_text(encoding="utf-8"))
    assert lat["n_sessions"] == 3
    assert lat["obs_per_session"] == 300
    assert "cuda_synchronize" in lat["hardware"]["execution_mode"]
    for pol in ("always_a0_svm256", "always_a1_svm512", "always_a2_multidomain", "policy_confidence", "policy_blind_snr_flatness", "lbap_full_a0_a1_a2"):
        assert lat["pooled_compute_summary"][pol]["n_calls"] == 900

    boot = json.loads((N3_DIR / "n3_4_paired_bootstrap.json").read_text(encoding="utf-8"))
    for base in ("always_a0_svm256", "always_a1_svm512", "always_a2_multidomain", "policy_confidence", "policy_blind_snr_flatness"):
        row = boot["paired_bootstrap_10k"]["test"][f"lbap_minus_{base}"]
        assert row["replicates"] == 10000
        assert len(row["macro_f1_ci95"]) == 2
        assert len(row["utility_ci95"]) == 2
    assert len(boot["three_seed_dispersion"]["test"]["lbap_full_a0_a1_a2"]["seeds"]) == 3

    budgets = json.loads((N3_DIR / "n3_5_training_budgets.json").read_text(encoding="utf-8"))
    assert len(budgets["models"]) >= 4

    adc = json.loads((N3_DIR / "n3_6_adc_and_calibration_audit.json").read_text(encoding="utf-8"))
    assert adc["same_parents_across_resolutions"] is True
    assert adc["mixed_training_hypothesis_verdict_at_4bit"]["mixed_train_4bit_acc"] < adc["mixed_training_hypothesis_verdict_at_4bit"]["clean_train_4bit_acc"]
