"""Tests de verificación para Nivel 5 (N5.1 diagnóstico controlado de generalización y clases difíciles)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
N5_DIR = ROOT_DIR / "results" / "nivel5_diagnostico"


def test_n5_1_diagnostic_artifacts_and_hypotheses() -> None:
    """Verifica el reporte controlado N5.1: H1 (CFO vs multipath), H2 (16QAM/64QAM) y H3 (GNU Radio QPSK->BPSK)."""
    rep_path = N5_DIR / "n5_1_diagnostic_report.json"
    csv_h1_path = N5_DIR / "n5_1_factorial_metrics_by_class_snr.csv"
    csv_h2_path = N5_DIR / "n5_1_qam_confusion_breakdown.csv"
    fig_path = N5_DIR / "n5_1_diagnostic_summary.png"

    assert rep_path.exists(), f"Falta reporte N5.1: {rep_path}"
    assert csv_h1_path.exists()
    assert csv_h2_path.exists()
    assert fig_path.exists() and fig_path.stat().st_size > 10_000

    rep = json.loads(rep_path.read_text(encoding="utf-8"))

    # H1: 720 padres pareados, 5 escenarios x 5 modelos, 10k réplicas bootstrap
    h1 = rep["h1_channel_shift_factorial"]
    assert h1["n_parents_per_scenario"] == 720
    assert set(h1["scenarios"].keys()) == {"id", "cfo_zero", "cfo_shift", "multipath_shift", "both_shift"}
    for sc_vals in h1["scenarios"].values():
        assert set(sc_vals.keys()) == {"svm256", "svm512", "cnn_original", "cnn_phase", "multidomain_a2"}
    assert h1["paired_effects_vs_id"]["cnn_original"]["cfo_shift_minus_id"]["replicates"] == 10000
    assert h1["paired_effects_vs_id"]["cnn_original"]["cfo_shift_minus_id"]["accuracy_ci95"][1] < 0.0
    assert h1["paired_effects_vs_id"]["svm256"]["multipath_shift_minus_id"]["accuracy_ci95"][1] < 0.0

    # H2: 240 padres pareados (120 16QAM + 120 64QAM) por factor en SNR >= 10 dB
    h2 = rep["h2_qam16_qam64_confusion"]
    assert h2["n_parents_per_factor"] == 240
    assert set(h2["by_factor"].keys()) == {"awgn_cfo0_no_mp", "cfo_only_id", "mp_only_id", "id_full", "mp_shift_cfo0"}
    for fac_dict in h2["by_factor"].values():
        for m_row in fac_dict.values():
            assert m_row["n_16qam"] == 120 and m_row["n_64qam"] == 120
            assert 0 <= m_row["16qam_pred_64qam_count"] <= 120
            assert 0 <= m_row["64qam_pred_16qam_count"] <= 120

    # H3: Diagnóstico de un factor GNU Radio QPSK -> BPSK (CFO=0 vs CFO=35 Hz y MultiDomainAMCNet)
    h3 = rep["h3_gnuradio_qpsk_anomaly"]
    assert h3["total_windows_256"] == 16
    hits_raw = h3["conditions"]["svm256_raw_cfo0_all16"]["qpsk_hits"]
    sweep = {row["cfo_hz"]: row["svm256_qpsk_hits"] for row in h3["cfo_injection_sweep_same_grc_signal"]}
    assert sweep[0.0] <= sweep[35.0]
    assert h3["conditions"]["multidomain_a2_raw_all16"]["steady_1_to_15_qpsk_hits"] >= hits_raw

    with csv_h1_path.open(encoding="utf-8", newline="") as f:
        assert len(list(csv.DictReader(f))) > 100
    with csv_h2_path.open(encoding="utf-8", newline="") as f:
        assert len(list(csv.DictReader(f))) == 20
