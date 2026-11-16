"""Pruebas de aceptación para Nivel 0 (N0.1-N0.2) y Nivel 1 (N1.1-N1.6)."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pytest

from experiments.verify_n0_protocol import check_tamper_detection_on_copy
from src.detection import calibrate_energy, detect_energy_file, energy_statistic
from src.gnuradio_interop import save_c64
from src.monitor import analyze_complete, load_signal
from src.snr_estimation import calibrate_noise_reference
from src.spectrum import estimate_spectrum


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_qpsk_burst(rng, n_samples=512, snr_db=15.0, noise_power=0.1):
    # Portadora QPSK sencilla de banda base con energía controlada para pruebas unitarias
    idx = np.arange(n_samples)
    syms = rng.choice([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], size=(n_samples // 8) + 2) / np.sqrt(2.0)
    clean = np.repeat(syms, 8)[:n_samples] * np.exp(2j * np.pi * 0.01 * idx)
    clean *= np.sqrt(noise_power * 10.0 ** (snr_db / 10.0) / energy_statistic(clean))
    noise = np.sqrt(noise_power / 2.0) * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
    return clean + noise


# --- Nivel 0 ---


def test_n0_1_manifest_and_tamper_detection():
    manifest_path = PROJECT_ROOT / "results/nivel0_evidencia/manifest_historical_state.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hist = manifest["historical_key_figures"]["adaptive_test_nominal_paper_research"]
    assert hist["svm256_pct"] == pytest.approx(43.75, abs=1e-4)
    assert hist["svm512_pct"] == pytest.approx(46.0417, abs=1e-4)
    assert hist["multidomain_net_pct"] == pytest.approx(51.5833, abs=1e-4)
    assert hist["lbap_full_pct"] == pytest.approx(51.625, abs=1e-4)

    tamper = check_tamper_detection_on_copy(manifest, PROJECT_ROOT)
    assert tamper["original_unmodified"] is True
    assert tamper["tamper_detected_on_copy"] is True


def test_n0_2_zero_shared_parents_and_exact_prefixes():
    audit_path = PROJECT_ROOT / "results/nivel0_evidencia/partition_independence_audit.json"
    assert audit_path.is_file()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["shared_parents_count"] == 0
    assert audit["all_prefixes_exact"] is True
    assert audit["all_internal_splits_disjoint"] is True
    assert audit["all_cross_datasets_disjoint"] is True


# --- N1.1 Detección calibrada e invarianza de escala ---


def test_n1_1_uncalibrated_returns_indeterminate_not_no_signal():
    rng = np.random.default_rng(1001)
    sig = _make_qpsk_burst(rng, n_samples=512, snr_db=15.0, noise_power=0.1)
    res = analyze_complete(sig, fs_hz=8000.0)
    assert res["detection"]["detected"] is None
    assert res["detection"]["status"] == "indeterminate_uncalibrated"
    assert res["classification"]["modulation"] == "INDETERMINATE_DETECTION"
    assert res["classification"]["confidence"] is None


def test_n1_1_scale_invariance_signal_and_reference():
    rng = np.random.default_rng(1002)
    noise_base = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)) / np.sqrt(2.0)
    sig_base = _make_qpsk_burst(rng, n_samples=512, snr_db=12.0, noise_power=1.0)

    ref_results = []
    for gain in (0.25, 1.0, 4.0):
        ref = calibrate_noise_reference(gain * noise_base, fs_hz=8000.0, gain=gain)
        res = analyze_complete(
            gain * sig_base,
            fs_hz=8000.0,
            noise_reference=ref,
            gain=gain,
        )
        ref_results.append(res)

    base_stat = ref_results[0]["detection"]["normalized_statistic"]
    base_snr = ref_results[0]["snr"]["snr_est_db"]
    for r in ref_results:
        assert r["detection"]["detected"] is True
        assert r["detection"]["normalized_statistic"] == pytest.approx(base_stat, rel=1e-9)
        assert r["snr"]["snr_est_db"] == pytest.approx(base_snr, rel=1e-9)


def test_n1_1_late_transmission_detected_and_file_pfa_threshold():
    rng = np.random.default_rng(1003)
    noise_pow = 0.2
    ref_iq = np.sqrt(noise_pow / 2.0) * (rng.standard_normal(4096) + 1j * rng.standard_normal(4096))
    ref = calibrate_noise_reference(ref_iq, fs_hz=8000.0)

    # 3 primeras ventanas solo ruido (0..767), últimas 2 ventanas con señal (768..1279)
    bg = np.sqrt(noise_pow / 2.0) * (rng.standard_normal(1280) + 1j * rng.standard_normal(1280))
    burst = _make_qpsk_burst(rng, n_samples=512, snr_db=15.0, noise_power=noise_pow)
    bg[768:1280] = burst

    res = analyze_complete(bg, fs_hz=8000.0, noise_reference=ref)
    assert res["detection"]["detected"] is True
    assert res["detection"]["num_windows_evaluated"] == 5
    assert res["detection"]["threshold_per_file"] > res["detection"]["threshold_per_window"]
    assert 3 in res["detection"]["active_window_indices"]
    assert 4 in res["detection"]["active_window_indices"]
    assert 0 not in res["detection"]["active_window_indices"]


# --- N1.2 Frecuencia de muestreo y compatibilidad ---


def test_n1_2_missing_and_invalid_fs_and_doubling():
    rng = np.random.default_rng(1004)
    sig = _make_qpsk_burst(rng, n_samples=512, snr_db=15.0, noise_power=0.1)
    ref = calibrate_noise_reference(rng.standard_normal(2048) + 1j * rng.standard_normal(2048))

    res_no_fs = analyze_complete(sig, fs_hz=None, noise_reference=ref)
    assert res_no_fs["spectrum"]["bandwidth_unit"] == "ciclos/muestra"
    assert res_no_fs["classification"]["modulation"] == "OUT_OF_SCOPE_MISSING_FS"
    assert res_no_fs["classification"]["confidence"] is None

    s1 = estimate_spectrum(sig, fs_hz=4000.0)
    s2 = estimate_spectrum(sig, fs_hz=8000.0)
    assert s2["total_power_bandwidth"] == pytest.approx(2.0 * s1["total_power_bandwidth"], rel=1e-12)
    assert s2["integrated_power"] == pytest.approx(s1["integrated_power"], rel=1e-12)

    for bad_fs in (0.0, -8000.0, np.nan, np.inf, True):
        with pytest.raises(ValueError):
            estimate_spectrum(sig, fs_hz=bad_fs)
        with pytest.raises(ValueError):
            analyze_complete(sig, fs_hz=bad_fs)

    res_bad_fs = analyze_complete(sig, fs_hz=16000.0, default_threshold=0.01)
    assert res_bad_fs["classification"]["modulation"] == "OUT_OF_SCOPE_INCOMPATIBLE_SAMPLING"

    res_bad_sps = analyze_complete(sig, fs_hz=8000.0, default_threshold=0.01, signal_metadata={"sps": 4})
    assert res_bad_sps["classification"]["modulation"] == "OUT_OF_SCOPE_INCOMPATIBLE_SAMPLING"


# --- N1.3 SNR estados explícitos ---


def test_n1_3_snr_states_and_incompatible_reference():
    rng = np.random.default_rng(1005)
    sig = _make_qpsk_burst(rng, n_samples=256, snr_db=10.0, noise_power=0.1)
    ref_8k = calibrate_noise_reference(rng.standard_normal(1024) + 1j * rng.standard_normal(1024), fs_hz=8000.0)
    ref_16k = calibrate_noise_reference(rng.standard_normal(1024) + 1j * rng.standard_normal(1024), fs_hz=16000.0)

    # Sin referencia
    r_no_ref = analyze_complete(sig, fs_hz=8000.0, default_threshold=0.01)
    assert r_no_ref["snr"]["snr_est_db"] is None
    assert r_no_ref["snr"]["status"] == "no_identificable_sin_supuestos"

    # Detector negativo
    r_neg = analyze_complete(sig * 0.01, fs_hz=8000.0, noise_reference=ref_8k)
    assert r_neg["detection"]["detected"] is False
    assert r_neg["snr"]["snr_est_db"] is None
    assert r_neg["snr"]["status"] == "no_signal"

    # Referencia con fs incompatible
    r_incompat = analyze_complete(sig, fs_hz=8000.0, noise_reference=ref_16k)
    assert r_incompat["detection"]["detected"] is None
    assert r_incompat["snr"]["snr_est_db"] is None


# --- N1.4 Calibración y ausencia de recorte artificial ---


def test_n1_4_calibrator_matching_and_no_artificial_clipping():
    cal_path = PROJECT_ROOT / "results/nivel1_monitor/monitor_calibrators.joblib"
    assert cal_path.is_file()
    rng = np.random.default_rng(1006)
    # Señal débil con umbral manual para forzar clasificación con margen bajo
    weak_noise = 0.01 * (rng.standard_normal(256) + 1j * rng.standard_normal(256))
    res_1w = analyze_complete(weak_noise, fs_hz=8000.0, default_threshold=1e-6)
    assert res_1w["classification"]["confidence_calibrated"] is True
    assert 0.0 <= res_1w["classification"]["confidence"] <= 1.0

    # Verificar que no existe recorte artificial a [0.20, 0.99]: usando el calibrador guardado con margen negativo extremo
    pkg = joblib.load(cal_path)
    cal_1w = pkg["calibrators"]["single_window_256"]["calibrator"]
    assert float(cal_1w.predict_proba([[-20.0]])[0, 1]) < 0.20

    # 3 ventanas de 256 -> longitud de agregación no calibrada para 3 ventanas -> confidence_calibrated debe ser False
    sig_3w = _make_qpsk_burst(rng, n_samples=768, snr_db=15.0, noise_power=0.1)
    res_3w = analyze_complete(sig_3w, fs_hz=8000.0, default_threshold=0.01)
    assert res_3w["classification"]["confidence_calibrated"] is False
    assert "uncalibrated" in res_3w["classification"]["confidence_status"]


# --- N1.5 Contratos de entrada, longitudes y errores CLI ---


def test_n1_5_formats_valid_and_invalid(tmp_path):
    iq = np.exp(2j * np.pi * 0.05 * np.arange(256)).astype(np.complex64)

    # .c64 válido vs truncado
    p_c64 = tmp_path / "ok.c64"
    save_c64(p_c64, iq)
    assert load_signal(p_c64).shape == (256,)
    p_c64_bad = tmp_path / "trunc.c64"
    p_c64_bad.write_bytes(p_c64.read_bytes()[:-3])
    with pytest.raises(ValueError):
        load_signal(p_c64_bad)

    # .bin válido vs vacío
    p_bin = tmp_path / "ok.bin"
    p_bin.write_bytes(iq.tobytes())
    assert load_signal(p_bin).shape == (256,)
    p_bin_empty = tmp_path / "empty.bin"
    p_bin_empty.write_bytes(b"")
    with pytest.raises(ValueError):
        load_signal(p_bin_empty)

    # .npy válido vs lote 2D complejo inválido
    p_npy = tmp_path / "ok.npy"
    np.save(p_npy, iq)
    assert load_signal(p_npy).shape == (256,)
    p_npy_batch = tmp_path / "batch.npy"
    np.save(p_npy_batch, np.vstack([iq, iq]))
    with pytest.raises(ValueError):
        load_signal(p_npy_batch)

    # .npz válido vs multi-clave inválido
    p_npz = tmp_path / "ok.npz"
    np.savez(p_npz, iq=iq, fs_hz=np.array(8000.0))
    loaded_iq, meta = load_signal(p_npz, return_metadata=True)
    assert loaded_iq.shape == (256,) and meta["fs_hz"] == 8000.0
    p_npz_multi = tmp_path / "multi.npz"
    np.savez(p_npz_multi, iq256=iq, iq512=np.tile(iq, 2))
    with pytest.raises(ValueError):
        load_signal(p_npz_multi)

    # .csv válido con cabecera vs columnas inconsistentes
    p_csv = tmp_path / "ok.csv"
    p_csv.write_text("I,Q\n" + "\n".join(f"{z.real},{z.imag}" for z in iq), encoding="utf-8")
    assert load_signal(p_csv).shape == (256,)
    p_csv_bad = tmp_path / "bad.csv"
    p_csv_bad.write_text("I,Q\n1.0,2.0\n3.0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_signal(p_csv_bad)

    # Extensión no soportada y valores no finitos
    p_txt = tmp_path / "bad.txt"
    p_txt.write_text("1,2\n3,4\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_signal(p_txt)
    p_nan = tmp_path / "nan.npy"
    bad_iq = iq.copy()
    bad_iq[10] = np.nan + 0j
    np.save(p_nan, bad_iq)
    with pytest.raises(ValueError):
        load_signal(p_nan)


def test_n1_5_lengths_255_256_512_and_missing_model_cli(tmp_path):
    rng = np.random.default_rng(1007)
    sig512 = _make_qpsk_burst(rng, n_samples=512, snr_db=15.0, noise_power=0.1)

    r255 = analyze_complete(sig512[:255], fs_hz=8000.0, default_threshold=0.01)
    assert r255["classification"]["modulation"] == "OUT_OF_SCOPE_SHORT_SIGNAL"
    assert r255["classification"]["confidence"] is None

    r256 = analyze_complete(sig512[:256], fs_hz=8000.0, default_threshold=0.01)
    assert r256["classification"]["modulation"] in ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "4PAM")
    assert r256["classification"]["confidence_calibrated"] is True

    r512 = analyze_complete(sig512, fs_hz=8000.0, default_threshold=0.01)
    assert r512["classification"]["windows_evaluated"] == 2
    assert r512["classification"]["confidence_calibrated"] is True

    # Modelo ausente por CLI no provoca TypeError secundario al imprimir confianza null
    c64_in = tmp_path / "sig.c64"
    save_c64(c64_in, sig512)
    out_json = tmp_path / "out.json"
    out_png = tmp_path / "out.png"
    proc_missing_model = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.monitor",
            "--input",
            str(c64_in),
            "--fs-hz",
            "8000",
            "--threshold",
            "0.01",
            "--model",
            str(tmp_path / "nonexistent_model.joblib"),
            "--output",
            str(out_json),
            "--plot",
            str(out_png),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc_missing_model.returncode == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["classification"]["status"] == "model_not_found"
    assert payload["classification"]["confidence"] is None

    # Archivo de entrada inválido -> código de salida 2
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("bad", encoding="utf-8")
    proc_bad = subprocess.run(
        [sys.executable, "-m", "src.monitor", "--input", str(bad_file)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc_bad.returncode == 2
    assert "ERROR:" in proc_bad.stderr


# --- N1.6 Espectro (Parseval <= 1e-6, pico de tono) y metadatos de alcance ---


def test_n1_6_tone_peak_parseval_double_precision_and_scope():
    fs, n = 8000.0, 512
    f0 = 1250.0
    x = np.exp(2j * np.pi * f0 * np.arange(n) / fs)
    spec = estimate_spectrum(x, fs_hz=fs, nfft=512)
    peak_idx = int(np.argmax(spec["psd"]))
    expected_idx = int(np.argmin(np.abs(spec["frequency"] - f0)))
    assert abs(peak_idx - expected_idx) <= 1

    # Identidad de Parseval exacta para ventana Hann periódica en doble precisión
    w = np.hanning(n + 1)[:-1]
    expected_weighted_power = float(np.sum(np.abs(x * w) ** 2) / np.sum(w**2))
    assert spec["integrated_power"] == pytest.approx(expected_weighted_power, rel=1e-6)
    assert spec["bandwidth_scope"] == "total_observed_power_includes_noise"

    res = analyze_complete(x, fs_hz=fs, default_threshold=0.01)
    assert res["spectrum"]["bandwidth_type"] == "total_power_99_includes_noise"
    assert res["classification"]["model_sha256"] is not None
    assert res["classification"]["scope"] == "signal_present_closed_set_6_classes"
