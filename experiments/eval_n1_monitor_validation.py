"""Validación Nivel 1 (N1.1, N1.3, N1.4): calibradores, evaluación SNR y demos con referencia."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.exp01_detection import _clean_window, _noise
from src.channels import add_awgn, apply_carrier_impairments, apply_multipath
from src.classification_data import load_classification_split
from src.conventional import CLASSES, load_model
from src.detection import calibrate_energy, detect_energy_file, energy_statistic
from src.features import FEATURE_VERSION
from src.gnuradio_interop import save_c64
from src.monitor import analyze_complete, load_signal
from src.signals import qam16_modulation, qpsk_modulation
from src.snr_estimation import calibrate_noise_reference, estimate_snr
from src.waveform import pulse_shape


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "nivel1_monitor"
RESERVED_SEED_N1 = 2026111601


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _calibration_metrics_10bins(prob: np.ndarray, correct: np.ndarray, n_boot: int = 2000, seed: int = RESERVED_SEED_N1):
    prob = np.asarray(prob, dtype=float)
    correct = np.asarray(correct, dtype=float)
    prob_clipped = np.clip(prob, 1e-8, 1.0 - 1e-8)

    brier = float(np.mean((prob - correct) ** 2))
    nll = float(-np.mean(correct * np.log(prob_clipped) + (1.0 - correct) * np.log(1.0 - prob_clipped)))

    edges = np.linspace(0.0, 1.0, 11)
    ece10 = 0.0
    bins_info = []
    n = len(prob)
    for b in range(10):
        lo, hi = float(edges[b]), float(edges[b + 1])
        mask = (prob >= lo) & (prob < hi if b < 9 else prob <= hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            mean_conf = float(np.mean(prob[mask]))
            emp_acc = float(np.mean(correct[mask]))
            ece10 += (cnt / n) * abs(mean_conf - emp_acc)
        else:
            mean_conf = None
            emp_acc = None
        bins_info.append(
            {
                "bin_index": b,
                "bin_lower": round(lo, 2),
                "bin_upper": round(hi, 2),
                "occupancy": cnt,
                "occupancy_fraction": float(cnt / n),
                "mean_confidence": mean_conf,
                "empirical_accuracy": emp_acc,
            }
        )

    rng = np.random.default_rng(seed)
    boot_brier = np.empty(n_boot, dtype=float)
    boot_nll = np.empty(n_boot, dtype=float)
    boot_ece = np.empty(n_boot, dtype=float)
    for r in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pb = prob[idx]
        pbc = prob_clipped[idx]
        cb = correct[idx]
        boot_brier[r] = np.mean((pb - cb) ** 2)
        boot_nll[r] = -np.mean(cb * np.log(pbc) + (1.0 - cb) * np.log(1.0 - pbc))
        eb = 0.0
        for b in range(10):
            lo, hi = edges[b], edges[b + 1]
            m = (pb >= lo) & (pb < hi if b < 9 else pb <= hi)
            if m.any():
                eb += m.mean() * abs(pb[m].mean() - cb[m].mean())
        boot_ece[r] = eb

    return {
        "n": int(n),
        "accuracy": float(np.mean(correct)),
        "brier": brier,
        "brier_ci95": np.percentile(boot_brier, [2.5, 97.5]).tolist(),
        "nll": nll,
        "nll_ci95": np.percentile(boot_nll, [2.5, 97.5]).tolist(),
        "ece10_equal_width": float(ece10),
        "ece10_ci95": np.percentile(boot_ece, [2.5, 97.5]).tolist(),
        "bins": bins_info,
    }


def _evaluate_svm_predictions(bundle: dict, iq: np.ndarray):
    pipe = bundle["pipeline"]
    n_samples = int(pipe.named_steps["features"].n_samples)
    preds = pipe.predict(iq[:, :n_samples]).astype(int)
    scores = pipe.decision_function(iq[:, :n_samples])
    sorted_scores = np.sort(scores, axis=1)[:, ::-1]
    margins = sorted_scores[:, 0] - sorted_scores[:, 1]
    return preds, scores, margins


def run_n1_4_calibration_study(output_dir: Path) -> dict:
    adaptive_root = PROJECT_ROOT / "data" / "adaptive_dataset"
    dev_parts = json.loads((PROJECT_ROOT / "results/adaptive/development_partitions.json").read_text(encoding="utf-8"))
    cal_parent_set = set(dev_parts["cal_parents"])

    m256_path = PROJECT_ROOT / "results/fase5/conventional_256.joblib"
    m512_path = PROJECT_ROOT / "results/fase5/conventional_512.joblib"
    b256 = load_model(m256_path)
    b512 = load_model(m512_path)
    sha256_m256 = _sha256(m256_path)
    sha256_m512 = _sha256(m512_path)

    val_data = load_classification_split(adaptive_root, "val", 512, allow_shift=False)
    cal_idx = np.array([i for i, r in enumerate(val_data["evaluation_rows"]) if r["parent_waveform_id"] in cal_parent_set], dtype=int)
    if len(cal_idx) != 1200:
        raise ValueError(f"Se esperaban 1200 padres en val_cal, encontrados {len(cal_idx)}.")

    test_id = load_classification_split(adaptive_root, "test", 512, allow_shift=False)
    test_shift = load_classification_split(adaptive_root, "test_shift", 512, allow_shift=True)

    splits_data = {
        "val_cal": {"iq": val_data["iq"][cal_idx], "targets": val_data["targets"][cal_idx]},
        "test_id": {"iq": test_id["iq"], "targets": test_id["targets"]},
        "test_shift": {"iq": test_shift["iq"], "targets": test_shift["targets"]},
    }

    # Calcular predicciones y características de calibración por split
    extracted = {}
    for split_name, d in splits_data.items():
        iq512 = d["iq"]
        y = d["targets"]

        p256_w0, s256_w0, m256_w0 = _evaluate_svm_predictions(b256, iq512[:, :256])
        p256_w1, s256_w1, m256_w1 = _evaluate_svm_predictions(b256, iq512[:, 256:512])
        p512, _, m512 = _evaluate_svm_predictions(b512, iq512)

        # Voto mayoritario 2x256 (desempate por suma de scores)
        n_obs = len(y)
        vote_pred = np.empty(n_obs, dtype=int)
        vote_frac = np.empty(n_obs, dtype=float)
        for i in range(n_obs):
            if p256_w0[i] == p256_w1[i]:
                vote_pred[i] = p256_w0[i]
                vote_frac[i] = 1.0
            else:
                vote_pred[i] = int(np.argmax(s256_w0[i] + s256_w1[i]))
                vote_frac[i] = 0.5
        mean_margin_2w = 0.5 * (m256_w0 + m256_w1)

        extracted[split_name] = {
            "y": y,
            "single_window_256": {
                "X": m256_w0[:, None],
                "correct": (p256_w0 == y).astype(float),
                "base_prob": 1.0 / (1.0 + np.exp(-0.8 * m256_w0)),
                "legacy_clipped": np.clip(1.0 / (1.0 + np.exp(-0.8 * m256_w0)), 0.20, 0.99),
            },
            "majority_vote_2x256": {
                "X": np.column_stack([mean_margin_2w, vote_frac]),
                "correct": (vote_pred == y).astype(float),
                "base_prob": 1.0 / (1.0 + np.exp(-0.8 * mean_margin_2w)),
                "legacy_clipped": np.clip((1.0 / (1.0 + np.exp(-0.8 * mean_margin_2w))) * vote_frac, 0.20, 0.99),
            },
            "single_window_512": {
                "X": m512[:, None],
                "correct": (p512 == y).astype(float),
                "base_prob": 1.0 / (1.0 + np.exp(-0.8 * m512)),
                "legacy_clipped": np.clip(1.0 / (1.0 + np.exp(-0.8 * m512)), 0.20, 0.99),
            },
        }

    # Ajustar exclusivamente en val_cal (1.200 padres independientes)
    cal_single_256 = LogisticRegression(random_state=RESERVED_SEED_N1)
    cal_single_256.fit(extracted["val_cal"]["single_window_256"]["X"], extracted["val_cal"]["single_window_256"]["correct"])

    cal_vote_2x256 = make_pipeline(StandardScaler(), LogisticRegression(random_state=RESERVED_SEED_N1))
    cal_vote_2x256.fit(extracted["val_cal"]["majority_vote_2x256"]["X"], extracted["val_cal"]["majority_vote_2x256"]["correct"])

    cal_single_512 = LogisticRegression(random_state=RESERVED_SEED_N1)
    cal_single_512.fit(extracted["val_cal"]["single_window_512"]["X"], extracted["val_cal"]["single_window_512"]["correct"])

    calibrator_pkg = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_dataset": "data/adaptive_dataset",
        "calibration_split": "val_cal",
        "n_calibration_parents": 1200,
        "feature_version": FEATURE_VERSION,
        "classes": CLASSES,
        "calibrators": {
            "single_window_256": {
                "model_sha256": sha256_m256,
                "window_len": 256,
                "aggregation_mode": "single_window",
                "expected_n_windows": 1,
                "feature_names": ["top2_margin"],
                "calibrator": cal_single_256,
            },
            "majority_vote_2x256": {
                "model_sha256": sha256_m256,
                "window_len": 256,
                "aggregation_mode": "majority_vote",
                "expected_n_windows": 2,
                "feature_names": ["mean_top2_margin", "vote_fraction"],
                "calibrator": cal_vote_2x256,
            },
            "single_window_512": {
                "model_sha256": sha256_m512,
                "window_len": 512,
                "aggregation_mode": "single_window",
                "expected_n_windows": 1,
                "feature_names": ["top2_margin"],
                "calibrator": cal_single_512,
            },
        },
    }
    cal_pkg_path = output_dir / "monitor_calibrators.joblib"
    joblib.dump(calibrator_pkg, cal_pkg_path, compress=3)

    fitted_models = {
        "single_window_256": cal_single_256,
        "majority_vote_2x256": cal_vote_2x256,
        "single_window_512": cal_single_512,
    }

    report = {
        "schema_version": 1,
        "calibrator_package_path": str(cal_pkg_path.relative_to(PROJECT_ROOT)),
        "calibrator_package_sha256": _sha256(cal_pkg_path),
        "calibration_units": 1200,
        "evaluation_units_per_regime": 2400,
        "artificial_clipping_applied": False,
        "modes": {},
    }
    bin_rows = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax_idx, mode in enumerate(("single_window_256", "majority_vote_2x256", "single_window_512")):
        mode_res = {}
        cal_model = fitted_models[mode]
        for eval_split in ("test_id", "test_shift"):
            item = extracted[eval_split][mode]
            prob_cal = cal_model.predict_proba(item["X"])[:, 1]
            m_base = _calibration_metrics_10bins(item["base_prob"], item["correct"])
            m_legacy = _calibration_metrics_10bins(item["legacy_clipped"], item["correct"])
            m_cal = _calibration_metrics_10bins(prob_cal, item["correct"])
            mode_res[eval_split] = {
                "base_unclipped_sigmoid": m_base,
                "legacy_clipped_heuristic": m_legacy,
                "calibrated": m_cal,
                "meets_ece10_le_0_05": bool(m_cal["ece10_equal_width"] <= 0.05),
                "brier_not_worse_than_base": bool(m_cal["brier"] <= m_base["brier"]),
            }
            for variant_name, m_dict in (("base_sigmoid", m_base), ("calibrated", m_cal)):
                for b in m_dict["bins"]:
                    bin_rows.append(
                        {
                            "mode": mode,
                            "split": eval_split,
                            "variant": variant_name,
                            **b,
                        }
                    )

        report["modes"][mode] = mode_res
        ax = axes[ax_idx]
        ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="Calibración perfecta")
        for split_label, color, marker in (("test_id", "royalblue", "o"), ("test_shift", "darkorange", "s")):
            bins = mode_res[split_label]["calibrated"]["bins"]
            xs = [b["mean_confidence"] for b in bins if b["occupancy"] > 0]
            ys = [b["empirical_accuracy"] for b in bins if b["occupancy"] > 0]
            ece_v = mode_res[split_label]["calibrated"]["ece10_equal_width"]
            ax.plot(xs, ys, marker=marker, color=color, lw=1.6, label=f"{split_label} (ECE={ece_v:.3f})")
        b_bins = mode_res["test_id"]["base_unclipped_sigmoid"]["bins"]
        ax.plot(
            [b["mean_confidence"] for b in b_bins if b["occupancy"] > 0],
            [b["empirical_accuracy"] for b in b_bins if b["occupancy"] > 0],
            marker="^",
            linestyle=":",
            color="gray",
            lw=1.2,
            label=f"base ID (ECE={mode_res['test_id']['base_unclipped_sigmoid']['ece10_equal_width']:.3f})",
        )
        ax.set_title(mode)
        ax.set_xlabel("Confianza predicha")
        ax.set_ylabel("Exactitud empírica")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=7.5, loc="upper left")

    fig.suptitle("N1.4 Curvas de confiabilidad (10 intervalos de igual anchura)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "n1_4_reliability_curves.png", dpi=160)
    plt.close(fig)

    (output_dir / "n1_4_calibration_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "n1_4_reliability_bins.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(bin_rows[0]))
        writer.writeheader()
        writer.writerows(bin_rows)

    return report


def run_n1_3_snr_evaluation(output_dir: Path) -> dict:
    e1_cfg = json.loads((PROJECT_ROOT / "configs/exp01_detection.json").read_text(encoding="utf-8"))
    n_samples = int(e1_cfg["n_samples"])
    noise_power = float(e1_cfg["noise_power"])
    fs_hz = float(e1_cfg["symbol_rate_hz"] * e1_cfg["sps"])

    root_ss = np.random.SeedSequence(RESERVED_SEED_N1)
    ref_seed, sig_seed, scale_seed = root_ss.spawn(3)

    ref_noise = _noise(np.random.default_rng(ref_seed), (5000, n_samples), noise_power)
    noise_ref = calibrate_noise_reference(ref_noise.ravel(), fs_hz=fs_hz, gain=1.0, window_len=n_samples)
    det_cal = calibrate_energy(ref_noise, pfa=0.01, fs_hz=fs_hz, gain=1.0)

    trials = int(e1_cfg["trials_per_class_snr_regime"])
    cells = [
        (regime, mod, snr)
        for regime in ("id", "shift")
        for mod in CLASSES
        for snr in e1_cfg["snr_db"]
    ]

    cell_rows = []
    agg_buckets = {(regime, snr): {"errors": [], "no_signal": 0, "nonpos": 0, "n_total": 0} for regime in ("id", "shift") for snr in e1_cfg["snr_db"]}

    for (regime, mod, snr), cseed in zip(cells, sig_seed.spawn(len(cells))):
        errors = []
        n_no_sig = 0
        n_nonpos = 0
        for pseed in cseed.spawn(trials):
            wseed, nseed = pseed.spawn(2)
            clean = _clean_window(e1_cfg, mod, regime, np.random.default_rng(wseed))
            scaled = clean * np.sqrt(noise_power * 10.0 ** (snr / 10.0) / energy_statistic(clean))
            noise = _noise(np.random.default_rng(nseed), n_samples, noise_power)
            obs = scaled + noise

            det = detect_energy_file(obs, window_len=n_samples, calibration=det_cal, fs_hz=fs_hz, gain=1.0)
            est = estimate_snr(obs, reference=noise_ref, detected=det["detected"], fs_hz=fs_hz, gain=1.0)
            realized_snr = float(10.0 * np.log10(energy_statistic(scaled) / energy_statistic(noise)))

            bkey = (regime, snr)
            agg_buckets[bkey]["n_total"] += 1
            if est["snr_db"] is not None:
                err = float(est["snr_db"] - realized_snr)
                errors.append(err)
                agg_buckets[bkey]["errors"].append(err)
            elif est["status"] == "no_signal":
                n_no_sig += 1
                agg_buckets[bkey]["no_signal"] += 1
            elif est["status"] == "nonpositive_signal_power":
                n_nonpos += 1
                agg_buckets[bkey]["nonpos"] += 1

        err_arr = np.asarray(errors, dtype=float)
        cell_rows.append(
            {
                "regime": regime,
                "modulation": mod,
                "snr_nominal_db": snr,
                "n_total": trials,
                "n_valid": int(len(err_arr)),
                "n_missing_no_signal": n_no_sig,
                "n_missing_nonpositive": n_nonpos,
                "coverage": float(len(err_arr) / trials),
                "bias_db": float(np.mean(err_arr)) if len(err_arr) else None,
                "mae_db": float(np.mean(np.abs(err_arr))) if len(err_arr) else None,
                "rmse_db": float(np.sqrt(np.mean(err_arr**2))) if len(err_arr) else None,
            }
        )

    agg_by_snr = []
    id_ge0_errors = []
    id_ge0_total = 0
    for regime in ("id", "shift"):
        for snr in e1_cfg["snr_db"]:
            b = agg_buckets[(regime, snr)]
            arr = np.asarray(b["errors"], dtype=float)
            if regime == "id" and snr >= 0:
                id_ge0_errors.extend(b["errors"])
                id_ge0_total += b["n_total"]
            agg_by_snr.append(
                {
                    "regime": regime,
                    "snr_nominal_db": snr,
                    "n_total": b["n_total"],
                    "n_valid": int(len(arr)),
                    "n_missing_no_signal": b["no_signal"],
                    "n_missing_nonpositive": b["nonpos"],
                    "coverage": float(len(arr) / b["n_total"]),
                    "bias_db": float(np.mean(arr)) if len(arr) else None,
                    "mae_db": float(np.mean(np.abs(arr))) if len(arr) else None,
                    "rmse_db": float(np.sqrt(np.mean(arr**2))) if len(arr) else None,
                }
            )

    id_ge0_arr = np.asarray(id_ge0_errors, dtype=float)
    id_ge0_coverage = float(len(id_ge0_arr) / id_ge0_total)
    id_ge0_mae = float(np.mean(np.abs(id_ge0_arr)))
    id_ge0_bias = float(np.mean(id_ge0_arr))

    # Verificación de invariancia de escala (ganancia 0.5, 1.0, 2.0)
    rng_s = np.random.default_rng(scale_seed)
    base_ref = _noise(rng_s, 2048, 1.0)
    base_sig = np.sqrt(10.0) * np.exp(2j * np.pi * 0.1 * np.arange(256)) + _noise(rng_s, 256, 1.0)
    scale_checks = []
    for g in (0.5, 1.0, 2.0):
        ref_g = calibrate_noise_reference(g * base_ref, fs_hz=fs_hz, gain=g)
        det_g = detect_energy_file(g * base_sig, window_len=256, noise_reference=ref_g, fs_hz=fs_hz, gain=g)
        snr_g = estimate_snr(g * base_sig, reference=ref_g, detected=det_g["detected"], fs_hz=fs_hz, gain=g)
        scale_checks.append(
            {
                "gain": g,
                "detected": det_g["detected"],
                "normalized_statistic": det_g["normalized_statistic"],
                "snr_db": snr_g["snr_db"],
            }
        )

    scale_invariant = all(
        c["detected"] == scale_checks[0]["detected"]
        and np.isclose(c["normalized_statistic"], scale_checks[0]["normalized_statistic"], rtol=1e-12)
        and np.isclose(c["snr_db"], scale_checks[0]["snr_db"], rtol=1e-12)
        for c in scale_checks
    )

    summary = {
        "schema_version": 1,
        "seed": RESERVED_SEED_N1,
        "n_samples": n_samples,
        "total_realizations": int(len(cells) * trials),
        "id_snr_ge_0db_aggregate": {
            "n_total": id_ge0_total,
            "n_valid": int(len(id_ge0_arr)),
            "coverage": id_ge0_coverage,
            "bias_db": id_ge0_bias,
            "mae_db": id_ge0_mae,
            "meets_coverage_ge_0_90": bool(id_ge0_coverage >= 0.90),
            "meets_mae_le_3_0_db": bool(id_ge0_mae <= 3.0),
        },
        "scale_invariance_verified": bool(scale_invariant),
        "scale_checks": scale_checks,
        "by_regime_and_snr": agg_by_snr,
    }

    (output_dir / "n1_3_snr_evaluation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "n1_3_snr_by_class_snr.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cell_rows[0]))
        writer.writeheader()
        writer.writerows(cell_rows)

    return summary


def run_demos_and_historical_audit(output_dir: Path) -> dict:
    # 1. Auditoría de las dos demos históricas SIN umbral manual (comportamiento predeterminado seguro)
    hist_paths = [
        PROJECT_ROOT / "results/gnuradio/py_qpsk_15db.c64",
        PROJECT_ROOT / "results/gnuradio/sim_run/grc_sim_output.c64",
    ]
    hist_audit = {}
    for hp in hist_paths:
        iq = load_signal(hp)
        res = analyze_complete(iq, signal_id=hp.stem, fs_hz=8000.0)
        hist_audit[hp.stem] = {
            "file": str(hp.relative_to(PROJECT_ROOT)),
            "detected": res["detection"]["detected"],
            "detection_status": res["detection"]["status"],
            "calibration_status": res["detection"]["calibration_status"],
            "modulation": res["classification"]["modulation"],
            "confidence": res["classification"]["confidence"],
        }
    (output_dir / "historical_demos_audit.json").write_text(
        json.dumps(hist_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # 2. Dos nuevas demos con referencias de ruido independientes (N1.1)
    demos_dir = output_dir / "demos"
    demos_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RESERVED_SEED_N1)

    # Demo A: QPSK 512 muestras (2 ventanas de 256 -> calibrado exacto majority_vote_2x256) a ~15 dB con potencia de ruido 0.05
    noise_pow_a = 0.05
    bits_a = rng.integers(0, 2, size=512)
    sym_a = qpsk_modulation(bits_a)
    shaped_a, h_a = pulse_shape(sym_a, sps=8, beta=0.35, span=10)
    carr_a = apply_carrier_impairments(shaped_a, fs_hz=8000.0, phase_rad=0.25, cfo_hz=35.0)
    mp_a = apply_multipath(carr_a, taps=[1.0, 0.0, 0.15j])
    clean_seg_a = mp_a[len(h_a) + 2 : len(h_a) + 2 + 512]
    clean_seg_a = clean_seg_a * np.sqrt((noise_pow_a * 10 ** (15.0 / 10.0)) / energy_statistic(clean_seg_a))
    noise_a = np.sqrt(noise_pow_a / 2.0) * (rng.standard_normal(512) + 1j * rng.standard_normal(512))
    rx_a = clean_seg_a + noise_a
    ref_a = np.sqrt(noise_pow_a / 2.0) * (rng.standard_normal(4096) + 1j * rng.standard_normal(4096))

    demo_a_path = demos_dir / "demo_py_qpsk_15db.c64"
    ref_a_path = demos_dir / "demo_py_noise_ref.c64"
    save_c64(demo_a_path, rx_a)
    save_c64(ref_a_path, ref_a)
    (demos_dir / "demo_py_qpsk_15db.c64.meta.json").write_text(
        json.dumps({"fs_hz": 8000.0, "sps": 8, "symbol_rate_hz": 1000.0}, indent=2) + "\n", encoding="utf-8"
    )

    # Demo B: Transmisión tardía en ventanas 3 y 4 (muestras 768..1280) tras 3 ventanas iniciales de solo ruido
    noise_pow_b = 0.25
    bits_b = rng.integers(0, 2, size=1024)
    sym_b = qam16_modulation(bits_b)
    shaped_b, h_b = pulse_shape(sym_b, sps=8, beta=0.35, span=10)
    clean_seg_b = shaped_b[len(h_b) : len(h_b) + 512]
    clean_seg_b = clean_seg_b * np.sqrt((noise_pow_b * 10 ** (12.0 / 10.0)) / energy_statistic(clean_seg_b))
    bg_noise_b = np.sqrt(noise_pow_b / 2.0) * (rng.standard_normal(1280) + 1j * rng.standard_normal(1280))
    rx_b = bg_noise_b.copy()
    rx_b[768:1280] += clean_seg_b
    ref_b = np.sqrt(noise_pow_b / 2.0) * (rng.standard_normal(4096) + 1j * rng.standard_normal(4096))

    demo_b_path = demos_dir / "demo_late_tx_16qam_12db.c64"
    ref_b_path = demos_dir / "demo_late_tx_noise_ref.c64"
    save_c64(demo_b_path, rx_b)
    save_c64(ref_b_path, ref_b)

    demo_results = {}
    for name, sig_p, ref_p, extra_args in (
        ("demo_py_qpsk_15db", demo_a_path, ref_a_path, []),
        ("demo_late_tx_16qam_12db", demo_b_path, ref_b_path, ["--fs-hz", "8000"]),
    ):
        out_json = demos_dir / f"{name}.json"
        out_png = demos_dir / f"{name}_psd.png"
        cmd = [
            sys.executable,
            "-m",
            "src.monitor",
            "--input",
            str(sig_p),
            "--noise-ref",
            str(ref_p),
            "--output",
            str(out_json),
            "--plot",
            str(out_png),
            *extra_args,
        ]
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=True)
        demo_results[name] = json.loads(out_json.read_text(encoding="utf-8"))

    return {"historical_audit": hist_audit, "new_demos": demo_results}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cal_report = run_n1_4_calibration_study(RESULTS_DIR)
    snr_report = run_n1_3_snr_evaluation(RESULTS_DIR)
    demos_report = run_demos_and_historical_audit(RESULTS_DIR)

    summary = {
        "n1_4_id_ece10": {
            mode: cal_report["modes"][mode]["test_id"]["calibrated"]["ece10_equal_width"]
            for mode in cal_report["modes"]
        },
        "n1_4_id_brier_cal_vs_base": {
            mode: {
                "calibrated": cal_report["modes"][mode]["test_id"]["calibrated"]["brier"],
                "base": cal_report["modes"][mode]["test_id"]["base_unclipped_sigmoid"]["brier"],
            }
            for mode in cal_report["modes"]
        },
        "n1_3_id_ge0db": snr_report["id_snr_ge_0db_aggregate"],
        "n1_1_scale_invariance": snr_report["scale_invariance_verified"],
        "historical_demos_uncalibrated_status": {
            k: v["detection_status"] for k, v in demos_report["historical_audit"].items()
        },
        "new_demos_summary": {
            k: {
                "detected": v["detection"]["detected"],
                "active_windows": v["detection"]["active_window_indices"],
                "snr_est_db": v["snr"]["snr_est_db"],
                "modulation": v["classification"]["modulation"],
                "confidence": v["classification"]["confidence"],
                "confidence_calibrated": v["classification"]["confidence_calibrated"],
            }
            for k, v in demos_report["new_demos"].items()
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
