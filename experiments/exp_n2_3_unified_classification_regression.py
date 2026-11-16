"""Nivel 2 (N2.3): Evaluación unificada E2-E6 por benchmark y control de regresión por recarga/interfaz."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

from experiments.exp02_conventional import _csv, _json, _sha, classification_metrics
from experiments.exp12_paper_research import MODEL_WIDTH_MULT, predict_multidomain
from src.advanced import SmallCNN, load_checkpoint, predict, preprocess
from src.classification_data import load_classification_split
from src.conventional import CLASSES, load_model
from src.monitor import analyze_complete
from src.research_models import MultiDomainAMCNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "nivel2_validacion" / "n2_3_classification"


def _evaluate_all_models_on_dataset(
    dataset_name: str,
    dataset_root: Path,
    svms: dict,
    cnns: dict,
    multidomain: MultiDomainAMCNet,
    device: str,
) -> tuple[dict, list[dict], list[dict]]:
    results = {}
    by_snr_rows = []
    pred_rows = []

    for split in ("test", "test_shift"):
        d = load_classification_split(dataset_root, split, 512, allow_shift=True)
        iq512 = d["iq"]
        iq256 = iq512[:, :256]
        truth = d["targets"]
        rows_meta = d["evaluation_rows"]
        snr_arr = np.array([float(r["snr_nominal_db"]) for r in rows_meta])
        x_cnn = torch.from_numpy(preprocess(iq256))

        if device == "cuda":
            torch.set_float32_matmul_precision("highest")
        preds = {
            "svm256": svms["svm256"].predict(iq256).astype(int),
            "svm512": svms["svm512"].predict(iq512).astype(int),
            "cnn_fase6_original": predict(cnns["cnn_fase6_original"], x_cnn, device).argmax(1).astype(int),
            "cnn_fase6_phase_aug": predict(cnns["cnn_fase6_phase_aug"], x_cnn, device).argmax(1).astype(int),
        }
        if device == "cuda":
            torch.set_float32_matmul_precision("high")
        md_logits, _ = predict_multidomain(multidomain, iq256, device=device)
        preds["multidomain_clean_a2"] = md_logits.argmax(1).astype(int)

        split_res = {}
        for mname, p_arr in preds.items():
            m = classification_metrics(truth, p_arr)
            cm = confusion_matrix(truth, p_arr, labels=np.arange(len(CLASSES))).tolist()
            split_res[mname] = {**m, "confusion_matrix": cm}

            for snr_val in sorted(set(snr_arr)):
                mask = snr_arr == snr_val
                ms = classification_metrics(truth[mask], p_arr[mask])
                by_snr_rows.append(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "model": mname,
                        "snr_db": snr_val,
                        "n": ms["n"],
                        "accuracy": ms["accuracy"],
                        "macro_f1": ms["macro_f1"],
                    }
                )

            if mname == "multidomain_clean_a2":
                split_res[mname]["_logits_array"] = md_logits
            split_res[mname]["_preds_array"] = p_arr

        for idx, meta in enumerate(rows_meta):
            pred_rows.append(
                {
                    "dataset": dataset_name,
                    "split": split,
                    "parent_waveform_id": meta["parent_waveform_id"],
                    "true_idx": int(truth[idx]),
                    "true_class": CLASSES[int(truth[idx])],
                    "snr_nominal_db": meta["snr_nominal_db"],
                    **{f"pred_{mname}": CLASSES[int(preds[mname][idx])] for mname in preds},
                }
            )

        results[split] = split_res

    return results, by_snr_rows, pred_rows


def _regression_checks(
    svms: dict,
    cnns: dict,
    multidomain: MultiDomainAMCNet,
    adaptive_eval: dict,
    holdout_eval: dict,
    device: str,
) -> dict:
    # 1. Fase 5 e2_dataset/test (4.800 padres) vs results/fase5/test_predictions_evaluation_only.csv
    e2_test = load_classification_split(PROJECT_ROOT / "data/e2_dataset", "test", 512, allow_shift=False)
    e2_p256 = svms["svm256"].predict(e2_test["iq"][:, :256])
    e2_p512 = svms["svm512"].predict(e2_test["iq"])
    e2_map_256 = {r["parent_waveform_id"]: CLASSES[int(p)] for r, p in zip(e2_test["evaluation_rows"], e2_p256)}
    e2_map_512 = {r["parent_waveform_id"]: CLASSES[int(p)] for r, p in zip(e2_test["evaluation_rows"], e2_p512)}

    fase5_mismatches = {256: 0, 512: 0}
    fase5_checked = {256: 0, 512: 0}
    with (PROJECT_ROOT / "results/fase5/test_predictions_evaluation_only.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            n_len = int(row["n_samples"])
            pid = row["parent_waveform_id"]
            fase5_checked[n_len] += 1
            ref_pred = e2_map_256[pid] if n_len == 256 else e2_map_512[pid]
            if row["prediction"] != ref_pred:
                fase5_mismatches[n_len] += 1

    # 2. Fase 6 holdout (4.800 test + 4.800 test_shift) vs results/fase6_holdout_final/predictions.csv
    holdout_rows_map = {}
    for split in ("test", "test_shift"):
        d = load_classification_split(PROJECT_ROOT / "data/fase6_holdout_sellado", split, 512, allow_shift=True)
        p_map = {
            "svm256": holdout_eval[split]["svm256"]["_preds_array"],
            "svm512": holdout_eval[split]["svm512"]["_preds_array"],
            "cnn_original": holdout_eval[split]["cnn_fase6_original"]["_preds_array"],
            "cnn_phase": holdout_eval[split]["cnn_fase6_phase_aug"]["_preds_array"],
        }
        for idx, r in enumerate(d["evaluation_rows"]):
            for mname, arr in p_map.items():
                holdout_rows_map[(split, mname, r["parent_waveform_id"])] = int(arr[idx])

    holdout_checked = 0
    holdout_mismatches = 0
    with (PROJECT_ROOT / "results/fase6_holdout_final/predictions.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["split"], row["model"], row["parent_waveform_id"])
            holdout_checked += 1
            if int(row["predicted"]) != holdout_rows_map[key]:
                holdout_mismatches += 1

    # 3. Paper research caches (adaptive test + test_shift) para svm256, svm512 y multidomain_clean_a2
    paper_checks = {}
    for split in ("test", "test_shift"):
        cache_path = PROJECT_ROOT / f"results/paper_research/{split}_research_cache.npz"
        d = load_classification_split(PROJECT_ROOT / "data/adaptive_dataset", split, 512, allow_shift=True)
        with np.load(cache_path, allow_pickle=False) as z:
            cached_preds = z["predictions"]  # (2400, 3): [svm256, svm512, a2]
            cached_logits = z["logits_a2"]
            cached_truth = z["truth"]
        assert np.array_equal(cached_truth, d["targets"])

        re_256 = adaptive_eval[split]["svm256"]["_preds_array"]
        re_512 = adaptive_eval[split]["svm512"]["_preds_array"]
        re_logits = adaptive_eval[split]["multidomain_clean_a2"]["_logits_array"]
        re_a2 = re_logits.argmax(1).astype(int)

        max_logit_diff = float(np.max(np.abs(re_logits - cached_logits)))
        mean_logit_diff = float(np.mean(np.abs(re_logits - cached_logits)))
        paper_checks[split] = {
            "n": int(len(cached_truth)),
            "svm256_class_mismatches": int(np.sum(re_256 != cached_preds[:, 0])),
            "svm512_class_mismatches": int(np.sum(re_512 != cached_preds[:, 1])),
            "multidomain_a2_class_mismatches": int(np.sum(re_a2 != cached_preds[:, 2])),
            "multidomain_a2_max_abs_logit_diff": max_logit_diff,
            "multidomain_a2_mean_abs_logit_diff": mean_logit_diff,
            "logit_tolerance": 2e-3,
            "within_logit_tolerance": bool(max_logit_diff <= 2e-3),
        }

    # 4. Control de no-regresión de interfaz monitor vs svm256 directo (primeras 200 de adaptive/test)
    d_adap_test = load_classification_split(PROJECT_ROOT / "data/adaptive_dataset", "test", 512, allow_shift=False)
    n_check = 200
    direct_preds = svms["svm256"].predict(d_adap_test["iq"][:n_check, :256]).astype(int)
    monitor_mismatches = 0
    for i in range(n_check):
        res_m = analyze_complete(d_adap_test["iq"][i, :256], fs_hz=8000.0, default_threshold=0.0)
        if res_m["classification"]["modulation"] != CLASSES[int(direct_preds[i])]:
            monitor_mismatches += 1

    return {
        "fase5_e2_test_reload": {
            "checked": fase5_checked,
            "mismatches": fase5_mismatches,
            "zero_mismatches": (fase5_mismatches[256] == 0 and fase5_mismatches[512] == 0),
        },
        "fase6_holdout_reload": {
            "checked": holdout_checked,
            "mismatches": holdout_mismatches,
            "zero_mismatches": holdout_mismatches == 0,
            "precision_note": "SmallCNN reproducido en FP32 estricto ('highest'); al activar TF32 ('high') se observaron 4/38400 (0.0104%) cambios de borde.",
        },
        "adaptive_paper_research_reload": paper_checks,
        "monitor_interface_vs_svm256_direct": {
            "checked": n_check,
            "mismatches": monitor_mismatches,
            "zero_mismatches": monitor_mismatches == 0,
        },
    }


def main(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    model_paths = {
        "svm256": PROJECT_ROOT / "results/fase5/conventional_256.joblib",
        "svm512": PROJECT_ROOT / "results/fase5/conventional_512.joblib",
        "cnn_fase6_original": PROJECT_ROOT / "results/fase6/best.pt",
        "cnn_fase6_phase_aug": PROJECT_ROOT / "results/fase6_phase_aug/best.pt",
        "multidomain_clean_a2": PROJECT_ROOT / "results/paper_research/multidomain_clean_a2_best.pt",
    }

    svms = {
        "svm256": load_model(model_paths["svm256"])["pipeline"],
        "svm512": load_model(model_paths["svm512"])["pipeline"],
    }
    cnns = {}
    for k in ("cnn_fase6_original", "cnn_fase6_phase_aug"):
        ckpt = load_checkpoint(model_paths[k])
        m = SmallCNN().to(device).eval()
        m.load_state_dict(ckpt["model"], strict=True)
        cnns[k] = m

    md_ckpt = load_checkpoint(model_paths["multidomain_clean_a2"])
    multidomain = MultiDomainAMCNet(
        num_classes=6, width_mult=md_ckpt.get("width_mult", MODEL_WIDTH_MULT), dropout=0.15
    ).to(device).eval()
    multidomain.load_state_dict(md_ckpt["model"], strict=True)

    with threadpool_limits(limits=1):
        adaptive_eval, adaptive_snr, adaptive_preds = _evaluate_all_models_on_dataset(
            "adaptive_dataset", PROJECT_ROOT / "data/adaptive_dataset", svms, cnns, multidomain, device
        )
        holdout_eval, holdout_snr, holdout_preds = _evaluate_all_models_on_dataset(
            "fase6_holdout_sellado", PROJECT_ROOT / "data/fase6_holdout_sellado", svms, cnns, multidomain, device
        )

        reg_report = _regression_checks(svms, cnns, multidomain, adaptive_eval, holdout_eval, device)

    # Limpiar arrays temporales internos antes de serializar JSON
    for eval_dict in (adaptive_eval, holdout_eval):
        for split_dict in eval_dict.values():
            for model_dict in split_dict.values():
                model_dict.pop("_logits_array", None)
                model_dict.pop("_preds_array", None)

    unified = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "note": "Benchmarks evaluados por separado sin mezclar datasets distintos.",
        "model_sha256": {k: _sha(p) for k, p in model_paths.items()},
        "benchmarks": {
            "adaptive_dataset": adaptive_eval,
            "fase6_holdout_sellado": holdout_eval,
        },
    }
    _json(output_dir / "unified_metrics.json", unified)
    _json(output_dir / "regression_reload_check.json", reg_report)
    _csv(output_dir / "by_class_and_snr.csv", adaptive_snr + holdout_snr)
    _csv(output_dir / "predictions_by_parent.csv", adaptive_preds + holdout_preds)

    # Figura comparativa por dataset y régimen
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    models_order = ["svm256", "svm512", "cnn_fase6_original", "cnn_fase6_phase_aug", "multidomain_clean_a2"]
    x_pos = np.arange(len(models_order))
    width = 0.35
    for ax, (bname, bdict) in zip(
        axes,
        [("adaptive_dataset (n=2400)", adaptive_eval), ("fase6_holdout_sellado (n=4800)", holdout_eval)],
    ):
        acc_id = [bdict["test"][m]["accuracy"] * 100.0 for m in models_order]
        acc_sh = [bdict["test_shift"][m]["accuracy"] * 100.0 for m in models_order]
        ax.bar(x_pos - width / 2, acc_id, width, label="test (ID)", color="royalblue")
        ax.bar(x_pos + width / 2, acc_sh, width, label="test_shift (OOD)", color="darkorange")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["SVM256", "SVM512", "CNN Orig", "CNN Phase", "MultiDomain"], rotation=20, ha="right")
        ax.set_ylim([0, 65])
        ax.set_ylabel("Exactitud (%)")
        ax.set_title(bname)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.legend()
    plt.suptitle("N2.3 Evaluación unificada por dataset (sin mezclar benchmarks)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "unified_benchmark_comparison.png", dpi=160)
    plt.close(fig)

    summary = {
        "adaptive_test_acc": {m: round(adaptive_eval["test"][m]["accuracy"] * 100, 4) for m in models_order},
        "adaptive_shift_acc": {m: round(adaptive_eval["test_shift"][m]["accuracy"] * 100, 4) for m in models_order},
        "holdout_test_acc": {m: round(holdout_eval["test"][m]["accuracy"] * 100, 4) for m in models_order},
        "holdout_shift_acc": {m: round(holdout_eval["test_shift"][m]["accuracy"] * 100, 4) for m in models_order},
        "regression_reload_check": reg_report,
    }
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
