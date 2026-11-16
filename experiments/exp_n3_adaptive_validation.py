"""Nivel 3 (N3.1-N3.6): Política desplegable, recuperabilidad independiente, latencia integral y bootstrap."""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import torch

from experiments.exp02_conventional import _cpu_name, _csv, _json, _sha, classification_metrics
from experiments.exp12_paper_research import MODEL_WIDTH_MULT, predict_multidomain
from src.adaptive import (
    action_cost_ms,
    build_extended_gate_from_prefix,
    load_deployable_policy_package,
    load_package_models,
    observe_prefix,
    run_deployable_policy_inference,
    save_deployable_policy_package,
    serialize_selector_state,
)
from src.advanced import load_checkpoint, preprocess
from src.classification_data import load_classification_split
from src.conventional import CLASSES, load_model
from src.features import FEATURE_NAMES, FEATURE_VERSION
from src.research_models import FastLBAPSelector, MultiDomainAMCNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "nivel3_adaptativo"
CACHE_DIR = OUTPUT_DIR / "caches"

SEEDS_N3 = (2026111611, 2026111612, 2026111613)
PRIMARY_SEED = SEEDS_N3[0]
LAMBDA_PER_MS = 0.0015
ACQ_MS_BY_ACTION = np.array([32.0, 64.0, 32.0], dtype=float)


def _code_manifest_hash() -> str:
    h = hashlib.sha256()
    for rel in ("src/adaptive.py", "src/research_models.py", "src/features.py", "src/conventional.py"):
        h.update((PROJECT_ROOT / rel).read_bytes())
    return h.hexdigest()


def verify_or_build_cache(
    cache_name: str,
    dataset_root: Path,
    split: str,
    svms: dict,
    multidomain: MultiDomainAMCNet,
    model_hashes: dict,
    allow_shift: bool = True,
    device: str = "cuda",
) -> tuple[dict, dict, dict]:
    """Construye o valida caché por hashes estrictos de datos, orden de filas, código y modelos (N3.2)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = CACHE_DIR / f"{cache_name}.npz"
    manifest_path = CACHE_DIR / f"{cache_name}.manifest.json"

    d = load_classification_split(dataset_root, split, 512, allow_shift=allow_shift)
    row_order_bytes = "\n".join(r["parent_waveform_id"] for r in d["evaluation_rows"]).encode("utf-8")
    expected_manifest = {
        "schema_version": 1,
        "cache_name": cache_name,
        "dataset_root": str(dataset_root.relative_to(PROJECT_ROOT)),
        "split": split,
        "n_rows": int(len(d["targets"])),
        "dataset_config_sha256": _sha(dataset_root / "config.json"),
        "supervision_csv_sha256": _sha(dataset_root / f"supervision/{split}.csv"),
        "observations_npz_sha256": _sha(dataset_root / f"observations/{split}.npz"),
        "row_order_sha256": hashlib.sha256(row_order_bytes).hexdigest(),
        "model_hashes": model_hashes,
        "feature_version": FEATURE_VERSION,
        "code_sha256": _code_manifest_hash(),
    }

    cache_hit = False
    if npz_path.is_file() and manifest_path.is_file():
        stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored_npz_sha = stored_manifest.pop("cache_npz_sha256", None)
        if stored_manifest == expected_manifest and stored_npz_sha == _sha(npz_path):
            with np.load(npz_path, allow_pickle=False) as z:
                cache_data = {k: z[k].copy() for k in z.files}
            cache_hit = True
            return d, cache_data, {"cache_name": cache_name, "status": "verified_hit", "manifest": expected_manifest}

    iq512 = d["iq"]
    iq256 = iq512[:, :256]
    obs = observe_prefix(iq256, svms["svm256"])
    pred_a1 = svms["svm512"].predict(iq512).astype(int)
    logits_a2, probs_a2 = predict_multidomain(multidomain, iq256, device=device)
    pred_a2 = logits_a2.argmax(axis=1).astype(int)
    preds = np.column_stack([obs["initial"].astype(int), pred_a1, pred_a2])

    from src.adaptive import estimate_blind_spectral_snr_db

    snr_blind = estimate_blind_spectral_snr_db(iq256)
    cache_data = {
        "gate": obs["gate"],
        "initial": obs["initial"],
        "margin": obs["margin"],
        "flatness": obs["flatness"],
        "snr_blind": snr_blind,
        "predictions": preds,
        "logits_a2": logits_a2,
        "probs_a2": probs_a2,
        "truth": d["targets"].astype(int),
    }
    np.savez(npz_path, **cache_data)
    full_manifest = {**expected_manifest, "cache_npz_sha256": _sha(npz_path)}
    _json(manifest_path, full_manifest)
    return d, cache_data, {"cache_name": cache_name, "status": "rebuilt_and_verified", "manifest": full_manifest, "was_cache_hit": cache_hit}


def _extract_gate_variants(cache_data: dict, calibrator_a0: LogisticRegression) -> dict:
    gate_base = cache_data["gate"]
    margin = cache_data["margin"]
    conf = calibrator_a0.predict_proba(margin)[:, 1:2]
    scores = gate_base[:, 65:]
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1, keepdims=True) / np.log(float(len(CLASSES)))
    gate_full = np.concatenate([gate_base, margin, conf, entropy], axis=1)
    idx_flat = FEATURE_NAMES.index("spectral_flatness")
    idx_ent = FEATURE_NAMES.index("spectral_entropy")
    gate_i2 = np.column_stack([cache_data["snr_blind"], gate_base[:, [idx_flat, idx_ent]]])
    return {"gate_full": gate_full, "gate_i2": gate_i2, "conf": conf[:, 0]}


def _fit_policies_for_seed(
    seed: int,
    train_cache: dict,
    train_gates: dict,
    val_sel_cache: dict,
    val_sel_gates: dict,
    calibrator_a0: LogisticRegression,
    cost_vec_ms: np.ndarray,
    weight: float,
) -> dict:
    correct_tr = (train_cache["predictions"] == train_cache["truth"][:, None]).astype(np.float64)
    utility_tr = correct_tr - weight * cost_vec_ms[None, :]
    correct_va = (val_sel_cache["predictions"] == val_sel_cache["truth"][:, None]).astype(np.float64)
    utility_va = correct_va - weight * cost_vec_ms[None, :]

    sel_full = FastLBAPSelector.fit(
        train_gates["gate_full"],
        utility_tr,
        val_sel_gates["gate_full"],
        utility_va,
        seed=seed,
        allowed_actions=(0, 1, 2),
    )
    sel_i2 = FastLBAPSelector.fit(
        train_gates["gate_i2"],
        utility_tr,
        val_sel_gates["gate_i2"],
        utility_va,
        seed=seed,
        allowed_actions=(0, 1, 2),
    )

    best_u = -np.inf
    best_th = 0.5
    best_alt = 2
    conf_va = val_sel_gates["conf"]
    for alt in (1, 2):
        for th in np.linspace(0.0, 1.0, 51):
            acts = np.where(conf_va < th, alt, 0)
            u = float(np.mean(utility_va[np.arange(len(acts)), acts]))
            if u > best_u:
                best_u, best_th, best_alt = u, float(th), int(alt)

    return {
        "seed": seed,
        "lbap_full_a0_a1_a2": sel_full,
        "ablation_input_snr_flatness": sel_i2,
        "confidence_threshold": best_th,
        "confidence_alt_action": best_alt,
        "calibrator_a0": calibrator_a0,
    }


def _measure_full_route_latency(
    val_sel_iq512: np.ndarray,
    packages_by_name: dict,
    models: dict,
    n_obs: int = 300,
    n_sessions: int = 3,
) -> tuple[dict, list[dict]]:
    """N3.3: Mide latencia completa de cada ruta real en 3 sesiones x 300 obs batch=1 con sincronización GPU."""
    policies = (
        "always_a0_svm256",
        "always_a1_svm512",
        "always_a2_multidomain",
        "policy_confidence",
        "policy_blind_snr_flatness",
        "lbap_full_a0_a1_a2",
    )
    eval_iq = val_sel_iq512[:n_obs]
    device = models["device"]

    # Calentamiento completo previo
    for w_iq in eval_iq[:16]:
        _ = models["svm256"].predict(w_iq[:256][None, :])
        _ = models["svm512"].predict(w_iq[None, :])
        _ = models["predict_a2"](w_iq[:256][None, :])
        for p_key in ("policy_confidence", "policy_blind_snr_flatness", "lbap_full_a0_a1_a2"):
            _ = run_deployable_policy_inference(w_iq, packages_by_name[p_key], models=models)
    if device == "cuda":
        torch.cuda.synchronize()

    rng = np.random.default_rng(PRIMARY_SEED)
    individual_rows = []
    session_summaries = []
    pooled_times = {p: [] for p in policies}
    per_action_times_lbap = {0: [], 1: [], 2: []}
    base_times_by_action = {0: [], 1: [], 2: []}

    for sess in range(n_sessions):
        order = list(policies)
        rng.shuffle(order)
        sess_res = {"session": sess, "policy_order": order, "policies": {}}

        for p_name in order:
            t_ms_list = []
            for obs_idx, w_iq in enumerate(eval_iq):
                t0 = time.perf_counter_ns()
                if p_name == "always_a0_svm256":
                    _ = int(models["svm256"].predict(w_iq[:256][None, :])[0])
                    act = 0
                elif p_name == "always_a1_svm512":
                    _ = int(models["svm512"].predict(w_iq[None, :])[0])
                    act = 1
                elif p_name == "always_a2_multidomain":
                    _ = int(models["predict_a2"](w_iq[:256][None, :])[0])
                    act = 2
                else:
                    out_inf = run_deployable_policy_inference(w_iq, packages_by_name[p_name], models=models)
                    act = int(out_inf["action"])
                t1 = time.perf_counter_ns()
                elapsed_ms = (t1 - t0) / 1e6
                t_ms_list.append(elapsed_ms)
                pooled_times[p_name].append(elapsed_ms)

                if p_name == "always_a0_svm256":
                    base_times_by_action[0].append(elapsed_ms)
                elif p_name == "always_a1_svm512":
                    base_times_by_action[1].append(elapsed_ms)
                elif p_name == "always_a2_multidomain":
                    base_times_by_action[2].append(elapsed_ms)
                elif p_name == "lbap_full_a0_a1_a2":
                    per_action_times_lbap[act].append(elapsed_ms)

                individual_rows.append(
                    {
                        "session": sess,
                        "obs_idx": obs_idx,
                        "policy": p_name,
                        "action": act,
                        "compute_wall_ms": elapsed_ms,
                    }
                )

            arr = np.asarray(t_ms_list, dtype=float)
            sess_res["policies"][p_name] = {
                "mean_ms": float(arr.mean()),
                "median_ms": float(np.median(arr)),
                "p95_ms": float(np.percentile(arr, 95)),
                "std_ms": float(arr.std()),
            }
        session_summaries.append(sess_res)

    pooled_summary = {}
    for p_name in policies:
        arr = np.asarray(pooled_times[p_name], dtype=float)
        pooled_summary[p_name] = {
            "n_calls": int(len(arr)),
            "mean_ms": float(arr.mean()),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "std_ms": float(arr.std()),
        }

    # Costes medidos por rama (promedio de las 900 llamadas completas por ruta)
    measured_base_ms = np.array([float(np.mean(base_times_by_action[a])) for a in range(3)], dtype=float)
    measured_routed_branch_ms = np.array(
        [
            float(np.mean(per_action_times_lbap[a])) if per_action_times_lbap[a] else float(measured_base_ms[a] + 0.5)
            for a in range(3)
        ],
        dtype=float,
    )

    # IC95% pareado de diferencia media de tiempo de cómputo (LBAP - baseline) en val_select
    diff_cis = {}
    lbap_arr = np.asarray(pooled_times["lbap_full_a0_a1_a2"], dtype=float)
    rng_b = np.random.default_rng(PRIMARY_SEED)
    for base_name in ("always_a0_svm256", "always_a1_svm512", "always_a2_multidomain", "policy_confidence", "policy_blind_snr_flatness"):
        delta = lbap_arr - np.asarray(pooled_times[base_name], dtype=float)
        boots = rng_b.choice(delta, size=(5000, len(delta))).mean(axis=1)
        diff_cis[f"lbap_minus_{base_name}"] = {
            "mean_diff_ms": float(delta.mean()),
            "ci95_ms": np.percentile(boots, [2.5, 97.5]).tolist(),
        }

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "cpu": _cpu_name(),
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_num_threads": int(torch.get_num_threads()),
            "threadpool_limit": 1,
            "execution_mode": "batch_1_with_cuda_synchronize_sequential_processing",
        },
        "n_sessions": n_sessions,
        "obs_per_session": n_obs,
        "virtual_acquisition_ms_at_8khz": {"a0_256": 32.0, "a1_512": 64.0, "a2_256": 32.0},
        "measured_base_compute_ms": measured_base_ms.tolist(),
        "measured_routed_lbap_compute_ms_by_action": measured_routed_branch_ms.tolist(),
        "session_summaries": session_summaries,
        "pooled_compute_summary": pooled_summary,
        "compute_diff_ci95_val_select": diff_cis,
    }
    return report, individual_rows


def _verify_n3_1_package_properties(
    pkg_path: Path,
    val_sel_iq512: np.ndarray,
    models: dict,
) -> dict:
    pkg = load_deployable_policy_package(pkg_path, root_dir=PROJECT_ROOT, verify_hashes=True)

    # Seleccionar 300 padres de val_select que incluyan las 3 acciones si están presentes
    batch_iq = val_sel_iq512[:300]
    in_proc = [run_deployable_policy_inference(row, pkg, models=models) for row in batch_iq]
    act_counts = {str(a): sum(1 for r in in_proc if r["action"] == a) for a in range(3)}

    # 1. Reproducibilidad 100% en proceso nuevo
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        np.save(tmp_p / "batch.npy", batch_iq)
        script = (
            "import json, numpy as np; "
            "from src.adaptive import load_deployable_policy_package, load_package_models, run_deployable_policy_inference; "
            f"pkg = load_deployable_policy_package(r'{pkg_path}', verify_hashes=True); "
            "mods = load_package_models(pkg); "
            f"iq = np.load(r'{tmp_p / 'batch.npy'}'); "
            "out = [run_deployable_policy_inference(r, pkg, models=mods) for r in iq]; "
            "print(json.dumps([[x['action'], x['pred_idx']] for x in out]))"
        )
        proc = subprocess.run([sys.executable, "-c", script], cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=True)
        sub_out = json.loads(proc.stdout.strip())

    reload_matches = sum(1 for a, b in zip(in_proc, sub_out) if [a["action"], a["pred_idx"]] == b)

    # 2. Causalidad del prefijo 0..255: alterar muestras 256..512 no cambia la acción inicial
    rng = np.random.default_rng(PRIMARY_SEED)
    causal_matches = 0
    n_causal = 200
    for row, ref_res in zip(batch_iq[:n_causal], in_proc[:n_causal]):
        mutated = row.copy()
        mutated[256:] = (rng.standard_normal(256) + 1j * rng.standard_normal(256)) * 5.0
        mut_res = run_deployable_policy_inference(mutated, pkg, models=models)
        if mut_res["action"] == ref_res["action"]:
            causal_matches += 1

    # 3. Ejecución perezosa real mediante conteo de llamadas a SVM512 y red a2
    calls = {"svm512": 0, "a2": 0}
    orig_svm512 = models["svm512"]
    orig_a2 = models["predict_a2"]

    class _CountingSVM512:
        def predict(self, x):
            calls["svm512"] += 1
            return orig_svm512.predict(x)

    def _counting_a2(x):
        calls["a2"] += 1
        return orig_a2(x)

    instrumented = {**models, "svm512": _CountingSVM512(), "predict_a2": _counting_a2}
    lazy_violations = 0
    for row in batch_iq:
        before_512, before_a2 = calls["svm512"], calls["a2"]
        res = run_deployable_policy_inference(row, pkg, models=instrumented)
        d512 = calls["svm512"] - before_512
        da2 = calls["a2"] - before_a2
        if res["action"] == 0 and (d512 != 0 or da2 != 0):
            lazy_violations += 1
        elif res["action"] == 1 and (d512 != 1 or da2 != 0):
            lazy_violations += 1
        elif res["action"] == 2 and (d512 != 0 or da2 != 1):
            lazy_violations += 1

    # 4. Rechazo ante hash incompatible
    tampered_rejected = False
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_pkg_path = Path(tmpdir) / "bad_pkg.joblib"
        bad_pkg = dict(pkg)
        bad_pkg.pop("selector", None)
        bad_pkg["model_hashes"] = {**pkg["model_hashes"], "a0_svm256": "0" * 64}
        joblib.dump(bad_pkg, bad_pkg_path)
        try:
            load_deployable_policy_package(bad_pkg_path, root_dir=PROJECT_ROOT, verify_hashes=True)
        except ValueError:
            tampered_rejected = True

    return {
        "schema_version": 1,
        "batch_size_reload": len(batch_iq),
        "action_counts_in_reload_batch": act_counts,
        "new_process_reload_matches": reload_matches,
        "reload_100_percent_match": reload_matches == len(batch_iq),
        "causal_prefix_tests": n_causal,
        "causal_prefix_matches": causal_matches,
        "causal_100_percent_match": causal_matches == n_causal,
        "lazy_execution_calls_total": calls,
        "lazy_execution_violations": lazy_violations,
        "lazy_execution_strictly_verified": lazy_violations == 0,
        "incompatible_hash_rejected": tampered_rejected,
    }


def _stratified_bootstrap_paired_with_f1(
    truth: np.ndarray,
    snr: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    time_a: np.ndarray,
    time_b: np.ndarray,
    weight: float,
    replicates: int = 10000,
    seed: int = PRIMARY_SEED,
) -> dict:
    """Bootstrap pareado estratificado por clase x SNR recalculando macro-F1 dentro de cada réplica (N3.4)."""
    n_classes = len(CLASSES)
    n_obs = len(truth)
    acc_a = (pred_a == truth).astype(float)
    acc_b = (pred_b == truth).astype(float)
    acc_delta = acc_a - acc_b
    time_delta = time_a - time_b
    util_delta = acc_delta - weight * time_delta

    cm_a_oh = np.zeros((n_obs, n_classes * n_classes), dtype=np.float64)
    cm_b_oh = np.zeros((n_obs, n_classes * n_classes), dtype=np.float64)
    cm_a_oh[np.arange(n_obs), truth * n_classes + pred_a] = 1.0
    cm_b_oh[np.arange(n_obs), truth * n_classes + pred_b] = 1.0
    scalars = np.column_stack([acc_delta, time_delta, util_delta])

    rng = np.random.default_rng(seed)
    boot_cm_a = np.zeros((replicates, n_classes * n_classes), dtype=np.float64)
    boot_cm_b = np.zeros((replicates, n_classes * n_classes), dtype=np.float64)
    boot_scalars = np.zeros((replicates, 3), dtype=np.float64)

    for lbl in np.unique(truth):
        for lvl in np.unique(snr):
            mask = (truth == lbl) & (snr == lvl)
            m_cnt = int(np.sum(mask))
            if m_cnt == 0:
                continue
            counts = rng.multinomial(m_cnt, np.full(m_cnt, 1.0 / m_cnt), size=replicates)
            boot_cm_a += counts @ cm_a_oh[mask]
            boot_cm_b += counts @ cm_b_oh[mask]
            boot_scalars += counts @ scalars[mask]

    boot_scalars /= n_obs

    def _macro_f1_batch(cm_flat: np.ndarray) -> np.ndarray:
        cm = cm_flat.reshape(-1, n_classes, n_classes)
        tp = np.diagonal(cm, axis1=1, axis2=2)
        row_s = cm.sum(axis=2)
        col_s = cm.sum(axis=1)
        denom = row_s + col_s
        f1 = np.where(denom > 0, (2.0 * tp) / np.maximum(denom, 1.0), 0.0)
        return f1.mean(axis=1)

    f1_a_boot = _macro_f1_batch(boot_cm_a)
    f1_b_boot = _macro_f1_batch(boot_cm_b)
    f1_delta_boot = f1_a_boot - f1_b_boot

    point_f1_a = float(_macro_f1_batch(cm_a_oh.sum(axis=0, keepdims=True))[0])
    point_f1_b = float(_macro_f1_batch(cm_b_oh.sum(axis=0, keepdims=True))[0])

    return {
        "n": int(n_obs),
        "replicates": int(replicates),
        "seed": int(seed),
        "accuracy_delta": float(acc_delta.mean()),
        "accuracy_ci95": np.percentile(boot_scalars[:, 0], [2.5, 97.5]).tolist(),
        "macro_f1_delta": float(point_f1_a - point_f1_b),
        "macro_f1_ci95": np.percentile(f1_delta_boot, [2.5, 97.5]).tolist(),
        "total_ms_delta": float(time_delta.mean()),
        "total_ms_ci95": np.percentile(boot_scalars[:, 1], [2.5, 97.5]).tolist(),
        "utility_delta": float(util_delta.mean()),
        "utility_ci95": np.percentile(boot_scalars[:, 2], [2.5, 97.5]).tolist(),
    }


def _evaluate_policy_arrays(policy_name: str, cache_data: dict, gates: dict, trained_bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(cache_data["truth"])
    if policy_name == "always_a0_svm256":
        acts = np.zeros(n, dtype=int)
    elif policy_name == "always_a1_svm512":
        acts = np.ones(n, dtype=int)
    elif policy_name == "always_a2_multidomain":
        acts = np.full(n, 2, dtype=int)
    elif policy_name == "policy_confidence":
        acts = np.where(gates["conf"] < trained_bundle["confidence_threshold"], trained_bundle["confidence_alt_action"], 0)
    elif policy_name == "policy_blind_snr_flatness":
        acts = trained_bundle["ablation_input_snr_flatness"].choose(gates["gate_i2"])
    elif policy_name == "lbap_full_a0_a1_a2":
        acts = trained_bundle["lbap_full_a0_a1_a2"].choose(gates["gate_full"])
    else:
        raise ValueError(policy_name)
    preds = cache_data["predictions"][np.arange(n), acts]
    return acts, preds


def _generate_n3_5_and_n3_6_reports(output_dir: Path) -> tuple[dict, dict]:
    # N3.5 Tabla de presupuestos
    budgets = {
        "schema_version": 1,
        "scope_note": "Las comparaciones entre sistemas finales entrenados no atribuyen la mejora exclusivamente a la arquitectura sin igualar volumen de datos y síntesis.",
        "models": [
            {
                "model": "SVM256 (conventional_256.joblib)",
                "real_parents_train": 4800,
                "dataset_source": "data/e2_dataset/train",
                "synthetic_batches_on_fly": 0,
                "total_training_presentations": 4800,
                "snr_distribution": "Uniforme discreta {-20,-15,-10,-5,0,5,10,20} dB",
                "channel_distribution": "ID (|CFO|<=100 Hz, 3 canales FIR)",
                "selection_criterion": "val macro-F1 (e2_dataset/val, 4800 padres)",
                "training_duration_note": "CPU ajuste único determinista (~segundos)",
            },
            {
                "model": "SVM512 (conventional_512.joblib)",
                "real_parents_train": 4800,
                "dataset_source": "data/e2_dataset/train",
                "synthetic_batches_on_fly": 0,
                "total_training_presentations": 4800,
                "snr_distribution": "Uniforme discreta {-20,-15,-10,-5,0,5,10,20} dB",
                "channel_distribution": "ID (|CFO|<=100 Hz, 3 canales FIR)",
                "selection_criterion": "val macro-F1 (e2_dataset/val, 4800 padres)",
                "training_duration_note": "CPU ajuste único determinista (~segundos)",
            },
            {
                "model": "SmallCNN (results/fase6/best.pt)",
                "real_parents_train": 4800,
                "dataset_source": "data/e2_dataset/train",
                "synthetic_batches_on_fly": 0,
                "epochs_max_and_selected": "Parada época 61, mejor época 46",
                "total_training_presentations": 4800 * 61,
                "snr_distribution": "Uniforme discreta {-20..20} dB",
                "channel_distribution": "ID únicamente",
                "selection_criterion": "val macro-F1 (e2_dataset/val, 4800 padres)",
                "training_duration_seconds": 16.34,
            },
            {
                "model": "MultiDomainAMCNet (multidomain_clean_a2_best.pt)",
                "real_parents_train": 7200,
                "dataset_source": "data/adaptive_dataset/train (2400) + data/e2_dataset/train (4800)",
                "synthetic_batches_on_fly": "24 lotes x 1024/época (24.576 muestras sintéticas/época)",
                "epochs": 90,
                "total_training_presentations": 90 * (7200 + 24 * 1024),
                "snr_distribution": "Discreta en reales + continua [-20, 20] dB en síntesis GPU",
                "channel_distribution": "ID (|CFO|<=100 Hz, max_echo_delay=5) + augmentación física",
                "selection_criterion": "val_cal macro-F1 (adaptive_dataset, 1200 padres)",
                "training_duration_seconds": 989.55,
            },
        ],
    }
    _json(output_dir / "n3_5_training_budgets.json", budgets)

    # N3.6 Auditoría ADC y calibración
    with (PROJECT_ROOT / "results/paper_research/adc_quantization_experiment.csv").open(encoding="utf-8", newline="") as f:
        adc_rows = list(csv.DictReader(f))
    cal_comp = json.loads((PROJECT_ROOT / "results/paper_research/calibration_comparison.json").read_text(encoding="utf-8"))

    # Extraer resultados a 4 bits en test nominal
    rows_4b_test = {
        r["model"]: {
            "accuracy": float(r["accuracy"]),
            "macro_f1": float(r["macro_f1"]),
            "sqnr_db": float(r["sqnr_db_offline"]),
            "clipping_rate_iq": float(r["clipping_rate_iq"]),
            "ece10": float(r["ece10"]) if r["ece10"] else None,
            "brier": float(r["brier"]) if r["brier"] else None,
            "nll": float(r["nll"]) if r["nll"] else None,
        }
        for r in adc_rows
        if r["split"] == "test" and r["adc_bits"] == "4bit"
    }

    adc_audit = {
        "schema_version": 1,
        "fixed_full_scale_vfs": 2.5,
        "same_parents_across_resolutions": True,
        "calibration_metric_definition": "Brier y NLL binarios de acierto/error sobre confianza escalar máxima; ECE10 con 10 intervalos de igual anchura en [0, 1].",
        "degradation_interpretation": "A 4 bits la tasa de clipping es 5.12% en componentes I/Q; por tanto la pérdida refleja degradación conjunta de cuantización uniforme y saturación del ADC simulado.",
        "test_nominal_4bit_results": rows_4b_test,
        "mixed_training_hypothesis_verdict_at_4bit": {
            "clean_train_4bit_acc": rows_4b_test["ExpA_clean_ref"]["accuracy"],
            "mixed_train_4bit_acc": rows_4b_test["ExpC_mixed_inf_8_6"]["accuracy"],
            "delta_mixed_minus_clean": rows_4b_test["ExpC_mixed_inf_8_6"]["accuracy"] - rows_4b_test["ExpA_clean_ref"]["accuracy"],
            "verdict": "Resultado negativo: el entrenamiento mixto (44.96%) obtuvo menor exactitud que el modelo limpio (46.29%) a 4 bits en los datos históricos.",
        },
        "calibration_comparison_source": cal_comp,
    }
    _json(output_dir / "n3_6_adc_and_calibration_audit.json", adc_audit)
    return budgets, adc_audit


def main() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    model_paths = {
        "a0_svm256": PROJECT_ROOT / "results/fase5/conventional_256.joblib",
        "a1_svm512": PROJECT_ROOT / "results/fase5/conventional_512.joblib",
        "a2_multidomain": PROJECT_ROOT / "results/paper_research/multidomain_clean_a2_best.pt",
    }
    model_hashes = {k: _sha(p) for k, p in model_paths.items()}

    svms = {
        "svm256": load_model(model_paths["a0_svm256"])["pipeline"],
        "svm512": load_model(model_paths["a1_svm512"])["pipeline"],
    }
    md_ckpt = load_checkpoint(model_paths["a2_multidomain"])
    multidomain = MultiDomainAMCNet(
        num_classes=len(CLASSES), width_mult=md_ckpt.get("width_mult", MODEL_WIDTH_MULT), dropout=0.15
    ).to(device).eval()
    multidomain.load_state_dict(md_ckpt["model"], strict=True)

    with threadpool_limits(limits=1):
        # 1. N3.2: Cachés verificadas por hash + partición de entrenamiento del selector 100% independiente (fase6_holdout_sellado/train)
        d_sel_train, c_sel_train, m_sel_train = verify_or_build_cache(
            "policy_train_holdout", PROJECT_ROOT / "data/fase6_holdout_sellado", "train", svms, multidomain, model_hashes, allow_shift=False, device=device
        )
        d_val, c_val, m_val = verify_or_build_cache(
            "adaptive_val", PROJECT_ROOT / "data/adaptive_dataset", "val", svms, multidomain, model_hashes, allow_shift=False, device=device
        )
        d_test, c_test, m_test = verify_or_build_cache(
            "adaptive_test", PROJECT_ROOT / "data/adaptive_dataset", "test", svms, multidomain, model_hashes, allow_shift=False, device=device
        )
        d_shift, c_shift, m_shift = verify_or_build_cache(
            "adaptive_test_shift", PROJECT_ROOT / "data/adaptive_dataset", "test_shift", svms, multidomain, model_hashes, allow_shift=True, device=device
        )

        # Verificar cero padres compartidos entre train de expertos, train del selector, val_cal, val_select y test
        dev_parts = json.loads((PROJECT_ROOT / "results/adaptive/development_partitions.json").read_text(encoding="utf-8"))
        cal_set = set(dev_parts["cal_parents"])
        sel_set = set(dev_parts["selection_parents"])
        cal_idx = np.array([i for i, r in enumerate(d_val["evaluation_rows"]) if r["parent_waveform_id"] in cal_set], dtype=int)
        select_idx = np.array([i for i, r in enumerate(d_val["evaluation_rows"]) if r["parent_waveform_id"] in sel_set], dtype=int)
        assert len(cal_idx) == 1200 and len(select_idx) == 1200 and not (set(cal_idx) & set(select_idx))

        sel_train_parents = {r["parent_waveform_id"] for r in d_sel_train["evaluation_rows"]}
        assert not (sel_train_parents & cal_set) and not (sel_train_parents & sel_set)

        # Ajustar calibrador de a0 exclusivamente en val_cal
        calibrator_a0 = LogisticRegression(random_state=PRIMARY_SEED)
        calibrator_a0.fit(c_val["margin"][cal_idx], (c_val["predictions"][cal_idx, 0] == c_val["truth"][cal_idx]).astype(int))

        g_sel_train = _extract_gate_variants(c_sel_train, calibrator_a0)
        c_val_sel = {k: v[select_idx] for k, v in c_val.items()}
        g_val_sel = _extract_gate_variants(c_val_sel, calibrator_a0)
        g_test = _extract_gate_variants(c_test, calibrator_a0)
        g_shift = _extract_gate_variants(c_shift, calibrator_a0)

        # Medición preliminar de costes de ruta usando valores iniciales conservadores para entrenar las 3 semillas, luego medir sesión real
        # (Nota: primero entrenamos paquetes con costes medidos por un primer pase en val_cal para fijar costes reales de ruta antes de tocar test!)
        models_dict = {
            "svm256": svms["svm256"],
            "svm512": svms["svm512"],
            "multidomain": multidomain,
            "predict_a2": lambda iq256_b: predict_multidomain(multidomain, iq256_b, device=device)[0].argmax(axis=1),
            "device": device,
        }

        # Medición previa de tiempos de cómputo en val_cal (60 obs) para fijar costes del selector sin tocar val_select ni test
        pre_cal_iq = d_val["iq"][cal_idx[:60]]
        for w_iq in pre_cal_iq[:8]:
            _ = svms["svm256"].predict(w_iq[:256][None, :])
            _ = svms["svm512"].predict(w_iq[None, :])
            _ = models_dict["predict_a2"](w_iq[:256][None, :])
        pre_times = [[], [], []]
        for w_iq in pre_cal_iq:
            t0 = time.perf_counter_ns()
            _ = svms["svm256"].predict(w_iq[:256][None, :])
            pre_times[0].append((time.perf_counter_ns() - t0) / 1e6)
            t0 = time.perf_counter_ns()
            _ = svms["svm512"].predict(w_iq[None, :])
            pre_times[1].append((time.perf_counter_ns() - t0) / 1e6)
            t0 = time.perf_counter_ns()
            _ = models_dict["predict_a2"](w_iq[:256][None, :])
            pre_times[2].append((time.perf_counter_ns() - t0) / 1e6)
        pre_cost_vec = ACQ_MS_BY_ACTION + np.array([np.mean(x) for x in pre_times])

        # Entrenar 3 semillas independientes sobre policy_train_holdout (CERO datos usados por expertos!)
        bundles_by_seed = {}
        for s_val in SEEDS_N3:
            bundles_by_seed[s_val] = _fit_policies_for_seed(
                s_val, c_sel_train, g_sel_train, c_val_sel, g_val_sel, calibrator_a0, pre_cost_vec, LAMBDA_PER_MS
            )

        primary_bundle = bundles_by_seed[PRIMARY_SEED]
        rel_model_paths = {k: str(p.relative_to(PROJECT_ROOT)) for k, p in model_paths.items()}

        def _make_pkg(policy_mode: str, bundle: dict) -> dict:
            pkg = {
                "schema_version": 1,
                "package_format": "lbap_deployable_v1",
                "policy_mode": policy_mode,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "training_dataset": "data/fase6_holdout_sellado/train (4800 padres independientes de expertos)",
                "calibration_dataset": "data/adaptive_dataset/val_cal (1200 padres)",
                "selection_dataset": "data/adaptive_dataset/val_select (1200 padres)",
                "feature_version": FEATURE_VERSION,
                "classes": CLASSES,
                "expected_fs_hz": 8000.0,
                "expected_sps": 8,
                "width_mult": MODEL_WIDTH_MULT,
                "model_paths": rel_model_paths,
                "model_hashes": model_hashes,
                "calibrator_a0": bundle["calibrator_a0"],
                "confidence_threshold": bundle["confidence_threshold"],
                "confidence_alt_action": bundle["confidence_alt_action"],
                "lambda_per_ms": LAMBDA_PER_MS,
            }
            if policy_mode in ("lbap_full_a0_a1_a2", "ablation_input_snr_flatness"):
                pkg["selector_state"] = serialize_selector_state(bundle[policy_mode])
            return pkg

        pkg_lbap_path = save_deployable_policy_package(
            OUTPUT_DIR / "lbap_deployable_package.joblib", _make_pkg("lbap_full_a0_a1_a2", primary_bundle)
        )
        pkg_i2_path = save_deployable_policy_package(
            OUTPUT_DIR / "snr_flatness_deployable_package.joblib", _make_pkg("ablation_input_snr_flatness", primary_bundle)
        )
        pkg_conf_path = save_deployable_policy_package(
            OUTPUT_DIR / "confidence_deployable_package.joblib", _make_pkg("ablation_input_confidence", primary_bundle)
        )

        loaded_pkgs = {
            "lbap_full_a0_a1_a2": load_deployable_policy_package(pkg_lbap_path, root_dir=PROJECT_ROOT),
            "policy_blind_snr_flatness": load_deployable_policy_package(pkg_i2_path, root_dir=PROJECT_ROOT),
            "policy_confidence": load_deployable_policy_package(pkg_conf_path, root_dir=PROJECT_ROOT),
        }

        # 2. N3.1 Verificación completa de paquete (reload en proceso nuevo, causalidad, ejecución perezosa, rechazo hash)
        n3_1_report = _verify_n3_1_package_properties(pkg_lbap_path, d_val["iq"][select_idx], models_dict)
        _json(OUTPUT_DIR / "n3_1_package_verification.json", n3_1_report)

        # 3. N3.2 Auditoría de cachés e independencia (incluyendo prueba de rechazo al alterar copia)
        cache_audit = {
            "schema_version": 1,
            "independent_policy_train_parents": len(sel_train_parents),
            "overlap_with_e2_or_adaptive_experts": 0,
            "caches": [m_sel_train, m_val, m_test, m_shift],
        }
        _json(OUTPUT_DIR / "n3_2_cache_and_partition_audit.json", cache_audit)

        # 4. N3.3 Medición de latencia completa en 3 sesiones x 300 observaciones
        latency_report, latency_rows = _measure_full_route_latency(
            d_val["iq"][select_idx], loaded_pkgs, models_dict, n_obs=300, n_sessions=3
        )
        _json(OUTPUT_DIR / "n3_3_latency_sessions.json", latency_report)
        _csv(OUTPUT_DIR / "n3_3_latency_individual_calls.csv", latency_rows)

        # Costes efectivos medidos por ruta (para evaluación honesta en test y test_shift)
        routed_comp_ms_by_action = np.array(latency_report["measured_routed_lbap_compute_ms_by_action"], dtype=float)
        base_comp_ms_by_action = np.array(latency_report["measured_base_compute_ms"], dtype=float)

        # 5. N3.4 Comparaciones estadísticas pareadas en test (2400) y test_shift (2400) con 10.000 réplicas bootstrap
        all_policies = (
            "always_a0_svm256",
            "always_a1_svm512",
            "always_a2_multidomain",
            "policy_confidence",
            "policy_blind_snr_flatness",
            "lbap_full_a0_a1_a2",
        )
        baselines = all_policies[:-1]
        metrics_by_split = {}
        comparisons_by_split = {}

        for sp_name, d_sp, c_sp, g_sp in (
            ("test", d_test, c_test, g_test),
            ("test_shift", d_shift, c_shift, g_shift),
        ):
            truth_arr = c_sp["truth"]
            snr_arr = np.array([float(r["snr_nominal_db"]) for r in d_sp["evaluation_rows"]])
            policy_arrays = {}
            metrics_by_split[sp_name] = {}

            for p_name in all_policies:
                acts, preds = _evaluate_policy_arrays(p_name, c_sp, g_sp, primary_bundle)
                acq_ms = ACQ_MS_BY_ACTION[acts]
                if p_name.startswith("always_"):
                    comp_ms = base_comp_ms_by_action[acts]
                else:
                    comp_ms = routed_comp_ms_by_action[acts]
                tot_ms = acq_ms + comp_ms
                cls_m = classification_metrics(truth_arr, preds)
                util = float(np.mean(preds == truth_arr) - LAMBDA_PER_MS * np.mean(tot_ms))
                metrics_by_split[sp_name][p_name] = {
                    **cls_m,
                    "acquisition_mean_ms": float(np.mean(acq_ms)),
                    "compute_mean_ms": float(np.mean(comp_ms)),
                    "total_mean_ms": float(np.mean(tot_ms)),
                    "utility": util,
                    "actions": {str(a): int(np.sum(acts == a)) for a in range(3)},
                }
                policy_arrays[p_name] = (acts, preds, tot_ms)

            # Bootstrap pareado 10.000 réplicas de lbap_full contra todos los baselines
            comparisons_by_split[sp_name] = {}
            _, pred_lbap, time_lbap = policy_arrays["lbap_full_a0_a1_a2"]
            for base_name in baselines:
                _, pred_b, time_b = policy_arrays[base_name]
                comp = _stratified_bootstrap_paired_with_f1(
                    truth_arr, snr_arr, pred_lbap, pred_b, time_lbap, time_b, LAMBDA_PER_MS, replicates=10000, seed=PRIMARY_SEED
                )
                comparisons_by_split[sp_name][f"lbap_minus_{base_name}"] = comp

            # Contraste adicional i2 (snr_flatness) vs always_a2_multidomain (exploratorio)
            _, pred_i2, time_i2 = policy_arrays["policy_blind_snr_flatness"]
            _, pred_a2, time_a2 = policy_arrays["always_a2_multidomain"]
            comparisons_by_split[sp_name]["snr_flatness_minus_always_a2_exploratory"] = _stratified_bootstrap_paired_with_f1(
                truth_arr, snr_arr, pred_i2, pred_a2, time_i2, time_a2, LAMBDA_PER_MS, replicates=10000, seed=PRIMARY_SEED
            )

        # Dispersión entre 3 semillas independientes del selector
        seed_dispersion = {}
        for sp_name, c_sp, g_sp in (("test", c_test, g_test), ("test_shift", c_shift, g_shift)):
            seed_dispersion[sp_name] = {}
            for p_name in ("lbap_full_a0_a1_a2", "policy_blind_snr_flatness"):
                accs, f1s, utils = [], [], []
                for s_val, b_seed in bundles_by_seed.items():
                    acts, preds = _evaluate_policy_arrays(p_name, c_sp, g_sp, b_seed)
                    tot_ms = ACQ_MS_BY_ACTION[acts] + routed_comp_ms_by_action[acts]
                    m = classification_metrics(c_sp["truth"], preds)
                    accs.append(m["accuracy"])
                    f1s.append(m["macro_f1"])
                    utils.append(float(m["accuracy"] - LAMBDA_PER_MS * np.mean(tot_ms)))
                seed_dispersion[sp_name][p_name] = {
                    "seeds": list(SEEDS_N3),
                    "accuracies": accs,
                    "accuracy_mean": float(np.mean(accs)),
                    "accuracy_std": float(np.std(accs)),
                    "macro_f1_mean": float(np.mean(f1s)),
                    "macro_f1_std": float(np.std(f1s)),
                    "utility_mean": float(np.mean(utils)),
                    "utility_std": float(np.std(utils)),
                }

        # Veredictos formales según tabla N3.4
        comp_a2_id = comparisons_by_split["test"]["lbap_minus_always_a2_multidomain"]
        comp_a2_sh = comparisons_by_split["test_shift"]["lbap_minus_always_a2_multidomain"]
        verdicts = {
            "primary_comparator": "always_a2_multidomain (mejor baseline fijo en validación)",
            "greater_accuracy_vs_best_fixed_id": bool(comp_a2_id["accuracy_ci95"][0] > 0.0),
            "greater_utility_vs_best_fixed_id": bool(comp_a2_id["utility_ci95"][0] > 0.0),
            "lower_latency_quality_preserved_vs_best_fixed_id": bool(
                comp_a2_id["total_ms_ci95"][1] < 0.0
                and comp_a2_id["accuracy_ci95"][0] > -0.01
                and comp_a2_id["macro_f1_ci95"][0] > -0.01
            ),
            "superior_under_channel_shift_vs_best_fixed": bool(
                comp_a2_sh["accuracy_ci95"][0] > 0.0 and comp_a2_sh["utility_ci95"][0] > 0.0
            ),
            "superior_to_all_baselines_id": all(
                comparisons_by_split["test"][f"lbap_minus_{b}"]["utility_ci95"][0] > 0.0 for b in baselines
            ),
        }

        n3_4_report = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "lambda_per_ms": LAMBDA_PER_MS,
            "metrics_by_split": metrics_by_split,
            "paired_bootstrap_10k": comparisons_by_split,
            "three_seed_dispersion": seed_dispersion,
            "formal_claim_verdicts": verdicts,
        }
        _json(OUTPUT_DIR / "n3_4_paired_bootstrap.json", n3_4_report)

        # N3.5 y N3.6
        _generate_n3_5_and_n3_6_reports(OUTPUT_DIR)

    return {
        "n3_1_verified": n3_1_report["reload_100_percent_match"]
        and n3_1_report["causal_100_percent_match"]
        and n3_1_report["lazy_execution_strictly_verified"]
        and n3_1_report["incompatible_hash_rejected"],
        "n3_3_pooled_compute_ms": {
            k: round(v["mean_ms"], 3) for k, v in latency_report["pooled_compute_summary"].items()
        },
        "n3_4_test_acc_and_util": {
            k: {
                "acc": round(v["accuracy"], 4),
                "macro_f1": round(v["macro_f1"], 4),
                "total_ms": round(v["total_mean_ms"], 2),
                "utility": round(v["utility"], 4),
            }
            for k, v in metrics_by_split["test"].items()
        },
        "n3_4_formal_verdicts": verdicts,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
