"""E2 CPU: selección val de baselines convencionales, test tras congelar N=256/512."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits, threadpool_info

from src.classification_data import load_classification_split, prepare_classification_dataset
from src.conventional import (CLASSES, complete_pipeline, load_model,
                              make_feature_classifier, save_model)
from src.features import CORE_INDICES, FEATURE_NAMES, FEATURE_VERSION, extract_feature_matrix


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classification_metrics(truth, predicted):
    """Macro sobre las seis clases fijas; cero explícito cuando no hay predicción."""
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=np.arange(len(CLASSES)), zero_division=0)
    return {"n": len(truth), "accuracy": float(accuracy_score(truth, predicted)),
            "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()),
            "per_class": [{"modulation": name, "precision": float(p), "recall": float(r),
                           "f1": float(f), "support": int(s)}
                          for name, p, r, f, s in zip(CLASSES, precision, recall, f1, support)]}


def _cpu_name():
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor()


def measure_latency(pipeline, iq, repeats, n_windows):
    """Una ventana por llamada, batch=1; incluye extracción de las 65 features.

    core selecciona después 49, por lo que también paga las 65. Se separa
    tiempo de adquisición virtual, medido solo como N/fs fuera de aquí.
    """
    x = iq[:min(n_windows, len(iq))]
    feature_step = pipeline.named_steps["features"]
    classifier = Pipeline(pipeline.steps[1:])
    pipeline.predict(x[:4])
    rows = []
    for repeat in range(repeats):
        for index, row in enumerate(x):
            cpu_start, start = time.process_time_ns(), time.perf_counter_ns()
            features = feature_step.transform(row[None, :])
            middle = time.perf_counter_ns()
            classifier.predict(features)
            end, cpu_end = time.perf_counter_ns(), time.process_time_ns()
            rows.append({"repeat": repeat, "window": index,
                         "features_wall_ms": (middle-start)/1e6,
                         "classifier_wall_ms": (end-middle)/1e6,
                         "total_wall_ms": (end-start)/1e6,
                         "total_process_cpu_ms": (cpu_end-cpu_start)/1e6})
    summary = {"batch_size": 1, "repeats": repeats, "distinct_windows": len(x),
               "calls": len(rows), "all_features_computed": len(FEATURE_NAMES)}
    for name in ("features_wall_ms", "classifier_wall_ms", "total_wall_ms", "total_process_cpu_ms"):
        values = np.asarray([r[name] for r in rows])
        summary[name] = {"mean": float(values.mean()), "median": float(np.median(values)),
                         "p95": float(np.percentile(values, 95)), "std": float(values.std())}
    return rows, summary


def _confusion_figure(matrix, n_samples, output):
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    normalized = matrix/np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    visual = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{matrix[i,j]}\n{normalized[i,j]:.1%}", ha="center", va="center",
                    color="white" if normalized[i,j] > .5 else "black", fontsize=8)
    ax.set(xticks=np.arange(len(CLASSES)), yticks=np.arange(len(CLASSES)),
           xticklabels=CLASSES, yticklabels=CLASSES, xlabel="Predicción", ylabel="Verdad (evaluación)",
           title=f"E2 test independiente, {n_samples} muestras; normalizada por clase")
    fig.colorbar(visual, ax=ax, label="Fracción de cada clase real")
    fig.savefig(output/f"confusion_{n_samples}.png", dpi=160)
    plt.close(fig)


def _validate_config(config):
    if (config["schema_version"] != 1 or config["window_lengths"] != [256, 512]
            or config["selection_metric"] != "macro_f1"
            or config["tie_break"] != "first_candidate_in_config"
            or config["fit_splits"] != ["train"] or config["selection_splits"] != ["val"]
            or config["evaluation_splits"] != ["test"]):
        raise ValueError("Protocolo E2 inválido; selección solo val y test_shift sellado.")
    if len({c["name"] for c in config["candidates"]}) != len(config["candidates"]) or not config["candidates"]:
        raise ValueError("Se requieren candidatos con nombres distintos.")
    for key in ("latency_repeats", "latency_windows"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} debe ser entero positivo.")


def run_experiment(config, output):
    _validate_config(config)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    dataset_root = Path(config["dataset_root"])
    dataset_config = json.loads(Path(config["dataset_config"]).read_text(encoding="utf-8"))
    if not dataset_root.exists():
        prepare_classification_dataset(dataset_config, dataset_root)
    if json.loads((dataset_root/"config.json").read_text(encoding="utf-8")) != dataset_config:
        raise ValueError("El dataset existente no coincide con la configuración declarada.")
    audit = json.loads((dataset_root/"audit.json").read_text(encoding="utf-8"))
    if not audit["splits_disjoint"] or not audit["prefixes_exact"]:
        raise ValueError("La auditoría de padres/prefijos no es válida.")
    output.mkdir(parents=True)
    _json(output/"config.json", config)
    events = []
    def event(stage, **values):
        events.append({"sequence": len(events), "stage": stage, **values})
        with (output/"events.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(events[-1])+"\n")
    event("protocol_fixed", selection="val_macro_f1_then_first_config_candidate", test_shift="sealed")
    # Solo dos matrices, cargadas una vez. Los prefijos se derivan en memoria.
    train = load_classification_split(dataset_root, "train", 512)
    event("split_loaded", split="train")
    val = load_classification_split(dataset_root, "val", 512)
    event("split_loaded", split="val")
    train_parents = {r["parent_waveform_id"] for r in train["evaluation_rows"]}
    val_parents = {r["parent_waveform_id"] for r in val["evaluation_rows"]}
    if train_parents & val_parents:
        raise ValueError("Padres compartidos entre train y val.")
    metrics, selections, reload_checks, selection_rows = {}, {}, {}, []
    fit_seeds = np.random.SeedSequence(config["seed"]).generate_state(2, dtype=np.uint32)
    with threadpool_limits(limits=1):
        for n_samples, seed in zip(config["window_lengths"], fit_seeds):
            start = time.perf_counter()
            x_train = extract_feature_matrix(train["iq"][:, :n_samples])
            train_feature_seconds = time.perf_counter()-start
            start = time.perf_counter()
            x_val = extract_feature_matrix(val["iq"][:, :n_samples])
            val_feature_seconds = time.perf_counter()-start
            winner, winner_f1, winner_candidate = None, -1., None
            for candidate in config["candidates"]:
                model = make_feature_classifier(candidate, int(seed))
                start = time.perf_counter()
                model.fit(x_train, train["targets"])
                fit_seconds = time.perf_counter()-start
                predicted = model.predict(x_val)
                scores = classification_metrics(val["targets"], predicted)
                row = {"n_samples": n_samples, "candidate": candidate["name"],
                       "feature_set": candidate["feature_set"],
                       "n_features_selected": len(FEATURE_NAMES) if candidate["feature_set"] == "full" else len(CORE_INDICES),
                       "fit_seconds": fit_seconds,
                       **{k: scores[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}}
                selection_rows.append(row)
                print(f"N={n_samples} val {candidate['name']}: F1macro={scores['macro_f1']:.4f}", flush=True)
                if scores["macro_f1"] > winner_f1:
                    winner, winner_f1, winner_candidate = model, scores["macro_f1"], candidate
            metadata = {"n_samples": n_samples, "selected_candidate": winner_candidate,
                        "selection_metric": "val_macro_f1", "selection_score": winner_f1,
                        "fit_split": "train", "selection_split": "val", "fit_seed": int(seed),
                        "dataset_config_sha256": _sha(dataset_root/"config.json"),
                        "n_train": len(x_train), "n_val": len(x_val),
                        "all_features_computed": len(FEATURE_NAMES),
                        "n_features_selected": len(FEATURE_NAMES) if winner_candidate["feature_set"] == "full" else len(CORE_INDICES)}
            model_path = output/f"conventional_{n_samples}.joblib"
            save_model(model_path, complete_pipeline(winner, n_samples), metadata)
            reloaded = load_model(model_path)
            cases = min(32, len(x_val))
            before = winner.predict(x_val[:cases])
            after = reloaded["pipeline"].predict(val["iq"][:cases, :n_samples])
            if not np.array_equal(before, after):
                raise RuntimeError("La inferencia recargada difiere del modelo seleccionado.")
            reload_checks[str(n_samples)] = {"validation_cases": cases, "predictions_identical": True}
            selections[str(n_samples)] = {**metadata, "model_sha256": _sha(model_path)}
            metrics[str(n_samples)] = {"feature_extraction_seconds": {"train": train_feature_seconds, "val": val_feature_seconds}}
            event("model_saved", n_samples=n_samples, model_sha256=_sha(model_path))
        # Ambos modelos juntos quedan congelados ANTES de cargar test.
        _json(output/"selection_frozen.json", selections)
        _json(output/"reload_validation.json", reload_checks)
        _csv(output/"candidate_validation.csv", selection_rows)
        event("both_models_frozen", selection_sha256=_sha(output/"selection_frozen.json"))
        test = load_classification_split(dataset_root, "test", 512)
        event("split_loaded", split="test")
        test_parents = {r["parent_waveform_id"] for r in test["evaluation_rows"]}
        if (train_parents | val_parents) & test_parents:
            raise ValueError("Padres compartidos con test.")
        predictions, by_snr, per_class_rows, confusion_rows, latency_rows = [], [], [], [], []
        for n_samples in config["window_lengths"]:
            artifact = load_model(output/f"conventional_{n_samples}.joblib")
            pipeline = artifact["pipeline"]
            iq = test["iq"][:, :n_samples]
            start = time.perf_counter()
            predicted = pipeline.predict(iq)
            test_prediction_seconds = time.perf_counter()-start
            scores = classification_metrics(test["targets"], predicted)
            matrix = confusion_matrix(test["targets"], predicted, labels=np.arange(len(CLASSES)))
            for item in scores["per_class"]:
                per_class_rows.append({"n_samples": n_samples, **item})
            for truth_index, truth_name in enumerate(CLASSES):
                for pred_index, pred_name in enumerate(CLASSES):
                    confusion_rows.append({"n_samples": n_samples, "truth": truth_name, "prediction": pred_name,
                                           "count": int(matrix[truth_index, pred_index])})
            snrs = np.asarray([float(r["snr_nominal_db"]) for r in test["evaluation_rows"]])
            for snr in dataset_config["snr_db"]:
                mask = snrs == snr
                cell = classification_metrics(test["targets"][mask], predicted[mask])
                by_snr.append({"n_samples": n_samples, "snr_nominal_db": snr,
                               **{k: v for k, v in cell.items() if k != "per_class"}})
                for item in cell["per_class"]:
                    per_class_rows.append({"n_samples": n_samples, "snr_nominal_db": snr, **item})
            for record, target, pred in zip(test["evaluation_rows"], test["targets"], predicted):
                predictions.append({"record_id": record["record_id"], "parent_waveform_id": record["parent_waveform_id"],
                                    "n_samples": n_samples, "truth_evaluation_only": CLASSES[int(target)],
                                    "prediction": CLASSES[int(pred)], "snr_nominal_evaluation_only_db": record["snr_nominal_db"]})
            raw_latency, latency = measure_latency(pipeline, iq, config["latency_repeats"], config["latency_windows"])
            latency_rows.extend({"n_samples": n_samples, **row} for row in raw_latency)
            metrics[str(n_samples)].update(test=scores, confusion_matrix=matrix.tolist(), latency=latency,
                                          test_batch_prediction_seconds=test_prediction_seconds,
                                          virtual_acquisition_seconds=n_samples/test["fs_hz"])
            _confusion_figure(matrix, n_samples, output)
            print(f"N={n_samples} TEST: accuracy={scores['accuracy']:.4f}, F1macro={scores['macro_f1']:.4f}", flush=True)
        event("evaluation_completed", test_shift="sealed")
        pools = threadpool_info()
    _csv(output/"test_predictions_evaluation_only.csv", predictions)
    _csv(output/"test_metrics_by_snr.csv", by_snr)
    # Esquema uniforme: agregado marcado explícitamente como ALL.
    for row in per_class_rows:
        row.setdefault("snr_nominal_db", "ALL")
    _csv(output/"test_metrics_by_class.csv", per_class_rows)
    _csv(output/"confusion_counts.csv", confusion_rows)
    _csv(output/"latency_batch1.csv", latency_rows)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for n_samples in config["window_lengths"]:
        rows = [r for r in by_snr if r["n_samples"] == n_samples]
        ax.plot([r["snr_nominal_db"] for r in rows], [r["accuracy"] for r in rows], "o-", label=f"{n_samples} muestras")
    ax.axhline(1/len(CLASSES), color="gray", linestyle=":", label="Azar balanceado 1/6")
    ax.set(xlabel="SNR nominal de evaluación (dB)", ylabel="Exactitud", ylim=(0, 1),
           title="E2: test independiente; modelos seleccionados solo con val")
    ax.grid(alpha=.3)
    ax.legend()
    fig.savefig(output/"accuracy_by_snr.png", dpi=160)
    plt.close(fig)
    root = Path(__file__).resolve().parents[1]
    source_names = ["src/features.py", "src/conventional.py", "src/classification_data.py", "src/dataset.py",
                    "src/signals.py", "src/waveform.py", "src/channels.py", "experiments/exp02_conventional.py"]
    summary = {"config": config, "dataset_config": dataset_config, "classes": CLASSES,
               "feature_names": FEATURE_NAMES, "feature_version": FEATURE_VERSION,
               "selection": selections, "metrics": metrics, "dataset_audit": audit,
               "split_counts": {split: {"signal_present": len(data["targets"]), "noise_only_excluded": data["n_noise_excluded"]}
                                for split, data in (("train", train), ("val", val), ("test", test))},
               "test_shift": "sealed_not_loaded_by_e2", "same_parents_for_both_lengths": True,
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                            "sklearn": sklearn.__version__, "joblib": joblib.__version__, "matplotlib": matplotlib.__version__},
               "execution": {"cpu": _cpu_name(), "platform": platform.platform(), "n_jobs": 1,
                             "threadpool_limit": 1, "threadpools_during_fit_and_timing": pools},
               "source_sha256": {name: _sha(root/name) for name in source_names},
               "limits": "AMC closed-set condicionada a presencia real en el benchmark, sin evaluar gating del detector. No produce confianza calibrada. ID comparte distribuciones de canal, no padres. test_shift reservado E6. Selección entre candidatos predefinidos solo por macro-F1 val; modelos ajustados solo train, sin refit train+val. Core selecciona49 de65 pero el pipeline calcula65: latencia incluye todo. SNR/IDs solo supervisión/evaluación. N/fs es adquisición virtual, separado de latencia CPU batch1. Tiempos son mediciones, no constantes reproducibles bit a bit ni consumo energético."}
    _json(output/"metricas.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp02_conventional.json"))
    parser.add_argument("--output", type=Path, default=Path("results/fase5"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_experiment(config, args.output)
    _json(args.output/"command.json", {"argv": [sys.executable, "-m", "experiments.exp02_conventional",
                                               "--config", str(args.config), "--output", str(args.output)], "cwd": str(Path.cwd())})


if __name__ == "__main__":
    main()
