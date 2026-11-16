"""Verifica ganadores congelados sin reabrir test; demo con nuevas IQ."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from src.classification_data import load_classification_split
from src.conventional import CLASSES, load_model, make_feature_classifier, predict_observation
from src.dataset import _generate_parent
from src.features import extract_feature_matrix


def verify(results, demo_seed):
    results = Path(results)
    config = json.loads((results/"config.json").read_text(encoding="utf-8"))
    selections = json.loads((results/"selection_frozen.json").read_text(encoding="utf-8"))
    dataset_config = json.loads(Path(config["dataset_config"]).read_text(encoding="utf-8"))
    if demo_seed in (config["seed"], dataset_config["seed"]):
        raise ValueError("La demostración requiere una semilla nueva.")
    train = load_classification_split(config["dataset_root"], "train", 512)
    checks, artifacts = {}, {}
    with threadpool_limits(limits=1):
        for n_samples in (256, 512):
            key = str(n_samples)
            model_path = results/f"conventional_{n_samples}.joblib"
            digest_before = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if digest_before != selections[key]["model_sha256"]:
                raise ValueError("El modelo ya no coincide con el congelado.")
            artifact = load_model(model_path)
            artifacts[n_samples] = artifact
            features = extract_feature_matrix(train["iq"][:, :n_samples])
            repeated = make_feature_classifier(selections[key]["selected_candidate"], selections[key]["fit_seed"])
            repeated.fit(features, train["targets"])
            original_steps = artifact["pipeline"].named_steps
            state_hashes = {}
            for name, estimator in repeated.steps:
                first, second = joblib.hash(original_steps[name]), joblib.hash(estimator)
                state_hashes[name] = {"frozen": first, "retrained": second, "identical": first == second}
                if first != second:
                    raise RuntimeError(f"Estado diferente al reentrenar {n_samples}/{name}.")
            predicted_new = repeated.predict(features)
            # Inferencia por el pipeline completo recargado, mismas IQ train.
            predicted_saved = artifact["pipeline"].predict(train["iq"][:, :n_samples])
            if not np.array_equal(predicted_new, predicted_saved):
                raise RuntimeError("El ganador reentrenado difiere en predicciones train.")
            if hashlib.sha256(model_path.read_bytes()).hexdigest() != digest_before:
                raise RuntimeError("Se modificó un artefacto congelado.")
            checks[key] = {"state_hashes": state_hashes, "train_predictions_identical": True,
                           "n_train_predictions": len(predicted_new), "original_model_unchanged": True,
                           "model_sha256": digest_before}
    # Realizaciones nuevas e independientes. La API recibe solamente IQ/fs.
    windows = []
    for index, (name, seed) in enumerate(zip(CLASSES, np.random.SeedSequence(demo_seed).spawn(len(CLASSES)))):
        record = {"record_id": f"demo_{index}", "parent_waveform_id": f"demo_parent_{index}",
                  "split": "test", "modulation": name, "snr": 10, "seed": seed}
        iq, _ = _generate_parent(dataset_config, record)
        windows.append(iq)
    fs = dataset_config["symbol_rate_hz"]*dataset_config["sps"]
    np.savez(results/"demo_new_observations.npz", iq512=np.asarray(windows))
    demo_rows = []
    with threadpool_limits(limits=1):
        for index, iq in enumerate(windows):
            for n_samples, artifact in artifacts.items():
                demo_rows.append({"observation_id": f"demo_{index}",
                                  **predict_observation(artifact, iq=iq[:n_samples], fs_hz=fs)})
    demo = {"seed": demo_seed, "fs_hz": fs, "api_arguments": ["iq", "fs_hz"],
            "scope": "Demostración de API, no métrica de exactitud; seis padres nuevos y sus dos prefijos.",
            "predictions": demo_rows}
    (results/"demo_predictions.json").write_text(json.dumps(demo, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    report = {"only_training_split_reopened": True, "test_and_test_shift_not_loaded": True,
              "checks": checks, "demo_seed": demo_seed,
              "command": [sys.executable, "-m", "experiments.verify_exp02_artifacts", "--results", str(results),
                          "--demo-seed", str(demo_seed)]}
    (results/"artifact_validation.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("Reentrenamiento de ganadores: estados/predicciones idénticos; demo IQ/fs guardada.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/fase5"))
    parser.add_argument("--demo-seed", type=int, default=2026110504)
    args = parser.parse_args()
    verify(args.results, args.demo_seed)


if __name__ == "__main__":
    main()
