"""Pruebas físicas, separación train/val/test y persistencia del baseline E2."""
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from experiments.exp02_conventional import classification_metrics, run_experiment
from src.classification_data import load_classification_split
from src.conventional import (CLASSES, complete_pipeline, load_model,
                              make_feature_classifier, predict_observation, save_model)
from src.features import (CORE_INDICES, FEATURE_NAMES, extract_feature_matrix,
                          extract_features)


def test_features_invariancia_ganancia_fase_y_amplitud_subnormal():
    rng = np.random.default_rng(501)
    iq = rng.normal(size=256)+1j*rng.normal(size=256)
    reference = extract_features(iq)
    for factor in (1e-310, 1e-150, 1e150, 2*np.exp(.73j)):
        np.testing.assert_allclose(extract_features(iq*factor), reference, rtol=1e-8, atol=1e-9)
    assert len(reference) == len(set(FEATURE_NAMES)) == 65
    assert len(CORE_INDICES) == 49
    assert np.isfinite(extract_features(np.zeros(256))).all()
    assert np.isfinite(extract_features(np.ones(256))).all()
    for invalid in ([], [1]*32, [[1]*256], [np.nan]*256, [np.inf]*256):
        with pytest.raises(ValueError):
            extract_features(invalid)


def test_cumulantes_constelaciones_y_fase_diferencial_observable():
    bpsk = np.tile([1., -1.], 512).astype(complex)
    qpsk = np.tile([1., 1j, -1., -1j], 256)
    bpsk_features, qpsk_features = map(extract_features, (bpsk, qpsk))
    indices = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    assert bpsk_features[indices["circularity"]] == pytest.approx(1)
    assert qpsk_features[indices["circularity"]] == pytest.approx(0)
    assert bpsk_features[indices["c42_real"]] == pytest.approx(-2)
    assert qpsk_features[indices["c42_real"]] == pytest.approx(-1)
    rng = np.random.default_rng(502)
    iq = rng.normal(size=256)+1j*rng.normal(size=256)
    shifted = iq*np.exp(2j*np.pi*.043*np.arange(256))
    first, second = map(extract_features, (iq, shifted))
    differential = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith("differential_")]
    np.testing.assert_allclose(first[differential], second[differential], atol=1e-12)


def test_scaler_solo_train_y_artefacto_recargado(tmp_path):
    rng = np.random.default_rng(503)
    iq = rng.normal(size=(36, 256))+1j*rng.normal(size=(36, 256))
    features = extract_feature_matrix(iq)
    targets = np.tile(np.arange(6), 6)
    candidate = {"family": "svm_rbf", "feature_set": "core", "C": 1.}
    model = make_feature_classifier(candidate, 504)
    model.fit(features, targets)
    scaler = model.named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, features[:, CORE_INDICES].mean(axis=0))
    original_mean = scaler.mean_.copy()
    model.predict(features+100)
    np.testing.assert_array_equal(scaler.mean_, original_mean)
    path = tmp_path/"model.joblib"
    pipeline = complete_pipeline(model, 256)
    save_model(path, pipeline, {"fit_split": "train"})
    artifact = load_model(path)
    np.testing.assert_array_equal(artifact["pipeline"].predict(iq), model.predict(features))
    result = predict_observation(artifact, iq[0], fs_hz=8000)
    assert result["modulation"] in CLASSES
    assert result["scores_calibrated"] is False
    with pytest.raises(ValueError):
        predict_observation(artifact, np.ones(512))
    artifact["classes"] = tuple(reversed(CLASSES))
    joblib.dump(artifact, tmp_path/"corrupt.joblib")
    with pytest.raises(ValueError):
        load_model(tmp_path/"corrupt.joblib")


def test_metricas_no_ocultan_clases_sin_prediccion():
    result = classification_metrics(np.arange(6), np.zeros(6, dtype=int))
    assert result["accuracy"] == pytest.approx(1/6)
    assert len(result["per_class"]) == 6
    assert result["per_class"][1]["precision"] == 0
    assert result["macro_recall"] == pytest.approx(1/6)


def test_e2_congela_ambos_modelos_antes_test_y_shift_sellado(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    dataset_config = json.loads((root/"configs/dataset_e2.json").read_text())
    dataset_config.update(realizations_per_class_snr=2, noise_only_per_split=2, snr_db=[0, 10])
    dataset_config_path = tmp_path/"dataset.json"
    dataset_config_path.write_text(json.dumps(dataset_config))
    config = json.loads((root/"configs/exp02_conventional.json").read_text())
    config.update(dataset_root=str(tmp_path/"dataset"), dataset_config=str(dataset_config_path),
                  latency_repeats=1, latency_windows=2)
    config["candidates"] = [{"name": "rf5", "family": "random_forest", "feature_set": "full", "n_estimators": 5},
                            {"name": "svm1", "family": "svm_rbf", "feature_set": "full", "C": 1.}]
    calls = []
    def checked_loader(dataset, split, n_samples, **kwargs):
        assert split != "test_shift"
        if split == "test":
            assert (tmp_path/"results/selection_frozen.json").exists()
            assert all((tmp_path/f"results/conventional_{n}.joblib").exists() for n in (256, 512))
        calls.append(split)
        return load_classification_split(dataset, split, n_samples, **kwargs)
    monkeypatch.setattr("experiments.exp02_conventional.load_classification_split", checked_loader)
    report = run_experiment(config, tmp_path/"results")
    assert calls == ["train", "val", "test"]
    assert report["test_shift"] == "sealed_not_loaded_by_e2"
    assert report["split_counts"]["test"] == {"signal_present": 24, "noise_only_excluded": 2}
    with pytest.raises(ValueError):
        load_classification_split(tmp_path/"dataset", "test_shift", 256)
    with (tmp_path/"results/test_predictions_evaluation_only.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 48
    short = [r["parent_waveform_id"] for r in rows if r["n_samples"] == "256"]
    long = [r["parent_waveform_id"] for r in rows if r["n_samples"] == "512"]
    assert short == long
    events = [json.loads(row) for row in (tmp_path/"results/events.jsonl").read_text().splitlines()]
    freeze_index = next(r["sequence"] for r in events if r["stage"] == "both_models_frozen")
    test_index = next(r["sequence"] for r in events if r.get("split") == "test")
    assert freeze_index < test_index
    for n_samples in (256, 512):
        assert sum(map(sum, report["metrics"][str(n_samples)]["confusion_matrix"])) == 24
