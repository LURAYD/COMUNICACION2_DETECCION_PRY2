"""Pipelines convencionales CPU y contrato de persistencia de Fase 5."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.features import (CORE_INDICES, FEATURE_NAMES, FEATURE_VERSION,
                          IQFeatureTransformer)


CLASSES = ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "4PAM")


class SelectObservableFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, feature_set="full"):
        self.feature_set = feature_set

    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, X):
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError("Matriz de características observable inválida.")
        if self.feature_set == "full":
            return x
        if self.feature_set == "core":
            return x[:, CORE_INDICES]
        raise ValueError("feature_set debe ser full o core.")


def make_feature_classifier(candidate, seed):
    """Fit de esta cadena solo con train. Los nombres fijan el desempate."""
    family = candidate["family"]
    steps = [("select", SelectObservableFeatures(candidate["feature_set"]))]
    if family == "svm_rbf":
        steps.extend([("scale", StandardScaler()),
                      ("classifier", SVC(C=candidate["C"], kernel="rbf", gamma="scale",
                                         probability=False, cache_size=256, random_state=seed))])
    elif family == "random_forest":
        steps.append(("classifier", RandomForestClassifier(n_estimators=candidate["n_estimators"],
                        max_features="sqrt", min_samples_leaf=2, n_jobs=1, random_state=seed)))
    else:
        raise ValueError("Familia convencional desconocida.")
    return Pipeline(steps)


def complete_pipeline(fitted_feature_classifier, n_samples):
    return Pipeline([("features", IQFeatureTransformer(n_samples)), *fitted_feature_classifier.steps])


def save_model(path, pipeline, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    artifact = {"schema_version": 1, "classes": CLASSES, "feature_version": FEATURE_VERSION,
                "feature_names": FEATURE_NAMES, "pipeline": pipeline, "metadata": metadata}
    _validate_artifact(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)


def _validate_artifact(artifact):
    if (artifact.get("schema_version") != 1 or artifact.get("classes") != CLASSES
            or artifact.get("feature_version") != FEATURE_VERSION
            or artifact.get("feature_names") != FEATURE_NAMES):
        raise ValueError("Esquema o clases del artefacto incompatibles.")
    pipeline = artifact["pipeline"]
    if not isinstance(pipeline, Pipeline) or "features" not in pipeline.named_steps:
        raise ValueError("Artefacto sin pipeline observable.")
    if pipeline.named_steps["features"].n_samples not in (256, 512):
        raise ValueError("Longitud de artefacto no compatible.")
    if not np.array_equal(pipeline.named_steps["classifier"].classes_, np.arange(len(CLASSES))):
        raise ValueError("Clases aprendidas ausentes o incompatibles.")


def load_model(path):
    artifact = joblib.load(path)
    _validate_artifact(artifact)
    return artifact


def predict_observation(artifact, iq, fs_hz=None):
    """Inferencia pública: observación nueva sin etiquetas ni metadatos oracle."""
    _validate_artifact(artifact)
    x = np.asarray(iq, dtype=np.complex128)
    if x.ndim != 1:
        raise ValueError("Se requiere un vector IQ.")
    if fs_hz is not None and (isinstance(fs_hz, bool) or not np.isscalar(fs_hz)
                             or not np.isrealobj(fs_hz) or not np.isfinite(fs_hz) or fs_hz <= 0):
        raise ValueError("fs_hz debe ser conocida positiva o None.")
    predicted = int(artifact["pipeline"].predict(x[None, :])[0])
    return {"modulation": CLASSES[predicted], "n_samples": len(x),
            "scope": "closed_set_signal_present", "scores_calibrated": False}
