"""Selector observable: solo prefijo 256 y cálculos efectivamente ejecutados; paquete desplegable N3.1."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
import torch

from src.advanced import load_checkpoint, preprocess
from src.conventional import CLASSES, load_model
from src.features import FEATURE_NAMES, FEATURE_VERSION, extract_feature_matrix
from src.research_models import FastLBAPSelector, MultiDomainAMCNet

FLATNESS_INDEX = FEATURE_NAMES.index("spectral_flatness")
ENTROPY_INDEX = FEATURE_NAMES.index("spectral_entropy")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def observe_prefix(iq, svm256):
    x = np.asarray(iq)
    if x.ndim != 2 or x.shape[1] != 256 or not np.iscomplexobj(x) or not np.isfinite(x).all():
        raise ValueError("El selector solo admite matrices IQ de 256 muestras finitas.")
    features = extract_feature_matrix(x)
    classifier = Pipeline(svm256.steps[1:])
    labels = classifier.predict(features)
    scores = classifier.decision_function(features)
    ordered = np.sort(scores, axis=1)
    return dict(
        gate=np.concatenate([features, scores], axis=1),
        initial=labels,
        margin=(ordered[:, -1] - ordered[:, -2])[:, None],
        flatness=features[:, FLATNESS_INDEX],
    )


def choose_actions(observation, kind, artifact):
    if kind == "confidence":
        confidence = artifact["calibrator"].predict_proba(observation["margin"])[:, 1]
        return np.where(confidence < artifact["confidence_threshold"], 1, 0)
    if kind == "flatness":
        return np.where(observation["flatness"] > artifact["flatness_threshold"], 2, 1)
    if kind == "selector":
        return artifact["selector"].predict(observation["gate"]).argmax(axis=1)
    raise ValueError("Política desconocida.")


def action_cost_ms(actions, base_ms, gate_ms=0.0, routed=True):
    actions = np.asarray(actions, dtype=int)
    if np.any((actions < 0) | (actions > 2)):
        raise ValueError("Acción inválida.")
    acquisition = np.where(actions == 1, 64.0, 32.0)
    if routed:
        compute = base_ms[0] + gate_ms + np.where(actions == 1, base_ms[1], np.where(actions == 2, base_ms[2], 0.0))
    else:
        compute = np.asarray(base_ms)[actions]
    return acquisition, compute


def estimate_blind_spectral_snr_db(iq_matrix):
    """Estimador ciego no-oracular de SNR (dB) por cuantiles del periodograma."""
    x = np.asarray(iq_matrix, dtype=np.complex128)
    n = x.shape[1]
    w = np.hanning(n)[None, :]
    w_norm = np.sum(w**2)
    psd = (np.abs(np.fft.fft(x * w, axis=1)) ** 2) / w_norm
    total_power = np.mean(psd, axis=1)
    noise_floor = np.quantile(psd, 0.35, axis=1) / 0.43
    sig_power = np.maximum(total_power - noise_floor, 1e-3 * noise_floor)
    snr_db = 10.0 * np.log10(sig_power / np.maximum(noise_floor, 1e-15))
    return np.clip(snr_db, -25.0, 30.0)


def build_extended_gate_from_prefix(iq256, svm256, calibrator_a0):
    """Construye el vector observable completo de 74 dimensiones desde prefijo de 256 muestras."""
    obs = observe_prefix(iq256, svm256)
    gate_base = obs["gate"]
    margin = obs["margin"]
    conf = calibrator_a0.predict_proba(margin)[:, 1:2]
    scores = gate_base[:, 65:]
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1, keepdims=True) / np.log(float(len(CLASSES)))
    gate_full = np.concatenate([gate_base, margin, conf, entropy], axis=1)
    snr_blind = estimate_blind_spectral_snr_db(iq256)
    gate_i2 = np.column_stack([snr_blind, gate_base[:, [FLATNESS_INDEX, ENTROPY_INDEX]]])
    return {
        "gate_full": gate_full,
        "gate_i2": gate_i2,
        "confidence": conf[:, 0],
        "initial": obs["initial"],
        "margin": margin[:, 0],
        "snr_blind": snr_blind,
    }


def serialize_selector_state(selector: FastLBAPSelector) -> dict:
    return {
        "mean": selector.mean.copy(),
        "scale": selector.scale.copy(),
        "w1": selector.w1.copy(),
        "b1": selector.b1.copy(),
        "w2": selector.w2.copy(),
        "b2": selector.b2.copy(),
        "w_lin": selector.w_lin.copy(),
        "b_lin": selector.b_lin.copy(),
        "alpha_mlp": float(selector.alpha_mlp),
        "allowed_actions": tuple(selector.allowed_actions),
    }


def deserialize_selector_state(state: dict) -> FastLBAPSelector:
    return FastLBAPSelector(
        mean=state["mean"],
        scale=state["scale"],
        w1=state["w1"],
        b1=state["b1"],
        w2=state["w2"],
        b2=state["b2"],
        w_lin=state["w_lin"],
        b_lin=state["b_lin"],
        alpha_mlp=state["alpha_mlp"],
        allowed_actions=state["allowed_actions"],
    )


def save_deployable_policy_package(path: str | Path, package: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, path, compress=3)
    return path


def load_deployable_policy_package(
    path: str | Path, root_dir: str | Path | None = None, verify_hashes: bool = True
) -> dict:
    pkg_path = Path(path)
    if not pkg_path.is_file():
        raise FileNotFoundError(f"Paquete de política no encontrado: {pkg_path}")
    pkg = joblib.load(pkg_path)
    if not isinstance(pkg, dict) or pkg.get("package_format") != "lbap_deployable_v1":
        raise ValueError("Formato de paquete de política incompatible.")
    if pkg.get("schema_version") != 1:
        raise ValueError("schema_version del paquete de política incompatible.")
    if pkg.get("feature_version") != FEATURE_VERSION or tuple(pkg.get("classes", ())) != CLASSES:
        raise ValueError("Versión de características o clases incompatible con el código actual.")

    if verify_hashes:
        base_root = Path(root_dir) if root_dir else Path(__file__).resolve().parents[1]
        expected_hashes = pkg.get("model_hashes", {})
        for key, rel_p in pkg.get("model_paths", {}).items():
            full_p = (base_root / rel_p) if not Path(rel_p).is_absolute() else Path(rel_p)
            if not full_p.is_file():
                raise ValueError(f"Modelo requerido '{key}' ausente en {full_p}.")
            actual = _sha256_file(full_p)
            if expected_hashes.get(key) != actual:
                raise ValueError(
                    f"Hash incompatible para modelo '{key}' ({rel_p}): esperado {expected_hashes.get(key)}, obtenido {actual}."
                )

    if "selector_state" in pkg and "selector" not in pkg:
        pkg["selector"] = deserialize_selector_state(pkg["selector_state"])
    return pkg


def load_package_models(package: dict, root_dir: str | Path | None = None, device: str | None = None) -> dict:
    base_root = Path(root_dir) if root_dir else Path(__file__).resolve().parents[1]
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    paths = package["model_paths"]
    svm256 = load_model(base_root / paths["a0_svm256"])["pipeline"]
    svm512 = load_model(base_root / paths["a1_svm512"])["pipeline"]
    ckpt = load_checkpoint(base_root / paths["a2_multidomain"])
    if dev == "cuda":
        torch.set_float32_matmul_precision("high")
    net = MultiDomainAMCNet(
        num_classes=len(CLASSES),
        width_mult=ckpt.get("width_mult", package.get("width_mult", 2.0)),
        dropout=0.15,
    ).to(dev).eval()
    net.load_state_dict(ckpt["model"], strict=True)

    def _predict_a2(iq256_batch: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(preprocess(iq256_batch)).to(dev)
        with torch.inference_mode():
            logits = net(x)
            if dev == "cuda":
                torch.cuda.synchronize()
            return logits.argmax(dim=1).cpu().numpy().astype(int)

    return {
        "svm256": svm256,
        "svm512": svm512,
        "multidomain": net,
        "predict_a2": _predict_a2,
        "device": dev,
    }


def run_deployable_policy_inference(
    iq,
    package: dict,
    models: dict | None = None,
    fs_hz: float = 8000.0,
    root_dir: str | Path | None = None,
) -> dict:
    """Ejecuta inferencia causal y perezosa: solo observa [:256] antes de decidir y solo ejecuta la rama elegida."""
    if fs_hz is None or not np.isfinite(fs_hz) or not np.isclose(float(fs_hz), float(package.get("expected_fs_hz", 8000.0))):
        raise ValueError(f"Frecuencia de muestreo incompatible; esperada {package.get('expected_fs_hz', 8000.0)} Hz.")

    x = np.asarray(iq, dtype=np.complex128)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Se requiere un vector IQ 1D finito.")
    if x.size < 256:
        raise ValueError("Se requieren al menos 256 muestras para observación inicial.")

    if models is None:
        models = load_package_models(package, root_dir=root_dir)

    # Paso 1 (Causal): entregar únicamente el prefijo de 256 muestras
    iq_prefix = x[:256][None, :]
    gate_info = build_extended_gate_from_prefix(iq_prefix, models["svm256"], package["calibrator_a0"])
    policy_mode = package.get("policy_mode", "lbap_full_a0_a1_a2")

    if policy_mode == "lbap_full_a0_a1_a2":
        action = int(package["selector"].choose(gate_info["gate_full"])[0])
    elif policy_mode == "ablation_input_snr_flatness":
        action = int(package["selector"].choose(gate_info["gate_i2"])[0])
    elif policy_mode == "ablation_input_confidence":
        th = float(package["confidence_threshold"])
        alt = int(package.get("confidence_alt_action", 2))
        action = alt if float(gate_info["confidence"][0]) < th else 0
    else:
        raise ValueError(f"Modo de política no soportado: {policy_mode}")

    # Paso 2 (Perezoso): ejecutar exclusivamente la rama seleccionada
    if action == 0:
        pred_idx = int(gate_info["initial"][0])
    elif action == 1:
        if x.size < 512:
            raise ValueError("La política seleccionó a1 (N=512), pero la señal tiene menos de 512 muestras.")
        pred_idx = int(models["svm512"].predict(x[:512][None, :])[0])
    elif action == 2:
        pred_idx = int(models["predict_a2"](iq_prefix)[0])
    else:
        raise ValueError(f"Acción fuera de rango: {action}")

    action_names = ("a0_svm256", "a1_svm512", "a2_multidomain")
    return {
        "action": action,
        "action_name": action_names[action],
        "pred_idx": pred_idx,
        "modulation": CLASSES[pred_idx],
        "initial_a0_modulation": CLASSES[int(gate_info["initial"][0])],
        "confidence_a0": round(float(gate_info["confidence"][0]), 4),
        "snr_blind_db": round(float(gate_info["snr_blind"][0]), 3),
        "policy_mode": policy_mode,
        "fs_hz": float(fs_hz),
    }


def main():
    parser = argparse.ArgumentParser(description="Inferencia desplegable con política adaptativa LBAP (Nivel 3)")
    parser.add_argument("--package", required=True, help="Ruta al paquete .joblib serializado")
    parser.add_argument("--input", "-i", required=True, help="Archivo IQ (.c64, .npy, .npz, .csv)")
    parser.add_argument("--fs-hz", type=float, default=8000.0, help="Frecuencia de muestreo en Hz")
    parser.add_argument("--output", "-o", default=None, help="Ruta opcional para guardar JSON")
    args = parser.parse_args()

    from src.monitor import load_signal

    try:
        pkg = load_deployable_policy_package(args.package, verify_hashes=True)
        iq = load_signal(args.input)
        res = run_deployable_policy_inference(iq, pkg, fs_hz=args.fs_hz)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    out_text = json.dumps(res, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out_text + "\n", encoding="utf-8")
    print(out_text)


if __name__ == "__main__":
    main()
