"""Experimento integral de investigación para el Paper IEEE (Secciones 9, 10, 12, 13 y 14 de GUIA_MAESTRA).

Ejecuta en la GPU NVIDIA RTX 4060:
1. Entrenamiento intensivo en GPU de MultiDomainAMCNet (a2) con síntesis física al vuelo
   (GPUChannelSynthesizer) + aumentación diferenciable en VRAM, seleccionando estrictamente
   en val_cal sin tocar val_select, test ni test_shift.
2. Estudio completo de Cuantización ADC Virtual (Fase 10):
   - Experimento A: Entrenamiento en referencia limpia (float32).
   - Experimento B8 / B6: Entrenamiento cuantizado fijo (8 bits y 6 bits).
   - Experimento C: Entrenamiento mixto multi-resolución (inf, 8, 6 bits).
   - Evaluación en resoluciones vistas (inf, 8, 6 bits) y NO VISTAS (5 y 4 bits), midiendo
     SQNR (offline), tasa de clipping, Accuracy, F1-macro, ECE-10, Brier y efecto sobre el selector.
3. Evaluación de Políticas Adaptativas de 3 Acciones y Suite de Ablaciones (Sección 14):
   - Baselines fijos: always_a0 (SVM256), always_a1 (SVM512), always_a2 (MultiDomainAMCNet-256).
   - Ablaciones de entrada: (I1) solo confianza calibrada, (I2) solo SNR ciega / planitud espectral,
     (I3) confianza + estadísticas compactas, (I4) vector completo LBAP (74 descriptores observables).
   - Ablaciones de espacio de acciones: (A1) solo {a0, a1} (adquisición adaptativa),
     (A2) solo {a0, a2} (escalamiento computacional tipo early-exit), (A3) conjunto {a0, a1, a2}.
   - Curvas de Pareto exactitud-costo barriendo lambda y sensibilidad a frecuencia de muestreo fs,
     con intervalos de confianza bootstrap pareados estratificados (10,000 réplicas).
"""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
import torch.nn.functional as F

from src.adc import evaluate_sqnr_db, quantize_iq
from src.adaptive import action_cost_ms, observe_prefix
from src.advanced import atomic_save, load_checkpoint, preprocess
from src.classification_data import load_classification_split
from src.conventional import CLASSES, load_model
from src.features import FEATURE_NAMES
from src.research_models import (
    FastLBAPSelector,
    GPUChannelSynthesizer,
    MultiDomainAMCNet,
    gpu_physical_augmentation,
)
from experiments.exp02_conventional import _csv, _json, _sha, classification_metrics
from experiments.exp10_adaptive import bootstrap_difference, calibration_metrics


def estimate_blind_spectral_snr_db(iq_matrix):
    """Estimador ciego no-oracular de SNR (dB) por cuantiles del periodograma (sps=8).

    Aprovecha que con sps=8 la señal ocupa una fracción de la banda de Nyquist;
    el cuantil inferior (35%) de los bins espectrales estima el piso de ruido por bin.
    """
    x = np.asarray(iq_matrix, dtype=np.complex128)
    n = x.shape[1]
    w = np.hanning(n)[None, :]
    w_norm = np.sum(w ** 2)
    psd = (np.abs(np.fft.fft(x * w, axis=1)) ** 2) / w_norm
    total_power = np.mean(psd, axis=1)
    noise_floor = np.quantile(psd, 0.35, axis=1) / 0.43  # Corrección sesgo exponencial cuantil 0.35
    sig_power = np.maximum(total_power - noise_floor, 1e-3 * noise_floor)
    snr_db = 10.0 * np.log10(sig_power / np.maximum(noise_floor, 1e-15))
    return np.clip(snr_db, -25.0, 30.0)


def build_extended_gate(obs_dict, calibrator):
    """Construye el vector observable completo de 74 dimensiones para LBAP en < 0.02 ms."""
    gate_base = obs_dict["gate"]  # (N, 71): 65 features + 6 SVM scores
    margin = obs_dict["margin"]   # (N, 1)
    conf = calibrator.predict_proba(margin)[:, 1:2]  # (N, 1)
    scores = gate_base[:, 65:]
    # Pseudo-probabilidades vía softmax de márgenes SVM para entropía de decisión
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1, keepdims=True) / np.log(6.0)
    return np.concatenate([gate_base, margin, conf, entropy], axis=1), conf[:, 0]


@torch.inference_mode()
def predict_multidomain(model, iq_complex, device="cuda", batch_size=512):
    """Predice logits y probabilidades para una matriz compleja (N, 256) en GPU."""
    model.eval()
    x = torch.from_numpy(preprocess(iq_complex))
    logits_list = []
    for i in range(0, len(x), batch_size):
        batch = x[i : i + batch_size].to(device, non_blocking=True)
        logits_list.append(model(batch).cpu())
    logits = torch.cat(logits_list, dim=0).numpy()
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_l = np.exp(shifted)
    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    return logits, probs


MODEL_WIDTH_MULT = 2.0
ARCH_VERSION = 2


def train_gpu_expert(
    name,
    x_train_gpu,
    y_train_gpu,
    x_val_gpu,
    y_val_gpu,
    synth,
    output_dir,
    epochs=90,
    batch_size=1024,
    synth_batches_per_epoch=24,
    lr=1.5e-3,
    weight_decay=2e-4,
    adc_bits_choices=None,
    full_scale=2.5,
    seed=20261111,
    force_retrain=False,
):
    """Entrena un modelo MultiDomainAMCNet intensivamente en la GPU RTX 4060 sin GradScaler ni fugas OOD."""
    ckpt_path = output_dir / f"{name}_best.pt"
    hist_path = output_dir / f"{name}_history.json"
    if not force_retrain and ckpt_path.exists() and hist_path.exists():
        saved = load_checkpoint(ckpt_path)
        if saved.get("arch_version") == ARCH_VERSION:
            print(f"[GPU] Cargando experto existente v{ARCH_VERSION}: {ckpt_path}", flush=True)
            model = MultiDomainAMCNet(
                num_classes=6, width_mult=saved.get("width_mult", MODEL_WIDTH_MULT), dropout=0.15
            ).cuda().eval()
            model.load_state_dict(saved["model"])
            return model, json.loads(hist_path.read_text(encoding="utf-8")), saved["train_seconds"]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    model = MultiDomainAMCNet(num_classes=6, width_mult=MODEL_WIDTH_MULT, dropout=0.15).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[GPU] Iniciando entrenamiento v{ARCH_VERSION} de {name} en {torch.cuda.get_device_name(0)} "
          f"({n_params:,} parámetros, width={MODEL_WIDTH_MULT}, batch={batch_size}, epochs={epochs})", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_epochs = min(5, max(1, epochs // 10))
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    best_f1 = -1.0
    best_state = None
    history = []
    t_start = time.perf_counter()

    n_real = len(x_train_gpu)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_t = torch.zeros((), device="cuda", dtype=torch.float32)
        epoch_samples = 0

        # 1. Paso sobre ventanas reales train con rotación de fase / conjugación exacta (sin doble corrupción)
        perm = torch.randperm(n_real, device="cuda")
        for i in range(0, n_real, batch_size):
            idx = perm[i : i + batch_size]
            xb = x_train_gpu[idx]
            yb = y_train_gpu[idx]
            xb = gpu_physical_augmentation(xb, p_conj=0.5, cfo_max_cycles_per_window=0.0, multipath_prob=0.0, noise_std_max=0.0)
            if adc_bits_choices is not None:
                choices_t = torch.randint(0, len(adc_bits_choices), (len(xb),), device="cuda")
                amp_phy = 0.5 + 1.0 * torch.rand((len(xb), 1, 1), device="cuda")
                xb_phy = xb * amp_phy
                xb_q = xb_phy.clone()
                for idx_b, b_choice in enumerate(adc_bits_choices):
                    if b_choice is None:
                        continue
                    mask_b = (choices_t == idx_b).view(-1, 1, 1)
                    levels = float(1 << int(b_choice))
                    delta = (2.0 * full_scale) / levels
                    clipped = torch.clamp(xb_phy, -full_scale, full_scale)
                    q_idx = torch.clamp(torch.floor((clipped + full_scale) / delta), 0.0, levels - 1.0)
                    q_val = -full_scale + (q_idx + 0.5) * delta
                    xb_q = torch.where(mask_b, q_val, xb_q)
                xb_q = xb_q - xb_q.mean(dim=-1, keepdim=True)
                pwr = torch.clamp((xb_q ** 2).sum(dim=1, keepdim=True).mean(dim=-1, keepdim=True), min=1e-7)
                xb = xb_q / torch.sqrt(pwr)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, label_smoothing=0.02)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            epoch_loss_t += loss.detach() * len(yb)
            epoch_samples += len(yb)

        # 2. Lotes sintetizados al vuelo en RTX 4060 estrictamente dentro del soporte ID (|CFO|<=100Hz, delay<=5)
        for _ in range(synth_batches_per_epoch):
            xb, yb = synth.sample_batch(
                batch_size=batch_size,
                n_samples=256,
                snr_range_db=(-20.0, 20.0),
                cfo_range_hz=(-100.0, 100.0),
                max_echo_delay=5,
                adc_bits_choices=adc_bits_choices,
                full_scale=full_scale,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, label_smoothing=0.02)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            epoch_loss_t += loss.detach() * len(yb)
            epoch_samples += len(yb)

        scheduler.step()
        epoch_loss = float(epoch_loss_t.item())

        # Evaluación rápida en val_cal (por bloques en GPU, sin tocar val_select/test/test_shift)
        model.eval()
        with torch.inference_mode():
            val_logits_chunks = [model(x_val_gpu[j : j + 512]) for j in range(0, len(x_val_gpu), 512)]
            val_logits = torch.cat(val_logits_chunks, dim=0)
            val_loss = float(F.cross_entropy(val_logits, y_val_gpu).item())
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
        val_scores = classification_metrics(y_val_gpu.cpu().numpy(), val_pred)
        val_f1 = val_scores["macro_f1"]
        val_acc = val_scores["accuracy"]

        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(epoch_samples, 1),
            "val_cal_loss": val_loss,
            "val_cal_accuracy": val_acc,
            "val_cal_macro_f1": val_f1,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - t_start,
        }
        history.append(row)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"  [{name}] Epoch {epoch:02d}/{epochs} | loss={row['train_loss']:.4f} | "
                f"val_cal_acc={val_acc:.4f} | val_cal_F1={val_f1:.4f} | best_F1={best_f1:.4f} | "
                f"t={row['elapsed_seconds']:.1f}s",
                flush=True,
            )

    train_seconds = time.perf_counter() - t_start
    model.load_state_dict(best_state)
    model.eval()
    atomic_save(
        ckpt_path,
        {
            "model": best_state,
            "name": name,
            "arch_version": ARCH_VERSION,
            "width_mult": MODEL_WIDTH_MULT,
            "best_val_cal_f1": best_f1,
            "train_seconds": train_seconds,
            "epochs": epochs,
            "batch_size": batch_size,
            "synth_batches_per_epoch": synth_batches_per_epoch,
            "adc_bits_choices": adc_bits_choices,
            "gpu": torch.cuda.get_device_name(0),
        },
    )
    _json(hist_path, history)
    return model, history, train_seconds


def fit_temperature_scaling(logits_cal, truth_cal):
    """Ajusta un escalar de temperatura T > 0 sobre val_cal minimizando NLL."""
    logits_t = torch.from_numpy(logits_cal.astype(np.float32))
    targets_t = torch.from_numpy(truth_cal.astype(np.int64))
    log_temp = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        temp = torch.exp(log_temp).clamp(min=0.05, max=10.0)
        loss = F.cross_entropy(logits_t / temp, targets_t)
        loss.backward()
        return loss

    opt.step(closure)
    temp = float(torch.exp(log_temp).clamp(min=0.05, max=10.0).item())
    return temp


def calibrated_probs_from_logits(logits, temperature):
    scaled = logits / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/paper_research")
    parser.add_argument("--epochs-main", type=int, default=85)
    parser.add_argument("--epochs-adc", type=int, default=55)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--synth-batches", type=int, default=20)
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "Se requiere GPU CUDA para exp12_paper_research."

    # 1. Cargar splits de data/adaptive_dataset y data/e2_dataset (solo train para ampliar base real)
    adap_root = Path("data/adaptive_dataset")
    e2_root = Path("data/e2_dataset")
    d_train = load_classification_split(adap_root, "train", 512, allow_shift=True)
    d_val = load_classification_split(adap_root, "val", 512, allow_shift=True)
    d_test = load_classification_split(adap_root, "test", 512, allow_shift=True)
    d_shift = load_classification_split(adap_root, "test_shift", 512, allow_shift=True)
    e2_train = load_classification_split(e2_root, "train", 512, allow_shift=False)

    # Verificar disjuntividad estricta de padres entre todos los splits de evaluación y el pool de train
    eval_parents = (
        {r["parent_waveform_id"] for r in d_val["evaluation_rows"]}
        | {r["parent_waveform_id"] for r in d_test["evaluation_rows"]}
        | {r["parent_waveform_id"] for r in d_shift["evaluation_rows"]}
    )
    train_parents = (
        {r["parent_waveform_id"] for r in d_train["evaluation_rows"]}
        | {r["parent_waveform_id"] for r in e2_train["evaluation_rows"]}
    )
    assert not (train_parents & eval_parents), "Fuga detectada entre train y val/test/test_shift."

    # Partición idéntica de val en val_cal (1200) y val_select (1200)
    seen = defaultdict(int)
    cal_idx, select_idx = [], []
    for i, row in enumerate(d_val["evaluation_rows"]):
        key = (row["modulation"], row["snr_nominal_db"])
        (cal_idx if seen[key] < 25 else select_idx).append(i)
        seen[key] += 1
    cal_idx = np.asarray(cal_idx, dtype=int)
    select_idx = np.asarray(select_idx, dtype=int)
    assert len(cal_idx) == 1200 and len(select_idx) == 1200 and not (set(cal_idx) & set(select_idx))

    # Preparar tensores en GPU para entrenamiento de expertos profundos
    iq_train_combined = np.concatenate([d_train["iq"][:, :256], e2_train["iq"][:, :256]], axis=0)
    y_train_combined = np.concatenate([d_train["targets"], e2_train["targets"]], axis=0)
    x_train_gpu = torch.from_numpy(preprocess(iq_train_combined)).cuda()
    y_train_gpu = torch.from_numpy(y_train_combined).long().cuda()
    x_val_cal_gpu = torch.from_numpy(preprocess(d_val["iq"][cal_idx, :256])).cuda()
    y_val_cal_gpu = torch.from_numpy(d_val["targets"][cal_idx]).long().cuda()

    synth = GPUChannelSynthesizer(device="cuda", sps=8, span=10, fs_hz=8000.0)

    # 2. Entrenar el experto principal MultiDomainAMCNet (Experimento A / Acción a2) y variantes ADC (Fase 10)
    model_clean, hist_clean, time_clean = train_gpu_expert(
        "multidomain_clean_a2",
        x_train_gpu,
        y_train_gpu,
        x_val_cal_gpu,
        y_val_cal_gpu,
        synth,
        out,
        epochs=args.epochs_main,
        batch_size=args.batch_size,
        synth_batches_per_epoch=args.synth_batches,
        adc_bits_choices=None,
        seed=20261111,
        force_retrain=args.force_retrain,
    )

    model_adc8, hist_adc8, time_adc8 = train_gpu_expert(
        "multidomain_adc8",
        x_train_gpu,
        y_train_gpu,
        x_val_cal_gpu,
        y_val_cal_gpu,
        synth,
        out,
        epochs=args.epochs_adc,
        batch_size=args.batch_size,
        synth_batches_per_epoch=16,
        adc_bits_choices=(8,),
        seed=20261112,
        force_retrain=args.force_retrain,
    )

    model_adc6, hist_adc6, time_adc6 = train_gpu_expert(
        "multidomain_adc6",
        x_train_gpu,
        y_train_gpu,
        x_val_cal_gpu,
        y_val_cal_gpu,
        synth,
        out,
        epochs=args.epochs_adc,
        batch_size=args.batch_size,
        synth_batches_per_epoch=16,
        adc_bits_choices=(6,),
        seed=20261113,
        force_retrain=args.force_retrain,
    )

    model_mixed, hist_mixed, time_mixed = train_gpu_expert(
        "multidomain_adc_mixed",
        x_train_gpu,
        y_train_gpu,
        x_val_cal_gpu,
        y_val_cal_gpu,
        synth,
        out,
        epochs=args.epochs_adc,
        batch_size=args.batch_size,
        synth_batches_per_epoch=16,
        adc_bits_choices=(None, 8, 6),
        seed=20261114,
        force_retrain=args.force_retrain,
    )

    # 3. Cargar modelos convencionales congelados (a0: SVM256, a1: SVM512) y SmallCNN previo para referencia
    svm256 = load_model("results/fase5/conventional_256.joblib")["pipeline"]
    svm512 = load_model("results/fase5/conventional_512.joblib")["pipeline"]

    # 4. Medir tiempos reales de inferencia en batch=1 (con warm-up) para a0, a1, a2 y el gate LBAP
    print("[Benchmark] Midiendo latencias reales batch=1...", flush=True)
    sample_iqs = d_val["iq"][cal_idx[:96]]
    with threadpool_limits(limits=1):
        # Warm-up
        for row_iq in sample_iqs[:8]:
            _ = svm256.predict(row_iq[None, :256])
            _ = svm512.predict(row_iq[None, :512])
            with torch.inference_mode():
                _ = model_clean(torch.from_numpy(preprocess(row_iq[None, :256])).cuda())

        lat_a0, lat_a1, lat_a2 = [], [], []
        for _ in range(3):
            for row_iq in sample_iqs:
                t0 = time.perf_counter_ns()
                _ = svm256.predict(row_iq[None, :256])
                lat_a0.append((time.perf_counter_ns() - t0) / 1e6)

                t0 = time.perf_counter_ns()
                _ = svm512.predict(row_iq[None, :512])
                lat_a1.append((time.perf_counter_ns() - t0) / 1e6)

                t0 = time.perf_counter_ns()
                with torch.inference_mode():
                    xt = torch.from_numpy(preprocess(row_iq[None, :256])).to("cuda", non_blocking=True)
                    _ = int(model_clean(xt).argmax(dim=1).item())
                lat_a2.append((time.perf_counter_ns() - t0) / 1e6)

    base_ms = [float(np.mean(lat_a0)), float(np.mean(lat_a1)), float(np.mean(lat_a2))]
    latency_summary = {
        "a0_svm256_ms": {
            "mean": float(np.mean(lat_a0)),
            "median": float(np.median(lat_a0)),
            "p95": float(np.percentile(lat_a0, 95)),
        },
        "a1_svm512_ms": {
            "mean": float(np.mean(lat_a1)),
            "median": float(np.median(lat_a1)),
            "p95": float(np.percentile(lat_a1, 95)),
        },
        "a2_multidomain_gpu_ms": {
            "mean": float(np.mean(lat_a2)),
            "median": float(np.median(lat_a2)),
            "p95": float(np.percentile(lat_a2, 95)),
        },
    }
    print(f"[Benchmark] Latencias medias batch=1 (ms): a0={base_ms[0]:.3f}, a1={base_ms[1]:.3f}, a2={base_ms[2]:.3f}", flush=True)

    # 5. Calcular predicciones base y observaciones de prefijo en train, val, test y test_shift
    splits_data = {"train": d_train, "val": d_val, "test": d_test, "test_shift": d_shift}
    cache_splits = {}
    for s_name, d_split in splits_data.items():
        cache_file = out / f"{s_name}_research_cache.npz"
        if not args.force_retrain and cache_file.exists():
            with np.load(cache_file) as f:
                cache_splits[s_name] = {k: f[k].copy() for k in f.files}
            continue
        print(f"[Caché] Evaluando acciones base en split={s_name}...", flush=True)
        obs = observe_prefix(d_split["iq"][:, :256], svm256)
        pred_a0 = obs["initial"]
        pred_a1 = svm512.predict(d_split["iq"][:, :512])
        logits_a2, probs_a2 = predict_multidomain(model_clean, d_split["iq"][:, :256], device="cuda")
        pred_a2 = logits_a2.argmax(axis=1)
        snr_blind = estimate_blind_spectral_snr_db(d_split["iq"][:, :256])
        preds = np.column_stack([pred_a0, pred_a1, pred_a2])
        payload = dict(
            **obs,
            predictions=preds,
            logits_a2=logits_a2,
            probs_a2=probs_a2,
            snr_blind=snr_blind,
            truth=d_split["targets"],
        )
        np.savez(cache_file, **payload)
        cache_splits[s_name] = payload

    ct = cache_splits["train"]
    cv = cache_splits["val"]

    # 6. Calibración probabilística en val_cal (sin tocar val_select ni test)
    calibrator_a0 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, random_state=20261111))
    calibrator_a0.fit(cv["margin"][cal_idx], (cv["initial"][cal_idx] == cv["truth"][cal_idx]).astype(int))
    temp_a2 = fit_temperature_scaling(cv["logits_a2"][cal_idx], cv["truth"][cal_idx])

    gate_train_full, conf_train = build_extended_gate(ct, calibrator_a0)
    gate_val_full, conf_val = build_extended_gate(cv, calibrator_a0)

    # Medir latencia real del selector LBAP en batch=1
    dummy_sel = FastLBAPSelector.fit(gate_train_full, np.zeros((len(gate_train_full), 3)), epochs=5)
    t0 = time.perf_counter_ns()
    for i in range(200):
        _ = dummy_sel.choose(gate_val_full[i : i + 1])
    gate_overhead_ms = (time.perf_counter_ns() - t0) / (200 * 1e6)
    latency_summary["lbap_gate_ms"] = float(gate_overhead_ms)
    _json(out / "latency_benchmark.json", latency_summary)

    # 7. Entrenar políticas y ablaciones de Sección 14 (con lambda nominal = 0.0015 ms^-1)
    weight = 0.0015
    acq_vec, comp_vec = action_cost_ms(np.arange(3), base_ms, gate_ms=gate_overhead_ms, routed=True)
    routed_total_cost = acq_vec + comp_vec  # Costo total por acción cuando se enruta desde el prefijo 256
    # Si la política decide a0, no paga un segundo clasificador
    routed_total_cost[0] = 32.0 + base_ms[0] + gate_overhead_ms

    correct_train = (ct["predictions"] == ct["truth"][:, None]).astype(np.float64)
    utility_train = correct_train - weight * routed_total_cost[None, :]
    correct_val = (cv["predictions"] == cv["truth"][:, None]).astype(np.float64)
    utility_val = correct_val - weight * routed_total_cost[None, :]

    # Subconjuntos de entrada para la ablación de entrada de política (Sección 14)
    # I1: Solo confianza calibrada de a0 -> elige entre a0 y la mejor alternativa en val_select
    # I2: Solo SNR ciega + spectral flatness
    # I3: Confianza + estadísticas espectrales/cumulantes compactas (sin márgenes por clase)
    # I4: Vector completo LBAP (74 dimensiones)
    idx_snr_flat = [FEATURE_NAMES.index("spectral_flatness"), FEATURE_NAMES.index("spectral_entropy")]
    gate_train_i2 = np.column_stack([ct["snr_blind"], ct["gate"][:, idx_snr_flat]])
    gate_val_i2 = np.column_stack([cv["snr_blind"], cv["gate"][:, idx_snr_flat]])

    stat_cols = list(range(25)) + [FEATURE_NAMES.index("spectral_flatness"), FEATURE_NAMES.index("spectral_entropy")]
    gate_train_i3 = np.column_stack([conf_train, ct["snr_blind"], ct["gate"][:, stat_cols]])
    gate_val_i3 = np.column_stack([conf_val, cv["snr_blind"], cv["gate"][:, stat_cols]])

    # Ajustar umbral óptimo de confianza en val_select
    best_conf_u = -np.inf
    best_conf_th = 0.5
    best_conf_alt = 2
    for alt in (1, 2):
        for th in np.linspace(0.0, 1.0, 51):
            acts = np.where(conf_val[select_idx] < th, alt, 0)
            u = float(np.mean(utility_val[select_idx, acts]))
            if u > best_conf_u:
                best_conf_u, best_conf_th, best_conf_alt = u, float(th), int(alt)

    # Ajustar selectores LBAP para cada ablación de entrada y de espacio de acciones
    sel_i2_snr = FastLBAPSelector.fit(
        gate_train_i2, utility_train, gate_val_i2[select_idx], utility_val[select_idx], seed=20261111
    )
    sel_i3_conf_stats = FastLBAPSelector.fit(
        gate_train_i3, utility_train, gate_val_i3[select_idx], utility_val[select_idx], seed=20261111
    )
    sel_a0_a1 = FastLBAPSelector.fit(
        gate_train_full, utility_train, gate_val_full[select_idx], utility_val[select_idx],
        seed=20261111, allowed_actions=(0, 1)
    )
    sel_a0_a2 = FastLBAPSelector.fit(
        gate_train_full, utility_train, gate_val_full[select_idx], utility_val[select_idx],
        seed=20261111, allowed_actions=(0, 2)
    )
    sel_full_3actions = FastLBAPSelector.fit(
        gate_train_full, utility_train, gate_val_full[select_idx], utility_val[select_idx],
        seed=20261111, allowed_actions=(0, 1, 2)
    )

    policy_names = (
        "always_a0_svm256",
        "always_a1_svm512",
        "always_a2_multidomain",
        "ablation_input_confidence",
        "ablation_input_snr_flatness",
        "ablation_input_conf_stats",
        "ablation_actions_a0_a1",
        "ablation_actions_a0_a2",
        "lbap_full_a0_a1_a2",
    )

    def evaluate_policy_actions(split_cache, p_name):
        n = len(split_cache["truth"])
        if p_name == "always_a0_svm256":
            return np.zeros(n, dtype=int), False
        if p_name == "always_a1_svm512":
            return np.ones(n, dtype=int), False
        if p_name == "always_a2_multidomain":
            return np.full(n, 2, dtype=int), False

        g_full, c_prob = build_extended_gate(split_cache, calibrator_a0)
        if p_name == "ablation_input_confidence":
            return np.where(c_prob < best_conf_th, best_conf_alt, 0), True
        if p_name == "ablation_input_snr_flatness":
            g_i2 = np.column_stack([split_cache["snr_blind"], split_cache["gate"][:, idx_snr_flat]])
            return sel_i2_snr.choose(g_i2), True
        if p_name == "ablation_input_conf_stats":
            g_i3 = np.column_stack([c_prob, split_cache["snr_blind"], split_cache["gate"][:, stat_cols]])
            return sel_i3_conf_stats.choose(g_i3), True
        if p_name == "ablation_actions_a0_a1":
            return sel_a0_a1.choose(g_full), True
        if p_name == "ablation_actions_a0_a2":
            return sel_a0_a2.choose(g_full), True
        if p_name == "lbap_full_a0_a1_a2":
            return sel_full_3actions.choose(g_full), True
        raise ValueError(f"Política desconocida: {p_name}")

    # 8. Evaluar todas las políticas y ablaciones en val_select, test y test_shift
    results_by_split = {}
    bootstrap_vs_a0 = {}
    bootstrap_vs_a1 = {}
    by_snr_rows = []

    for split_name in ("val_select", "test", "test_shift"):
        if split_name == "val_select":
            raw_c = cache_splits["val"]
            c = {k: v[select_idx] for k, v in raw_c.items()}
            eval_rows = [d_val["evaluation_rows"][i] for i in select_idx]
        else:
            c = cache_splits[split_name]
            eval_rows = splits_data[split_name]["evaluation_rows"]

        snr_arr = np.asarray([float(r["snr_nominal_db"]) for r in eval_rows])
        truth_arr = c["truth"]
        results_by_split[split_name] = {}

        # Oracle upper bounds (descriptivos, nunca ejecutables en inferencia)
        c0 = c["predictions"][:, 0] == truth_arr
        c1 = c["predictions"][:, 1] == truth_arr
        c2 = c["predictions"][:, 2] == truth_arr
        results_by_split[split_name]["oracle_bounds_descriptive"] = {
            "oracle_a0_a1": float(np.mean(c0 | c1)),
            "oracle_a0_a2": float(np.mean(c0 | c2)),
            "oracle_a0_a1_a2": float(np.mean(c0 | c1 | c2)),
        }

        policy_arrays = {}
        for p_name in policy_names:
            actions, is_routed = evaluate_policy_actions(c, p_name)
            preds = c["predictions"][np.arange(len(truth_arr)), actions]
            acq_ms, comp_ms = action_cost_ms(actions, base_ms, gate_ms=gate_overhead_ms if is_routed else 0.0, routed=is_routed)
            tot_ms = acq_ms + comp_ms
            cls_m = classification_metrics(truth_arr, preds)
            utility_val_split = float(np.mean(preds == truth_arr) - weight * np.mean(tot_ms))

            results_by_split[split_name][p_name] = {
                **cls_m,
                "acquisition_mean_ms": float(np.mean(acq_ms)),
                "compute_mean_ms": float(np.mean(comp_ms)),
                "total_mean_ms": float(np.mean(tot_ms)),
                "utility": utility_val_split,
                "actions": {str(a): int(np.sum(actions == a)) for a in range(3)},
                "action_fractions": {str(a): float(np.mean(actions == a)) for a in range(3)},
            }
            policy_arrays[p_name] = (actions, preds, tot_ms)

            if split_name in ("test", "test_shift"):
                for snr_level in sorted(np.unique(snr_arr)):
                    m_snr = snr_arr == snr_level
                    by_snr_rows.append(
                        {
                            "split": split_name,
                            "policy": p_name,
                            "snr_db": float(snr_level),
                            "n": int(np.sum(m_snr)),
                            "accuracy": float(np.mean(preds[m_snr] == truth_arr[m_snr])),
                            "total_ms": float(np.mean(tot_ms[m_snr])),
                            "frac_a0": float(np.mean(actions[m_snr] == 0)),
                            "frac_a1": float(np.mean(actions[m_snr] == 1)),
                            "frac_a2": float(np.mean(actions[m_snr] == 2)),
                        }
                    )

        if split_name in ("test", "test_shift"):
            bootstrap_vs_a0[split_name] = {}
            bootstrap_vs_a1[split_name] = {}
            _, pred_0, time_0 = policy_arrays["always_a0_svm256"]
            _, pred_1, time_1 = policy_arrays["always_a1_svm512"]
            for p_name in policy_names[3:]:
                _, pred_p, time_p = policy_arrays[p_name]
                bootstrap_vs_a0[split_name][p_name] = bootstrap_difference(
                    truth_arr, snr_arr, pred_p == truth_arr, pred_0 == truth_arr, time_p, time_0, weight
                )
                bootstrap_vs_a1[split_name][p_name] = bootstrap_difference(
                    truth_arr, snr_arr, pred_p == truth_arr, pred_1 == truth_arr, time_p, time_1, weight
                )

    _json(out / "ablation_and_policy_metrics.json", results_by_split)
    _json(out / "bootstrap_vs_a0.json", bootstrap_vs_a0)
    _json(out / "bootstrap_vs_a1.json", bootstrap_vs_a1)
    _csv(out / "policy_by_snr.csv", by_snr_rows)

    # Exportar desglose por clase y taxonomía contrafactual C11/C01/C10/C00 en test y test_shift
    diag_report = {}
    for split_name in ("test", "test_shift"):
        c = cache_splits[split_name]
        eval_rows = splits_data[split_name]["evaluation_rows"]
        snr_arr = np.asarray([float(r["snr_nominal_db"]) for r in eval_rows])
        truth_arr = c["truth"]
        c0 = c["predictions"][:, 0] == truth_arr
        c1 = c["predictions"][:, 1] == truth_arr
        c2 = c["predictions"][:, 2] == truth_arr
        acts_lbap, _ = evaluate_policy_actions(c, "lbap_full_a0_a1_a2")
        preds_lbap = c["predictions"][np.arange(len(truth_arr)), acts_lbap]
        per_class = {}
        high_snr_mask = snr_arr >= 10.0
        for cls_idx, cls_name in enumerate(CLASSES):
            m_all = truth_arr == cls_idx
            m_hi = m_all & high_snr_mask
            per_class[cls_name] = {
                "all_snr": {
                    "a0": float(np.mean(c0[m_all])),
                    "a1": float(np.mean(c1[m_all])),
                    "a2": float(np.mean(c2[m_all])),
                    "lbap": float(np.mean(preds_lbap[m_all] == truth_arr[m_all])),
                },
                "snr_ge_10db": {
                    "a0": float(np.mean(c0[m_hi])),
                    "a1": float(np.mean(c1[m_hi])),
                    "a2": float(np.mean(c2[m_hi])),
                    "lbap": float(np.mean(preds_lbap[m_hi] == truth_arr[m_hi])),
                },
            }
        diag_report[split_name] = {
            "per_class": per_class,
            "counterfactual_a0_vs_a1": {
                "C11_both_ok": int(np.sum(c0 & c1)),
                "C01_recoverable": int(np.sum((~c0) & c1)),
                "C10_overthinking": int(np.sum(c0 & (~c1))),
                "C00_irrecoverable": int(np.sum((~c0) & (~c1))),
            },
            "counterfactual_a0_vs_a2": {
                "C11_both_ok": int(np.sum(c0 & c2)),
                "C01_recoverable": int(np.sum((~c0) & c2)),
                "C10_overthinking": int(np.sum(c0 & (~c2))),
                "C00_irrecoverable": int(np.sum((~c0) & (~c2))),
            },
            "counterfactual_a0_vs_best_alt": {
                "C11_both_ok": int(np.sum(c0 & (c1 | c2))),
                "C01_recoverable": int(np.sum((~c0) & (c1 | c2))),
                "C10_only_a0_ok": int(np.sum(c0 & (~c1) & (~c2))),
                "C00_irrecoverable": int(np.sum((~c0) & (~c1) & (~c2))),
            },
        }
    _json(out / "per_class_and_counterfactual.json", diag_report)

    # 9. Barrido de Pareto exactitud-costo variando lambda y frecuencia de muestreo fs
    pareto_rows = []
    for fs_khz in (8.0, 32.0, 128.0, 1000.0):
        acq_256 = 256.0 / fs_khz  # ms
        acq_512 = 512.0 / fs_khz  # ms
        for lam in (0.0, 0.0002, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005):
            cost_vec = np.array(
                [
                    acq_256 + base_ms[0] + gate_overhead_ms,
                    acq_512 + base_ms[0] + gate_overhead_ms + base_ms[1],
                    acq_256 + base_ms[0] + gate_overhead_ms + base_ms[2],
                ]
            )
            u_tr = correct_train - lam * cost_vec[None, :]
            u_val = correct_val - lam * cost_vec[None, :]
            sel_lam = FastLBAPSelector.fit(
                gate_train_full, u_tr, gate_val_full[select_idx], u_val[select_idx],
                epochs=250, seed=20261111, allowed_actions=(0, 1, 2)
            )
            for sp in ("test", "test_shift"):
                c = cache_splits[sp]
                g_sp, _ = build_extended_gate(c, calibrator_a0)
                acts = sel_lam.choose(g_sp)
                preds = c["predictions"][np.arange(len(c["truth"])), acts]
                tot_ms = cost_vec[acts]
                pareto_rows.append(
                    {
                        "fs_khz": fs_khz,
                        "lambda_per_ms": lam,
                        "split": sp,
                        "accuracy": float(np.mean(preds == c["truth"])),
                        "total_ms": float(np.mean(tot_ms)),
                        "utility": float(np.mean(preds == c["truth"]) - lam * np.mean(tot_ms)),
                        "frac_a0": float(np.mean(acts == 0)),
                        "frac_a1": float(np.mean(acts == 1)),
                        "frac_a2": float(np.mean(acts == 2)),
                    }
                )
    _csv(out / "pareto_sweep_lambda_fs.csv", pareto_rows)

    # 10. Evaluación exhaustiva de Cuantización ADC Virtual (Fase 10: vistas inf, 8, 6 y no vistas 5, 4 bits)
    print("[Fase 10 ADC] Evaluando robustez a cuantización ADC y entrenamiento mixto...", flush=True)
    adc_models = {
        "ExpA_clean_ref": model_clean,
        "ExpB_quant8": model_adc8,
        "ExpB_quant6": model_adc6,
        "ExpC_mixed_inf_8_6": model_mixed,
    }
    adc_temperatures = {}
    for m_name, m_obj in adc_models.items():
        logits_cal, _ = predict_multidomain(m_obj, d_val["iq"][cal_idx, :256], device="cuda")
        adc_temperatures[m_name] = fit_temperature_scaling(logits_cal, d_val["targets"][cal_idx])

    adc_rows = []
    for sp_name, d_sp in [("test", d_test), ("test_shift", d_shift)]:
        iq256_ref = d_sp["iq"][:, :256]
        iq512_ref = d_sp["iq"][:, :512]
        truth_sp = d_sp["targets"]
        for bits in (None, 8, 6, 5, 4):
            q256, diag256 = quantize_iq(iq256_ref, bits=bits, full_scale=2.5, return_diagnostics=True)
            q512 = quantize_iq(iq512_ref, bits=bits, full_scale=2.5, return_diagnostics=False)
            sqnr_info = (
                {"sqnr_db": float("inf"), "quantization_error_power": 0.0}
                if bits is None
                else evaluate_sqnr_db(iq256_ref, q256)
            )
            bits_label = "inf_float32" if bits is None else f"{bits}bit"
            is_unseen_bits = bits in (5, 4)

            # Evaluar SVM256 y SVM512 bajo cuantización ADC
            obs_q = observe_prefix(q256, svm256)
            pred_svm256 = obs_q["initial"]
            pred_svm512 = svm512.predict(q512)
            for conv_name, conv_pred in [("SVM256_ref", pred_svm256), ("SVM512_ref", pred_svm512)]:
                cm = classification_metrics(truth_sp, conv_pred)
                adc_rows.append(
                    {
                        "split": sp_name,
                        "adc_bits": bits_label,
                        "unseen_resolution": is_unseen_bits,
                        "model": conv_name,
                        "sqnr_db_offline": sqnr_info["sqnr_db"],
                        "clipping_rate_iq": diag256["clipping_rate_iq"],
                        "sample_saturation_rate": diag256["sample_saturation_rate"],
                        "accuracy": cm["accuracy"],
                        "macro_f1": cm["macro_f1"],
                        "ece10": None,
                        "brier": None,
                        "nll": None,
                    }
                )

            # Evaluar los 4 modelos profundos GPU (ExpA, ExpB8, ExpB6, ExpC) y la política LBAP acoplada
            for m_name, m_obj in adc_models.items():
                logits_q, _ = predict_multidomain(m_obj, q256, device="cuda")
                preds_q = logits_q.argmax(axis=1)
                probs_cal = calibrated_probs_from_logits(logits_q, adc_temperatures[m_name])
                max_prob = probs_cal.max(axis=1)
                cal_m = calibration_metrics(max_prob, (preds_q == truth_sp).astype(float))
                cm = classification_metrics(truth_sp, preds_q)
                adc_rows.append(
                    {
                        "split": sp_name,
                        "adc_bits": bits_label,
                        "unseen_resolution": is_unseen_bits,
                        "model": m_name,
                        "sqnr_db_offline": sqnr_info["sqnr_db"],
                        "clipping_rate_iq": diag256["clipping_rate_iq"],
                        "sample_saturation_rate": diag256["sample_saturation_rate"],
                        "accuracy": cm["accuracy"],
                        "macro_f1": cm["macro_f1"],
                        "ece10": cal_m["ece10"],
                        "brier": cal_m["brier"],
                        "nll": cal_m["nll"],
                    }
                )

            # Evaluar el selector LBAP (usando ExpA vs ExpC como acción a2) bajo cuantización ADC
            gate_q, _ = build_extended_gate(obs_q, calibrator_a0)
            acts_q = sel_full_3actions.choose(gate_q)
            for m_tag, m_obj in [("LBAP_with_ExpA", model_clean), ("LBAP_with_ExpC_mixed", model_mixed)]:
                logits_q, _ = predict_multidomain(m_obj, q256, device="cuda")
                preds_stack = np.column_stack([pred_svm256, pred_svm512, logits_q.argmax(axis=1)])
                routed_preds = preds_stack[np.arange(len(truth_sp)), acts_q]
                cm = classification_metrics(truth_sp, routed_preds)
                adc_rows.append(
                    {
                        "split": sp_name,
                        "adc_bits": bits_label,
                        "unseen_resolution": is_unseen_bits,
                        "model": m_tag,
                        "sqnr_db_offline": sqnr_info["sqnr_db"],
                        "clipping_rate_iq": diag256["clipping_rate_iq"],
                        "sample_saturation_rate": diag256["sample_saturation_rate"],
                        "accuracy": cm["accuracy"],
                        "macro_f1": cm["macro_f1"],
                        "ece10": None,
                        "brier": None,
                        "nll": None,
                    }
                )

    _csv(out / "adc_quantization_experiment.csv", adc_rows)

    # 11. Calibración detallada de a0 (SVM calibrado) vs a2 (MultiDomainAMCNet antes y después de Temp Scaling)
    cal_report = {}
    for sp_name in ("test", "test_shift"):
        c = cache_splits[sp_name]
        truth_sp = c["truth"]
        # a0 calibrado por margen logístico
        prob_a0 = calibrator_a0.predict_proba(c["margin"])[:, 1]
        corr_a0 = (c["predictions"][:, 0] == truth_sp).astype(float)
        # a2 max-softmax sin calibrar vs temperature-scaled
        prob_a2_raw = c["probs_a2"].max(axis=1)
        prob_a2_cal = calibrated_probs_from_logits(c["logits_a2"], temp_a2).max(axis=1)
        corr_a2 = (c["predictions"][:, 2] == truth_sp).astype(float)
        cal_report[sp_name] = {
            "a0_logistic_margin": calibration_metrics(prob_a0, corr_a0),
            "a2_multidomain_uncalibrated_max_softmax": calibration_metrics(prob_a2_raw, corr_a2),
            "a2_multidomain_temperature_scaled": {
                **calibration_metrics(prob_a2_cal, corr_a2),
                "temperature": temp_a2,
            },
        }
    _json(out / "calibration_comparison.json", cal_report)

    # 12. Generar figuras de calidad publicación (IEEE 300 DPI)
    _generate_ieee_figures(out, results_by_split, by_snr_rows, pareto_rows, adc_rows)

    summary_manifest = {
        "completed": True,
        "gpu_device": torch.cuda.get_device_name(0),
        "train_seconds": {
            "multidomain_clean_a2": time_clean,
            "multidomain_adc8": time_adc8,
            "multidomain_adc6": time_adc6,
            "multidomain_adc_mixed": time_mixed,
            "total_gpu_seconds": time_clean + time_adc8 + time_adc6 + time_mixed,
        },
        "latency_ms": latency_summary,
        "temperature_a2": temp_a2,
        "confidence_threshold_a0": best_conf_th,
    }
    _json(out / "experiment_manifest.json", summary_manifest)
    print("\n[COMPLETADO] Resumen de métricas principales (test y test_shift):", flush=True)
    for sp in ("test", "test_shift"):
        print(f"\n--- {sp.upper()} ---", flush=True)
        for p_name in policy_names:
            m = results_by_split[sp][p_name]
            print(
                f"  {p_name:28s}: Acc={m['accuracy']:.4f} | F1={m['macro_f1']:.4f} | "
                f"Total={m['total_mean_ms']:.2f}ms | U={m['utility']:.4f} | actions={m['actions']}",
                flush=True,
            )


def _generate_ieee_figures(out, results_by_split, by_snr_rows, pareto_rows, adc_rows):
    """Genera las figuras de alta resolución (300 DPI) para el artículo IEEE."""
    # Figura 1: Curvas de Accuracy vs SNR y distribución de acciones por SNR en test y test_shift
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    highlight_policies = [
        ("always_a0_svm256", "Always $a_0$ (SVM-256)", "o--"),
        ("always_a1_svm512", "Always $a_1$ (SVM-512)", "s--"),
        ("always_a2_multidomain", "Always $a_2$ (MultiDomainNet-256)", "^-"),
        ("lbap_full_a0_a1_a2", "Proposed LBAP ($\\{a_0,a_1,a_2\\}$)", "D-"),
    ]
    for col, sp in enumerate(("test", "test_shift")):
        ax_acc = axes[0, col]
        for key, label, style in highlight_policies:
            sub = sorted([r for r in by_snr_rows if r["split"] == sp and r["policy"] == key], key=lambda r: r["snr_db"])
            ax_acc.plot([r["snr_db"] for r in sub], [r["accuracy"] for r in sub], style, linewidth=2, label=label)
        ax_acc.set_title(f"Classification Accuracy vs. SNR ({'In-Distribution' if sp == 'test' else 'OOD Channel Shift'})")
        ax_acc.set_xlabel("Nominal SNR (dB)")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0.1, 1.02)
        ax_acc.grid(alpha=0.35)
        ax_acc.legend(fontsize=8.5)

        ax_act = axes[1, col]
        sub_lbap = sorted(
            [r for r in by_snr_rows if r["split"] == sp and r["policy"] == "lbap_full_a0_a1_a2"],
            key=lambda r: r["snr_db"],
        )
        snrs = [str(int(r["snr_db"])) for r in sub_lbap]
        f0 = np.array([r["frac_a0"] for r in sub_lbap])
        f1 = np.array([r["frac_a1"] for r in sub_lbap])
        f2 = np.array([r["frac_a2"] for r in sub_lbap])
        ax_act.bar(snrs, f0, label="$a_0$: Fast SVM-256", color="#4c72b0")
        ax_act.bar(snrs, f1, bottom=f0, label="$a_1$: Extend N=512", color="#55a868")
        ax_act.bar(snrs, f2, bottom=f0 + f1, label="$a_2$: MultiDomainNet-256", color="#c44e52")
        ax_act.set_title(f"LBAP Action Allocation by SNR ({'ID Test' if sp == 'test' else 'OOD Shift'})")
        ax_act.set_xlabel("Nominal SNR (dB)")
        ax_act.set_ylabel("Action Selection Fraction")
        ax_act.set_ylim(0, 1.05)
        ax_act.grid(axis="y", alpha=0.3)
        ax_act.legend(fontsize=8.5)

    fig.savefig(out / "fig1_snr_accuracy_and_actions.png", dpi=300)
    plt.close(fig)

    # Figura 2: Frontera de Pareto Riesgo-Costo y Ablación de Acciones/Entradas
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.8), constrained_layout=True)
    offsets_by_split = {
        "test": {
            "always_a0_svm256": (6, 3),
            "always_a1_svm512": (-46, -12),
            "always_a2_multidomain": (6, -9),
            "ablation_input_confidence": (6, 4),
            "ablation_input_snr_flatness": (6, 2),
            "ablation_input_conf_stats": (6, 3),
            "ablation_actions_a0_a1": (6, 3),
            "ablation_actions_a0_a2": (-62, 5),
            "lbap_full_a0_a1_a2": (-72, -13),
        },
        "test_shift": {
            "always_a0_svm256": (6, 3),
            "always_a1_svm512": (-46, -12),
            "always_a2_multidomain": (6, 5),
            "ablation_input_confidence": (6, -10),
            "ablation_input_snr_flatness": (6, 3),
            "ablation_input_conf_stats": (6, 3),
            "ablation_actions_a0_a1": (6, 3),
            "ablation_actions_a0_a2": (-56, 6),
            "lbap_full_a0_a1_a2": (-10, -14),
        },
    }
    legend_handles = {}
    for ax, sp in zip(axes, ("test", "test_shift")):
        m_dict = results_by_split[sp]
        for p_key, m in m_dict.items():
            if p_key.startswith("oracle"):
                continue
            err = 1.0 - m["accuracy"]
            cost = m["total_mean_ms"]
            marker = "D" if p_key == "lbap_full_a0_a1_a2" else ("o" if p_key.startswith("always") else "s")
            size = 90 if p_key == "lbap_full_a0_a1_a2" else 55
            sc = ax.scatter(cost, err, s=size, marker=marker, label=p_key, zorder=4)
            legend_handles.setdefault(p_key, sc)
            dx, dy = offsets_by_split[sp].get(p_key, (5, 3))
            short_lbl = p_key.replace("ablation_", "").replace("always_", "")
            ax.annotate(short_lbl, (cost, err), fontsize=7.6, xytext=(dx, dy), textcoords="offset points")
        # Curva de barrido lambda para fs=8 kHz
        sweep = sorted([r for r in pareto_rows if r["split"] == sp and r["fs_khz"] == 8.0], key=lambda r: r["total_ms"])
        ln, = ax.plot([r["total_ms"] for r in sweep], [1.0 - r["accuracy"] for r in sweep], "k--", alpha=0.65, label="LBAP Pareto Sweep ($\\lambda$)", zorder=2)
        legend_handles.setdefault("LBAP Pareto Sweep ($\\lambda$)", ln)
        ax.set_xlabel("Total Latency: Virtual Acquisition $N/f_s$ + Inference (ms)")
        ax.set_ylabel("Classification Error Rate ($1 - \\mathrm{Acc}$)")
        ax.set_title(f"Risk-Latency Pareto Frontier ({'ID Test' if sp == 'test' else 'OOD Shift'})")
        ax.margins(x=0.08, y=0.10)
        ax.grid(alpha=0.35)
    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="outside lower center",
        ncol=5,
        fontsize=7.6,
        frameon=True,
    )
    fig.savefig(out / "fig2_pareto_ablation_frontier.png", dpi=300)
    plt.close(fig)

    # Figura 3: Robustez a Cuantización ADC Virtual (Fase 10)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    bit_order = ["inf_float32", "8bit", "6bit", "5bit", "4bit"]
    bit_labels = ["Float32\n(Ref)", "8-bit\n(Seen)", "6-bit\n(Seen)", "5-bit\n(Unseen)", "4-bit\n(Unseen)"]
    models_to_plot = [
        ("SVM256_ref", "SVM-256 (Baseline $a_0$)", "o--"),
        ("ExpA_clean_ref", "MultiDomainNet (Exp A: Clean Train)", "s-"),
        ("ExpB_quant8", "MultiDomainNet (Exp B: 8-bit Train)", "^--"),
        ("ExpB_quant6", "MultiDomainNet (Exp B: 6-bit Train)", "v--"),
        ("ExpC_mixed_inf_8_6", "MultiDomainNet (Exp C: Mixed Train)", "D-"),
        ("LBAP_with_ExpC_mixed", "Adaptive LBAP + Exp C Mixed", "P-"),
    ]
    for ax, sp in zip(axes, ("test", "test_shift")):
        for m_key, m_label, style in models_to_plot:
            vals = [
                next(r["accuracy"] for r in adc_rows if r["split"] == sp and r["model"] == m_key and r["adc_bits"] == b)
                for b in bit_order
            ]
            ax.plot(range(len(bit_order)), vals, style, linewidth=2, label=m_label)
        ax.set_xticks(range(len(bit_order)))
        ax.set_xticklabels(bit_labels)
        ax.set_ylabel("Classification Accuracy")
        ax.set_title(f"ADC Quantization Robustness ({'ID Test' if sp == 'test' else 'OOD Shift'})")
        ax.grid(alpha=0.35)
        ax.legend(fontsize=8)
    fig.savefig(out / "fig3_adc_quantization_robustness.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
