"""Pruebas unitarias para MultiDomainAMCNet, GPUChannelSynthesizer y FastLBAPSelector."""
import time
import numpy as np
import pytest
import torch

from src.research_models import (
    FastLBAPSelector,
    GPUChannelSynthesizer,
    MultiDomainAMCNet,
    extract_multidomain_tensors,
)


def test_multidomain_batch_independence_and_cuda_forward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(20261111)
    model = MultiDomainAMCNet(num_classes=6, width_mult=1.0).to(device).eval()
    x = torch.randn(8, 2, 256, device=device)
    with torch.inference_mode():
        logits_batch = model(x)
        logits_single = torch.cat([model(x[i : i + 1]) for i in range(4)], dim=0)
    assert logits_batch.shape == (8, 6)
    assert torch.isfinite(logits_batch).all()
    # Independencia estricta por ventana (cero BatchNorm leakage, tolerancia float32/TF32)
    assert torch.allclose(logits_batch[:4], logits_single, atol=1e-3)


def test_multidomain_tensors_global_phase_invariance():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(123)
    x = torch.randn(4, 2, 256, device=device)
    # Rotar todas las muestras por un ángulo constante arbitrario phi = 1.234 rad
    phi = torch.tensor(1.2345, device=device)
    c, s = torch.cos(phi), torch.sin(phi)
    x_rot = torch.stack([x[:, 0] * c - x[:, 1] * s, x[:, 0] * s + x[:, 1] * c], dim=1)

    t1, sp1, _, st1 = extract_multidomain_tensors(x)
    t2, sp2, _, st2 = extract_multidomain_tensors(x_rot)
    # Los espectros y el vector de cumulantes/estadísticas son invariantes a rotación global de fase
    assert torch.allclose(sp1, sp2, atol=2e-3)
    assert torch.allclose(st1, st2, atol=2e-3)
    # Envolvente y productos diferenciales (canales 4..11) son invariantes a fase global
    assert torch.allclose(t1[:, 4:, :], t2[:, 4:, :], atol=2e-3)


def test_gpu_channel_synthesizer_shapes_and_quantization():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    synth = GPUChannelSynthesizer(device=device)
    x, y = synth.sample_batch(batch_size=60, n_samples=256, adc_bits_choices=(8, 6, None))
    assert x.shape == (60, 2, 256)
    assert y.shape == (60,)
    assert torch.isfinite(x).all()
    assert set(y.cpu().numpy().tolist()) == {0, 1, 2, 3, 4, 5}


def test_fast_lbap_selector_latency_and_action_constraints():
    rng = np.random.default_rng(7)
    gate_tr = rng.standard_normal((400, 71))
    u_tr = rng.standard_normal((400, 3))
    selector = FastLBAPSelector.fit(gate_tr, u_tr, gate_tr[:100], u_tr[:100], epochs=20)
    one_row = gate_tr[:1]
    # Warm-up
    for _ in range(10):
        selector.choose(one_row)
    t0 = time.perf_counter_ns()
    for _ in range(200):
        act = selector.choose(one_row)
    elapsed_ms = (time.perf_counter_ns() - t0) / (200 * 1e6)
    assert act.shape == (1,)
    assert int(act[0]) in (0, 1, 2)
    assert elapsed_ms < 0.5  # Ultra-rápido frente a ~18 ms de RandomForestRegressor

    # Restricción de subconjunto de acciones {a0, a2}
    sel_02 = FastLBAPSelector.fit(gate_tr, u_tr, allowed_actions=(0, 2), epochs=10)
    chosen_02 = sel_02.choose(gate_tr[:50])
    assert set(chosen_02.tolist()).issubset({0, 2})
