"""Modelos profundos multi-dominio informados por física en GPU y selector LBAP (Fases 9-10 / Paper IEEE).

Diseño estrictamente libre de fugas (zero-leakage):
- Opera únicamente sobre ventanas I/Q observables (N=256 o N=512).
- No recibe SNR verdadera, CFO verdadero, fase inicial ni taps de canal.
- Normalización exclusivamente por ventana individual + GroupNorm/LayerNorm (cero BatchNorm),
  garantizando que la inferencia en batch=1 y batch=K sea bit a bit idéntica.
- Incluye sintetizador físico nativo en GPU (GPUChannelSynthesizer) exclusivamente para
  generar realizaciones independientes ilimitadas durante el entrenamiento sin cuellos
  de botella de CPU/disco, manteniendo val, test y test_shift 100% sellados e intactos.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.waveform import rrc_filter


class SqueezeExcite1d(nn.Module):
    """Atención de canal tipo Squeeze-and-Excitation para representaciones 1D."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.net(x).unsqueeze(-1)
        return x * scale


class ResBlock1d(nn.Module):
    """Bloque residual 1D de alta capacidad con GroupNorm y Squeeze-and-Excitation."""
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        groups = 16 if out_ch % 16 == 0 else 8
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_ch)
        self.se = SqueezeExcite1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.silu(self.gn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.gn2(self.conv2(out))
        out = self.se(out)
        return F.silu(out + residual)


def _complex_cumulants_and_moments(z, eps=1e-7):
    """Calcula cumulantes de orden superior y resultantes de fase por ventana en GPU."""
    mag2 = z.real * z.real + z.imag * z.imag
    power = torch.clamp(mag2.mean(dim=-1, keepdim=True), min=eps)
    zn = z / torch.sqrt(power)
    env = torch.sqrt(torch.clamp(zn.real * zn.real + zn.imag * zn.imag, min=eps))

    z2 = zn * zn
    z4 = z2 * z2
    m20 = z2.mean(dim=-1)
    m40 = z4.mean(dim=-1)
    m42 = (env ** 4).mean(dim=-1)
    m63 = (env ** 6).mean(dim=-1)

    c20_abs = torch.abs(m20)
    c40_abs = torch.abs(m40 - 3.0 * (m20 * m20))
    c42_real = m42 - (c20_abs * c20_abs) - 2.0
    env_mean = env.mean(dim=-1)
    env_std = env.std(dim=-1, unbiased=False)
    env_kurt = m42 / torch.clamp(env.pow(2).mean(dim=-1).pow(2), min=eps)

    unit = zn / env
    u2 = unit * unit
    u4 = u2 * u2
    u8 = u4 * u4
    r1 = torch.abs(unit.mean(dim=-1))
    r2 = torch.abs(u2.mean(dim=-1))
    r4 = torch.abs(u4.mean(dim=-1))
    r8 = torch.abs(u8.mean(dim=-1))

    return torch.stack(
        [c20_abs, c40_abs, c42_real, m42, m63, env_mean, env_std, env_kurt, r1, r2, r4, r8],
        dim=-1,
    )


_MULTIDOMAIN_KERNEL_CACHE = {}


def _get_multidomain_kernels(device, sps=8, n_fft=512, n_samples=256):
    """Cachea filtros RRC matched-filter, ventana Hann y bancos en GPU sin sincronización."""
    key = (str(device), int(sps), int(n_fft), int(n_samples))
    cached = _MULTIDOMAIN_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    dev = torch.device(device)
    betas = (0.25, 0.35, 0.50)
    span_mf = 6
    taps_list = [rrc_filter(beta=b, sps=sps, span=span_mf).astype(np.float32) for b in betas]
    rrc_bank = torch.from_numpy(np.stack(taps_list, axis=0)).to(dev).unsqueeze(1)  # (3, 1, K)
    pad_mf = rrc_bank.shape[-1] // 2

    # Normalización de borde para evitar atenuación artificial en extremos de ventana finita
    ones = torch.ones((1, 1, n_samples), device=dev, dtype=torch.float32)
    edge_norms = []
    for i in range(len(betas)):
        h2 = (rrc_bank[i : i + 1] ** 2)
        e = F.conv1d(ones, h2, padding=pad_mf).clamp(min=1e-6).sqrt()
        edge_norms.append(e.squeeze(1))  # (1, N)
    edge_norm_bank = torch.cat(edge_norms, dim=0)  # (3, N)

    # Plantilla espectral RRC aproximada para bloqueo grueso de CFO inmune a ecos multicamino
    w_bins = max(int(round(n_fft / sps)), 8)
    pass_taps = np.hanning(max(w_bins + 4, 8)).astype(np.float32)
    pass_taps = np.clip(pass_taps + 0.35, 0.0, 1.0)
    pass_taps = (pass_taps / np.sum(pass_taps)).astype(np.float32)
    pass_kernel = torch.from_numpy(pass_taps).to(dev).view(1, 1, -1)

    hann = torch.hann_window(n_samples, periodic=True, device=dev, dtype=torch.float32).unsqueeze(0)
    fine_dfs = torch.linspace(-48.0, 48.0, 25, device=dev, dtype=torch.float32)
    idx_n = torch.arange(n_samples, device=dev, dtype=torch.float32).unsqueeze(0)
    q_levels = torch.tensor([0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.85, 0.95], device=dev, dtype=torch.float32)

    payload = {
        "rrc_bank": rrc_bank,
        "pad_mf": pad_mf,
        "edge_norm_bank": edge_norm_bank,
        "pass_kernel": pass_kernel,
        "pass_pad": pass_kernel.shape[-1] // 2,
        "hann": hann,
        "fine_dfs": fine_dfs,
        "idx_n": idx_n,
        "q_levels": q_levels,
    }
    _MULTIDOMAIN_KERNEL_CACHE[key] = payload
    return payload


def extract_multidomain_tensors(x, sps=8, eps=1e-7):
    """Transforma un batch (B, 2, N) en vistas temporal, espectral, polifásica y estadística en FP32 sobre GPU."""
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        return _extract_multidomain_tensors_fp32(x, sps=sps, eps=eps)


def _extract_multidomain_tensors_fp32(x, sps=8, eps=1e-7):
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError("Se espera tensor (B, 2, N) con canales I y Q.")
    bsz, _, n_samples = x.shape
    n_fft = max(512, 1 << int(np.ceil(np.log2(max(n_samples * 2, 256)))))
    kbuf = _get_multidomain_kernels(x.device, sps=sps, n_fft=n_fft, n_samples=n_samples)

    z_raw = torch.complex(x[:, 0, :].float(), x[:, 1, :].float())

    # 1. Centrado y normalización de potencia por ventana
    mean_z = z_raw.mean(dim=-1, keepdim=True)
    z = z_raw - mean_z
    raw_power = torch.clamp((z_raw.real ** 2 + z_raw.imag ** 2).mean(dim=-1, keepdim=True), min=eps)
    dc_frac = (torch.abs(mean_z) ** 2 / raw_power).squeeze(-1)
    power = torch.clamp((z.real ** 2 + z.imag ** 2).mean(dim=-1, keepdim=True), min=eps)
    z = z / torch.sqrt(power)
    env_raw = torch.sqrt(torch.clamp(z.real ** 2 + z.imag ** 2, min=eps))

    # 2. Estimador grueso de CFO por correlación de banda pasante (inmune al sesgo multicamino en lag-1)
    hann = kbuf["hann"]
    spec_pad = torch.fft.fftshift(torch.fft.fft(z * hann, n=n_fft, dim=-1), dim=-1)
    psd_pad = spec_pad.real ** 2 + spec_pad.imag ** 2
    psd_med = torch.clamp(psd_pad.median(dim=-1, keepdim=True).values, min=eps)
    band_ind = torch.tanh(0.35 * psd_pad / psd_med).unsqueeze(1)
    pass_score = F.conv1d(band_ind, kbuf["pass_kernel"], padding=kbuf["pass_pad"]).squeeze(1)
    if pass_score.shape[-1] > n_fft:
        pass_score = pass_score[:, :n_fft]

    # Restringir búsqueda al rango físico observable +/- 0.15 * fs (+/- 1200 Hz a fs=8 kHz)
    center_bin = n_fft // 2
    max_shift_bins = max(int(round(0.15 * n_fft)), 8)
    lo = center_bin - max_shift_bins
    hi = center_bin + max_shift_bins + 1
    sub_score = pass_score[:, lo:hi]
    rel_peak = sub_score.argmax(dim=-1)
    peak_bin = lo + rel_peak

    # Interpolación parabólica sub-bin para evitar saltos discretos de 15.6 Hz
    left_bin = torch.clamp(peak_bin - 1, min=0)
    right_bin = torch.clamp(peak_bin + 1, max=n_fft - 1)
    s_l = pass_score.gather(1, left_bin.unsqueeze(1)).squeeze(1)
    s_c = pass_score.gather(1, peak_bin.unsqueeze(1)).squeeze(1)
    s_r = pass_score.gather(1, right_bin.unsqueeze(1)).squeeze(1)
    denom = s_l - 2.0 * s_c + s_r
    delta_bin = torch.where(
        torch.abs(denom) > 1e-6,
        torch.clamp(0.5 * (s_l - s_r) / torch.where(torch.abs(denom) > 1e-6, denom, torch.ones_like(denom)), -0.5, 0.5),
        torch.zeros_like(denom),
    )
    f_norm_coarse = (peak_bin.float() + delta_bin - float(center_bin)) / float(n_fft)
    omega_coarse = (2.0 * np.pi * f_norm_coarse).unsqueeze(1)
    idx_n = kbuf["idx_n"]
    z_coarse = z * torch.complex(torch.cos(omega_coarse * idx_n), -torch.sin(omega_coarse * idx_n))

    # 3. Banco de Filtros Acoplados RRC en GPU (betas = 0.25, 0.35, 0.50) sobre z_coarse
    rrc_bank = kbuf["rrc_bank"]  # (3, 1, K)
    pad_mf = kbuf["pad_mf"]
    edge_norm_bank = kbuf["edge_norm_bank"]  # (3, N)
    mf_re = F.conv1d(z_coarse.real.unsqueeze(1), rrc_bank, padding=pad_mf)
    mf_im = F.conv1d(z_coarse.imag.unsqueeze(1), rrc_bank, padding=pad_mf)
    z_mf_bank = torch.complex(mf_re, mf_im) / edge_norm_bank.unsqueeze(0)
    # Normalizar cada rama matched-filter a potencia unitaria
    p_bank = torch.clamp((z_mf_bank.real ** 2 + z_mf_bank.imag ** 2).mean(dim=-1, keepdim=True), min=eps)
    z_mf_bank = z_mf_bank / torch.sqrt(p_bank)
    z_mf35 = z_mf_bank[:, 1, :]  # Filtro nominal beta=0.35
    z_mf25 = z_mf_bank[:, 0, :]
    z_mf50 = z_mf_bank[:, 2, :]

    # Razón energía fuera de banda / energía total tras filtrado acoplado (proxy ciego de SNR)
    mf_raw_re = mf_re[:, 1, :]
    mf_raw_im = mf_im[:, 1, :]
    mf_energy_ratio = torch.clamp((mf_raw_re ** 2 + mf_raw_im ** 2).mean(dim=-1), min=eps)

    # 4. Selección de fase de muestreo óptima y búsqueda fina de CFO residual en GPU
    n_sym = max(n_samples // sps, 1)
    batch_indices = torch.arange(bsz, device=x.device)
    if n_samples >= sps:
        z_mf_poly0 = z_mf35[:, : n_sym * sps].reshape(bsz, n_sym, sps).permute(0, 2, 1)  # (B, sps, n_sym)
        poly_stats0 = _complex_cumulants_and_moments(z_mf_poly0, eps=eps)
        # Máxima energía + menor varianza diferencial de fase entre símbolos = mejor apertura de ojo
        sym_pow = (z_mf_poly0.real ** 2 + z_mf_poly0.imag ** 2).mean(dim=-1)
        eye_score0 = sym_pow + 0.5 * torch.abs(poly_stats0[:, :, 1]) + 0.25 * poly_stats0[:, :, 2]
        best_phase0 = eye_score0.argmax(dim=1)
        best_sym0 = z_mf_poly0[batch_indices, best_phase0, :]  # (B, n_sym)

        # Búsqueda vectorizada de CFO fino (-48 a +48 Hz a fs=8000) sobre secuencia de símbolos
        sym_idx = (torch.arange(n_sym, device=x.device, dtype=torch.float32) * float(sps) + best_phase0.float().unsqueeze(1))
        cand_omega = (2.0 * np.pi * (kbuf["fine_dfs"] / 8000.0)).view(1, -1, 1)  # (1, K_f, 1)
        rot_phase = cand_omega * sym_idx.unsqueeze(1)  # (B, K_f, n_sym)
        cand_sym = best_sym0.unsqueeze(1) * torch.complex(torch.cos(rot_phase), -torch.sin(rot_phase))
        cand_env = torch.sqrt(torch.clamp(cand_sym.real ** 2 + cand_sym.imag ** 2, min=eps))
        cand_u = cand_sym / cand_env
        s2 = torch.abs((cand_u * cand_u).mean(dim=-1)) ** 2
        s4 = torch.abs((cand_sym ** 4).mean(dim=-1)) ** 2
        s8 = torch.abs((cand_u ** 8).mean(dim=-1)) ** 2
        fine_score = s2 + s4 + 0.5 * s8
        best_fine_idx = fine_score.argmax(dim=1)
        omega_fine = (2.0 * np.pi * (kbuf["fine_dfs"][best_fine_idx] / 8000.0)).unsqueeze(1)
        z_fine = z_mf35 * torch.complex(torch.cos(omega_fine * idx_n), -torch.sin(omega_fine * idx_n))
    else:
        z_fine = z_mf35

    env_fine = torch.sqrt(torch.clamp(z_fine.real ** 2 + z_fine.imag ** 2, min=eps))
    env_mf25 = torch.sqrt(torch.clamp(z_mf25.real ** 2 + z_mf25.imag ** 2, min=eps))
    env_mf50 = torch.sqrt(torch.clamp(z_mf50.real ** 2 + z_mf50.imag ** 2, min=eps))

    psi2 = 0.5 * torch.angle((z_fine ** 2).mean(dim=-1, keepdim=True))
    psi4 = 0.25 * torch.angle((z_fine ** 4).mean(dim=-1, keepdim=True))
    z_align2 = z_fine * torch.exp(-1j * psi2)
    z_align4 = z_fine * torch.exp(-1j * psi4)

    # 5. Productos diferenciales invariantes a fase global en lags 1, 4 y 8 (1 símbolo) sobre señal filtrada
    def _diff_channels(sig, lag):
        d = sig[:, lag:] * torch.conj(sig[:, :-lag])
        d_pad = F.pad(d, (lag, 0))
        return d_pad.real, d_pad.imag

    d1_r, d1_i = _diff_channels(z_fine, 1)
    d4_r, d4_i = _diff_channels(z_fine, 4)
    d8_r, d8_i = _diff_channels(z_fine, sps)

    temporal = torch.stack(
        [
            z_align4.real,
            z_align4.imag,
            z_align2.real,
            z_align2.imag,
            env_raw - 1.0,
            env_fine - 1.0,
            env_mf25 - 1.0,
            env_mf50 - 1.0,
            (env_fine * env_fine) - 1.0,
            (env_raw * env_raw) - (env_fine * env_fine),
            d1_r,
            d1_i,
            d4_r,
            d4_i,
            d8_r,
            d8_i,
        ],
        dim=1,
    )

    # 6. Espectros multi-armónicos (B, 6, N) con ventana Hann
    def _norm_mag_fft(sig):
        spec = torch.fft.fftshift(torch.fft.fft(sig * hann, dim=-1), dim=-1)
        mag = torch.abs(spec)
        mag_norm = mag / torch.clamp(mag.mean(dim=-1, keepdim=True), min=eps)
        return torch.log1p(mag_norm)

    z_sq_norm = (z_fine ** 2) / env_fine
    z_4th_norm = (z_fine ** 4) / (env_fine ** 3)
    z_8th_norm = (z_fine ** 8) / (env_fine ** 7)
    spectral = torch.stack(
        [
            _norm_mag_fft(z),
            _norm_mag_fft(z_coarse),
            _norm_mag_fft(z_fine),
            _norm_mag_fft(z_sq_norm),
            _norm_mag_fft(z_4th_norm),
            _norm_mag_fft(z_8th_norm),
        ],
        dim=1,
    )

    # 7. Banco polifásico de símbolos tras Matched Filter (B, 25, n_sym)
    z_poly = z_align4[:, : n_sym * sps].reshape(bsz, n_sym, sps).permute(0, 2, 1)
    poly_stats = _complex_cumulants_and_moments(z_poly, eps=eps)
    sym_pow_f = (z_poly.real ** 2 + z_poly.imag ** 2).mean(dim=-1)
    eye_score = sym_pow_f + 0.5 * torch.abs(poly_stats[:, :, 1]) + 0.25 * poly_stats[:, :, 2]
    best_phase_idx = eye_score.argmax(dim=1)
    best_poly_seq = z_poly[batch_indices, best_phase_idx, :]
    best_poly_diff = best_poly_seq[:, 1:] * torch.conj(best_poly_seq[:, :-1])
    best_poly_diff = F.pad(best_poly_diff, (1, 0))
    best_poly_env = torch.abs(best_poly_seq)

    z_raw_poly = env_raw[:, : n_sym * sps].reshape(bsz, n_sym, sps).permute(0, 2, 1)
    best_raw_env = z_raw_poly[batch_indices, best_phase_idx, :]

    # Perfil ordenado de anillos y ejes invariante a pi/2 (clave para 16QAM vs 64QAM vs QPSK vs 4PAM)
    abs_re = torch.abs(best_poly_seq.real)
    abs_im = torch.abs(best_poly_seq.imag)
    axis_max = torch.maximum(abs_re, abs_im)
    axis_min = torch.minimum(abs_re, abs_im)
    sorted_amp = torch.sort(best_poly_env, dim=-1).values
    sorted_axmax = torch.sort(axis_max, dim=-1).values
    sorted_axmin = torch.sort(axis_min, dim=-1).values

    poly_tensor = torch.cat(
        [
            z_poly.real,
            z_poly.imag,
            best_poly_seq.real.unsqueeze(1),
            best_poly_seq.imag.unsqueeze(1),
            (best_poly_env - 1.0).unsqueeze(1),
            best_poly_diff.real.unsqueeze(1),
            best_poly_diff.imag.unsqueeze(1),
            sorted_amp.unsqueeze(1),
            sorted_axmax.unsqueeze(1),
            sorted_axmin.unsqueeze(1),
            (best_raw_env - 1.0).unsqueeze(1),
        ],
        dim=1,
    )

    # 8. Vector estadístico multi-escala en GPU (120 descriptores invariantes a fase global)
    global_raw_stats = _complex_cumulants_and_moments(z, eps=eps)
    global_fine_stats = _complex_cumulants_and_moments(z_fine, eps=eps)
    blocks64 = z_fine[:, : (n_samples // 64) * 64].reshape(bsz, -1, 64)
    block64_stats = _complex_cumulants_and_moments(blocks64, eps=eps).mean(dim=1)
    poly_mean_stats = poly_stats.mean(dim=1)
    poly_max_stats = poly_stats.max(dim=1).values
    poly_min_stats = poly_stats.min(dim=1).values
    best_phase_stats = poly_stats[batch_indices, best_phase_idx, :]

    spec_center = spectral[:, 2, :]
    spec_flatness = torch.exp(torch.log(torch.clamp(spec_center, min=eps)).mean(dim=-1)) / torch.clamp(spec_center.mean(dim=-1), min=eps)
    spec_peak2 = spectral[:, 3, :].max(dim=-1).values / torch.clamp(spectral[:, 3, :].mean(dim=-1), min=eps)
    spec_peak4 = spectral[:, 4, :].max(dim=-1).values / torch.clamp(spectral[:, 4, :].mean(dim=-1), min=eps)
    spec_peak8 = spectral[:, 5, :].max(dim=-1).values / torch.clamp(spectral[:, 5, :].mean(dim=-1), min=eps)
    lag_corrs = torch.stack(
        [
            torch.abs((z_fine[:, lag:] * torch.conj(z_fine[:, :-lag])).mean(dim=-1))
            for lag in (1, 2, 4, 8, 12, 16)
        ],
        dim=-1,
    )
    extra_stats = torch.cat(
        [
            dc_frac.unsqueeze(-1),
            mf_energy_ratio.unsqueeze(-1),
            spec_flatness.unsqueeze(-1),
            spec_peak2.unsqueeze(-1),
            spec_peak4.unsqueeze(-1),
            spec_peak8.unsqueeze(-1),
            lag_corrs,
        ],
        dim=-1,
    )

    q_levels = kbuf["q_levels"]
    q_amp = torch.quantile(best_poly_env, q_levels, dim=-1).T
    q_axmax = torch.quantile(axis_max, q_levels, dim=-1).T
    q_axmin = torch.quantile(axis_min, q_levels, dim=-1).T

    stat_vector = torch.cat(
        [
            global_raw_stats,
            global_fine_stats,
            block64_stats,
            poly_mean_stats,
            poly_max_stats,
            poly_min_stats,
            best_phase_stats,
            extra_stats,
            q_amp,
            q_axmax,
            q_axmin,
        ],
        dim=-1,
    )
    stat_vector = torch.clamp(stat_vector, min=-25.0, max=25.0)
    return temporal, spectral, poly_tensor, stat_vector


class MultiDomainAMCNet(nn.Module):
    """Red neuronal profunda multi-dominio (ResNet1D + SpectralNet + Polyphase Transformer + Cumulant MLP).

    Diseñada para aprovechar los núcleos CUDA/Tensor de la GPU en entrenamiento e inferencia,
    manteniendo independencia estricta por ventana (sin BatchNorm).
    """
    def __init__(self, num_classes=6, width_mult=1.5, dropout=0.15):
        super().__init__()
        c1 = int(64 * width_mult)
        c2 = int(128 * width_mult)
        c3 = int(192 * width_mult)
        emb_dim = 192

        # Experto 1: Temporal multi-canal (16 canales -> ResNet 1D profunda)
        self.temporal_stem = nn.Sequential(
            nn.Conv1d(16, c1, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(16 if c1 % 16 == 0 else 8, c1),
            nn.SiLU(),
        )
        self.temporal_blocks = nn.Sequential(
            ResBlock1d(c1, c1, kernel_size=5, stride=1, dilation=1, dropout=dropout),
            ResBlock1d(c1, c2, kernel_size=5, stride=2, dilation=1, dropout=dropout),
            ResBlock1d(c2, c2, kernel_size=5, stride=1, dilation=2, dropout=dropout),
            ResBlock1d(c2, c3, kernel_size=3, stride=2, dilation=2, dropout=dropout),
            ResBlock1d(c3, c3, kernel_size=3, stride=2, dilation=4, dropout=dropout),
        )

        # Experto 2: Espectral multi-armónico (6 canales -> ResNet 1D)
        self.spectral_stem = nn.Sequential(
            nn.Conv1d(6, c1, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(16 if c1 % 16 == 0 else 8, c1),
            nn.SiLU(),
        )
        self.spectral_blocks = nn.Sequential(
            ResBlock1d(c1, c1, kernel_size=5, stride=2, dilation=1, dropout=dropout),
            ResBlock1d(c1, c2, kernel_size=5, stride=2, dilation=1, dropout=dropout),
            ResBlock1d(c2, c2, kernel_size=3, stride=2, dilation=2, dropout=dropout),
        )

        # Experto 3: Secuencia polifásica de símbolos tras MF (25 canales -> ResNet + Transformer Encoder)
        self.poly_stem = nn.Sequential(
            nn.Conv1d(25, c2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16 if c2 % 16 == 0 else 8, c2),
            nn.SiLU(),
        )
        self.poly_blocks = nn.Sequential(
            ResBlock1d(c2, c2, kernel_size=3, stride=1, dilation=1, dropout=dropout),
            ResBlock1d(c2, c2, kernel_size=3, stride=1, dilation=2, dropout=dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c2, nhead=8, dim_feedforward=c2 * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.poly_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=False)

        # Experto 4: MLP profundo con conexiones residuales sobre vector estadístico/cumulante (120 dims)
        self.stat_in = nn.Sequential(
            nn.LayerNorm(120),
            nn.Linear(120, 256),
            nn.SiLU(),
        )
        self.stat_res = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
        )
        self.stat_out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(256, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.SiLU(),
        )

        self.temp_proj = nn.Sequential(nn.Linear(c3 * 2, emb_dim), nn.LayerNorm(emb_dim), nn.SiLU())
        self.spec_proj = nn.Sequential(nn.Linear(c2 * 2, emb_dim), nn.LayerNorm(emb_dim), nn.SiLU())
        self.poly_proj = nn.Sequential(nn.Linear(c2 * 2, emb_dim), nn.LayerNorm(emb_dim), nn.SiLU())

        self.domain_gate = nn.Sequential(
            nn.Linear(emb_dim * 4, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, 4),
        )

        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 4, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _avg_max_pool(feat):
        return torch.cat([feat.mean(dim=-1), feat.amax(dim=-1)], dim=-1)

    def forward_with_embeddings(self, x):
        temporal, spectral, poly, stats = extract_multidomain_tensors(x)
        t_feat = self.temporal_blocks(self.temporal_stem(temporal))
        s_feat = self.spectral_blocks(self.spectral_stem(spectral))
        p_conv = self.poly_blocks(self.poly_stem(poly))
        # Transformer sobre la secuencia de símbolos (B, n_sym, c2)
        p_seq = self.poly_transformer(p_conv.transpose(1, 2)).transpose(1, 2)

        e_temp = self.temp_proj(self._avg_max_pool(t_feat))
        e_spec = self.spec_proj(self._avg_max_pool(s_feat))
        e_poly = self.poly_proj(self._avg_max_pool(p_seq))
        s_hidden = self.stat_in(stats)
        e_stat = self.stat_out(s_hidden + self.stat_res(s_hidden))

        concat = torch.cat([e_temp, e_spec, e_poly, e_stat], dim=-1)
        weights = F.softmax(self.domain_gate(concat), dim=-1)
        gated = torch.cat(
            [
                e_temp * (1.0 + weights[:, 0:1]),
                e_spec * (1.0 + weights[:, 1:2]),
                e_poly * (1.0 + weights[:, 2:3]),
                e_stat * (1.0 + weights[:, 3:4]),
            ],
            dim=-1,
        )
        logits = self.classifier(gated)
        return logits, weights

    def forward(self, x):
        logits, _ = self.forward_with_embeddings(x)
        return logits


class GPUChannelSynthesizer:
    """Sintetizador físico de formas de onda y canales 100% en GPU (PyTorch CUDA).

    Genera lotes masivos de señales moduladas (BPSK, QPSK, 8PSK, 16QAM, 64QAM, 4PAM)
    con filtrado RRC (betas 0.25, 0.35, 0.50), canales multicamino FIR, CFO, fase aleatoria,
    AWGN exacto y cuantización ADC opcional directamente en la memoria de la RTX 4060.
    USO EXCLUSIVO PARA ENTRENAMIENTO EN GPU (jamás sustituye test ni test_shift).
    """
    def __init__(self, device="cuda", sps=8, span=10, fs_hz=8000.0, betas=(0.25, 0.35, 0.50)):
        self.device = torch.device(device)
        self.sps = sps
        self.span = span
        self.fs_hz = float(fs_hz)
        self.betas = tuple(betas)

        # Pre-calcular filtros RRC en GPU
        taps = []
        for beta in self.betas:
            h = rrc_filter(beta=beta, sps=sps, span=span).astype(np.float32)
            taps.append(torch.from_numpy(h).to(self.device))
        self.rrc_bank = torch.stack(taps, dim=0)  # (len(betas), L_rrc)

        # Constelaciones normalizadas a potencia unitaria en GPU (mismo orden que CLASSES)
        bpsk = torch.tensor([-1.0 + 0j, 1.0 + 0j], dtype=torch.complex64, device=self.device)
        qpsk = torch.tensor(
            [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=torch.complex64, device=self.device
        ) / np.sqrt(2.0)
        angles8 = torch.arange(8, device=self.device, dtype=torch.float32) * (2.0 * np.pi / 8.0)
        psk8 = torch.complex(torch.cos(angles8), torch.sin(angles8))
        g4 = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=self.device) / np.sqrt(10.0)
        gx4, gy4 = torch.meshgrid(g4, g4, indexing="ij")
        qam16 = torch.complex(gx4.reshape(-1), gy4.reshape(-1))
        g8 = torch.tensor([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0], device=self.device) / np.sqrt(42.0)
        gx8, gy8 = torch.meshgrid(g8, g8, indexing="ij")
        qam64 = torch.complex(gx8.reshape(-1), gy8.reshape(-1))
        pam4 = torch.tensor([-3.0 + 0j, -1.0 + 0j, 1.0 + 0j, 3.0 + 0j], dtype=torch.complex64, device=self.device) / np.sqrt(5.0)
        self.constellations = (bpsk, qpsk, psk8, qam16, qam64, pam4)

    @torch.no_grad()
    def sample_batch(self, batch_size=512, n_samples=256, snr_range_db=(-20.0, 20.0),
                     cfo_range_hz=(-100.0, 100.0), max_echo_delay=5, adc_bits_choices=None, full_scale=2.5):
        """Sintetiza un minibatch balanceado (batch_size, 2, n_samples) y targets (batch_size,) en GPU.

        Estrictamente acotado al soporte In-Distribution (|CFO| <= 100 Hz, retardos <= 5) y sin
        llamadas sincronizadoras `.item()` para mantener alta ocupación CUDA en RTX 4060.
        """
        max_echo_delay = min(int(max_echo_delay), 5)
        n_sym = (n_samples // self.sps) + 2 * self.span + 4
        per_class = batch_size // 6
        rem = batch_size - 6 * per_class
        counts = [per_class + (1 if i < rem else 0) for i in range(6)]

        sym_list = []
        y_list = []
        for cls_idx, count in enumerate(counts):
            if count == 0:
                continue
            const = self.constellations[cls_idx]
            idx = torch.randint(0, len(const), (count, n_sym), device=self.device)
            sym_list.append(const[idx])
            y_list.append(torch.full((count,), cls_idx, dtype=torch.long, device=self.device))

        symbols = torch.cat(sym_list, dim=0)
        targets = torch.cat(y_list, dim=0)
        perm = torch.randperm(batch_size, device=self.device)
        symbols = symbols[perm]
        targets = targets[perm]

        # Upsampling por factor sps=8 en GPU
        up = torch.zeros((batch_size, n_sym * self.sps), dtype=torch.complex64, device=self.device)
        up[:, ::self.sps] = symbols

        # Filtrado RRC 100% vectorizado en GPU (sin sincronización CPU)
        beta_choices = torch.randint(0, len(self.betas), (batch_size,), device=self.device)
        pad = self.rrc_bank.shape[1] // 2
        kernels = self.rrc_bank.unsqueeze(1)
        re_bank = F.conv1d(up.real.unsqueeze(1), kernels, padding=pad)
        im_bank = F.conv1d(up.imag.unsqueeze(1), kernels, padding=pad)
        row_idx = torch.arange(batch_size, device=self.device)
        clean = torch.complex(re_bank[row_idx, beta_choices], im_bank[row_idx, beta_choices])

        # Canal multicamino estrictamente acotado al soporte ID (70% canales exactos ID + 30% variación ID delay<=5)
        faded = clean.clone()
        ch_draw = torch.rand((batch_size, 1), device=self.device)
        # Eco de retardo 2 exacto ID: 0.35j (cuando ch_draw in [0.233, 0.467))
        m_id1 = (ch_draw >= 0.233) & (ch_draw < 0.467)
        if max_echo_delay >= 2:
            d2 = F.pad(clean[:, :-2], (2, 0))
            faded = faded + m_id1.float() * (0.35j) * d2
        # Eco de retardo 5 exacto ID: 0.15 - 0.15j (cuando ch_draw in [0.467, 0.700))
        m_id2 = (ch_draw >= 0.467) & (ch_draw < 0.700)
        if max_echo_delay >= 5:
            d5 = F.pad(clean[:, :-5], (5, 0))
            faded = faded + m_id2.float() * (0.15 - 0.15j) * d5
        # Variación continua ID-acotada (30%): un único eco causal con delay in 1..max_echo_delay
        m_var = ch_draw >= 0.700
        var_delays = torch.randint(1, max_echo_delay + 1, (batch_size,), device=self.device)
        var_amp = 0.05 + 0.33 * torch.rand((batch_size, 1), device=self.device)
        var_phase = 2.0 * torch.pi * torch.rand((batch_size, 1), device=self.device)
        var_tap = m_var.float() * torch.complex(var_amp * torch.cos(var_phase), var_amp * torch.sin(var_phase))
        for d in range(1, max_echo_delay + 1):
            md = (var_delays == d).unsqueeze(1).float()
            faded = faded + (md * var_tap) * F.pad(clean[:, :-d], (d, 0))

        # Escalado de amplitud física en [0.5, 1.5]
        amp_scale = 0.5 + 1.0 * torch.rand((batch_size, 1), device=self.device)
        faded = faded * amp_scale

        # Recorte libre de transitorios con fase de muestreo aleatoria k0 in {0..sps-1}
        start_base = self.span * self.sps + max_echo_delay
        k0 = torch.randint(0, self.sps, (batch_size,), device=self.device)
        time_idx = torch.arange(n_samples, device=self.device).unsqueeze(0) + (start_base + k0).unsqueeze(1)
        window = torch.gather(faded, 1, time_idx)

        # Aplicar fase inicial y CFO estrictamente dentro de cfo_range_hz
        phi0 = (2.0 * torch.rand((batch_size, 1), device=self.device) - 1.0) * torch.pi
        cfo_hz = cfo_range_hz[0] + (cfo_range_hz[1] - cfo_range_hz[0]) * torch.rand((batch_size, 1), device=self.device)
        n_vec = torch.arange(n_samples, device=self.device, dtype=torch.float32).unsqueeze(0)
        rot_angle = phi0 + 2.0 * torch.pi * (cfo_hz / self.fs_hz) * n_vec
        rotated = window * torch.complex(torch.cos(rot_angle), torch.sin(rot_angle))

        # Mezcla equilibrada de SNR: 40% grilla ID [-20..20] + 60% foco discriminable [-5..22] dB
        sig_power = torch.clamp((rotated.real ** 2 + rotated.imag ** 2).mean(dim=-1, keepdim=True), min=1e-7)
        u_snr = torch.rand((batch_size, 1), device=self.device)
        snr_full = snr_range_db[0] + (snr_range_db[1] - snr_range_db[0]) * torch.rand((batch_size, 1), device=self.device)
        hi_lo = max(snr_range_db[0], -5.0)
        hi_hi = max(snr_range_db[1], 22.0)
        snr_high = hi_lo + (hi_hi - hi_lo) * torch.rand((batch_size, 1), device=self.device)
        snr_db = torch.where(u_snr < 0.45, snr_full, snr_high)
        noise_power = sig_power / torch.pow(10.0, snr_db / 10.0)
        sigma = torch.sqrt(noise_power / 2.0)
        noisy = rotated + sigma * torch.complex(torch.randn_like(rotated.real), torch.randn_like(rotated.imag))

        # Cuantización ADC virtual vectorizada sin .item() (cada muestra sortea resolución)
        if adc_bits_choices is not None:
            choices_t = torch.randint(0, len(adc_bits_choices), (batch_size,), device=self.device)
            q_re = noisy.real.clone()
            q_im = noisy.imag.clone()
            for idx_b, bits in enumerate(adc_bits_choices):
                if bits is None:
                    continue
                mask_b = (choices_t == idx_b).unsqueeze(1)
                levels = float(1 << int(bits))
                delta = (2.0 * full_scale) / levels
                c_re = -full_scale + (torch.clamp(torch.floor((torch.clamp(noisy.real, -full_scale, full_scale) + full_scale) / delta), 0.0, levels - 1.0) + 0.5) * delta
                c_im = -full_scale + (torch.clamp(torch.floor((torch.clamp(noisy.imag, -full_scale, full_scale) + full_scale) / delta), 0.0, levels - 1.0) + 0.5) * delta
                q_re = torch.where(mask_b, c_re, q_re)
                q_im = torch.where(mask_b, c_im, q_im)
            noisy = torch.complex(q_re, q_im)

        # Preprocesado estándar por ventana (centrado y potencia unitaria)
        z = noisy - noisy.mean(dim=-1, keepdim=True)
        p = torch.clamp((z.real ** 2 + z.imag ** 2).mean(dim=-1, keepdim=True), min=1e-7)
        z = z / torch.sqrt(p)
        return torch.stack([z.real, z.imag], dim=1), targets


def gpu_physical_augmentation(x, p_conj=0.5, cfo_max_cycles_per_window=0.0, multipath_prob=0.0, noise_std_max=0.0):
    """Aumentación física exacta y no destructiva en GPU sobre ventanas reales (B, 2, N).

    Aplica rotación aleatoria de fase global y conjugación compleja opcional (simetrías exactas
    de las 6 constelaciones BPSK/QPSK/8PSK/16QAM/64QAM/4PAM), evitando doble corrupción de SNR o eco.
    """
    bsz, _, _ = x.shape
    device = x.device
    z = torch.complex(x[:, 0, :], x[:, 1, :])

    if p_conj > 0:
        flip = torch.rand((bsz, 1), device=device) < p_conj
        z = torch.where(flip, torch.conj(z), z)

    phi0 = (2.0 * torch.pi * torch.rand((bsz, 1), device=device)) - torch.pi
    z = z * torch.complex(torch.cos(phi0), torch.sin(phi0))

    if cfo_max_cycles_per_window > 0:
        cfo_cycles = (2.0 * torch.rand((bsz, 1), device=device) - 1.0) * cfo_max_cycles_per_window
        n_idx = torch.linspace(0.0, 1.0, z.shape[-1], device=device).unsqueeze(0)
        z = z * torch.complex(torch.cos(2.0 * torch.pi * cfo_cycles * n_idx), torch.sin(2.0 * torch.pi * cfo_cycles * n_idx))

    if noise_std_max > 0:
        sigma = torch.rand((bsz, 1), device=device) * noise_std_max
        noise = sigma * torch.complex(torch.randn_like(z.real), torch.randn_like(z.imag)) / np.sqrt(2.0)
        z = z + noise

    z = z - z.mean(dim=-1, keepdim=True)
    power = torch.clamp((z.real ** 2 + z.imag ** 2).mean(dim=-1, keepdim=True), min=1e-7)
    z = z / torch.sqrt(power)
    return torch.stack([z.real, z.imag], dim=1)


class FastLBAPSelector:
    """Predictor ligero de beneficio/utilidad (LBAP) en NumPy puro (< 0.05 ms por ventana)."""
    def __init__(self, mean, scale, w1, b1, w2, b2, w_lin, b_lin, alpha_mlp=0.5, allowed_actions=(0, 1, 2)):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.w1 = np.asarray(w1, dtype=np.float64)
        self.b1 = np.asarray(b1, dtype=np.float64)
        self.w2 = np.asarray(w2, dtype=np.float64)
        self.b2 = np.asarray(b2, dtype=np.float64)
        self.w_lin = np.asarray(w_lin, dtype=np.float64)
        self.b_lin = np.asarray(b_lin, dtype=np.float64)
        self.alpha_mlp = float(alpha_mlp)
        self.allowed_actions = tuple(int(a) for a in allowed_actions)

    def predict_utilities(self, gate_features):
        x = (np.asarray(gate_features, dtype=np.float64) - self.mean) / self.scale
        x = np.clip(x, -10.0, 10.0)
        h = np.maximum(0.0, x @ self.w1 + self.b1)
        u_mlp = h @ self.w2 + self.b2
        u_lin = x @ self.w_lin + self.b_lin
        return self.alpha_mlp * u_mlp + (1.0 - self.alpha_mlp) * u_lin

    def choose(self, gate_features):
        utilities = self.predict_utilities(gate_features)
        if len(self.allowed_actions) == 3:
            return utilities.argmax(axis=1)
        sub = utilities[:, self.allowed_actions]
        idx = sub.argmax(axis=1)
        return np.asarray(self.allowed_actions, dtype=int)[idx]

    @classmethod
    def fit(cls, gate_train, utility_targets_train, gate_val=None, utility_targets_val=None,
            hidden_dim=64, ridge_l2=40.0, epochs=400, lr=3e-3, weight_decay=1e-3, seed=20261111,
            allowed_actions=(0, 1, 2)):
        """Entrena el predictor LBAP en GPU sobre train y selecciona hiperparámetros en val_cal."""
        x_tr = np.asarray(gate_train, dtype=np.float64)
        y_tr = np.asarray(utility_targets_train, dtype=np.float64)
        mean = x_tr.mean(axis=0)
        scale = np.where(x_tr.std(axis=0) > 1e-8, x_tr.std(axis=0), 1.0)
        xn = np.clip((x_tr - mean) / scale, -10.0, 10.0)

        d = xn.shape[1]
        xtx = xn.T @ xn + ridge_l2 * np.eye(d)
        w_lin = np.linalg.solve(xtx, xn.T @ (y_tr - y_tr.mean(axis=0)))
        b_lin = y_tr.mean(axis=0)

        torch.manual_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        xt = torch.from_numpy(xn.astype(np.float32)).to(device)
        yt = torch.from_numpy(y_tr.astype(np.float32)).to(device)

        fc1 = nn.Linear(d, hidden_dim).to(device)
        fc2 = nn.Linear(hidden_dim, y_tr.shape[1]).to(device)
        opt = torch.optim.AdamW(list(fc1.parameters()) + list(fc2.parameters()), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            pred = fc2(F.relu(fc1(xt)))
            loss = F.smooth_l1_loss(pred, yt)
            loss.backward()
            opt.step()

        w1 = fc1.weight.detach().cpu().numpy().T
        b1 = fc1.bias.detach().cpu().numpy()
        w2 = fc2.weight.detach().cpu().numpy().T
        b2 = fc2.bias.detach().cpu().numpy()

        best_alpha = 0.5
        if gate_val is not None and utility_targets_val is not None:
            best_score = -np.inf
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                cand = cls(mean, scale, w1, b1, w2, b2, w_lin, b_lin, alpha_mlp=alpha, allowed_actions=allowed_actions)
                chosen = cand.choose(gate_val)
                score = float(np.mean(utility_targets_val[np.arange(len(chosen)), chosen]))
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
        return cls(mean, scale, w1, b1, w2, b2, w_lin, b_lin, alpha_mlp=best_alpha, allowed_actions=allowed_actions)
