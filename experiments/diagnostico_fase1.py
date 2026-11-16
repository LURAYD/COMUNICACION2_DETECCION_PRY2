"""Generar figuras y métricas: python -m experiments.diagnostico_fase1."""
import argparse
from itertools import product
import json
import hashlib
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import signals
from src.waveform import pulse_shape, rrc_filter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/fase1"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    configurations = [("BPSK", signals.bpsk_modulation, 1),
                      ("QPSK", signals.qpsk_modulation, 2),
                      ("8PSK", signals.psk8_modulation, 3),
                      ("16-QAM", signals.qam16_modulation, 4),
                      ("64-QAM", signals.qam64_modulation, 6),
                      ("4-PAM", signals.pam4_modulation, 2)]
    root = Path(__file__).resolve().parents[1]
    source_files = ["src/signals.py", "src/waveform.py", "experiments/diagnostico_fase1.py"]
    metrics = {"seed": args.seed, "constelaciones": {},
               "versiones": {"python": platform.python_version(), "numpy": np.__version__,
                             "matplotlib": matplotlib.__version__},
               "fuentes_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                  for name in source_files}}
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for ax, (name, mapper, k) in zip(axes.ravel(), configurations):
        labels = np.array(list(product([0, 1], repeat=k)))
        symbols = mapper(labels.ravel())
        ax.scatter(symbols.real, symbols.imag, s=26)
        if k <= 4:
            for bits, symbol in zip(labels, symbols):
                ax.annotate("".join(map(str, bits)), (symbol.real, symbol.imag),
                            xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set(title=name, xlabel="I", ylabel="Q", xlim=(-1.65, 1.65), ylim=(-1.65, 1.65), aspect="equal")
        ax.grid(alpha=.3)
        metrics["constelaciones"][name] = {"puntos": len(symbols), "bits_por_simbolo": k,
                                            "energia_equiprobable": float(np.mean(abs(symbols)**2))}
    fig.savefig(args.output / "constelaciones.png", dpi=160)
    plt.close(fig)

    beta, sps, span = .35, 8, 10
    h = rrc_filter(beta=beta, sps=sps, span=span)
    matched = np.convolve(h, h[::-1])
    delay = (len(h)-1)//2
    samples = matched[::sps].copy()
    samples[len(samples)//2] = 0
    nfft = 65536
    frequency = np.fft.fftshift(np.fft.fftfreq(nfft, d=1/sps))
    power = abs(np.fft.fftshift(np.fft.fft(h, nfft)))**2/sps
    f = abs(frequency)
    ideal = np.where(f <= (1-beta)/2, 1.,
                     np.where(f <= (1+beta)/2, .5*(1+np.cos(np.pi/beta*(f-(1-beta)/2))), 0.))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].plot((np.arange(len(h))-delay)/sps, h)
    axes[0].set(title="Pulso RRC, energía discreta 1", xlabel="t/T", ylabel="h[n]")
    axes[1].plot((np.arange(len(matched))-2*delay)/sps, matched)
    axes[1].scatter((np.arange(0,len(matched),sps)-2*delay)/sps, matched[::sps], s=16)
    axes[1].set(title="RRC + filtro adaptado", xlabel="t/T", ylabel="Respuesta")
    axes[2].plot(frequency, power, label="|H|² / sps (RRC truncado)")
    axes[2].plot(frequency, ideal, "--", label="RC ideal")
    axes[2].set(title="Respuesta espectral", xlabel="f / Rs", ylabel="Ganancia normalizada", xlim=(-1,1))
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.3)
    fig.savefig(args.output / "rrc_diagnostico.png", dpi=160)
    plt.close(fig)

    symbols = signals.qpsk_modulation(rng.integers(0, 2, size=64))
    iq, _ = pulse_shape(symbols, sps=sps, beta=beta, span=span)
    received = np.convolve(iq, h[::-1].conj())
    recovered = received[2*delay:2*delay+len(symbols)*sps:sps]
    error = recovered-symbols
    fig, ax = plt.subplots(figsize=(12,4), constrained_layout=True)
    time = np.arange(len(iq))/sps
    ax.plot(time, iq.real, label="I")
    ax.plot(time, iq.imag, label="Q", alpha=.8)
    ax.axvline(delay/sps, color="black", linestyle=":", label="Retardo TX")
    ax.set(title="QPSK conformada con RRC, transitorios incluidos", xlabel="n / sps (períodos de símbolo)", ylabel="Amplitud")
    ax.legend()
    ax.grid(alpha=.3)
    fig.savefig(args.output / "iq_qpsk.png", dpi=160)
    plt.close(fig)
    metrics["rrc"] = {"beta": beta, "sps": sps, "span": span, "coeficientes": len(h),
                      "energia_discreta": float(np.sum(h*h)), "retardo_tx_muestras": delay,
                      "retardo_tx_rx_muestras": 2*delay,
                      "isi_residual_max": float(abs(samples).max()),
                      "error_espectral_max": float(abs(power-ideal).max()),
                      "nfft": nfft, "eje_frecuencia": "f/Rs", "normalizacion_espectral": "|H|^2/sps"}
    metrics["trama_qpsk"] = {"simbolos": len(symbols), "muestras": len(iq),
                             "energia_media_simbolos": float(np.mean(abs(symbols)**2)),
                             "energia_total_iq": float(np.sum(abs(iq)**2)),
                             "error_recuperacion_max": float(abs(error).max()),
                             "error_recuperacion_rms": float(np.sqrt(np.mean(abs(error)**2))),
                             "potencia_media_iq_con_transitorios": float(np.mean(abs(iq)**2))}
    (args.output / "metricas.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"Diagnósticos guardados en {args.output.resolve()}")


if __name__ == "__main__":
    main()
