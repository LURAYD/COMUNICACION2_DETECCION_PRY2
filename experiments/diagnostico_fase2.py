"""Canal de ejemplo reproducible: python -m experiments.diagnostico_fase2."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.channels import add_awgn, apply_carrier_impairments, apply_multipath
from src.signals import qpsk_modulation
from src.waveform import pulse_shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/fase2"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rs, sps, beta, span = 1000, 8, .35, 10
    fs = rs*sps
    phase, cfo, snr = .6, 75., 10.
    taps = np.array([1, 0, .35j, 0, 0, .15*np.exp(-1j*np.pi/3)])
    symbols = qpsk_modulation(rng.integers(0, 2, 8192))
    clean, h = pulse_shape(symbols, sps=sps, beta=beta, span=span)
    faded = apply_multipath(clean, taps)
    rotated = apply_carrier_impairments(faded, fs, phase, cfo)
    received, noise = add_awgn(rotated, snr, rng)
    power = lambda x: float(np.mean(abs(x)**2))
    measured = float(10*np.log10(power(rotated)/power(noise)))
    root = Path(__file__).resolve().parents[1]
    names = ['src/signals.py', 'src/waveform.py', 'src/channels.py', 'experiments/diagnostico_fase2.py']
    rows = []
    for snr_db in [-10, 0, 10, 20]:
        for amplitude in [.5, 1., 1.5]:
            # Generador compartido avanza: realizaciones de ruido distintas.
            _, trial_noise = add_awgn(amplitude*rotated, snr_db, rng)
            rows.append({"snr_nominal_db": snr_db, "amplitud": amplitude,
                         "snr_medida_db": float(10*np.log10(power(amplitude*rotated)/power(trial_noise)))})
    metrics = {"seed": args.seed,
               "versiones": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
               "fuentes_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},
               "parametros": {"rs_baud": rs, "sps": sps, "fs_hz": fs, "beta": beta, "span": span,
                              "phase_rad": phase, "cfo_hz": cfo, "snr_db": snr, "sample_offset": 0,
                              "taps_real_imag": [[float(z.real), float(z.imag)] for z in taps],
                              "retardos_no_nulos_muestras": [0, 2, 5]},
               "longitudes": {"simbolos": len(symbols), "rrc": len(h), "limpia": len(clean), "canal": len(received)},
               "potencias": {"limpia": power(clean), "multipath": power(faded), "portadora": power(rotated),
                             "ruido": power(noise), "recibida": power(received)},
               "awgn": {"snr_medida_db": measured, "varianza_i": float(np.var(noise.real)),
                        "varianza_q": float(np.var(noise.imag)), "varianza_objetivo_componente": power(rotated)/10**(snr/10)/2},
               "continuidad_portadora_error_max": float(abs(rotated-np.r_[
                   apply_carrier_impairments(faded[:512], fs, phase, cfo),
                   apply_carrier_impairments(faded[512:], fs, phase, cfo, 512)]).max()),
               "barrido_awgn": rows,
               "nota": "Parámetros de ingeniería ilustrativos; métricas oracle solo para validar el simulador. No es un dataset."}
    (args.output/'metricas.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].stem(np.arange(len(taps)), abs(taps))
    axes[0, 0].set(title="FIR: magnitud de los taps (sin renormalizar)", xlabel="Retardo (muestras)", ylabel="|tap|")
    n = np.arange(256)
    axes[0, 1].plot(n/fs*1000, phase+2*np.pi*cfo*n/fs)
    axes[0, 1].set(title="Fase impuesta por el oscilador", xlabel="Tiempo (ms)", ylabel="Fase acumulada (rad)")
    start, stop = 80, 280
    time = np.arange(start, stop)/fs*1000
    axes[1, 0].plot(time, rotated[start:stop].real, label="I antes de AWGN")
    axes[1, 0].plot(time, received[start:stop].real, alpha=.65, label="I recibida")
    axes[1, 0].set(title=f"AWGN: nominal {snr:g} dB; medido {measured:.2f} dB", xlabel="Tiempo (ms)", ylabel="Amplitud")
    axes[1, 0].legend(fontsize=8)
    for amplitude in [.5, 1., 1.5]:
        group = [row for row in rows if row['amplitud'] == amplitude]
        axes[1, 1].plot([row['snr_nominal_db'] for row in group],
                        [row['snr_medida_db']-row['snr_nominal_db'] for row in group], 'o-', label=f"Amplitud {amplitude:g}")
    axes[1, 1].axhline(0, color='black', linewidth=.8)
    axes[1, 1].set(title="Error estadístico de SNR (sin forzar potencia exacta)", xlabel="SNR nominal (dB)", ylabel="Medida − nominal (dB)")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=.3)
    fig.savefig(args.output/'canal_diagnostico.png', dpi=160)
    plt.close(fig)
    print(f"Diagnóstico guardado en {args.output.resolve()}")


if __name__ == '__main__':
    main()
