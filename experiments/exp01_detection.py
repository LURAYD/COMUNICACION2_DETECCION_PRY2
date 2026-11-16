"""E1 controlado: piso de ruido fijo, calibración y evaluación independientes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.channels import apply_carrier_impairments, apply_multipath
from src.dataset import MAPPERS
from src.detection import calibrate_energy, detect_energy, energy_statistic
from src.spectrum import estimate_spectrum
from src.snr_estimation import calibrate_noise_reference, estimate_snr
from src.waveform import pulse_shape


def wilson_interval(successes, trials, z=1.959963984540054):
    """Wilson bilateral 95%: https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm"""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Conteos binomiales inválidos.")
    proportion = successes/trials
    denominator = 1+z*z/trials
    center = (proportion+z*z/(2*trials))/denominator
    radius = z*np.sqrt(proportion*(1-proportion)/trials+z*z/(4*trials*trials))/denominator
    return [float(max(0, center-radius)), float(min(1, center+radius))]


def _noise(rng, shape, power):
    return np.sqrt(power/2)*(rng.standard_normal(shape)+1j*rng.standard_normal(shape))


def _clean_window(config, modulation, regime, rng):
    mapper, k = MAPPERS[modulation]
    symbols = mapper(rng.integers(0, 2, config["symbols_per_parent"]*k))
    beta = float(rng.choice(config["betas"]))
    clean, h = pulse_shape(symbols, config["sps"], beta, config["span"])
    bank = config["shift_channels" if regime == "shift" else "id_channels"]
    pairs = np.asarray(bank[int(rng.integers(len(bank)))])
    taps = pairs[:, 0]+1j*pairs[:, 1]
    faded = apply_multipath(clean, taps)
    cfo = (float(rng.uniform(*config["shift_cfo_abs_range_hz"]))*int(rng.choice([-1, 1]))
           if regime == "shift" else float(rng.uniform(*config["id_cfo_range_hz"])))
    fs = config["symbol_rate_hz"]*config["sps"]
    rotated = apply_carrier_impairments(faded, fs, float(rng.uniform(-np.pi, np.pi)), cfo)
    start = len(h)-1+len(taps)-1+int(rng.integers(config["sps"]))
    stop = start+config["n_samples"]
    if stop > (config["symbols_per_parent"]-1)*config["sps"]:
        raise ValueError("Trama demasiado corta para ventana sin transitorios.")
    return rotated[start:stop]


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _error_metrics(errors, n_total, rng, bootstrap_resamples):
    errors = np.asarray(errors, dtype=float)
    metrics = {"n_total": n_total, "n_valid": len(errors),
            "mae_db": float(np.mean(abs(errors))) if len(errors) else None,
            "bias_db": float(np.mean(errors)) if len(errors) else None,
            "rmse_db": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            "error_p05_db": float(np.percentile(errors, 5)) if len(errors) else None,
            "error_p50_db": float(np.percentile(errors, 50)) if len(errors) else None,
            "error_p95_db": float(np.percentile(errors, 95)) if len(errors) else None}
    for name in ("bias", "mae", "rmse"):
        metrics[name+"_bootstrap95_low_db"] = None
        metrics[name+"_bootstrap95_high_db"] = None
    metrics["bootstrap_status"] = "insufficient_valid_parents" if len(errors) <= 1 else "ok_conditional_on_n_valid"
    if len(errors) > 1:
        # Una observación independiente por padre. Se remuestrean los padres
        # válidos; IC percentil condicional a n_valid y a calibración fija.
        draws = rng.choice(errors, size=(bootstrap_resamples, len(errors)))
        for name, statistic in (("bias", np.mean(draws, axis=1)),
                                ("mae", np.mean(abs(draws), axis=1)),
                                ("rmse", np.sqrt(np.mean(draws**2, axis=1)))):
            low, high = np.percentile(statistic, [2.5, 97.5])
            metrics[name+"_bootstrap95_low_db"] = float(low)
            metrics[name+"_bootstrap95_high_db"] = float(high)
    return metrics


def run_experiment(config, output):
    for name in ("seed", "n_samples", "n_calibration", "n_noise_evaluation",
                 "trials_per_class_snr_regime", "drift_signal_trials_per_snr", "bootstrap_resamples"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "seed" else 2):
            raise ValueError(f"{name} debe ser un entero válido.")
    if config["drift_noise_power_factor"] != 2:
        raise ValueError("El ensayo de deriva E1 requiere duplicar la potencia de ruido.")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Ya existe {output}; use otra carpeta.")
    output.mkdir(parents=True)
    n = config["n_samples"]
    power = config["noise_power"]
    if n < 2 or power <= 0 or not np.isfinite(power):
        raise ValueError("Longitud o potencia de ruido inválida.")
    cal_seed, noise_seed, signal_seed, drift_seed, bootstrap_seed = np.random.SeedSequence(config["seed"]).spawn(5)
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    def error_metrics(errors, n_total):
        return _error_metrics(errors, n_total, bootstrap_rng, config["bootstrap_resamples"])
    calibration_noise = _noise(np.random.default_rng(cal_seed), (config["n_calibration"], n), power)
    calibration = calibrate_energy(calibration_noise, config["pfa_target"])
    reference = calibrate_noise_reference(calibration_noise.ravel())
    eval_noise = _noise(np.random.default_rng(noise_seed), (config["n_noise_evaluation"], n), power)
    false_alarms = sum(detect_energy(row, calibration)["detected"] for row in eval_noise)
    rows, aggregates, snr_rows, trial_rows = [], [], [], []
    pooled_errors = {(regime, snr): [] for regime in ("id", "shift") for snr in config["snr_db"]}
    cells = [(regime, modulation, snr) for regime in ("id", "shift") for modulation in MAPPERS for snr in config["snr_db"]]
    example = None
    trials = config["trials_per_class_snr_regime"]
    for (regime, modulation, snr), cell_seed in zip(cells, signal_seed.spawn(len(cells))):
        detections = 0
        measured_snrs = []
        errors = []
        for parent_seed in cell_seed.spawn(trials):
            waveform_seed, awgn_seed = parent_seed.spawn(2)
            clean = _clean_window(config, modulation, regime, np.random.default_rng(waveform_seed))
            # Escalado EXPERIMENTAL explícito: SNR local controlada con piso fijo.
            scaled = clean*np.sqrt(power*10**(snr/10)/energy_statistic(clean))
            noise = _noise(np.random.default_rng(awgn_seed), n, power)
            observed = scaled+noise
            detected = detect_energy(observed, calibration)["detected"]
            detections += detected
            estimate = estimate_snr(observed, reference, detected)
            # A partir de aquí, verdad reservada exclusivamente a evaluación.
            measured_snr = 10*np.log10(energy_statistic(scaled)/energy_statistic(noise))
            measured_snrs.append(measured_snr)
            if estimate["snr_db"] is not None:
                errors.append(estimate["snr_db"]-measured_snr)
            trial_rows.append({"parent_spawn_key": ".".join(map(str, parent_seed.spawn_key)),
                               "regime": regime, "modulation": modulation, "snr_nominal_db": snr,
                               "detected": detected, "energy": energy_statistic(observed),
                               "signal_power_evaluation_only": energy_statistic(scaled),
                               "noise_power_evaluation_only": energy_statistic(noise),
                               "snr_realized_evaluation_only_db": float(measured_snr),
                               "snr_estimate_db": estimate["snr_db"], "snr_status": estimate["status"]})
            if example is None and regime == "id" and modulation == "QPSK" and snr == 10:
                example = observed.copy()
        lower, upper = wilson_interval(detections, trials)
        rows.append({"regime": regime, "modulation": modulation, "snr_nominal_db": snr,
                     "trials": trials, "detections": detections, "pd": detections/trials,
                     "wilson95_low": lower, "wilson95_high": upper,
                     "mean_measured_snr_db": float(np.mean(measured_snrs))})
        pooled_errors[(regime, snr)].extend(errors)
        snr_rows.append({"regime": regime, "modulation": modulation, "snr_nominal_db": snr,
                         **error_metrics(errors, trials)})
    for regime in ("id", "shift"):
        for snr in config["snr_db"]:
            selected = [r for r in rows if r["regime"] == regime and r["snr_nominal_db"] == snr]
            total = sum(r["trials"] for r in selected)
            hits = sum(r["detections"] for r in selected)
            lower, upper = wilson_interval(hits, total)
            aggregates.append({"regime": regime, "snr_nominal_db": snr, "trials": total,
                               "detections": hits, "pd": hits/total,
                               "wilson95_low": lower, "wilson95_high": upper})
    _write_csv(output/"pd_por_clase.csv", rows)
    _write_csv(output/"trial_results_evaluation_only.csv", trial_rows)
    _write_csv(output/"pd_agregado.csv", aggregates)
    _write_csv(output/"snr_errores_por_clase.csv", snr_rows)
    aggregate_snr = [{"regime": regime, "snr_nominal_db": snr,
                      **error_metrics(errors, trials*len(MAPPERS))}
                     for (regime, snr), errors in pooled_errors.items()]
    _write_csv(output/"snr_errores_agregados.csv", aggregate_snr)
    # Ensayo deliberado de fallo: mismo detector/referencia, ruido real duplicado.
    drift_noise_seed, drift_signal_seed = drift_seed.spawn(2)
    drift_power = power*config["drift_noise_power_factor"]
    drift_noise = _noise(np.random.default_rng(drift_noise_seed), (config["n_noise_evaluation"], n), drift_power)
    drift_false_alarms = sum(detect_energy(row, calibration)["detected"] for row in drift_noise)
    drift_rows = []
    drift_trials = config["drift_signal_trials_per_snr"]
    for snr, cell_seed in zip([0, 10], drift_signal_seed.spawn(2)):
        errors = []
        hits = 0
        for seed in cell_seed.spawn(drift_trials):
            waveform_seed, noise_child = seed.spawn(2)
            clean = _clean_window(config, "QPSK", "id", np.random.default_rng(waveform_seed))
            scaled = clean*np.sqrt(drift_power*10**(snr/10)/energy_statistic(clean))
            noise = _noise(np.random.default_rng(noise_child), n, drift_power)
            observed = scaled+noise
            detected = detect_energy(observed, calibration)["detected"]
            hits += detected
            estimate = estimate_snr(observed, reference, detected)
            measured_snr = 10*np.log10(energy_statistic(scaled)/energy_statistic(noise))
            if estimate["snr_db"] is not None:
                errors.append(estimate["snr_db"]-measured_snr)
        drift_rows.append({"regime": "noise_drift_x2", "modulation": "QPSK", "snr_nominal_db": snr,
                           "detections": hits, **error_metrics(errors, drift_trials)})
    _write_csv(output/"snr_noise_drift.csv", drift_rows)
    fa_interval = wilson_interval(false_alarms, len(eval_noise))
    source_root = Path(__file__).resolve().parents[1]
    source_names = ["src/detection.py", "src/spectrum.py", "src/snr_estimation.py", "src/monitor.py", "src/dataset.py", "src/channels.py", "src/signals.py",
                    "src/waveform.py", "experiments/exp01_detection.py"]
    summary = {"config": config, "calibration": calibration, "noise_reference": reference,
               "seed_spawn_keys": {"calibration": list(cal_seed.spawn_key), "noise_evaluation": list(noise_seed.spawn_key), "signals": list(signal_seed.spawn_key), "drift": list(drift_seed.spawn_key), "bootstrap": list(bootstrap_seed.spawn_key)},
               "false_alarms": false_alarms, "noise_evaluation_trials": len(eval_noise),
               "pfa_measured": false_alarms/len(eval_noise), "pfa_wilson95": fa_interval,
               "signal_trials": len(cells)*trials,
               "noise_drift": {"noise_power_factor": config["drift_noise_power_factor"],
                               "false_alarms": drift_false_alarms, "trials": len(drift_noise),
                               "pfa_measured": drift_false_alarms/len(drift_noise),
                               "pfa_wilson95": wilson_interval(drift_false_alarms, len(drift_noise)),
                               "snr_results": drift_rows},
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
               "source_sha256": {name: hashlib.sha256((source_root/name).read_bytes()).hexdigest() for name in source_names},
               "limits": "E1 controlado: señal escalada por ventana y ruido fijo. IC Wilson condicional al umbral; no incluye incertidumbre entre calibraciones. Errores SNR solo sobre detecciones con estimación válida: sesgo de selección a baja SNR. Percentiles p05/p50/p95 son dispersión de errores. IC bootstrap percentil de bias/MAE/RMSE: remuestreo iid de padres válidos, condicional a n_valid y calibración fija; no cubre deriva ni incertidumbre entre calibraciones. Con n_valid<=1 no hay IC; con pocos válidos la cobertura nominal es aproximada. Referencia estacionaria, sin detección automática de deriva. No evalúa clasificadores."}
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for regime in ("id", "shift"):
        selected = [r for r in aggregates if r["regime"] == regime]
        x = [r["snr_nominal_db"] for r in selected]
        y = [r["pd"] for r in selected]
        ax.plot(x, y, "o-", label=regime.upper())
        ax.fill_between(x, [r["wilson95_low"] for r in selected], [r["wilson95_high"] for r in selected], alpha=.15)
    ax.axhline(summary["pfa_measured"], color="gray", linestyle=":", label="P_FA medido")
    ax.set(xlabel="SNR local nominal (dB)", ylabel="Probabilidad de detección", ylim=(-.02, 1.02),
           title=f"E1: N={n}, piso fijo; P_FA={summary['pfa_measured']:.4f}")
    ax.grid(alpha=.3)
    ax.legend()
    fig.savefig(output/"pd_vs_snr.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    for regime in ("id", "shift"):
        selected = [r for r in aggregate_snr if r["regime"] == regime and r["n_valid"] > 1]
        x = [r["snr_nominal_db"] for r in selected]
        for ax, metric, label in zip(axes, ("bias", "mae"), ("Sesgo (dB)", "MAE (dB)")):
            ax.plot(x, [r[metric+"_db"] for r in selected], "o-", label=regime.upper())
            ax.fill_between(x, [r[metric+"_bootstrap95_low_db"] for r in selected],
                            [r[metric+"_bootstrap95_high_db"] for r in selected], alpha=.15)
            ax.set_ylabel(label)
            ax.grid(alpha=.3)
    axes[0].set_title("SNR: error condicionado a detección válida; IC bootstrap 95%")
    axes[0].axhline(0, color="gray", linestyle=":")
    axes[0].legend()
    axes[1].set_xlabel("SNR local nominal (dB); conteos n_valid en CSV")
    fig.savefig(output/"snr_error_condicional.png", dpi=160)
    plt.close(fig)
    if example is not None:
        spectral = estimate_spectrum(example, config["symbol_rate_hz"]*config["sps"], nfft=1024)
        summary["spectral_example"] = {key: value for key, value in spectral.items() if key not in ("frequency", "psd", "fft")}
        _write_csv(output/"psd_ejemplo.csv", [{"frequency_hz": float(f), "psd_power_per_hz": float(p),
                                            "fft_amplitude_real": float(t.real), "fft_amplitude_imag": float(t.imag)}
                                           for f, p, t in zip(spectral["frequency"], spectral["psd"], spectral["fft"])])
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.plot(spectral["frequency"], 10*np.log10(np.maximum(spectral["psd"], np.finfo(float).tiny)))
        ax.set(xlabel="Frecuencia (Hz)", ylabel="PSD (dB, referencia 1 unidad de potencia/Hz)", title="QPSK recibida, SNR nominal 10 dB; Hann")
        ax.grid(alpha=.3)
        ax.axvspan(*spectral["total_power_bounds"], alpha=.12, color="orange",
                   label=f"99% potencia observada (incluye ruido): {spectral['total_power_bandwidth']:.1f} Hz")
        ax.legend(fontsize=8)
        fig.savefig(output/"psd_ejemplo.png", dpi=160)
        plt.close(fig)
    (output/"config.json").write_text(json.dumps(config, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output/"metricas.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp01_detection.json"))
    parser.add_argument("--output", type=Path, default=Path("results/fase4"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = run_experiment(config, args.output)
    (args.output/"command.json").write_text(json.dumps({"argv": [sys.executable, "-m", "experiments.exp01_detection", "--config", str(args.config), "--output", str(args.output)],
                                                    "cwd": str(Path.cwd())}, indent=2)+"\n", encoding="utf-8")
    print(f"E1: {summary['signal_trials']} tramas; P_FA={summary['pfa_measured']:.6f}; {args.output.resolve()}")


if __name__ == "__main__":
    main()
