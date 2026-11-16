"""Nivel 2 (N2.1): Banco independiente de validación de detección (ventana/archivo) y SNR."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.exp01_detection import _noise, wilson_interval
from src.channels import apply_carrier_impairments, apply_multipath
from src.dataset import MAPPERS
from src.detection import calibrate_energy, detect_energy_file, energy_statistic
from src.snr_estimation import calibrate_noise_reference, estimate_snr
from src.waveform import pulse_shape


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp_n2_1_detection_snr_bank.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "nivel2_validacion" / "n2_1_bank"


def _clean_parent_waveform(config: dict, modulation: str, regime: str, rng: np.random.Generator, total_len: int) -> np.ndarray:
    mapper, k = MAPPERS[modulation]
    symbols = mapper(rng.integers(0, 2, config["symbols_per_parent"] * k))
    beta = float(rng.choice(config["betas"]))
    clean, h = pulse_shape(symbols, config["sps"], beta, config["span"])
    bank = config["shift_channels" if regime == "shift" else "id_channels"]
    pairs = np.asarray(bank[int(rng.integers(len(bank)))])
    taps = pairs[:, 0] + 1j * pairs[:, 1]
    faded = apply_multipath(clean, taps)
    cfo = (
        float(rng.uniform(*config["shift_cfo_abs_range_hz"])) * int(rng.choice([-1, 1]))
        if regime == "shift"
        else float(rng.uniform(*config["id_cfo_range_hz"]))
    )
    fs = float(config["fs_hz"])
    rotated = apply_carrier_impairments(faded, fs, float(rng.uniform(-np.pi, np.pi)), cfo)
    start = len(h) - 1 + len(taps) - 1 + int(rng.integers(config["sps"]))
    stop = start + total_len
    if stop > (config["symbols_per_parent"] - 1) * config["sps"]:
        raise ValueError("Trama demasiado corta para ventana sin transitorios.")
    return rotated[start:stop]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_n2_1(config_path: Path = CONFIG_PATH, output_dir: Path = OUTPUT_DIR) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    window_len = int(config["window_len"])
    file_windows = int(config["file_windows"])
    file_len = window_len * file_windows
    fs_hz = float(config["fs_hz"])
    pfa_target = float(config["pfa_target"])
    base_noise_power = float(config["base_noise_power"])
    gains = [float(g) for g in config["gains"]]

    root_ss = np.random.SeedSequence(int(config["seed"]))
    cal_ss, eval_noise_ss, sig_ss, drift_cal_ss, drift_eval_ss, drift_sig_ss = root_ss.spawn(6)

    # 1. Ruido independiente de calibración y evaluación (base, luego escalado coherentemente por ganancia)
    base_cal_noise = _noise(np.random.default_rng(cal_ss), (int(config["n_noise_calibration"]), file_len), base_noise_power)
    base_eval_noise = _noise(np.random.default_rng(eval_noise_ss), (int(config["n_noise_evaluation"]), file_len), base_noise_power)

    cal_by_gain = {}
    ref_by_gain = {}
    pfa_rows = []
    for g in gains:
        cal_g = g * base_cal_noise
        eval_g = g * base_eval_noise
        # Calibración por ventana de 256 (usando todas las ventanas de calibración 10000x256)
        cal_win = calibrate_energy(cal_g.reshape(-1, window_len), pfa=pfa_target, fs_hz=fs_hz, gain=g)
        ref_g = calibrate_noise_reference(cal_g.ravel(), fs_hz=fs_hz, gain=g, window_len=window_len)
        cal_by_gain[g] = cal_win
        ref_by_gain[g] = ref_g

        fa_win = 0
        fa_file = 0
        for row in eval_g:
            res_win = detect_energy_file(row[:window_len], window_len=window_len, calibration=cal_win, pfa_file=pfa_target, fs_hz=fs_hz, gain=g)
            res_file = detect_energy_file(row, window_len=window_len, calibration=cal_win, pfa_file=pfa_target, fs_hz=fs_hz, gain=g)
            fa_win += int(res_win["detected"])
            fa_file += int(res_file["detected"])

        n_eval = len(eval_g)
        for mode_label, hits in (("single_window_256", fa_win), ("file_2x256_sidak", fa_file)):
            lo, hi = wilson_interval(hits, n_eval)
            pfa_rows.append(
                {
                    "condition": "nominal",
                    "gain": g,
                    "mode": mode_label,
                    "trials": n_eval,
                    "false_alarms": hits,
                    "pfa": hits / n_eval,
                    "wilson95_low": lo,
                    "wilson95_high": hi,
                    "meets_wilson_upper_le_0_02": bool(hi <= 0.02),
                    "clipping_fraction": 0.0,
                }
            )

    # 2. 9.600 padres de señal (6 clases x 8 SNR x 2 regímenes x 100)
    classes = tuple(MAPPERS.keys())
    cells = [(regime, mod, snr) for regime in ("id", "shift") for mod in classes for snr in config["snr_db"]]
    trials = int(config["trials_per_class_snr_regime"])

    pd_cell_rows = []
    snr_cell_rows = []
    trial_obs_rows = []
    pooled_pd = {(regime, snr): {"win_hits": 0, "file_hits": 0, "n": 0} for regime in ("id", "shift") for snr in config["snr_db"]}
    pooled_snr = {(regime, snr): {"err": [], "no_sig": 0, "nonpos": 0, "n": 0} for regime in ("id", "shift") for snr in config["snr_db"]}
    gain_pd_consistency = {g: 0 for g in gains}
    total_parents = len(cells) * trials

    for (regime, mod, snr), cseed in zip(cells, sig_ss.spawn(len(cells))):
        win_hits = 0
        file_hits = 0
        snr_errors = []
        n_no_sig = 0
        n_nonpos = 0

        for idx_p, pseed in enumerate(cseed.spawn(trials)):
            wseed, nseed = pseed.spawn(2)
            clean = _clean_parent_waveform(config, mod, regime, np.random.default_rng(wseed), file_len)
            scaled = clean * np.sqrt(base_noise_power * (10.0 ** (snr / 10.0)) / energy_statistic(clean))
            noise = _noise(np.random.default_rng(nseed), file_len, base_noise_power)
            obs_base = scaled + noise

            # Evaluación principal (gain=1.0) + verificación coherente en ganancias [0.5, 2.0] sobre la misma realización
            det_win_by_g = {}
            det_file_by_g = {}
            snr_by_g = {}
            for g in gains:
                obs_g = g * obs_base
                rw = detect_energy_file(obs_g[:window_len], window_len=window_len, calibration=cal_by_gain[g], pfa_file=pfa_target, fs_hz=fs_hz, gain=g)
                rf = detect_energy_file(obs_g, window_len=window_len, calibration=cal_by_gain[g], pfa_file=pfa_target, fs_hz=fs_hz, gain=g)
                es = estimate_snr(obs_g[:window_len], reference=ref_by_gain[g], detected=rw["detected"], fs_hz=fs_hz, gain=g)
                det_win_by_g[g] = bool(rw["detected"])
                det_file_by_g[g] = bool(rf["detected"])
                snr_by_g[g] = es
                if rf["detected"]:
                    gain_pd_consistency[g] += 1

            # Usar ganancia 1.0 como referencia nominal por padre (las ganancias 0.5 y 2.0 son físicamente idénticas)
            d_w = det_win_by_g[1.0]
            d_f = det_file_by_g[1.0]
            win_hits += int(d_w)
            file_hits += int(d_f)
            est = snr_by_g[1.0]
            realized_snr = float(10.0 * np.log10(energy_statistic(scaled[:window_len]) / energy_statistic(noise[:window_len])))

            pooled_pd[(regime, snr)]["win_hits"] += int(d_w)
            pooled_pd[(regime, snr)]["file_hits"] += int(d_f)
            pooled_pd[(regime, snr)]["n"] += 1
            pooled_snr[(regime, snr)]["n"] += 1

            if est["snr_db"] is not None:
                err = float(est["snr_db"] - realized_snr)
                snr_errors.append(err)
                pooled_snr[(regime, snr)]["err"].append(err)
            elif est["status"] == "no_signal":
                n_no_sig += 1
                pooled_snr[(regime, snr)]["no_sig"] += 1
            elif est["status"] == "nonpositive_signal_power":
                n_nonpos += 1
                pooled_snr[(regime, snr)]["nonpos"] += 1

            parent_id = f"{regime}_{mod}_{snr:+d}_{idx_p:03d}"
            trial_obs_rows.append(
                {
                    "parent_id": parent_id,
                    "regime": regime,
                    "modulation": mod,
                    "snr_nominal_db": snr,
                    "snr_realized_256_db": round(realized_snr, 4),
                    "detected_window_256": int(d_w),
                    "detected_file_2x256": int(d_f),
                    "snr_est_256_db": round(est["snr_db"], 4) if est["snr_db"] is not None else "",
                    "snr_status": est["status"],
                }
            )

        lo_w, hi_w = wilson_interval(win_hits, trials)
        lo_f, hi_f = wilson_interval(file_hits, trials)
        pd_cell_rows.append(
            {
                "regime": regime,
                "modulation": mod,
                "snr_nominal_db": snr,
                "trials": trials,
                "detections_window_256": win_hits,
                "pd_window_256": win_hits / trials,
                "wilson95_low_window": lo_w,
                "wilson95_high_window": hi_w,
                "detections_file_2x256": file_hits,
                "pd_file_2x256": file_hits / trials,
                "wilson95_low_file": lo_f,
                "wilson95_high_file": hi_f,
            }
        )
        err_arr = np.asarray(snr_errors, dtype=float)
        snr_cell_rows.append(
            {
                "regime": regime,
                "modulation": mod,
                "snr_nominal_db": snr,
                "n_total": trials,
                "n_valid": int(len(err_arr)),
                "n_missing_no_signal": n_no_sig,
                "n_missing_nonpositive": n_nonpos,
                "coverage": len(err_arr) / trials,
                "bias_db": float(np.mean(err_arr)) if len(err_arr) else None,
                "mae_db": float(np.mean(np.abs(err_arr))) if len(err_arr) else None,
                "rmse_db": float(np.sqrt(np.mean(err_arr**2))) if len(err_arr) else None,
            }
        )

    pd_snr_rows = []
    snr_snr_rows = []
    for regime in ("id", "shift"):
        for snr in config["snr_db"]:
            p = pooled_pd[(regime, snr)]
            lo_w, hi_w = wilson_interval(p["win_hits"], p["n"])
            lo_f, hi_f = wilson_interval(p["file_hits"], p["n"])
            pd_snr_rows.append(
                {
                    "regime": regime,
                    "snr_nominal_db": snr,
                    "trials": p["n"],
                    "pd_window_256": p["win_hits"] / p["n"],
                    "wilson95_low_window": lo_w,
                    "wilson95_high_window": hi_w,
                    "pd_file_2x256": p["file_hits"] / p["n"],
                    "wilson95_low_file": lo_f,
                    "wilson95_high_file": hi_f,
                }
            )
            s = pooled_snr[(regime, snr)]
            arr = np.asarray(s["err"], dtype=float)
            snr_snr_rows.append(
                {
                    "regime": regime,
                    "snr_nominal_db": snr,
                    "n_total": s["n"],
                    "n_valid": int(len(arr)),
                    "n_missing_no_signal": s["no_sig"],
                    "n_missing_nonpositive": s["nonpos"],
                    "coverage": len(arr) / s["n"],
                    "bias_db": float(np.mean(arr)) if len(arr) else None,
                    "mae_db": float(np.mean(np.abs(arr))) if len(arr) else None,
                    "rmse_db": float(np.sqrt(np.mean(arr**2))) if len(arr) else None,
                }
            )

    # 3. Ensayo de deriva x2 (sin recalibrar vs recalibrado)
    drift_pow = base_noise_power * float(config["drift_noise_power_factor"])
    drift_cal_noise = _noise(np.random.default_rng(drift_cal_ss), (int(config["n_noise_calibration"]), file_len), drift_pow)
    drift_eval_noise = _noise(np.random.default_rng(drift_eval_ss), (int(config["n_noise_evaluation"]), file_len), drift_pow)
    cal_drift_recal = calibrate_energy(drift_cal_noise.reshape(-1, window_len), pfa=pfa_target, fs_hz=fs_hz, gain=1.0)
    ref_drift_recal = calibrate_noise_reference(drift_cal_noise.ravel(), fs_hz=fs_hz, gain=1.0, window_len=window_len)

    drift_rows = []
    for scenario_name, cal_used, ref_used in (
        ("drift_x2_uncalibrated_stale_ref", cal_by_gain[1.0], ref_by_gain[1.0]),
        ("drift_x2_recalibrated_ref", cal_drift_recal, ref_drift_recal),
    ):
        fa_win = 0
        fa_file = 0
        for row in drift_eval_noise:
            fa_win += int(detect_energy_file(row[:window_len], window_len=window_len, calibration=cal_used, pfa_file=pfa_target, fs_hz=fs_hz, gain=1.0)["detected"])
            fa_file += int(detect_energy_file(row, window_len=window_len, calibration=cal_used, pfa_file=pfa_target, fs_hz=fs_hz, gain=1.0)["detected"])
        lo_w, hi_w = wilson_interval(fa_win, len(drift_eval_noise))
        lo_f, hi_f = wilson_interval(fa_file, len(drift_eval_noise))
        drift_rows.append(
            {
                "scenario": scenario_name,
                "metric_type": "pfa_window_256",
                "snr_nominal_db": "",
                "trials": len(drift_eval_noise),
                "hits_or_valid": fa_win,
                "rate": fa_win / len(drift_eval_noise),
                "wilson95_low": lo_w,
                "wilson95_high": hi_w,
                "bias_db": "",
                "mae_db": "",
            }
        )
        drift_rows.append(
            {
                "scenario": scenario_name,
                "metric_type": "pfa_file_2x256",
                "snr_nominal_db": "",
                "trials": len(drift_eval_noise),
                "hits_or_valid": fa_file,
                "rate": fa_file / len(drift_eval_noise),
                "wilson95_low": lo_f,
                "wilson95_high": hi_f,
                "bias_db": "",
                "mae_db": "",
            }
        )

    # Señales bajo deriva x2 (SNR nominal respecto al nuevo piso 2.0)
    drift_cells = [(mod, snr) for mod in classes for snr in config["drift_snr_db"]]
    drift_trials = int(config["drift_trials_per_class_snr"])
    for scenario_name, cal_used, ref_used in (
        ("drift_x2_uncalibrated_stale_ref", cal_by_gain[1.0], ref_by_gain[1.0]),
        ("drift_x2_recalibrated_ref", cal_drift_recal, ref_drift_recal),
    ):
        errs_by_snr = {s: [] for s in config["drift_snr_db"]}
        for (mod, snr), cseed in zip(drift_cells, drift_sig_ss.spawn(len(drift_cells))):
            for pseed in cseed.spawn(drift_trials):
                wseed, nseed = pseed.spawn(2)
                clean = _clean_parent_waveform(config, mod, "id", np.random.default_rng(wseed), window_len)
                scaled = clean * np.sqrt(drift_pow * (10.0 ** (snr / 10.0)) / energy_statistic(clean))
                noise = _noise(np.random.default_rng(nseed), window_len, drift_pow)
                obs = scaled + noise
                det = detect_energy_file(obs, window_len=window_len, calibration=cal_used, pfa_file=pfa_target, fs_hz=fs_hz, gain=1.0)
                est = estimate_snr(obs, reference=ref_used, detected=det["detected"], fs_hz=fs_hz, gain=1.0)
                if est["snr_db"] is not None:
                    realized = float(10.0 * np.log10(energy_statistic(scaled) / energy_statistic(noise)))
                    errs_by_snr[snr].append(float(est["snr_db"] - realized))
        for snr in config["drift_snr_db"]:
            arr = np.asarray(errs_by_snr[snr], dtype=float)
            n_tot = len(classes) * drift_trials
            drift_rows.append(
                {
                    "scenario": scenario_name,
                    "metric_type": "snr_estimation",
                    "snr_nominal_db": snr,
                    "trials": n_tot,
                    "hits_or_valid": int(len(arr)),
                    "rate": len(arr) / n_tot,
                    "wilson95_low": "",
                    "wilson95_high": "",
                    "bias_db": float(np.mean(arr)) if len(arr) else None,
                    "mae_db": float(np.mean(np.abs(arr))) if len(arr) else None,
                }
            )

    # Guardar archivos
    _write_csv(output_dir / "pfa_by_gain_and_mode.csv", pfa_rows)
    _write_csv(output_dir / "pd_by_cell.csv", pd_cell_rows)
    _write_csv(output_dir / "pd_by_snr.csv", pd_snr_rows)
    _write_csv(output_dir / "snr_by_cell.csv", snr_cell_rows)
    _write_csv(output_dir / "snr_by_snr.csv", snr_snr_rows)
    _write_csv(output_dir / "drift_results.csv", drift_rows)
    _write_csv(output_dir / "trial_observations.csv", trial_obs_rows)

    # Gráfica Pd y MAE SNR
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for regime, col, ls in (("id", "royalblue", "-"), ("shift", "darkorange", "--")):
        sub = [r for r in pd_snr_rows if r["regime"] == regime]
        xs = [r["snr_nominal_db"] for r in sub]
        axes[0].plot(xs, [r["pd_window_256"] for r in sub], marker="o", color=col, linestyle=ls, label=f"{regime} (ventana 256)")
        axes[0].plot(xs, [r["pd_file_2x256"] for r in sub], marker="s", color=col, linestyle=":", alpha=0.8, label=f"{regime} (archivo 2x256)")
    axes[0].axhline(0.95, color="green", linestyle="--", lw=1.0, label="Meta 95%")
    axes[0].set_title("N2.1 Curva P_D vs SNR nominal")
    axes[0].set_xlabel("SNR nominal (dB)")
    axes[0].set_ylabel("Probabilidad de detección (P_D)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8)

    for regime, col, ls in (("id", "royalblue", "-"), ("shift", "darkorange", "--")):
        sub = [r for r in snr_snr_rows if r["regime"] == regime and r["mae_db"] is not None]
        axes[1].plot([r["snr_nominal_db"] for r in sub], [r["mae_db"] for r in sub], marker="o", color=col, linestyle=ls, label=f"{regime} MAE (dB)")
    axes[1].axhline(3.0, color="crimson", linestyle="--", lw=1.0, label="Meta MAE <= 3 dB (SNR>=0)")
    axes[1].set_title("N2.1 Error absoluto medio SNR (casos válidos)")
    axes[1].set_xlabel("SNR nominal (dB)")
    axes[1].set_ylabel("MAE (dB)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "pd_and_snr_curves.png", dpi=160)
    plt.close(fig)

    id_ge0_cells = [r for r in pd_cell_rows if r["regime"] == "id" and r["snr_nominal_db"] >= 0]
    min_wilson_low_win = min(r["wilson95_low_window"] for r in id_ge0_cells)
    min_wilson_low_file = min(r["wilson95_low_file"] for r in id_ge0_cells)

    id_ge0_snr_rows = [r for r in snr_snr_rows if r["regime"] == "id" and r["snr_nominal_db"] >= 0]
    id_ge0_total = sum(r["n_total"] for r in id_ge0_snr_rows)
    id_ge0_valid = sum(r["n_valid"] for r in id_ge0_snr_rows)
    id_ge0_mae_weighted = sum(r["mae_db"] * r["n_valid"] for r in id_ge0_snr_rows) / id_ge0_valid

    metrics_summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(config["seed"]),
        "n_noise_calibration": int(config["n_noise_calibration"]),
        "n_noise_evaluation": int(config["n_noise_evaluation"]),
        "total_signal_parents": total_parents,
        "pfa_all_nominal_wilson_upper_le_0_02": all(r["meets_wilson_upper_le_0_02"] for r in pfa_rows),
        "pfa_rows": pfa_rows,
        "id_snr_ge_0db_24_cells_min_wilson95_low_window": min_wilson_low_win,
        "id_snr_ge_0db_24_cells_min_wilson95_low_file": min_wilson_low_file,
        "all_id_ge_0db_cells_meet_wilson_low_ge_0_95": bool(min_wilson_low_win >= 0.95 and min_wilson_low_file >= 0.95),
        "gain_pd_consistency_counts": {str(g): int(c) for g, c in gain_pd_consistency.items()},
        "gain_scale_invariance_exact_across_all_9600": len(set(gain_pd_consistency.values())) == 1,
        "snr_id_ge_0db_coverage": id_ge0_valid / id_ge0_total,
        "snr_id_ge_0db_mae_db": id_ge0_mae_weighted,
        "snr_meets_n1_3_targets": bool((id_ge0_valid / id_ge0_total) >= 0.90 and id_ge0_mae_weighted <= 3.0),
        "drift_rows": drift_rows,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metrics_summary


if __name__ == "__main__":
    summary = run_n2_1()
    print(json.dumps({k: v for k, v in summary.items() if k not in ("pfa_rows", "drift_rows")}, indent=2, ensure_ascii=False))
