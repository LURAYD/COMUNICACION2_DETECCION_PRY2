"""Regresión física de E1 y API de Fase 4 con observaciones públicas."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.exp01_detection import _noise, run_experiment
from src.dataset import load_observation
from src.detection import calibrate_energy, energy_statistic
from src.monitor import analyze
from src.snr_estimation import calibrate_noise_reference


def test_monitor_observacion_publica_sin_oracle(tmp_path):
    (tmp_path/"observations").mkdir()
    noise = _noise(np.random.default_rng(301), (500, 256), 1)
    iq = np.sqrt(10)*np.exp(2j*np.pi*.07*np.arange(256))+noise[0]
    np.savez(tmp_path/"observations/test.npz", iq256=iq[None, :])
    (tmp_path/"public_metadata.json").write_text(json.dumps({"fs_hz": 8000}), encoding="utf-8")
    with (tmp_path/"index.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["record_id", "row", "observation_path"])
        writer.writeheader()
        writer.writerow({"record_id": "opaque", "row": 0, "observation_path": "observations/test.npz"})
    observation = load_observation(tmp_path, "opaque")
    reference_iq = noise[1:]
    result = analyze(**observation, detection_calibration=calibrate_energy(reference_iq),
                     noise_reference=calibrate_noise_reference(reference_iq.ravel()))
    assert result["detection"]["detected"]
    assert result["snr"]["snr_db"] == pytest.approx(10, abs=1)
    assert result["spectrum"]["frequency_unit"] == "Hz"
    assert analyze(**observation)["snr"]["status"] == "detection_unknown"
    assert not (tmp_path/"oracle").exists()


def test_e1_reproducible_potencia_local_y_awgn_no_forzado(tmp_path):
    config = json.loads((Path(__file__).resolve().parents[1]/"configs/exp01_detection.json").read_text())
    config.update(n_calibration=64, n_noise_evaluation=64,
                  trials_per_class_snr_regime=2, drift_signal_trials_per_snr=2,
                  bootstrap_resamples=50, snr_db=[-20, 0, 10])
    first = run_experiment(config, tmp_path/"first")
    second = run_experiment(config, tmp_path/"second")
    assert first == second
    assert first["signal_trials"] == 72
    assert first["seed_spawn_keys"] == {"calibration": [0], "noise_evaluation": [1],
                                         "signals": [2], "drift": [3], "bootstrap": [4]}
    with (tmp_path/"first/trial_results_evaluation_only.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len({r["parent_spawn_key"] for r in rows}) == 72
    for row in rows:
        assert float(row["signal_power_evaluation_only"]) == pytest.approx(10**(float(row["snr_nominal_db"])/10))
    noise_powers = np.array([float(r["noise_power_evaluation_only"]) for r in rows])
    assert np.std(noise_powers) > .02  # AWGN no se normaliza por realización.
    assert np.mean(noise_powers) == pytest.approx(1, abs=.04)
    assert first["spectral_example"]["total_power_bandwidth"] > 0
    with pytest.raises(FileExistsError):
        run_experiment(config, tmp_path/"first")


def test_awgn_complejo_circular_independiente():
    noise = _noise(np.random.default_rng(302), 100000, 1)
    assert energy_statistic(noise) == pytest.approx(1, abs=.015)
    assert np.var(noise.real) == pytest.approx(.5, abs=.01)
    assert np.var(noise.imag) == pytest.approx(.5, abs=.01)
    assert abs(np.mean(noise**2)) < .015
    assert abs(np.mean(noise[1:]*noise[:-1].conj())) < .015
