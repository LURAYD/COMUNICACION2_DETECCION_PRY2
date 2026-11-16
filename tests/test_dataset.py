"""Pruebas de integridad, reproducibilidad y aislamiento público del smoke."""
import copy
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from src.dataset import (MAPPERS, SPLITS, audit_dataset, generate_dataset,
                         load_observation, load_truth, validate_config)


@pytest.fixture(scope="module")
def config():
    path = Path(__file__).resolve().parents[1]/"configs/dataset_smoke.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dataset(tmp_path_factory, config):
    root = tmp_path_factory.mktemp("dataset_parent")/"smoke"
    audit = generate_dataset(config, root)
    return root, audit


def _index(root):
    with (root/"index.csv").open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_index(root, rows):
    with (root/"index.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_truth(root, rows):
    (root/"oracle/manifest.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")


def test_plan_balance_y_prefijos(dataset):
    root, audit = dataset
    assert audit["parents"] == 224
    assert audit["windows"] == 448
    rows = _index(root)
    parents = [{r["parent_waveform_id"] for r in rows if r["split"] == split} for split in SPLITS]
    assert all(len(group) == 56 for group in parents)
    assert len(set.union(*parents)) == 224
    for split in SPLITS:
        summary = audit["splits"][split]
        assert summary["signal_parents"] == 48
        assert summary["noise_only_parents"] == 8
        assert len(summary["class_snr_counts"]) == 24
        assert all(cell["count"] == 2 for cell in summary["class_snr_counts"])
        with np.load(root/f"observations/{split}.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["iq256"], arrays["iq512"][:, :256])
    # La fila no codifica un bloque ordenado por etiqueta.
    truth = load_truth(root)
    train_labels = [r["modulation"] or "noise" for r in truth if r["split"] == "train"]
    assert train_labels != sorted(train_labels)


def test_reproducibilidad_semilla_y_namespace(dataset, config, tmp_path):
    root, audit = dataset
    repeated = tmp_path/"repeated"
    repeat_audit = generate_dataset(config, repeated)
    assert repeat_audit == audit
    assert (root/"index.csv").read_bytes() == (repeated/"index.csv").read_bytes()
    assert load_truth(root) == load_truth(repeated)
    changed = copy.deepcopy(config)
    changed["seed"] += 1
    other = tmp_path/"different_seed"
    other_audit = generate_dataset(changed, other)
    assert all(audit["splits"][s]["iq512_sha256"] != other_audit["splits"][s]["iq512_sha256"] for s in SPLITS)
    # Mismo seed, distinta configuración física: tampoco reutiliza IDs de padres.
    changed["seed"] = config["seed"]
    changed["betas"] = [.2, .35, .5]
    alternate = tmp_path/"different_config"
    generate_dataset(changed, alternate)
    assert not ({r["parent_waveform_id"] for r in _index(root)} &
                {r["parent_waveform_id"] for r in _index(alternate)})


def test_api_publica_funciona_sin_oracle(dataset, tmp_path):
    root, _ = dataset
    public = tmp_path/"public"
    public.mkdir()
    shutil.copytree(root/"observations", public/"observations")
    for name in ("index.csv", "public_metadata.json"):
        shutil.copy2(root/name, public/name)
    record_id = _index(public)[0]["record_id"]
    short = load_observation(public, record_id)
    long = load_observation(public, record_id, 512)
    assert set(short) == set(long) == {"iq", "fs_hz"}
    assert short["fs_hz"] == 8000
    np.testing.assert_array_equal(short["iq"], long["iq"][:256])
    with pytest.raises(KeyError):
        load_observation(public, "missing")
    with pytest.raises(ValueError):
        load_observation(public, record_id, 128)


def test_noise_only_estadistica_y_nulos(dataset):
    root, _ = dataset
    truths = [row for row in load_truth(root) if not row["has_signal"]]
    normalized = []
    for truth in truths:
        assert truth["modulation"] is truth["snr_true_db"] is truth["snr_measured_window_db"] is None
        assert truth["taps_real_imag"] is truth["cfo_hz"] is truth["beta"] is None
        expected_power = truth["noise_scale_reference_amplitude"]**2/(8*10**(truth["noise_scale_reference_db"]/10))
        assert truth["noise_power_target"] == pytest.approx(expected_power)
        iq = load_observation(root, truth["record_id"], 512)["iq"]
        normalized.append(iq/np.sqrt(expected_power))
    z = np.concatenate(normalized)
    # 32 realizaciones independientes, 16384 muestras; tolerancias estadísticas.
    assert np.mean(abs(z)**2) == pytest.approx(1, abs=.035)
    assert np.var(z.real) == pytest.approx(.5, abs=.025)
    assert np.var(z.imag) == pytest.approx(.5, abs=.025)
    assert abs(np.mean(z)) < .03
    assert abs(np.corrcoef(z.real, z.imag)[0, 1]) < .03


def test_shift_y_recorte_soporte_completo(dataset, config):
    root, _ = dataset
    id_delays, shift_delays, streams = set(), set(), []
    for truth in load_truth(root):
        streams.extend(tuple(v) for v in truth["stream_spawn_keys"].values())
        if not truth["has_signal"]:
            continue
        delay = len(truth["taps_real_imag"])-1
        assert truth["crop_start"] == 80+delay+truth["timing_offset_samples"]
        assert truth["crop_stop"] == truth["crop_start"]+512
        assert truth["crop_stop"] <= (config["symbols_per_parent"]-1)*8
        if truth["split"] == "test_shift":
            shift_delays.add(delay)
            assert 200 <= abs(truth["cfo_hz"]) <= 300
        else:
            id_delays.add(delay)
            assert abs(truth["cfo_hz"]) <= 100
    assert id_delays == {0, 2, 5}
    assert shift_delays == {9, 13}
    assert len(streams) == len(set(streams)) == 224*3


@pytest.mark.parametrize("kind", ["parent", "row", "path", "prefix", "duplicate_iq", "streams", "metadata", "crop", "shift_taps"])
def test_auditoria_detecta_corrupcion(dataset, tmp_path, kind):
    original, _ = dataset
    root = tmp_path/"corrupt"
    shutil.copytree(original, root)
    if kind in {"parent", "row", "path"}:
        rows = _index(root)
        if kind == "parent":
            rows[1]["parent_waveform_id"] = rows[0]["parent_waveform_id"]
        elif kind == "row":
            rows[1]["row"] = rows[0]["row"]
        else:
            rows[0]["observation_path"] = "observations/test.npz"
        _write_index(root, rows)
    elif kind in {"prefix", "duplicate_iq"}:
        path = root/"observations/train.npz"
        with np.load(path) as arrays:
            short, long = arrays["iq256"].copy(), arrays["iq512"].copy()
        if kind == "prefix":
            short[0, 0] += 1
        else:
            short[1] = short[0]
            long[1] = long[0]
        np.savez(path, iq256=short, iq512=long)
    elif kind == "metadata":
        (root/"public_metadata.json").write_text(json.dumps({"fs_hz": 16000, "window_lengths": [256, 512]}))
    else:
        rows = load_truth(root)
        if kind == "streams":
            rows[0]["stream_spawn_keys"]["noise"] = rows[1]["stream_spawn_keys"]["noise"]
        elif kind == "crop":
            next(r for r in rows if r["has_signal"])["crop_start"] = 0
        else:
            next(r for r in rows if r["has_signal"] and r["split"] == "test_shift")["taps_real_imag"] = [[1, 0]]
        _write_truth(root, rows)
    with pytest.raises(ValueError):
        audit_dataset(root)


def test_config_invalida_y_no_sobrescritura(dataset, config, tmp_path):
    root, _ = dataset
    with pytest.raises(FileExistsError):
        generate_dataset(config, root)
    for key, value in [("symbols_per_parent", 64), ("shift_cfo_abs_range_hz", [50, 100]),
                       ("shift_channels", [[[1, 0]]]), ("betas", [0]), ("sps", True)]:
        bad = copy.deepcopy(config)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_config(bad)
