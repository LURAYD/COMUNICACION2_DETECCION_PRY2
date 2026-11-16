"""Dataset de humo reproducible; no entrena modelos ni estima SNR a ciegas.

CLI: python -m src.dataset --config configs/dataset_smoke.json --output data/fase3_smoke
La API load_observation solo entrega iq/fs_hz. Los archivos oracle e index
son administrativos: su separación es lógica, no un control de acceso seguro.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from src import signals
from src.channels import add_awgn, apply_carrier_impairments, apply_multipath
from src.waveform import pulse_shape


SPLITS = ("train", "val", "test", "test_shift")
MAPPERS = {"BPSK": (signals.bpsk_modulation, 1), "QPSK": (signals.qpsk_modulation, 2),
           "8PSK": (signals.psk8_modulation, 3), "16QAM": (signals.qam16_modulation, 4),
           "64QAM": (signals.qam64_modulation, 6), "4PAM": (signals.pam4_modulation, 2)}


def _json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _taps(pairs):
    array = np.asarray(pairs, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError("Cada canal requiere pares [real, imag] finitos y no vacíos.")
    taps = array[:, 0] + 1j*array[:, 1]
    if not np.any(taps) or taps[-1] == 0:
        raise ValueError("El canal debe ser no nulo, sin ceros finales redundantes.")
    return taps


def validate_config(config):
    """Valida soporte, unidades y margen suficiente para eliminar transitorios."""
    if config.get("schema_version") != 1:
        raise ValueError("schema_version debe ser 1.")
    for key in ("seed", "sps", "span", "symbols_per_parent", "realizations_per_class_snr", "noise_only_per_split"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "seed" else 1):
            raise ValueError(f"{key} debe ser entero válido.")
    if config["sps"] < 2 or config["span"] % 2:
        raise ValueError("sps>=2 y span par son necesarios para RRC.")
    if config["window_lengths"] != [256, 512]:
        raise ValueError("Este contrato requiere ventanas [256, 512].")
    rs = config["symbol_rate_hz"]
    if isinstance(rs, bool) or not isinstance(rs, (int, float)) or not np.isfinite(rs) or rs <= 0:
        raise ValueError("symbol_rate_hz debe ser positiva y finita.")
    for key in ("betas", "snr_db"):
        values = config[key]
        if (not isinstance(values, list) or not values or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{key} requiere valores reales finitos distintos.")
    if not all(0 < b <= 1 for b in config["betas"]):
        raise ValueError("Los beta deben cumplir 0<beta<=1.")
    for key in ("amplitude_range", "phase_range_rad", "id_cfo_range_hz", "shift_cfo_abs_range_hz"):
        values = config[key]
        if (not isinstance(values, list) or len(values) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in values)
                or values[0] >= values[1]):
            raise ValueError(f"{key} requiere límites reales crecientes.")
    if config["amplitude_range"][0] <= 0:
        raise ValueError("La amplitud debe ser positiva.")
    id_max = max(abs(v) for v in config["id_cfo_range_hz"])
    if config["shift_cfo_abs_range_hz"][0] <= id_max:
        raise ValueError("El CFO shift debe quedar fuera del soporte ID.")
    fs = rs*config["sps"]
    if max(id_max, config["shift_cfo_abs_range_hz"][1]) + rs*(1+max(config["betas"]))/2 >= fs/2:
        raise ValueError("El espectro nominal desplazado debe caber dentro de Nyquist.")
    banks = {}
    for key in ("id_channels", "shift_channels"):
        if not isinstance(config[key], list) or not config[key]:
            raise ValueError(f"{key} no puede estar vacío.")
        banks[key] = [_taps(pairs) for pairs in config[key]]
    max_id_delay = max(len(t)-1 for t in banks["id_channels"])
    if any(len(t)-1 <= max_id_delay for t in banks["shift_channels"]):
        raise ValueError("Cada canal shift debe incluir un retardo fuera del soporte ID.")
    max_delay = max(len(t)-1 for bank in banks.values() for t in bank)
    # Última muestra de ventana antes del último impulso emitido: condición conservadora.
    latest_stop = config["span"]*config["sps"] + max_delay + config["sps"]-1 + 512
    if latest_stop > (config["symbols_per_parent"]-1)*config["sps"]:
        raise ValueError("La trama es demasiado corta para recortar transitorios y 512 muestras.")
    return config


def _make_plan(config):
    """Asigna split, clase/SNR y semilla madre antes de generar observaciones."""
    root = np.random.SeedSequence(config["seed"])
    namespace = _canonical_hash(config)[:16]
    plan = []
    for split, split_seed in zip(SPLITS, root.spawn(len(SPLITS))):
        descriptors = [(name, snr) for name in MAPPERS for snr in config["snr_db"]
                       for _ in range(config["realizations_per_class_snr"])]
        descriptors += [(None, None)]*config["noise_only_per_split"]
        order_seed, parents_seed = split_seed.spawn(2)
        parents = parents_seed.spawn(len(descriptors))
        # IDs opacos no incorporan clase ni SNR. La semilla completa está solo en oracle.
        records = []
        for (name, snr), seed in zip(descriptors, parents):
            opaque = namespace + "_" + seed.generate_state(4).tobytes().hex()
            records.append({"split": split, "modulation": name, "snr": snr,
                            "parent_waveform_id": "p_"+opaque, "record_id": "r_"+opaque,
                            "seed": seed})
        order = np.random.default_rng(order_seed).permutation(len(records))
        plan.extend(records[i] for i in order)
    return plan


def _generate_parent(config, record):
    streams = record["seed"].spawn(3)
    params, bits_rng, noise_rng = [np.random.default_rng(seed) for seed in streams]
    snr = record["snr"]
    name = record["modulation"]
    fs = config["symbol_rate_hz"]*config["sps"]
    truth = {key: record[key] for key in ("record_id", "parent_waveform_id", "split", "modulation")}
    truth.update({"has_signal": name is not None, "snr_true_db": snr,
                  "snr_measured_window_db": None, "seed_entropy": config["seed"],
                  "parent_spawn_key": list(record["seed"].spawn_key),
                  "stream_spawn_keys": {label: list(seed.spawn_key) for label, seed in zip(("parameters", "bits", "noise"), streams)}})
    if name is None:
        reference_amplitude = float(params.uniform(*config["amplitude_range"]))
        reference_db = float(params.choice(config["snr_db"]))
        noise_power = reference_amplitude**2/(config["sps"]*10**(reference_db/10))
        iq = np.sqrt(noise_power/2)*(noise_rng.standard_normal(512)+1j*noise_rng.standard_normal(512))
        truth.update({"noise_power_target": noise_power,
                      "noise_scale_reference_amplitude": reference_amplitude,
                      "noise_scale_reference_db": reference_db,
                      "beta": None, "amplitude": None, "phase_rad": None, "cfo_hz": None,
                      "channel_id": None, "taps_real_imag": None,
                      "crop_start": 0, "crop_stop": 512, "full_length": 512,
                      "noise_power_measured_window": float(np.mean(abs(iq)**2))})
        return iq, truth
    beta = float(params.choice(config["betas"]))
    amplitude = float(params.uniform(*config["amplitude_range"]))
    phase = float(params.uniform(*config["phase_range_rad"]))
    shifted = record["split"] == "test_shift"
    cfo = (float(params.uniform(*config["shift_cfo_abs_range_hz"]))*int(params.choice([-1, 1]))
           if shifted else float(params.uniform(*config["id_cfo_range_hz"])))
    bank_name = "shift_channels" if shifted else "id_channels"
    channel_index = int(params.integers(len(config[bank_name])))
    pairs = config[bank_name][channel_index]
    taps = _taps(pairs)
    mapper, k = MAPPERS[name]
    symbols = mapper(bits_rng.integers(0, 2, config["symbols_per_parent"]*k))
    clean, h = pulse_shape(symbols, config["sps"], beta, config["span"])
    faded = apply_multipath(amplitude*clean, taps)
    rotated = apply_carrier_impairments(faded, fs, phase, cfo)
    received, noise = add_awgn(rotated, snr, noise_rng)
    timing_offset = int(params.integers(config["sps"]))
    start = len(h)-1 + len(taps)-1 + timing_offset
    stop = start+512
    signal_window = rotated[start:stop]
    noise_window = noise[start:stop]
    signal_power = float(np.mean(abs(signal_window)**2))
    noise_power = float(np.mean(abs(noise_window)**2))
    truth.update({"beta": beta, "amplitude": amplitude, "phase_rad": phase, "cfo_hz": cfo,
                  "channel_id": f"{bank_name}_{channel_index}", "taps_real_imag": pairs,
                  "symbol_rate_hz": config["symbol_rate_hz"], "sps": config["sps"], "span": config["span"],
                  "rrc_length": len(h), "rrc_group_delay": (len(h)-1)//2,
                  "timing_offset_samples": timing_offset,
                  "crop_start": start, "crop_stop": stop, "full_length": len(received),
                  "noise_power_target": float(np.mean(abs(rotated)**2)/10**(snr/10)),
                  "signal_power_window": signal_power, "noise_power_measured_window": noise_power,
                  "snr_measured_window_db": float(10*np.log10(signal_power/noise_power))})
    return received[start:stop], truth


def generate_dataset(config, output):
    """Crea una carpeta nueva; nunca reemplaza un dataset ya existente."""
    validate_config(config)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Ya existe {output}; use una carpeta nueva.")
    plan = _make_plan(config)
    output.mkdir(parents=True)
    (output/"observations").mkdir()
    (output/"oracle").mkdir()
    _json_write(output/"config.json", config)
    _json_write(output/"public_metadata.json", {"fs_hz": config["symbol_rate_hz"]*config["sps"],
                                               "window_lengths": config["window_lengths"]})
    index, truths = [], []
    for split in SPLITS:
        arrays = []
        for row, record in enumerate(r for r in plan if r["split"] == split):
            iq, truth = _generate_parent(config, record)
            arrays.append(iq)
            truths.append(truth)
            index.append({"record_id": record["record_id"], "parent_waveform_id": record["parent_waveform_id"],
                          "split": split, "row": row, "observation_path": f"observations/{split}.npz"})
        iq512 = np.asarray(arrays, dtype=np.complex128)
        np.savez(output/f"observations/{split}.npz", iq256=iq512[:, :256], iq512=iq512)
    with (output/"index.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    with (output/"oracle/manifest.jsonl").open("w", encoding="utf-8") as file:
        for truth in truths:
            file.write(json.dumps(truth, ensure_ascii=False, allow_nan=False)+"\n")
    audit = audit_dataset(output)
    _json_write(output/"audit.json", audit)
    return audit


def _read_index(root):
    with (Path(root)/"index.csv").open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_observation(root, record_id, n_samples=256):
    """Solo I/Q y fs conocida; no abre config ni oracle ni entrega índices."""
    if isinstance(n_samples, bool) or n_samples not in (256, 512):
        raise ValueError("n_samples debe ser 256 o 512.")
    root = Path(root)
    row = next((row for row in _read_index(root) if row["record_id"] == record_id), None)
    if row is None:
        raise KeyError(record_id)
    metadata = json.loads((root/"public_metadata.json").read_text(encoding="utf-8"))
    with np.load(root/row["observation_path"], allow_pickle=False) as arrays:
        iq = arrays[f"iq{n_samples}"][int(row["row"])].copy()
    return {"iq": iq, "fs_hz": metadata["fs_hz"]}


def load_truth(root):
    """Acceso oracle explícito para supervisión/evaluación, nunca para monitor."""
    with (Path(root)/"oracle/manifest.jsonl").open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def audit_dataset(root):
    """Comprueba integridad, independencia de IDs/streams y prefijos; no prueba ML."""
    root = Path(root)
    config = validate_config(json.loads((root/"config.json").read_text(encoding="utf-8")))
    index = _read_index(root)
    truth = load_truth(root)
    metadata = json.loads((root/"public_metadata.json").read_text(encoding="utf-8"))
    expected_metadata = {"fs_hz": config["symbol_rate_hz"]*config["sps"], "window_lengths": [256, 512]}
    if metadata != expected_metadata:
        raise ValueError("Metadatos públicos incorrectos o contienen campos adicionales.")
    planned = _make_plan(config)
    if len(index) != len(truth):
        raise ValueError("Índice y oracle tienen distinto tamaño.")
    ids = [row["record_id"] for row in index]
    parents = [row["parent_waveform_id"] for row in index]
    seeds = [tuple(row["parent_spawn_key"]) for row in truth]
    if len(set(ids)) != len(ids) or len(set(parents)) != len(parents) or len(set(seeds)) != len(seeds):
        raise ValueError("Hay IDs/padres/semillas madre repetidos.")
    truth_by_id = {row["record_id"]: row for row in truth}
    if set(truth_by_id) != set(ids):
        raise ValueError("Oracle e índice no coinciden.")
    if len(index) != len(planned) or set(ids) != {r["record_id"] for r in planned}:
        raise ValueError("IDs de registros no corresponden al plan/configuración.")
    all_stream_keys = []
    for expected in planned:
        item = truth_by_id[expected["record_id"]]
        expected_keys = {label: list(seed.spawn_key) for label, seed in
                         zip(("parameters", "bits", "noise"), expected["seed"].spawn(3))}
        if (item["seed_entropy"] != config["seed"] or item["parent_spawn_key"] != list(expected["seed"].spawn_key)
                or item["stream_spawn_keys"] != expected_keys):
            raise ValueError("Linaje de semillas/streams incorrecto.")
        all_stream_keys.extend(tuple(key) for key in item["stream_spawn_keys"].values())
        if (item["parent_waveform_id"] != expected["parent_waveform_id"]
                or item["modulation"] != expected["modulation"] or item["snr_true_db"] != expected["snr"]
                or item["has_signal"] != (expected["modulation"] is not None)):
            raise ValueError("Etiquetas/SNR no corresponden al plan.")
    if len(set(all_stream_keys)) != len(all_stream_keys):
        raise ValueError("Hay streams aleatorios reutilizados.")
    summaries = {}
    waveform_hashes = set()
    for split in SPLITS:
        records = [row for row in index if row["split"] == split]
        with np.load(root/f"observations/{split}.npz", allow_pickle=False) as data:
            if set(data.files) != {"iq256", "iq512"}:
                raise ValueError("Las observaciones solo deben contener I/Q.")
            short, long = data["iq256"], data["iq512"]
            if short.shape != (len(records), 256) or long.shape != (len(records), 512):
                raise ValueError("Longitudes de observación incorrectas.")
            if short.dtype != np.complex128 or long.dtype != np.complex128:
                raise ValueError("El almacenamiento I/Q requiere complex128.")
            if not np.isfinite(long).all() or not np.array_equal(short, long[:, :256]):
                raise ValueError("Prefijos inconsistentes o I/Q no finita.")
            payload_hash = hashlib.sha256(long.tobytes()).hexdigest()
            for iq in long:
                row_hash = hashlib.sha256(iq.tobytes()).hexdigest()
                if row_hash in waveform_hashes:
                    raise ValueError("Hay I/Q512 exactamente duplicada entre padres.")
                waveform_hashes.add(row_hash)
        counts = Counter()
        local_snr = []
        noise_count = 0
        expected_split = [r for r in planned if r["split"] == split]
        for position, row in enumerate(records):
            if (row["row"] != str(position) or row["observation_path"] != f"observations/{split}.npz"
                    or row["record_id"] != expected_split[position]["record_id"]):
                raise ValueError("Fila/ruta/orden del índice inconsistente con el plan.")
            item = truth_by_id[row["record_id"]]
            if item["split"] != split or item["parent_waveform_id"] != row["parent_waveform_id"]:
                raise ValueError("Un padre o registro pertenece a otro split.")
            if item["has_signal"]:
                counts[(item["modulation"], item["snr_true_db"])] += 1
                local_snr.append(item["snr_measured_window_db"])
                cfo = item["cfo_hz"]
                if split == "test_shift":
                    lo, hi = config["shift_cfo_abs_range_hz"]
                    if not lo <= abs(cfo) <= hi or not item["channel_id"].startswith("shift_channels_"):
                        raise ValueError("Canal shift fuera del soporte configurado.")
                elif not config["id_cfo_range_hz"][0] <= cfo <= config["id_cfo_range_hz"][1] or not item["channel_id"].startswith("id_channels_"):
                    raise ValueError("Canal ID fuera del soporte configurado.")
                bank_name = "shift_channels" if split == "test_shift" else "id_channels"
                valid_channels = {f"{bank_name}_{i}": pairs for i, pairs in enumerate(config[bank_name])}
                if (item["channel_id"] not in valid_channels
                        or item["taps_real_imag"] != valid_channels[item["channel_id"]]):
                    raise ValueError("Los taps reales no corresponden al canal configurado.")
                taps = _taps(item["taps_real_imag"])
                expected_rrc_length = config["span"]*config["sps"]+1
                timing = item["timing_offset_samples"]
                if (isinstance(timing, bool) or not isinstance(timing, int) or not 0 <= timing < config["sps"]
                        or item["rrc_length"] != expected_rrc_length
                        or item["rrc_group_delay"] != (expected_rrc_length-1)//2
                        or item["crop_start"] != expected_rrc_length-1+len(taps)-1+timing
                        or item["crop_stop"] != item["crop_start"]+512
                        or item["full_length"] != config["symbols_per_parent"]*config["sps"]+expected_rrc_length-1+len(taps)-1):
                    raise ValueError("Recorte/transitorios o longitudes incorrectos.")
                if item["crop_stop"] > (config["symbols_per_parent"]-1)*config["sps"]:
                    raise ValueError("Ventana alcanza transitorios finales.")
            else:
                noise_count += 1
                if item["modulation"] is not None or item["snr_true_db"] is not None or item["snr_measured_window_db"] is not None:
                    raise ValueError("Noise-only no tiene modulación ni SNR de señal.")
                for field in ("beta", "amplitude", "phase_rad", "cfo_hz", "channel_id", "taps_real_imag"):
                    if item[field] is not None:
                        raise ValueError("Noise-only no debe tener parámetros de señal.")
        expected = Counter({(name, snr): config["realizations_per_class_snr"] for name in MAPPERS for snr in config["snr_db"]})
        if counts != expected or noise_count != config["noise_only_per_split"]:
            raise ValueError("Clases/SNR o noise-only desbalanceados.")
        summaries[split] = {"parents": len(records), "signal_parents": sum(counts.values()), "noise_only_parents": noise_count,
                            "class_snr_counts": [{"modulation": name, "snr_db": snr, "count": count} for (name, snr), count in sorted(counts.items())],
                            "measured_window_snr_db_min": min(local_snr), "measured_window_snr_db_max": max(local_snr),
                            "iq512_sha256": payload_hash}
    sources = ["src/dataset.py", "src/signals.py", "src/waveform.py", "src/channels.py"]
    source_root = Path(__file__).resolve().parents[1]
    return {"schema_version": 1, "parents": len(index), "windows": 2*len(index),
            "splits_disjoint": True, "prefixes_exact": True,
            "config_sha256": _canonical_hash(config), "splits": summaries,
            "versions": {"python": platform.python_version(), "numpy": np.__version__},
            "sources_sha256": {name: hashlib.sha256((source_root/name).read_bytes()).hexdigest() for name in sources},
            "limits": "Smoke, no accuracy ni estimación ciega de SNR. IDs/semillas separados no demuestran independencia estadística absoluta."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_smoke.json"))
    parser.add_argument("--output", type=Path, default=Path("data/fase3_smoke"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    audit = generate_dataset(config, args.output)
    print(f"Dataset generado: {audit['parents']} padres, {audit['windows']} ventanas; {args.output.resolve()}")


if __name__ == "__main__":
    main()
