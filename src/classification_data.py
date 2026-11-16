"""Supervisión administrativa E2/E3 separada de la entrada observable."""
import csv
import json
from pathlib import Path

import numpy as np

from src.conventional import CLASSES
from src.dataset import SPLITS, generate_dataset


def prepare_classification_dataset(config, output):
    """Generador Fase3 intacto, más manifiestos de supervisión por split."""
    output = Path(output)
    audit = generate_dataset(config, output)
    (output/"supervision").mkdir()
    with (output/"oracle/manifest.jsonl").open(encoding="utf-8") as file:
        truth = {row["record_id"]: row for row in map(json.loads, file)}
    with (output/"index.csv").open(newline="", encoding="utf-8") as file:
        index = list(csv.DictReader(file))
    for split in SPLITS:
        with (output/f"supervision/{split}.csv").open("w", newline="", encoding="utf-8") as file:
            fields = ["row", "record_id", "parent_waveform_id", "has_signal", "modulation", "snr_nominal_db"]
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for record in (r for r in index if r["split"] == split):
                target = truth[record["record_id"]]
                writer.writerow({"row": record["row"], "record_id": record["record_id"],
                                 "parent_waveform_id": record["parent_waveform_id"],
                                 "has_signal": int(target["has_signal"]), "modulation": target["modulation"],
                                 "snr_nominal_db": target["snr_true_db"]})
    return audit


def load_classification_split(root, split, n_samples, allow_shift=False):
    """Carga una matriz una vez. Noise-only se excluye de los targets AMC.

    Los IDs/SNR se devuelven solo como metadatos administrativos/evaluación.
    Únicamente iq puede entregarse al extractor de características.
    """
    if split not in SPLITS or (split == "test_shift" and not allow_shift):
        raise ValueError("test_shift permanece sellado hasta E6.")
    if isinstance(n_samples, bool) or n_samples not in (256, 512):
        raise ValueError("n_samples debe ser 256 o 512.")
    root = Path(root)
    with (root/f"supervision/{split}.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if any(r["has_signal"] not in ("0", "1") for r in rows):
        raise ValueError("Presencia inválida en manifiesto de supervisión.")
    if any(r["modulation"] or r["snr_nominal_db"] for r in rows if r["has_signal"] == "0"):
        raise ValueError("Noise-only no tiene target de modulación ni SNR.")
    selected = [r for r in rows if r["has_signal"] == "1"]
    if not selected or len({r["parent_waveform_id"] for r in rows}) != len(rows):
        raise ValueError("Supervisión vacía o padres duplicados.")
    if [int(r["row"]) for r in rows] != list(range(len(rows))):
        raise ValueError("Filas de supervisión inconsistentes.")
    with np.load(root/f"observations/{split}.npz", allow_pickle=False) as data:
        iq = data[f"iq{n_samples}"][[int(r["row"]) for r in selected]].copy()
    metadata = json.loads((root/"public_metadata.json").read_text(encoding="utf-8"))
    return {"iq": iq, "targets": np.asarray([CLASSES.index(r["modulation"]) for r in selected]),
            "evaluation_rows": selected, "fs_hz": metadata["fs_hz"],
            "n_noise_excluded": len(rows)-len(selected)}
