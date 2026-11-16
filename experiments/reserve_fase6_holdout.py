"""Reserva previa a ajustes: generación/auditoría, sin inferencia ni métricas."""
import json
from pathlib import Path
from src.classification_data import prepare_classification_dataset
from experiments.exp03_advanced import provenance


if __name__ == '__main__':
    config = json.loads(Path('configs/dataset_e2.json').read_text())
    config['seed'] = 2026110609
    destination = Path('data/fase6_holdout_sellado')
    prepare_classification_dataset(config, destination)
    Path('results/fase6_diagnostico/holdout_reservation.json').write_text(json.dumps({
        'status': 'sealed_no_predictions', 'dataset_root': str(destination),
        'seed': config['seed'], 'evaluation_splits_reserved': ['test', 'test_shift'],
        'unused_splits': ['train', 'val'], 'hashes': provenance(destination),
        'rule': 'No abrir para seleccionar arquitectura, hiperparámetros o política. Reservado para evaluación final posterior.'
    }, indent=2), encoding='utf-8')
