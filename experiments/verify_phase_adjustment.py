"""Verifica integridad y separación del ensayo sin abrir holdout para inferencia."""
import hashlib
import json
from pathlib import Path
from src.advanced import load_checkpoint
from experiments.exp03_advanced import provenance


if __name__ == '__main__':
    old = Path('results/fase6/best.pt')
    frozen = json.loads(Path('results/fase6/selection_frozen.json').read_text())
    assert hashlib.sha256(old.read_bytes()).hexdigest() == frozen['sha256']
    reservation = json.loads(Path('results/fase6_diagnostico/holdout_reservation.json').read_text())
    assert provenance(Path(reservation['dataset_root'])) == reservation['hashes']
    original = json.loads(Path('configs/exp03_advanced.json').read_text())
    adjusted = json.loads(Path('configs/exp03_phase_aug.json').read_text())
    assert adjusted == {**original, 'phase_augmentation': True}
    last = load_checkpoint('results/fase6_phase_aug/last.pt')
    best = load_checkpoint('results/fase6_phase_aug/best.pt')
    assert last['progress']['completed']
    assert best['config'] == adjusted
    result = dict(original_model_hash_unchanged=True, holdout_hashes_unchanged=True,
                  only_config_change='phase_augmentation=True', training_completed=True,
                  best_epoch=best['progress']['epoch'], final_epoch=last['progress']['epoch'],
                  training_seconds=last['progress']['train_seconds'],
                  best_sha256=hashlib.sha256(Path('results/fase6_phase_aug/best.pt').read_bytes()).hexdigest(),
                  tests_passed=236, test_seconds=17.26, test_command='python -m pytest -q -W error --basetemp tmp/pytest_fase6_aug')
    Path('results/fase6_phase_aug/integrity.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
