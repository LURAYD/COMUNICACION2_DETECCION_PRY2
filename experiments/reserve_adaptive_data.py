"""Fija datos nuevos y reserva test antes del desarrollo del selector."""
import json
from pathlib import Path
from src.classification_data import prepare_classification_dataset
from experiments.exp03_advanced import provenance
from experiments.exp02_conventional import _json,_sha


if __name__=='__main__':
    config=json.loads(Path('configs/adaptive_experiment.json').read_text())
    base=json.loads(Path('configs/dataset_e2.json').read_text())
    base.update(seed=config['dataset_seed'],realizations_per_class_snr=config['parents_per_class_snr'])
    root=Path('data/adaptive_dataset')
    out=Path('results/adaptive'); out.mkdir(exist_ok=True)
    if not root.exists():
        prepare_classification_dataset(base,root)
    assert json.loads((root/'config.json').read_text())==base
    reservation=dict(config=config,dataset_hashes=provenance(root),
        model_hashes={p:_sha(p) for p in [config['cnn'],'results/fase5/conventional_256.joblib','results/fase5/conventional_512.joblib']},
        test_status='sealed_before_selector_development',validation_partition='first25_per_class_snr_cal_rest25_selection_in_stored_order')
    path=out/'reservation.json'
    if path.exists():
        assert json.loads(path.read_text())==reservation
    else:
        _json(path,reservation)
    print('Dataset/reserva verificados; no predicciones test.')
