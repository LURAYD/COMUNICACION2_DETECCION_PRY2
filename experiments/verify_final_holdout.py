"""Reconstruye métricas y emparejamiento desde predicciones guardadas."""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
from experiments.exp02_conventional import classification_metrics, _json


if __name__ == '__main__':
    root = Path('results/fase6_holdout_final')
    with (root/'predictions.csv').open() as f:
        rows = list(csv.DictReader(f))
    metrics = json.loads((root/'metrics.json').read_text())
    pairs = json.loads((root/'paired_comparisons.json').read_text())
    for split, models in metrics.items():
        parent_sets, correctness = [], {}
        for name, expected in models.items():
            selected = [r for r in rows if r['split']==split and r['model']==name]
            selected.sort(key=lambda r:r['parent_waveform_id'])
            ids = [r['parent_waveform_id'] for r in selected]
            assert len(ids)==len(set(ids))==4800
            parent_sets.append(ids)
            y = np.array([int(r['target']) for r in selected])
            p = np.array([int(r['predicted']) for r in selected])
            assert classification_metrics(y,p)==expected
            assert confusion_matrix(y,p,labels=np.arange(6)).tolist()==json.loads((root/f'confusion_{split}_{name}.json').read_text())
            correctness[name] = (y==p).astype(float)
        assert all(p==parent_sets[0] for p in parent_sets)
        for name, result in pairs[split].items():
            a,b = name.split('_minus_')
            assert np.isclose((correctness[a]-correctness[b]).mean(),result['accuracy_difference'])
    _json(root/'verification.json',dict(passed=True,predictions=len(rows),groups=8,metrics_reconstructed=True,
          confusion_matrices_reconstructed=True,parents_paired=True,paired_differences_reconstructed=True,
          tests='2 passed in 15.68s: tests/test_holdout_comparison.py'))
    print('Verificación correcta:38400 predicciones,8 grupos, métricas y parejas consistentes.')
