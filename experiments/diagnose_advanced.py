"""Diagnóstico train/val; nunca abre test ni test_shift."""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.advanced import SmallCNN, load_checkpoint, predict, preprocess
from src.classification_data import load_classification_split
from experiments.exp02_conventional import classification_metrics, _csv, _json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-root', default='results/fase6')
    parser.add_argument('--output', default='results/fase6_diagnostico')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    checkpoint = load_checkpoint(Path(args.model_root)/'best.pt')
    model = SmallCNN().cuda()
    model.load_state_dict(checkpoint['model'])
    summary, rows = {}, []
    for split in ('train', 'val'):
        d = load_classification_split('data/e2_dataset', split, 256)
        x = torch.from_numpy(preprocess(d['iq']))
        logits = predict(model, x, 'cuda')
        prediction = logits.argmax(1)
        summary[split] = classification_metrics(d['targets'], prediction)
        summary[split]['cross_entropy_eval'] = torch.nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(d['targets'])).item()
        snr = np.array([float(r['snr_nominal_db']) for r in d['evaluation_rows']])
        for s in sorted(set(snr)):
            metrics = classification_metrics(d['targets'][snr == s], prediction[snr == s])
            rows.append(dict(split=split, snr_db=s, **{k:v for k,v in metrics.items() if k != 'per_class'}))
        if split == 'val':
            rotations = []
            for angle in (np.pi/4, np.pi/2, np.pi):
                p = predict(model, torch.from_numpy(preprocess(d['iq']*np.exp(1j*angle))), 'cuda').argmax(1)
                rotations.append(dict(radians=angle, prediction_changed=float(np.mean(p != prediction)), accuracy=float(np.mean(p == d['targets']))))
            summary['val_rotations'] = rotations
    history = json.loads((Path(args.model_root)/'history.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for key in ('train_loss', 'val_loss'):
        axes[0].plot([r['epoch'] for r in history], [r[key] for r in history], label=key)
    axes[0].set(xlabel='Época', ylabel='Entropía cruzada')
    axes[0].legend()
    for split in ('train', 'val'):
        r = [v for v in rows if v['split'] == split]
        axes[1].plot([v['snr_db'] for v in r], [v['accuracy'] for v in r], 'o-', label=split)
    axes[1].set(xlabel='SNR nominal (solo diagnóstico)', ylabel='Exactitud', ylim=(0,1))
    axes[1].legend()
    fig.savefig(out/'learning_diagnostics.png', dpi=150)
    _json(out/'train_val.json', summary)
    _csv(out/'by_snr.csv', rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
