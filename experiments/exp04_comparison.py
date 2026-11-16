"""E3–E6 con modelo congelado; CPU/GPU y adquisición separados."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits
from src.advanced import SmallCNN, load_checkpoint, predict, preprocess
from src.classification_data import load_classification_split
from src.conventional import CLASSES, load_model
from experiments.exp02_conventional import classification_metrics, _csv, _json, _sha, _cpu_name
from experiments.exp03_advanced import provenance


def main(output):
    output = Path(output)
    last = load_checkpoint(output/'last.pt')
    if not last['progress']['completed']:
        raise RuntimeError('Entrenamiento aún no terminado.')
    best = load_checkpoint(output/'best.pt')
    root = Path(best['config']['dataset_root'])
    if provenance(root) != best['hashes']:
        raise ValueError('Dataset modificado.')
    torch.set_num_threads(1)
    model = SmallCNN().cuda().eval()
    model.load_state_dict(best['model'])
    _json(output/'selection_frozen.json', dict(best_epoch=best['progress']['epoch'],
          val_macro_f1=best['progress']['best_f1'], sha256=_sha(output/'best.pt'), selection='val_only'))
    svms = {n: load_model(f'results/fase5/conventional_{n}.joblib')['pipeline'] for n in (256, 512)}
    metrics, predictions, snr_rows, latency = {}, [], [], []
    seen = set()
    for split in ('train', 'val'):
        import csv
        with (root/f'supervision/{split}.csv').open() as f:
            seen.update(r['parent_waveform_id'] for r in csv.DictReader(f))
    with threadpool_limits(limits=1):
        for split in ('test', 'test_shift'):
            data = load_classification_split(root, split, 512, allow_shift=True)
            ids = {r['parent_waveform_id'] for r in data['evaluation_rows']}
            assert not seen & ids
            seen.update(ids)
            truth = data['targets']
            iq = data['iq']
            results = {'cnn256': predict(model, torch.from_numpy(preprocess(iq[:, :256])), 'cuda').argmax(1)}
            for n, svm in svms.items():
                results[f'svm{n}'] = svm.predict(iq[:, :n])
            metrics[split] = {}
            snrs = np.asarray([float(r['snr_nominal_db']) for r in data['evaluation_rows']])
            for name, pred in results.items():
                metrics[split][name] = classification_metrics(truth, pred)
                matrix = confusion_matrix(truth, pred, labels=np.arange(6))
                _json(output/f'confusion_{split}_{name}.json', matrix.tolist())
                fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
                ax.imshow(matrix/matrix.sum(1, keepdims=True), vmin=0, vmax=1, cmap='Blues')
                for i in range(6):
                    for j in range(6):
                        ax.text(j, i, str(matrix[i, j]), ha='center', va='center')
                ax.set(xticks=range(6), yticks=range(6), xticklabels=CLASSES, yticklabels=CLASSES,
                       xlabel='Predicción', ylabel='Verdad', title=f'{split}: {name}')
                fig.savefig(output/f'confusion_{split}_{name}.png', dpi=150)
                plt.close(fig)
                for snr in sorted(set(snrs)):
                    m = classification_metrics(truth[snrs == snr], pred[snrs == snr])
                    snr_rows.append(dict(split=split, model=name, snr_db=float(snr), **{k:v for k,v in m.items() if k != 'per_class'}))
                for i, row in enumerate(data['evaluation_rows']):
                    predictions.append(dict(split=split, model=name, parent_waveform_id=row['parent_waveform_id'],
                                            target=int(truth[i]), predicted=int(pred[i]), snr_db=snrs[i]))
            if split == 'test':
                sample = iq[:128]
    # Llamada completa desde IQ CPU; GPU incluye transferencia y sincronización.
    with threadpool_limits(limits=1), torch.inference_mode():
        for name in ('svm256', 'svm512', 'cnn256_cpu', 'cnn256_cuda'):
            device = 'cuda' if name.endswith('cuda') else 'cpu'
            if name.startswith('cnn'):
                model.to(device).eval()
            def infer(row):
                if name.startswith('svm'):
                    n = int(name[3:])
                    return svms[n].predict(row[None, :n])
                x = torch.from_numpy(preprocess(row[None, :256])).to(device)
                result = model(x).argmax(1).cpu().numpy()
                if device == 'cuda':
                    torch.cuda.synchronize()
                return result
            for row in sample[:16]:
                infer(row)
            for repeat in range(5):
                for i, row in enumerate(sample):
                    start = time.perf_counter_ns()
                    infer(row)
                    latency.append(dict(model=name, repeat=repeat, window=i, total_wall_ms=(time.perf_counter_ns()-start)/1e6))
    latency_summary = {}
    for name in sorted({r['model'] for r in latency}):
        values = [r['total_wall_ms'] for r in latency if r['model'] == name]
        latency_summary[name] = dict(median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)))
    for split in ('test', 'test_shift'):
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for name in ('svm256', 'svm512', 'cnn256'):
            rows = [r for r in snr_rows if r['split'] == split and r['model'] == name]
            ax.plot([r['snr_db'] for r in rows], [r['accuracy'] for r in rows], 'o-', label=name)
        ax.set(xlabel='SNR nominal (dB), solo evaluación', ylabel='Exactitud', ylim=(0, 1), title=split)
        ax.legend()
        ax.grid(alpha=.3)
        fig.savefig(output/f'accuracy_by_snr_{split}.png', dpi=150)
        plt.close(fig)
    _csv(output/'predictions.csv', predictions)
    _csv(output/'metrics_by_snr.csv', snr_rows)
    _csv(output/'latency_batch1.csv', latency)
    _json(output/'metricas.json', dict(metrics=metrics, latency=latency_summary,
          parameters=sum(p.numel() for p in model.parameters()), train_seconds=last['progress']['train_seconds'],
          epochs=last['progress']['epoch'], best_epoch=best['progress']['epoch'],
          cpu=_cpu_name(), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
          cuda=torch.version.cuda, cpu_threads=1, batch_size_latency=1, warmup=16,
          acquisition_ms={'256':32, '512':64}, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='results/fase6')
    main(p.parse_args().output)
