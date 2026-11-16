"""Holdout independiente congelado: cuatro modelos, bootstrap pareado por celda."""
import csv
import argparse
import hashlib
import json
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
from experiments.exp02_conventional import _sha, _json, _csv, classification_metrics
from experiments.exp03_advanced import provenance


def paired_bootstrap(truth, snr, a, b, seed=2026110610, replicates=10000):
    """IC percentil de accuracy(A)-accuracy(B), estratos clase/SNR, padres pareados."""
    delta = (np.asarray(a) == truth).astype(float)-(np.asarray(b) == truth)
    rng = np.random.default_rng(seed)
    values = np.zeros(replicates)
    for label in np.unique(truth):
        for level in np.unique(snr):
            d = delta[(truth == label) & (snr == level)]
            if len(d):
                # Equivale a remuestrear padres por sus tres diferencias posibles.
                counts = np.array([(d == v).sum() for v in (-1, 0, 1)])
                draws = rng.multinomial(len(d), counts/len(d), size=replicates)
                values += draws[:, 2]-draws[:, 0]
    values /= len(truth)
    return dict(accuracy_difference=float(delta.mean()),
                ci95_percentile=np.percentile(values, [2.5, 97.5]).tolist(),
                a_correct_b_wrong=int((delta == 1).sum()), a_wrong_b_correct=int((delta == -1).sum()),
                bootstrap_replicates=replicates, seed=seed, strata='true_class_x_snr',
                scope='fixed_models_single_training_seed_conditional_on_simulator')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/fase6_holdout_final')
    out = Path(parser.parse_args().output)
    if out.exists():
        raise FileExistsError(out)
    reservation = json.loads(Path('results/fase6_diagnostico/holdout_reservation.json').read_text())
    root = Path(reservation['dataset_root'])
    assert reservation['status'] == 'sealed_no_predictions'
    assert provenance(root) == reservation['hashes']
    audit = json.loads((root/'audit.json').read_text())
    assert audit['splits_disjoint'] and audit['prefixes_exact']
    old_config = json.loads(Path('data/e2_dataset/config.json').read_text())
    new_config = json.loads((root/'config.json').read_text())
    assert old_config['seed'] != new_config['seed'] == 2026110609
    assert {**old_config, 'seed': new_config['seed']} == new_config
    old_ids = set()
    with Path('data/e2_dataset/index.csv').open() as f:
        old_ids = {r['parent_waveform_id'] for r in csv.DictReader(f)}
    with (root/'index.csv').open() as f:
        assert not old_ids & {r['parent_waveform_id'] for r in csv.DictReader(f)}
    old_waveforms = set()
    for path in Path('data/e2_dataset/observations').glob('*.npz'):
        with np.load(path) as observations:
            old_waveforms.update(hashlib.sha256(row.tobytes()).digest() for row in observations['iq256'])
    for split in ('test', 'test_shift'):
        with np.load(root/f'observations/{split}.npz') as observations:
            assert not old_waveforms & {hashlib.sha256(row.tobytes()).digest() for row in observations['iq256']}
    paths = dict(cnn_original='results/fase6/best.pt', cnn_phase='results/fase6_phase_aug/best.pt',
                 svm256='results/fase5/conventional_256.joblib', svm512='results/fase5/conventional_512.joblib')
    checkpoints = {k:load_checkpoint(v) for k,v in paths.items() if k.startswith('cnn')}
    assert checkpoints['cnn_original']['progress']['epoch'] == 46
    assert checkpoints['cnn_phase']['progress']['epoch'] == 24
    assert _sha(paths['cnn_original']) == json.loads(Path('results/fase6/selection_frozen.json').read_text())['sha256']
    assert _sha(paths['cnn_phase']) == json.loads(Path('results/fase6_phase_aug/integrity.json').read_text())['best_sha256']
    svms = {k:load_model(v)['pipeline'] for k,v in paths.items() if k.startswith('svm')}
    out.mkdir()
    _json(out/'protocol_frozen.json', dict(models={k:dict(path=v, sha256=_sha(v)) for k,v in paths.items()},
          dataset_hashes=reservation['hashes'], independence='different_seed_same_config_disjoint_parent_ids_no_identical_iq256_against_all_old_splits',
          fit=False, selection_after_evaluation=False, splits=['test','test_shift'],
          primary_comparison='cnn_phase_minus_cnn_original', other_comparisons='descriptive_no_multiplicity_correction',
          bootstrap=dict(replicates=10000, seed=2026110610, stratify='class_x_snr', unit='parent_paired')))
    torch.set_num_threads(1)
    metrics, rows, snr_rows, comparisons = {}, [], [], {}
    seen = set()
    with threadpool_limits(limits=1):
        for split in ('test', 'test_shift'):
            d = load_classification_split(root, split, 512, allow_shift=True)
            ids = [r['parent_waveform_id'] for r in d['evaluation_rows']]
            assert not seen & set(ids)
            seen.update(ids)
            truth = d['targets']
            snr = np.array([float(r['snr_nominal_db']) for r in d['evaluation_rows']])
            x = torch.from_numpy(preprocess(d['iq'][:, :256]))
            predictions = {}
            for name, checkpoint in checkpoints.items():
                model = SmallCNN().cuda().eval()
                model.load_state_dict(checkpoint['model'], strict=True)
                predictions[name] = predict(model, x, 'cuda').argmax(1)
            for name, model in svms.items():
                predictions[name] = model.predict(d['iq'][:, :int(name[3:])])
            metrics[split] = {}
            fig, axes = plt.subplots(2,2, figsize=(11,10), constrained_layout=True)
            for ax, (name, pred) in zip(axes.flat, predictions.items()):
                metrics[split][name] = classification_metrics(truth, pred)
                matrix = confusion_matrix(truth,pred,labels=np.arange(6))
                _json(out/f'confusion_{split}_{name}.json', matrix.tolist())
                normalized = matrix/matrix.sum(1,keepdims=True)
                ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues')
                for i in range(6):
                    for j in range(6):
                        ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=8,
                                color='white' if normalized[i,j]>.5 else 'black')
                ax.set(xticks=range(6),yticks=range(6),xticklabels=CLASSES,yticklabels=CLASSES,
                       xlabel='Predicción',ylabel='Verdad',title=name)
                for level in sorted(set(snr)):
                    m = classification_metrics(truth[snr==level],pred[snr==level])
                    snr_rows.append(dict(split=split,model=name,snr_db=level,**{k:v for k,v in m.items() if k!='per_class'}))
                for i, parent in enumerate(ids):
                    rows.append(dict(split=split,model=name,parent_waveform_id=parent,target=int(truth[i]),
                                     predicted=int(pred[i]),snr_db=snr[i]))
            fig.suptitle(f'Holdout nuevo: {split}')
            fig.savefig(out/f'confusions_{split}.png',dpi=140)
            plt.close(fig)
            comparisons[split] = {}
            for a,b in [('cnn_phase','cnn_original'),('cnn_original','svm256'),('cnn_phase','svm256'),('cnn_phase','svm512'),('svm512','svm256')]:
                comparisons[split][a+'_minus_'+b] = paired_bootstrap(truth,snr,predictions[a],predictions[b])
    for split in metrics:
        fig, ax = plt.subplots(figsize=(8,5),constrained_layout=True)
        for name in paths:
            r = [r for r in snr_rows if r['split']==split and r['model']==name]
            ax.plot([v['snr_db'] for v in r],[v['accuracy'] for v in r],'o-',label=name)
        ax.set(xlabel='SNR nominal, solo evaluación (dB)',ylabel='Exactitud',ylim=(0,1),title=f'Holdout nuevo: {split}')
        ax.legend(); ax.grid(alpha=.3)
        fig.savefig(out/f'accuracy_{split}.png',dpi=150); plt.close(fig)
    _json(out/'metrics.json',metrics)
    _json(out/'paired_comparisons.json',comparisons)
    _csv(out/'predictions.csv',rows)
    _csv(out/'by_snr.csv',snr_rows)
    _json(out/'completed.json',dict(completed=True,models_unchanged=all(_sha(v)==json.loads((out/'protocol_frozen.json').read_text())['models'][k]['sha256'] for k,v in paths.items()),
          n_predictions=len(rows),train_used=False,holdout_opened_for_final_evaluation=True))
    print(json.dumps({s:{n:{k:v for k,v in m.items() if k!='per_class'} for n,m in ms.items()} for s,ms in metrics.items()},indent=2))


if __name__ == '__main__':
    main()
