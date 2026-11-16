"""Entrenamiento E3 finito y reanudable incluso dentro de una época."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from src.advanced import (SmallCNN, atomic_save, load_checkpoint, predict,
                          preprocess, random_phase_rotation, restore_rng, rng_state, seed_everything)
from src.classification_data import load_classification_split
from experiments.exp02_conventional import classification_metrics


def provenance(root):
    paths = [*root.glob('*.json'), root/'index.csv', *root.glob('supervision/*.csv'),
             *root.glob('observations/*.npz')]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def train(config, output, resume=False, stop_after_steps=None, limit=None):
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    device = 'cuda'
    if not torch.cuda.is_available():
        raise RuntimeError('Se requiere GPU CUDA verificada.')
    torch.set_num_threads(1)
    seed_everything(config['seed'])
    hashes = provenance(Path(config['dataset_root']))
    data = {s: load_classification_split(config['dataset_root'], s, 256) for s in ('train', 'val')}
    parents = [{r['parent_waveform_id'] for r in data[s]['evaluation_rows']} for s in data]
    assert not parents[0] & parents[1]
    x, y = {}, {}
    for s, d in data.items():
        x[s] = torch.from_numpy(preprocess(d['iq'][:limit]))
        y[s] = torch.from_numpy(d['targets'][:limit]).long()
    model = SmallCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=5)
    state = dict(epoch=0, next_batch=0, step=0, order=None, best_f1=-1., bad_epochs=0,
                 history=[], epoch_loss=0., epoch_count=0, completed=False, train_seconds=0.)
    if resume:
        saved = load_checkpoint(output/'last.pt')
        if saved['config'] != config or saved['hashes'] != hashes or saved['limit'] != limit:
            raise ValueError('Configuración/dataset distintos del checkpoint.')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        state = saved['progress']
        restore_rng(saved['rng'])
    started = time.perf_counter()
    prior_seconds = state['train_seconds']
    (output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    def save(name='last.pt'):
        state['train_seconds'] = prior_seconds + time.perf_counter()-started
        atomic_save(output/name, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), scaler=None, rng=rng_state(), progress=state,
                    config=config, hashes=hashes, limit=limit, torch_version=torch.__version__,
                    cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name()))
        (output/'history.json').write_text(json.dumps(state['history'], indent=2), encoding='utf-8')
        (output/'ESTADO_REANUDACION.md').write_text(
            '# Estado de entrenamiento\n\n' + f"Época completada: {state['epoch']}; siguiente minibatch: {state['next_batch']}; pasos: {state['step']}; terminado: {state['completed']}.\n\n"
            + 'Checkpoint last.pt atómico: modelo, optimizador, scheduler, RNG Python/NumPy/Torch CPU/CUDA, orden/paso, historial, configuración y hashes de datos. Float32; scaler no aplica.\n\n'
            + f"Reanudar desde raíz: `.\\.venv\\Scripts\\python.exe -m experiments.exp03_advanced --config {output.as_posix()}/config.json --output {output.as_posix()} --resume" + (f" --limit {limit}" if limit else '') + "`\n\n"
            + 'No hay detector de cuota. Si el proceso se interrumpe abruptamente, se repiten como máximo checkpoint_steps minibatches desde el último guardado. Evaluación E3–E6 y reporte pendientes hasta existir metricas.json.\n', encoding='utf-8')
    save()
    while state['epoch'] < config['max_epochs'] and not state['completed']:
        if state['order'] is None:
            state['order'] = torch.randperm(len(x['train']))
        model.train()
        for start in range(state['next_batch'], len(x['train']), config['batch_size']):
            ids = state['order'][start:start+config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            batch = x['train'][ids].to(device)
            if config.get('phase_augmentation', False):
                batch = random_phase_rotation(batch)
            loss = torch.nn.functional.cross_entropy(model(batch), y['train'][ids].to(device))
            loss.backward()
            optimizer.step()
            state['step'] += 1
            state['next_batch'] = start + len(ids)
            state['epoch_loss'] += loss.item()*len(ids)
            state['epoch_count'] += len(ids)
            if state['step'] % config['checkpoint_steps'] == 0:
                save()
            if stop_after_steps is not None and state['step'] >= stop_after_steps:
                save()
                return state
        logits = predict(model, x['val'], device)
        val_loss = torch.nn.functional.cross_entropy(torch.from_numpy(logits), y['val']).item()
        f1 = classification_metrics(y['val'].numpy(), logits.argmax(1))['macro_f1']
        scheduler.step(val_loss)
        state['epoch'] += 1
        row = dict(epoch=state['epoch'], train_loss=state['epoch_loss']/state['epoch_count'],
                   val_loss=val_loss, val_macro_f1=f1, lr=optimizer.param_groups[0]['lr'])
        state['history'].append(row)
        improved = f1 > state['best_f1']
        if improved:
            state['best_f1'], state['bad_epochs'] = f1, 0
        else:
            state['bad_epochs'] += 1
        state.update(order=None, next_batch=0, epoch_loss=0., epoch_count=0)
        state['completed'] = state['bad_epochs'] >= config['patience'] or state['epoch'] >= config['max_epochs']
        if improved:
            save('best.pt')
        save()
        print(json.dumps(row), flush=True)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/exp03_advanced.json')
    parser.add_argument('--output', default='results/fase6')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-steps', type=int)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    train(json.loads(Path(args.config).read_text()), args.output, args.resume, args.stop_after_steps, args.limit)
