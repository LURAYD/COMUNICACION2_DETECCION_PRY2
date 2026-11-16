"""Dos procesos frente a corrida continua: misma trayectoria GPU bit a bit."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
import torch
from src.advanced import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/exp03_advanced.json')
    parser.add_argument('--output', default='results/fase6_resume_verified')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    config = json.loads(Path(args.config).read_text())
    config.update(max_epochs=2, checkpoint_steps=1)
    cfg = root/'config.json'
    cfg.write_text(json.dumps(config))
    base = [sys.executable, '-m', 'experiments.exp03_advanced', '--config', str(cfg), '--limit', '256']
    subprocess.run(base+['--output', str(root/'continuous')], check=True)
    subprocess.run(base+['--output', str(root/'interrupted'), '--stop-after-steps', '1'], check=True)
    paused = load_checkpoint(root/'interrupted/last.pt')
    assert paused['progress']['step'] == 1 and paused['progress']['next_batch'] == 128
    subprocess.run(base+['--output', str(root/'interrupted'), '--resume'], check=True)
    a = load_checkpoint(root/'continuous/last.pt')
    b = load_checkpoint(root/'interrupted/last.pt')
    assert all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
    assert a['progress']['history'] == b['progress']['history']
    for k in a['optimizer']['state']:
        for name, value in a['optimizer']['state'][k].items():
            assert torch.equal(value, b['optimizer']['state'][k][name])
    result = dict(passed=True, separate_processes=True, interrupted_step=1,
                  model_and_optimizer_bitwise_equal=True, history_identical=True,
                  torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
    (root/'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
