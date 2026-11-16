"""CNN moderada sobre IQ observable; persistencia atómica y RNG completo."""
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


def preprocess(iq):
    x = np.asarray(iq)
    if x.ndim != 2 or x.shape[1] != 256 or not np.iscomplexobj(x) or not np.isfinite(x).all():
        raise ValueError("Se requieren ventanas IQ complejas finitas de 256 muestras.")
    z = x - x.mean(axis=1, keepdims=True)
    power = np.mean(abs(z)**2, axis=1, keepdims=True)
    if np.any(power <= 0):
        raise ValueError("Potencia centrada nula.")
    z = z / np.sqrt(power)
    return np.stack([z.real, z.imag], axis=1).astype(np.float32)


def random_phase_rotation(x):
    """Una rotación global por ventana; RNG Torch guardado en checkpoint."""
    angle = 2 * torch.pi * torch.rand((len(x), 1), device=x.device)
    cosine, sine = angle.cos(), angle.sin()
    return torch.stack((x[:, 0]*cosine - x[:, 1]*sine,
                        x[:, 0]*sine + x[:, 1]*cosine), dim=1)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(), nn.AvgPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.AvgPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.AvgPool1d(8))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512, 128), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(128, 6))

    def forward(self, x):
        return self.head(self.features(x))


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as file:
        torch.save(state, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def load_checkpoint(path):
    # Solo artefactos locales propios; pickle nunca se carga de fuentes desconocidas.
    return torch.load(path, map_location='cpu', weights_only=False)


@torch.inference_mode()
def predict(model, x, device, batch_size=256):
    model.eval()
    return torch.cat([model(x[i:i+batch_size].to(device)).cpu()
                      for i in range(0, len(x), batch_size)]).numpy()
