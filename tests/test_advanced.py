"""Contratos observables y guardado seguro; GPU se verifica en smoke separado."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from src.advanced import SmallCNN, atomic_save, load_checkpoint, preprocess, random_phase_rotation


def test_observable_normalization_and_shape():
    rng = np.random.default_rng(3)
    iq = rng.normal(size=(3, 256)) + 1j*rng.normal(size=(3, 256))
    x = preprocess(iq)
    np.testing.assert_allclose(preprocess(3*iq+2), x, atol=1e-6)
    assert SmallCNN()(torch.from_numpy(x)).shape == (3, 6)
    assert x.dtype == np.float32


def test_invalid_observation():
    for x in [np.ones((2, 256), complex), np.ones((2, 512), complex), np.ones((2, 256))]:
        with pytest.raises(ValueError):
            preprocess(x)


def test_atomic_replacement(tmp_path):
    path = tmp_path/'last.pt'
    atomic_save(path, {'step': 1})
    atomic_save(path, {'step': 2})
    assert load_checkpoint(path)['step'] == 2
    assert not path.with_suffix('.pt.tmp').exists()


def test_phase_rotation_preserves_energy_and_rng_replay():
    x = torch.randn(4, 2, 256)
    state = torch.get_rng_state()
    y = random_phase_rotation(x)
    torch.set_rng_state(state)
    assert torch.equal(y, random_phase_rotation(x))
    torch.testing.assert_close(y.square().sum(1), x.square().sum(1))
    # La misma rotación se aplica a toda la ventana, no a cada muestra.
    before = x[:, 0, 1:]*x[:, 0, :-1] + x[:, 1, 1:]*x[:, 1, :-1]
    after = y[:, 0, 1:]*y[:, 0, :-1] + y[:, 1, 1:]*y[:, 1, :-1]
    torch.testing.assert_close(before, after)
