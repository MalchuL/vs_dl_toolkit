import numpy as np
import torch

from dl_toolkit.modules.utils.math import logit


def test_logit():
    x, y = [0.5, 0.1, 0.95], [0.0, -2.197224577336, 2.944438979166]
    for xi, yi in zip(x, y):
        assert abs(logit(xi) - yi) < 1e-6, f"{x}: {logit(xi)} != {yi}"


    x_np, y_np = np.array(x), np.array(y)
    assert np.allclose(logit(x_np), y_np)

    x_reshaped = x_np.reshape(-1, 1, 1).tolist()
    y_reshaped = y_np.reshape(-1, 1, 1)
    assert np.allclose(logit(x_reshaped), y_reshaped)  # Batched variant

    x_torch = torch.tensor(x)
    y_torch = torch.tensor(y)
    for xi, yi in zip(x_torch, y_torch):
        assert isinstance(logit(xi), torch.Tensor)
        assert abs(logit(xi) - yi) < 1e-6, f"{x}: {logit(xi)} != {yi}"
    assert torch.allclose(logit(x_torch), y_torch)  # Batched variant