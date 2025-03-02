import numpy as np
import pytest
import torch

from dl_toolkit.modules.layers.embeddings.fourier_embedding import (
    SQRT_2,
    FourierEmbedding,
)


@pytest.fixture
def sample_input():
    return torch.randn(16)  # Batch of 3 samples with 16 features each


def test_initialization():
    num_channels = 32
    layer = FourierEmbedding(num_channels)
    assert layer.freqs.shape == (num_channels // 2,)
    assert torch.allclose(layer.multiplier, torch.tensor(2 * np.pi))
    assert layer.freqs.requires_grad is False
    assert layer.multiplier.requires_grad is False


def test_output_shape(sample_input):
    num_channels = 64
    layer = FourierEmbedding(num_channels)
    output = layer(sample_input)
    assert output.shape == (sample_input.shape[0], num_channels)


def test_frequency_scaling(monkeypatch):
    num_channels = 16
    scale = 5.0

    # Mock random initialization
    def mock_randn(size):
        return torch.ones(size)

    monkeypatch.setattr(torch, "randn", mock_randn)

    layer = FourierEmbedding(num_channels, scale=scale)
    assert torch.allclose(layer.freqs, torch.ones(8) * scale)


def test_device_consistency():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = FourierEmbedding(32).to(device)
    x = torch.randn(10).to(device)
    output = layer(x)
    assert output.device.type == device


def test_gradient_flow(sample_input):
    layer = FourierEmbedding(64)
    output = layer(sample_input)
    loss = output.sum()
    with pytest.raises(RuntimeError):
        loss.backward()
