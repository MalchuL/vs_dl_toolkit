from typing import Iterable

import pytest
import torch

from dl_toolkit.modules.layers.clipper import (
    Clipper,
    ClipperChannelwise1D,
    ClipperChannelwise2D,
    ClipperWrapper,
)
from dl_toolkit.modules.toolkit_module import ToolkitModule
from dl_toolkit.modules.utils.clipping_utils import z_clip


@pytest.fixture
def sample_tensor1d():
    return torch.tensor(range(10 * 10 * 10)).reshape(10, 10, 10).float()


@pytest.fixture
def sample_tensor2d():
    return torch.tensor(range(10 * 10 * 10 * 10)).reshape(10, 10, 10, 10).float()


# Test Clipper
def test_clipper_forward(sample_tensor1d, sample_tensor2d):
    clipper = Clipper(z_score=2.0)
    output = clipper(sample_tensor1d)
    output = clipper(sample_tensor2d)


# Test ClipperChannelwise1D
def test_clipper_channelwise1d_forward(sample_tensor1d):
    clipper = ClipperChannelwise1D(z_score=2.0)
    output = clipper(sample_tensor1d)
    expected_output = z_clip(sample_tensor1d, z_value=2.0, dims=(2,))
    assert torch.allclose(output, expected_output)


# Test ClipperChannelwise2D
def test_clipper_channelwise2d_forward(sample_tensor2d):
    clipper = ClipperChannelwise2D(z_score=2.0)
    output = clipper(sample_tensor2d)
    expected_output = z_clip(sample_tensor2d, z_value=2.0, dims=(2, 3))
    assert torch.allclose(output, expected_output)


# Test ClipperWrapper
def test_clipper_wrapper(sample_tensor2d):
    # Create a dummy module that returns the input tensor
    class DummyModule(ToolkitModule):
        def forward(self, x):
            return x

    dummy_module = DummyModule()
    clipper = Clipper(z_score=2.0)
    wrapper = ClipperWrapper(module=dummy_module, clipper=clipper)

    output = wrapper(sample_tensor2d)
    expected_output = z_clip(sample_tensor2d, z_value=2.0, dims=None)
    assert torch.allclose(output, expected_output)


# Test ClipperWrapper with ClipperChannelwise1D
def test_clipper_wrapper_with_channelwise1d(sample_tensor1d):
    class DummyModule(ToolkitModule):
        def forward(self, x):
            return x

    dummy_module = DummyModule()
    clipper = ClipperChannelwise1D(z_score=2.0)
    wrapper = ClipperWrapper(module=dummy_module, clipper=clipper)

    output = wrapper(sample_tensor1d)
    expected_output = z_clip(sample_tensor1d, z_value=2.0, dims=(2,))
    assert torch.allclose(output, expected_output)


# Test ClipperWrapper with ClipperChannelwise2D
def test_clipper_wrapper_with_channelwise2d(sample_tensor2d):
    class DummyModule(ToolkitModule):
        def forward(self, x):
            return x

    dummy_module = DummyModule()
    clipper = ClipperChannelwise2D(z_score=2.0)
    wrapper = ClipperWrapper(module=dummy_module, clipper=clipper)

    output = wrapper(sample_tensor2d)
    expected_output = z_clip(sample_tensor2d, z_value=2.0, dims=(2, 3))
    assert torch.allclose(output, expected_output)
