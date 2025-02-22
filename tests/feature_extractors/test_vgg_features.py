import pytest
import torch

from dl_toolkit.modules.feature_extractors.vgg_features import (
    NETWORKS_CONFIGS,
    PaddingType,
    VGGFeatures,
)


@pytest.mark.parametrize("model_name", NETWORKS_CONFIGS.keys())
@pytest.mark.parametrize("layers", [[15], [22, 30], [15, 22, 30]])
@pytest.mark.slow
@torch.no_grad()
def test_each_model(model_name, layers):
    x = torch.rand(1, 3, 192, 224)  # Test non typical tensor

    features = VGGFeatures(network=model_name, layers=layers)
    output = features(x)
    assert len(output) == len(layers)


@pytest.mark.parametrize("model_name", NETWORKS_CONFIGS.keys())
@pytest.mark.parametrize("padding", list(PaddingType))
@pytest.mark.slow
@torch.no_grad()
def test_same_output(model_name, padding):
    x = torch.rand(1, 3, 192, 192)

    key_id = 15
    model = VGGFeatures(network=model_name, layers=[key_id], padding_type=padding)
    output_zeros = model(x)[key_id]
    output_zeros2 = model(x)[key_id]
    assert torch.allclose(
        output_zeros, output_zeros2
    )  # We should keep in mind that output will be same
    model = VGGFeatures(network=model_name, layers=[key_id], padding_type=padding)
    output_zeros3 = model(x)[key_id]
    assert torch.allclose(
        output_zeros, output_zeros3
    )  # We should keep in mind that output will be same


def test_padding():
    model_name = "vgg16"
    x = torch.rand(1, 3, 192, 192)

    key_id = 15

    model = VGGFeatures(network=model_name, layers=[key_id], padding_type=PaddingType.ZEROS)
    output_zeros = model(x)[key_id]

    model = VGGFeatures(network=model_name, layers=[key_id], padding_type=PaddingType.REFLECT)
    output_reflect = model(x)[key_id]
    assert not torch.allclose(
        output_zeros, output_reflect
    )  # We should keep in mind that output will be same
    assert output_zeros.shape == output_reflect.shape
    model = VGGFeatures(network=model_name, layers=[key_id], padding_type=PaddingType.VALID)
    output_valid = model(x)[key_id]
    assert output_zeros.shape != output_valid.shape


@pytest.mark.parametrize("model_name", ["vgg16", "vgg19"])
def test_different_shapes(model_name):
    model_name = "vgg16"
    x = torch.rand(1, 3, 192, 192)

    ids = [3, 10]  # Must have different shapes

    model = VGGFeatures(network=model_name, layers=ids, padding_type=PaddingType.ZEROS)
    outputs = model(x)

    assert len(outputs) == len(ids)
    assert outputs[ids[0]].shape != outputs[ids[1]].shape


def test_valid_decreasing():
    x = torch.rand(1, 3, 192, 192)

    model = VGGFeatures(network="vgg16", layers=list(range(15)), padding_type=PaddingType.VALID)
    outputs = model(x)
    for output in outputs.values():
        print(output.shape)

    assert outputs[1].shape[3] == outputs[2].shape[3] + 2
    assert outputs[4].shape[3] == outputs[5].shape[3] + 2
    assert outputs[9].shape[3] == outputs[10].shape[3] + 2
