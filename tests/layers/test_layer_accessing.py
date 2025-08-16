import pytest
import torch
from torch import nn

from dl_toolkit.modules.utils.layer_accessing import (
    get_module_by_pattern,
    get_modules_by_type,
    get_modules_by_type_name
)

class BN(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        return self.bn(x)
    
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            BN(128),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        return x


@pytest.fixture
def model():
    return SimpleNet()


class TestGetModuleByPattern:
    def test_exact_match(self, model):
        # Test exact path matching
        modules = get_module_by_pattern(model, 'conv1')
        assert len(modules) == 1
        assert isinstance(modules['conv1'], nn.Conv2d)

    def test_wildcard_match(self, model):
        # Test wildcard pattern matching
        modules = get_module_by_pattern(model, '*1')
        # Should match ['conv1', 'bn1', 'relu1', 'layer1', 'layer1.1', 'layer2.1']
        assert len(modules) == 6, modules.keys()  
        assert isinstance(modules['conv1'], nn.Conv2d)
        assert isinstance(modules['bn1'], nn.BatchNorm2d)
        assert isinstance(modules['relu1'], nn.ReLU)

    def test_nested_match(self, model):
        # Test nested module matching
        modules = get_module_by_pattern(model, 'layer1.*')
        assert len(modules) == 4  # Should match all submodules in layer1

    def test_dot_match_mode(self, model):
        # Test matching with match_each_dot=True
        modules = get_module_by_pattern(model, 'layer1.*.bn*', match_each_dot=True)
        assert len(modules) == 1
        assert all(isinstance(m, nn.BatchNorm2d) for m in modules.values())

    def test_no_match(self, model):
        # Test when no modules match the pattern
        modules = get_module_by_pattern(model, 'nonexistent')
        assert len(modules) == 0

    def test_multiple_wildcards(self, model):
        # Test multiple wildcard patterns
        modules = get_module_by_pattern(model, 'layer*.*')
        assert len(modules) == 7  # Should match all submodules in both layers


class TestGetModulesByType:
    def test_single_type(self, model):
        # Test finding all Conv2d layers
        modules = get_modules_by_type(model, nn.Conv2d)
        assert len(modules) == 3
        assert all(isinstance(m, nn.Conv2d) for m in modules.values())

    def test_multiple_types(self, model):
        # Test finding all BatchNorm2d layers
        modules = get_modules_by_type(model, nn.BatchNorm2d)
        assert len(modules) == 3
        assert all(isinstance(m, nn.BatchNorm2d) for m in modules.values())

    def test_base_type(self, model):
        # Test finding all activation layers using base class
        modules = get_modules_by_type(model, nn.ReLU)
        assert len(modules) == 3
        assert all(isinstance(m, nn.ReLU) for m in modules.values())

    def test_container_type(self, model):
        # Test finding Sequential containers
        modules = get_modules_by_type(model, nn.Sequential)
        assert len(modules) == 2
        assert all(isinstance(m, nn.Sequential) for m in modules.values())

    def test_invalid_input(self, model):
        # Test with invalid module input
        with pytest.raises(ValueError, match="Expected nn.Module"):
            get_modules_by_type("not a module", nn.Conv2d)


class TestGetModulesByTypeName:
    def test_exact_name(self, model):
        # Test exact class name matching
        modules = get_modules_by_type_name(model, 'Conv2d')
        assert len(modules) == 3
        assert all(m.__class__.__name__ == 'Conv2d' for m in modules.values())

    def test_wildcard_name(self, model):
        # Test wildcard in class name
        modules = get_modules_by_type_name(model, 'Conv*')
        assert len(modules) == 3
        assert all('Conv' in m.__class__.__name__ for m in modules.values())

    def test_multiple_types(self, model):
        # Test matching multiple types with pattern
        modules = get_modules_by_type_name(model, '*Norm*')
        assert len(modules) == 3
        assert all('Norm' in m.__class__.__name__ for m in modules.values())

    def test_no_match(self, model):
        # Test when no modules match the name pattern
        modules = get_modules_by_type_name(model, 'NonexistentLayer')
        assert len(modules) == 0

    def test_invalid_input(self, model):
        # Test with invalid module input
        with pytest.raises(ValueError, match="Expected nn.Module"):
            get_modules_by_type_name("not a module", "Conv2d")


def test_complex_model():
    # Test with a more complex model structure
    class ComplexNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3)
            )
            self.classifier = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )

        def forward(self, x):
            x = self.feature_extractor(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    model = ComplexNet()

    # Test pattern matching in nested structure
    conv_layers = get_module_by_pattern(model, '*.1*')
    assert len(conv_layers) == 2

    # Test type matching in nested structure
    linear_layers = get_modules_by_type(model, nn.Linear)
    assert len(linear_layers) == 2

    # Test name matching in nested structure
    relu_layers = get_modules_by_type_name(model, 'ReLU')
    assert len(relu_layers) == 2
