# DL-Toolkit

A comprehensive PyTorch-based toolkit for building effective deep learning models, with a focus on computer vision tasks. This toolkit provides a collection of reusable components, losses, and utilities commonly used in deep learning research and development.

## Features

### Core Components

- **Modular Architecture**: Built on a versioned module system for consistent versioning and compatibility
- **Experiment Tracking**: Tools for tracking experiments, including:
  - CLI arguments logging
  - Git diff tracking
  - Package version management
  - System information logging
  - Source code versioning

### Neural Network Components

#### Layers
- **Convolution Modules**:
  - ConvBNReLU blocks
  - Separable convolutions
  - Coordinate convolution
  - Residual blocks (v1 and v2)
  - Up/downsampling layers

- **Representation Layers**:
  - Style representation (based on [paper](https://arxiv.org/pdf/2207.02426.pdf))
  - Color shift
  - Guided filter
  - USM sharpening
  - Sobel filter

- **Normalization**:
  - Group normalization
  - Custom normalization layers

#### Loss Functions
- **Classification**:
  - Focal loss
- **Distribution**:
  - KL divergence loss
- **GAN-specific**:
  - Standard GAN loss
  - RP-GAN loss
- **Image-specific**:
  - Perceptual loss
  - Structure loss
  - Total Variation (TV) loss
- **Regression**:
  - Charbonnier loss
  - T-Clip loss
- **Other**:
  - Identity loss
  - Merging loss

### Feature Extractors
- VGG-based feature extraction with customizable padding modes

## Installation

### Requirements
- Python >= 3.10
- PyTorch >= 2.6.0
- torchvision >= 0.21.0

### Install from Source
```bash
# Clone the repository
git clone [repository-url]
cd dl-toolkit

# Install with development dependencies
pip install -e ".[dev]"
```

## Development Setup

### Dependencies
The project uses the following development tools:
- JupyterLab for experimentation
- pytest for testing
- pre-commit hooks for code quality

### Setting up Development Environment
```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests
```bash
pytest
```

## Project Structure
```
dl_toolkit/
├── src/dl_toolkit/          # Main package source
│   ├── modules/            # Core neural network modules
│   │   ├── layers/        # Neural network layers
│   │   ├── losses/        # Loss functions
│   │   └── feature_extractors/ # Feature extraction modules
│   ├── experiment_tracking/ # Experiment logging and tracking
│   └── utils/             # Utility functions
├── tests/                  # Test suite
└── notebooks/             # Example notebooks
```

## License
[Add License Information]

## Contributing
[Add Contributing Guidelines]

## Authors
- Vlad Sorokin (neuromancer.ai.lover@gmail.com)

## Citation
If you use this toolkit in your research, please cite:
[Add Citation Information]
