import os
import sys
from pathlib import Path

import pytest

from dl_toolkit.experiment_tracking.cli_arguments_dump import CLIArgumentsDumper


class TestCLIArgumentsDumper:
    """Test suite for CLIArgumentsDumper functionality."""

    @pytest.fixture
    def default_output(self, tmpdir):
        return Path(tmpdir) / "test_script.sh"

    def test_basic_capture(self, default_output, monkeypatch):
        """Test basic argument capture without CUDA devices."""
        monkeypatch.setattr(sys, "argv", ["main.py", "--epochs", "50"])
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        dumper = CLIArgumentsDumper(default_output)
        dumper()

        content = default_output.read_text()
        assert content == "python main.py --epochs 50"

    def test_cuda_devices_capture(self, default_output, monkeypatch):
        """Test CUDA_VISIBLE_DEVICES inclusion."""
        monkeypatch.setattr(sys, "argv", ["train.py"])
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")

        CLIArgumentsDumper(default_output)()

        content = default_output.read_text()
        assert content.startswith("CUDA_VISIBLE_DEVICES=0,1,2 python train.py")

    def test_special_characters(self, default_output, monkeypatch):
        """Test handling of special characters in arguments."""
        monkeypatch.setattr(sys, "argv", ["script.py", "--name", "Test Model v2.0"])
        CLIArgumentsDumper(default_output)()

        content = default_output.read_text()
        assert "Test Model v2.0" in content

    def test_output_directory_creation(self, tmpdir, monkeypatch):
        """Test writing to nested directories."""
        output_path = Path(tmpdir) / "nested/dir/script.sh"
        monkeypatch.setattr(sys, "argv", ["test.py"])

        CLIArgumentsDumper(output_path)()

        assert output_path.exists()
        assert "python test.py" in output_path.read_text()

    def test_multiple_calls(self, default_output, monkeypatch):
        """Test overwriting behavior on subsequent calls."""
        monkeypatch.setattr(sys, "argv", ["first.py"])
        dumper = CLIArgumentsDumper(default_output)
        dumper()

        monkeypatch.setattr(sys, "argv", ["second.py"])
        dumper()

        assert "second.py" in default_output.read_text()
