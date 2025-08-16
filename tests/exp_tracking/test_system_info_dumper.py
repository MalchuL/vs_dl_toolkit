import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dl_toolkit.experiment_tracking.system_info_dumper import (CPUInfo,
                                                             MemoryInfo,
                                                             SystemInfo,
                                                             SystemInfoDumper)


@pytest.fixture
def output_dir(tmp_path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_cpu_info() -> CPUInfo:
    """Create mock CPU information."""
    return CPUInfo(
        processor="AMD Ryzen 9 5950X",
        cores_physical=16,
        cores_total=32,
        usage_percent=25.5,
        frequency_current=3800.0,
        frequency_min=2200.0,
        frequency_max=4900.0,
        temperature=65.5
    )


@pytest.fixture
def mock_memory_info() -> MemoryInfo:
    """Create mock memory information."""
    return MemoryInfo(
        total=32 * 1024**3,  # 32 GB
        available=24 * 1024**3,  # 24 GB
        used=8 * 1024**3,  # 8 GB
        percent=25.0
    )




def test_init_with_str_path(output_dir):
    """Test initialization with string path."""
    dumper = SystemInfoDumper(str(output_dir))
    assert isinstance(dumper.output_folder, Path)
    assert dumper.output_folder == output_dir


def test_init_with_path_object(output_dir):
    """Test initialization with Path object."""
    dumper = SystemInfoDumper(output_dir)
    assert isinstance(dumper.output_folder, Path)
    assert dumper.output_folder == output_dir


@patch("psutil.cpu_count", return_value=16)
@patch("psutil.cpu_freq")
@patch("psutil.cpu_percent", return_value=25.5)
@patch("psutil.sensors_temperatures")
@patch("platform.processor", return_value="AMD Ryzen 9 5950X")
def test_get_cpu_info(mock_processor, mock_temps, mock_percent, mock_freq, mock_count):
    """Test CPU information collection."""
    # Mock CPU frequency
    freq = MagicMock()
    freq.current = 3800.0
    freq.min = 2200.0
    freq.max = 4900.0
    mock_freq.return_value = freq

    # Mock temperature sensors
    mock_temps.return_value = {
        "k10temp": [MagicMock(current=65.5)]
    }

    dumper = SystemInfoDumper("output")
    cpu_info = dumper._get_cpu_info()

    assert cpu_info.processor == "AMD Ryzen 9 5950X"
    assert cpu_info.cores_total == 16
    assert cpu_info.usage_percent == 25.5
    assert cpu_info.frequency_current == 3800.0
    assert cpu_info.frequency_min == 2200.0
    assert cpu_info.frequency_max == 4900.0
    assert cpu_info.temperature == 65.5


@patch("psutil.virtual_memory")
def test_get_memory_info(mock_memory):
    """Test memory information collection."""
    mem = MagicMock()
    mem.total = 32 * 1024**3  # 32 GB
    mem.available = 24 * 1024**3  # 24 GB
    mem.used = 8 * 1024**3  # 8 GB
    mem.percent = 25.0
    mock_memory.return_value = mem

    dumper = SystemInfoDumper("output")
    memory_info = dumper._get_memory_info()

    assert memory_info.total == 32 * 1024**3
    assert memory_info.available == 24 * 1024**3
    assert memory_info.used == 8 * 1024**3
    assert memory_info.percent == 25.0



def test_dump_info(output_dir, mock_cpu_info, mock_memory_info):
    """Test system information dumping to files."""
    with patch.multiple(
        "dl_toolkit.experiment_tracking.system_info_dumper.SystemInfoDumper",
        _get_cpu_info=lambda self: mock_cpu_info,
        _get_memory_info=lambda self: mock_memory_info,
    ):
        dumper = SystemInfoDumper(output_dir)
        dumper.dump_info()

        # Check text file
        text_file = output_dir / "system_info.txt"
        assert text_file.exists()
        content = text_file.read_text()
        assert "System Information:" in content
        assert mock_cpu_info.processor in content

        # Check JSON file
        json_file = output_dir / "system_info.json"
        assert json_file.exists()
        data = json.loads(json_file.read_text())
        assert data["cpu"]["processor"] == mock_cpu_info.processor


