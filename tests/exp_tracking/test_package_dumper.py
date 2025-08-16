from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from dl_toolkit.experiment_tracking.package_dumper import PackageDumper


@pytest.fixture
def output_dir(tmp_path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_package():
    """Create a mock package."""
    mock = Mock()
    mock.key = "test-package"
    mock.version = "1.0.0"
    mock.location = "/path/to/package"
    mock.requires.return_value = ["dependency1>=1.0", "dependency2>=2.0"]
    return mock


def test_init_with_str_path(output_dir):
    """Test initialization with string path."""
    dumper = PackageDumper(str(output_dir))
    assert isinstance(dumper.output_folder, Path)
    assert dumper.output_folder == output_dir


def test_init_with_path_object(output_dir):
    """Test initialization with Path object."""
    dumper = PackageDumper(output_dir)
    assert isinstance(dumper.output_folder, Path)
    assert dumper.output_folder == output_dir


def test_get_package_info(output_dir, mock_package):
    """Test package information collection."""
    with patch("pkg_resources.working_set", [mock_package]):
        dumper = PackageDumper(output_dir)
        packages = dumper._get_package_info()
        
        assert "test-package" in packages
        assert packages["test-package"]["version"] == "1.0.0"
        assert packages["test-package"]["location"] == "/path/to/package"
        assert packages["test-package"]["requires"] == "dependency1>=1.0, dependency2>=2.0"


def test_format_package_info(output_dir, mock_package):
    """Test package information formatting."""
    with patch("pkg_resources.working_set", [mock_package]):
        dumper = PackageDumper(output_dir)
        packages = dumper._get_package_info()
        content = dumper._format_package_info(packages)
        
        assert "Python Packages:" in content
        assert "Python version:" in content
        assert "Package: test-package" in content
        assert "Version: 1.0.0" in content
        assert "Location: /path/to/package" in content
        assert "Requires: dependency1>=1.0, dependency2>=2.0" in content


def test_format_requirements(output_dir, mock_package):
    """Test requirements.txt formatting."""
    with patch("pkg_resources.working_set", [mock_package]):
        dumper = PackageDumper(output_dir)
        packages = dumper._get_package_info()
        content = dumper._format_requirements(packages)
        
        assert content == "test-package==1.0.0"


def test_dump_packages(output_dir, mock_package):
    """Test package dumping to files."""
    with patch("pkg_resources.working_set", [mock_package]):
        dumper = PackageDumper(output_dir)
        dumper.dump_packages()
        
        # Check packages.txt
        packages_file = output_dir / "packages.txt"
        assert packages_file.exists()
        content = packages_file.read_text()
        assert "Python Packages:" in content
        assert "Package: test-package" in content
        assert "Version: 1.0.0" in content
        
        # Check requirements.txt
        requirements_file = output_dir / "requirements.txt"
        assert requirements_file.exists()
        content = requirements_file.read_text()
        assert content == "test-package==1.0.0"


def test_multiple_packages(output_dir):
    """Test handling multiple packages."""
    mock_pkg1 = Mock(key="pkg1", version="1.0.0", location="/path/to/pkg1")
    mock_pkg1.requires.return_value = []
    
    mock_pkg2 = Mock(key="pkg2", version="2.0.0", location="/path/to/pkg2")
    mock_pkg2.requires.return_value = ["pkg1>=1.0"]
    
    with patch("pkg_resources.working_set", [mock_pkg1, mock_pkg2]):
        dumper = PackageDumper(output_dir)
        dumper.dump_packages()
        
        # Check requirements.txt
        requirements_file = output_dir / "requirements.txt"
        content = requirements_file.read_text()
        assert "pkg1==1.0.0\npkg2==2.0.0" == content