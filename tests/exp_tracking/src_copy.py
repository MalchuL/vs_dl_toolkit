"""Source code copying utility with pattern filtering."""

import fnmatch
import os
import tempfile
from pathlib import Path

import pytest

from dl_toolkit.experiment_tracking.src_code_copy import CopySrcCode


class TestCopySrcCode:
    """Test suite for CopySrcCode functionality."""

    def create_test_structure(self, tmpdir):
        """Create test directory structure."""
        base = Path(tmpdir.mkdir("src"))
        (base / "module").mkdir()
        (base / "module" / "__init__.py").touch()
        (base / "module" / "code.py").touch()
        (base / "config.yaml").touch()
        (base / "scripts" / "util").mkdir(parents=True)
        (base / "scripts" / "util" / "tools.py").touch()
        return base

    @pytest.mark.parametrize(
        "extensions,exclude,expected",
        [
            ([".py"], ["*__init__.py"], ["code.py", "tools.py"]),
            ([".py", ".yaml"], [], ["__init__.py", "code.py", "config.yaml", "tools.py"]),
            ([".yaml"], None, ["config.yaml"]),
        ],
    )
    def test_basic_copy(self, tmpdir, extensions, exclude, expected):
        """Test core copying functionality with different filters.

        Args:
            extensions: File extensions to include
            exclude: Patterns to exclude
            expected: Expected copied files
        """
        src = self.create_test_structure(tmpdir)
        dest = Path(tmpdir.mkdir("dest"))

        copier = CopySrcCode(
            src_folder=str(src),
            output_folder=str(dest),
            file_extensions=extensions,
            exclude_patterns=exclude,
        )
        copier()

        found = [str(p.relative_to(dest).name) for p in dest.rglob("*") if p.is_file()]
        assert sorted(found) == sorted(expected)

    def test_nonexistent_source(self):
        """Test invalid source directory handling."""
        with pytest.raises(ValueError):
            CopySrcCode(src_folder="/invalid/path", output_folder=".")

    def test_empty_extensions(self):
        """Test empty file extensions validation."""
        with pytest.raises(ValueError):
            CopySrcCode(src_folder=".", output_folder=".", file_extensions=[])

    def test_directory_structure(self, tmpdir):
        """Verify directory hierarchy preservation."""
        src = self.create_test_structure(tmpdir)
        dest = Path(tmpdir.mkdir("dest"))

        copier = CopySrcCode(src_folder=str(src), output_folder=str(dest), file_extensions=[".py"])
        copier()

        expected_dirs = [dest / "module", dest / "scripts" / "util"]
        for d in expected_dirs:
            assert d.exists()

    def test_overwrite_existing(self, tmpdir):
        """Test existing file overwriting."""
        src = self.create_test_structure(tmpdir)
        dest = Path(tmpdir.mkdir("dest"))
        (dest / "module").mkdir(parents=True)
        (dest / "module" / "code.py").touch()  # Existing dir

        copier = CopySrcCode(src_folder=str(src), output_folder=str(dest), file_extensions=[".py"])
        copier()

        assert (dest / "module" / "code.py").is_file()
