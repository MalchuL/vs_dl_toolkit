import os
from pathlib import Path

import pytest

from dl_toolkit.experiment_tracking.src_code_copy import CopySrcCode


@pytest.fixture
def output_dir(tmp_path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def src_dir(tmp_path) -> Path:
    """Create a temporary source directory with test files."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    # Create test files
    (src_dir / "file1.py").write_text("content1")
    (src_dir / "file2.txt").write_text("content2")
    (src_dir / "test").mkdir()
    (src_dir / "test" / "test_file.py").write_text("test_content")
    (src_dir / "__init__.py").write_text("")

    return src_dir


@pytest.fixture
def src_dir2(tmp_path) -> Path:
    """Create a second temporary source directory with test files."""
    src_dir = tmp_path / "src2"
    src_dir.mkdir()

    # Create test files with some overlapping names
    (src_dir / "file3.py").write_text("content3")
    (src_dir / "file2.py").write_text("content2_py")
    (src_dir / "test").mkdir()
    (src_dir / "test" / "another_test.py").write_text("another_test_content")

    return src_dir


def test_init_with_str_path(src_dir, output_dir):
    """Test initialization with string paths."""
    copier = CopySrcCode(str(src_dir), str(output_dir))
    assert isinstance(copier.src_folders[0], Path)
    assert isinstance(copier.output_folder, Path)
    assert copier.src_folders[0] == src_dir
    assert copier.output_folder == output_dir


def test_init_with_path_objects(src_dir, output_dir):
    """Test initialization with Path objects."""
    copier = CopySrcCode(src_dir, output_dir)
    assert isinstance(copier.src_folders[0], Path)
    assert isinstance(copier.output_folder, Path)
    assert copier.src_folders[0] == src_dir
    assert copier.output_folder == output_dir


def test_init_with_multiple_sources(src_dir, src_dir2, output_dir):
    """Test initialization with multiple source directories."""
    copier = CopySrcCode([src_dir, src_dir2], output_dir)
    assert len(copier.src_folders) == 2
    assert all(isinstance(p, Path) for p in copier.src_folders)
    assert copier.src_folders[0] == src_dir
    assert copier.src_folders[1] == src_dir2


def test_init_with_invalid_source():
    """Test initialization with non-existent source directory."""
    with pytest.raises(ValueError, match="Source folder must exist"):
        CopySrcCode("/nonexistent/path", "/some/output")


def test_copy_single_source(src_dir, output_dir):
    """Test copying files from a single source directory."""
    copier = CopySrcCode(src_dir, output_dir)
    copier.dump_src()

    # Check files were copied with correct structure
    src_output = output_dir / src_dir.name
    assert (src_output / "file1.py").exists()
    assert not (src_output / "file2.txt").exists()  # Not a .py file
    assert (src_output / "test" / "test_file.py").exists()
    assert (src_output / "__init__.py").exists()


def test_copy_multiple_sources(src_dir, src_dir2, output_dir):
    """Test copying files from multiple source directories."""
    copier = CopySrcCode([src_dir, src_dir2], output_dir)
    copier.dump_src()

    # Check files from first source
    src1_output = output_dir / src_dir.name
    assert (src1_output / "file1.py").exists()
    assert (src1_output / "test" / "test_file.py").exists()

    # Check files from second source
    src2_output = output_dir / src_dir2.name
    assert (src2_output / "file3.py").exists()
    assert (src2_output / "file2.py").exists()
    assert (src2_output / "test" / "another_test.py").exists()


def test_file_extensions_filter(src_dir, output_dir):
    """Test filtering by file extensions."""
    copier = CopySrcCode(src_dir, output_dir, file_extensions=[".txt"])
    copier.dump_src()

    src_output = output_dir / src_dir.name
    assert not (src_output / "file1.py").exists()
    assert (src_output / "file2.txt").exists()


def test_exclude_patterns(src_dir, output_dir):
    """Test excluding files by patterns."""
    copier = CopySrcCode(
        src_dir, output_dir,
        exclude_patterns=["*__init__.py", "*/test_*.py"]
    )
    copier.dump_src()

    src_output = output_dir / src_dir.name
    assert (src_output / "file1.py").exists()
    assert not (src_output / "__init__.py").exists()
    assert not (src_output / "test" / "test_file.py").exists()


def test_empty_file_extensions():
    """Test initialization with empty file extensions list."""
    with pytest.raises(ValueError, match="Source folder must exist: /some/path"):
        CopySrcCode("/some/path", "/output", file_extensions=[])


def create_test_structure(tmpdir):
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
def test_basic_copy(tmpdir, extensions, exclude, expected):
    """Test core copying functionality with different filters.
    Args:
        extensions: File extensions to include
        exclude: Patterns to exclude
        expected: Expected copied files
    """
    src = create_test_structure(tmpdir)
    dest = Path(tmpdir.mkdir("dest"))
    copier = CopySrcCode(
        src_folders=str(src),
        output_folder=str(dest),
        file_extensions=extensions,
        exclude_patterns=exclude,
    )
    copier()
    found = [str(p.relative_to(dest).name) for p in dest.rglob("*") if p.is_file()]
    assert sorted(found) == sorted(expected)

def test_nonexistent_source():
    """Test invalid source directory handling."""
    with pytest.raises(ValueError):
        CopySrcCode(src_folders="/invalid/path", output_folder=".")

def test_empty_extensions():
    """Test empty file extensions validation."""
    with pytest.raises(ValueError):
        CopySrcCode(src_folders=".", output_folder=".", file_extensions=[])

def test_directory_structure(tmpdir):
    """Verify directory hierarchy preservation."""
    src = create_test_structure(tmpdir)
    dest = Path(tmpdir.mkdir("dest"))
    copier = CopySrcCode(src_folders=str(src), output_folder=str(dest), file_extensions=[".py"])
    copier()
    expected_dirs = [dest / "src" / "module", dest / "src" / "scripts" / "util"]
    for d in expected_dirs:
        assert d.exists(), os.listdir(dest)

def test_overwrite_existing(tmpdir):
    """Test existing file overwriting."""
    src = create_test_structure(tmpdir)
    dest = Path(tmpdir.mkdir("dest"))
    (dest / "module").mkdir(parents=True)
    (dest / "module" / "code.py").touch()  # Existing dir
    copier = CopySrcCode(src_folders=str(src), output_folder=str(dest), file_extensions=[".py"])
    copier()
    assert (dest / "module" / "code.py").is_file()