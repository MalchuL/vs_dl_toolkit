import os
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from git import Repo, Diff
from git.objects import Commit

from dl_toolkit.experiment_tracking.git_diff_saver import GitDiffSaver


@pytest.fixture
def temp_git_repo(tmp_path) -> Path:
    """Create a temporary git repository for testing."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Initialize git repo
    repo = Repo.init(repo_dir)
    
    # Create some test files
    (repo_dir / "file1.txt").write_text("content1")
    (repo_dir / "file2.txt").write_text("content2")
    (repo_dir / "subdir").mkdir()
    (repo_dir / "subdir" / "file3.txt").write_text("content3")
    
    # Add and commit files
    repo.index.add(["file1.txt", "file2.txt", "subdir/file3.txt"])
    repo.index.commit("Initial commit")
    
    return repo_dir


@pytest.fixture
def output_dir(tmp_path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_repo():
    """Create a mock git repository."""
    mock = MagicMock(spec=Repo)
    mock.head.commit = MagicMock(spec=Commit)
    return mock


def test_init_with_str_paths(temp_git_repo, output_dir):
    """Test initialization with string paths."""
    saver = GitDiffSaver(str(temp_git_repo), str(output_dir))
    assert isinstance(saver.repo_dir, Path)
    assert isinstance(saver.output_folder, Path)
    assert saver.repo_dir == temp_git_repo
    assert saver.output_folder == output_dir


def test_init_with_path_objects(temp_git_repo, output_dir):
    """Test initialization with Path objects."""
    saver = GitDiffSaver(temp_git_repo, output_dir)
    assert isinstance(saver.repo_dir, Path)
    assert isinstance(saver.output_folder, Path)
    assert saver.repo_dir == temp_git_repo
    assert saver.output_folder == output_dir


def test_tracking_files_filter(temp_git_repo, output_dir):
    """Test file tracking filter functionality."""
    tracking_files = ["subdir/"]
    saver = GitDiffSaver(temp_git_repo, output_dir, tracking_files=tracking_files)
    
    mock_diff = Mock(spec=Diff)
    mock_diff.a_path = "subdir/file3.txt"
    assert saver._should_track(mock_diff) == True
    


def test_git_diff_processing(temp_git_repo, output_dir, mock_repo):
    """Test git diff processing and file copying."""
    with patch('git.Repo', return_value=mock_repo):
        saver = GitDiffSaver(temp_git_repo, output_dir)
        
        # Mock diff data
        mock_diff = Mock(spec=Diff)
        mock_diff.a_path = "file1.txt"
        mock_diff.b_path = None
        mock_diff.change_type = "M"
        
        mock_repo.head.commit.diff.return_value.iter_change_type.return_value = [mock_diff]
        mock_repo.git.diff.return_value = "mock diff content"
        mock_repo.git.status.return_value = "mock status"
        
        # Execute diff saving
        saver.dump_git_data()
        
        # Check output files
        assert (output_dir / "git_diff.txt").exists()
        assert (output_dir / "git_status.txt").exists()
        assert (output_dir / "git_revision.txt").exists()




