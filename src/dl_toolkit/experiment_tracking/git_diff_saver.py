import os
import shutil
from pathlib import Path
from typing import List, Union, Dict, Literal

from git import Repo
from git.diff import Diff
from git.objects import Commit


class GitDiffSaver:
    """Git repository differ and archiver.

    This class captures and archives git repository changes including modified,
    added, and renamed files along with git status, full diff, and revision information.

    Args:
        repo_dir: Path to Git repository. Can be string or Path object.
        output_folder: Destination for archived files. Can be string or Path object.
        tracking_files: Path prefixes to include. If None, all files are included.

    Attributes:
        repo_dir (Path): Configured repository path.
        output_folder (Path): Configured output path.
        tracking_files (List[str]): Active file filters.
        repo (git.Repo): GitPython repository instance.
    """

    def __init__(
        self,
        repo_dir: Union[str, Path],
        output_folder: Union[str, Path],
        tracking_files: List[str] | None = None,
    ):
        """Initialize differ with repository path and filters.

        Args:
            repo_dir: Path to Git repository.
            output_folder: Destination for archived files.
            tracking_files: Path prefixes to include. If None, all files are included.
        """
        self.repo_dir = Path(repo_dir)
        self.repo = Repo(self.repo_dir)
        self.tracking_files = tracking_files or [""]
        self.output_folder = Path(output_folder)

    def dump_git_data(self) -> None:
        """Main diff processing and archiving method.

        This method performs the following operations:
            1. Collects modified/added/renamed files
            2. Copies changed files to output folder
            3. Writes status, diff and revision files
        """
        hcommit: Commit = self.repo.head.commit
        git_diff = hcommit.diff(None)
        git_status_data: List[str] = []

        # Configure change types to process
        change_types: Dict[Literal["M", "A", "R", "U"], str] = {
            "M": "Modified",
            "A": "Added",
            "R": "Renamed"
        }

        os.makedirs(self.output_folder, exist_ok=True)

        # Process each change type
        for mod, mod_name in change_types.items():
            for diff in git_diff.iter_change_type(mod):
                if not self._should_track(diff):
                    continue

                self._record_change(git_status_data, diff, mod_name)
                self._copy_modified_file(diff)

        # Write metadata files
        self._write_file("git_status.txt", "\n".join(git_status_data))
        self._write_file("git_diff.txt", self.repo.git.diff(hcommit))
        self._write_file(
            "git_revision.txt", f"{hcommit}\n{self.repo.git.status(hcommit).splitlines()[0]}"
        )

    def _should_track(self, diff: Diff) -> bool:
        """Determine if a diff should be tracked based on filters.

        Args:
            diff: The git diff object to check.

        Returns:
            bool: True if the diff should be tracked, False otherwise.
        """
        paths = [p for p in [diff.a_path, diff.b_path] if p]
        return any(p.startswith(t) for t in self.tracking_files for p in paths)

    def _record_change(self, status_data: List[str], diff: Diff, mod_name: str) -> None:
        """Record change in status data list.

        Args:
            status_data: List to store status information.
            diff: The git diff object containing change information.
            mod_name: The name of the modification type.
        """
        if diff.change_type == "R":
            status_data.append(f"{mod_name:9}:  {diff.a_path} -> {diff.b_path}")
        else:
            status_data.append(f"{mod_name:9}:  {diff.a_path}")

    def _copy_modified_file(self, diff: Diff) -> None:
        """Copy modified file to output directory.

        Args:
            diff: The git diff object containing file information.
        """
        path = diff.b_path if diff.change_type == "R" else diff.a_path
        if path is None:
            return
            
        src_path = self.repo_dir / path
        if src_path.exists():
            dest_dir = self.output_folder / "modified_files" / Path(path).parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(
                src_path,
                dest_dir / Path(path).name
            )

    def _write_file(self, filename: Union[str, Path], content: str) -> None:
        """Helper to write text files to output folder.

        Args:
            filename: Name or path of the file to write.
            content: Content to write to the file.
        """
        file_path = self.output_folder / filename
        file_path.write_text(content)

    def __call__(self) -> None:
        """Execute the diff saving process.

        This is a convenience method that calls dump_git_data().
        """
        self.dump_git_data()
