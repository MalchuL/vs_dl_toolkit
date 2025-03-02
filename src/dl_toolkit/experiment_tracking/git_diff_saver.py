import os
import shutil

from git import Repo


class GitDiffSaver:
    """Git repository differ and archiver.

    Captures and saves:
    - Modified/added/renamed files
    - Git status summary
    - Full git diff
    - Revision information

    Args:
        repo_dir (str): Path to Git repository
        output_folder (str): Destination for archived files
        tracking_files (list): Path prefixes to include (None=all)
        save_untracked (bool): Whether to include untracked files

    Attributes:
        repo_dir (str): Configured repository path
        output_folder (str): Configured output path
        tracking_files (list): Active file filters
        save_untracked (bool): Untracked file inclusion flag
        repo (git.Repo): GitPython repository instance
    """

    def __init__(self, repo_dir: str, output_folder: str,
                 tracking_files: list = None, save_untracked: bool = False):
        """Initialize differ with repository path and filters."""
        self.repo_dir = repo_dir
        self.repo = Repo(self.repo_dir)
        self.tracking_files = tracking_files or [""]
        self.save_untracked = save_untracked
        self.output_folder = output_folder

    def dump_git_data(self) -> None:
        """Main diff processing and archiving method.

        Performs:
        1. Collects modified/added/renamed files
        2. Copies changed files to output folder
        3. Writes status, diff and revision files
        """
        hcommit = self.repo.head.commit
        git_diff = hcommit.diff(None)
        git_status_data = []

        # Configure change types to process
        change_types = {"M": "Modified", "A": "Added", "R": "Renamed"}
        if self.save_untracked:
            change_types["U"] = "Untracked"

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
        self._write_file("git_revision.txt",
                        f"{hcommit}\n{self.repo.git.status(hcommit).splitlines()[0]}")

    def _should_track(self, diff) -> bool:
        """Determine if a diff should be tracked based on filters."""
        paths = [p for p in [diff.a_path, diff.b_path] if p]
        return any(p.startswith(t) for t in self.tracking_files for p in paths)

    def _record_change(self, status_data, diff, mod_name):
        """Record change in status data list."""
        if diff.change_type == "R":
            status_data.append(f"{mod_name:9}:  {diff.a_path} -> {diff.b_path}")
        else:
            status_data.append(f"{mod_name:9}:  {diff.a_path}")

    def _copy_modified_file(self, diff):
        """Copy modified file to output directory."""
        path = diff.b_path if diff.change_type == "R" else diff.a_path
        src_path = os.path.join(self.repo_dir, path)

        if os.path.exists(src_path):
            dest_dir = os.path.join(self.output_folder, "modified_files", os.path.dirname(path))
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(src_path, os.path.join(dest_dir, os.path.basename(path)))

    def _write_file(self, filename, content):
        """Helper to write text files to output folder."""
        with open(os.path.join(self.output_folder, filename), "w") as f:
            f.write(content)

    def __call__(self) -> None:
        """Execute the diff saving process."""
        self.dump_git_data()