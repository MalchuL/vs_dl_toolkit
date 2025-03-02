import fnmatch
import os
import shutil
from pathlib import Path
from typing import List, Optional

from dl_toolkit.utils.path_utils import iterate_files_with_creating_structure


class CopySrcCode:
    """Source code copier with extension filtering and pattern exclusion.

    Copies files from source directory to destination while:
    - Maintaining directory structure
    - Filtering by file extensions
    - Excluding files matching specified patterns

    Args:
        src_folder (str): Source directory to copy from
        output_folder (str): Destination directory to copy to
        file_extensions (List[str]): Allowed file extensions (e.g. ['.py', '.txt'])
        exclude_patterns (Optional[List[str]]): Unix-style exclusion patterns

    Attributes:
        src_folder (str): Configured source directory path
        output_folder (str): Configured destination path
        file_extensions (List[str]): Active file extension filters
        exclude_patterns (List[str]): Active exclusion patterns
    """

    def __init__(
        self,
        src_folder: str,
        output_folder: str,
        file_extensions: Optional[List[str]] = (".py",),
        exclude_patterns: Optional[List[str]] = ("*__init__.py",),
    ):
        """Initialize copier with validation checks."""
        assert os.path.isdir(src_folder) and os.path.exists(src_folder), "Source folder must exist"
        self.src_folder = src_folder
        self.output_folder = output_folder or "."

        assert (
            file_extensions is not None and len(file_extensions) > 0
        ), "At least one file extension required"
        self.file_extensions = file_extensions
        self.exclude_patterns = exclude_patterns

    @staticmethod
    def dump_src(
        src_folder: str,
        output_folder: str,
        file_extensions: List[str],
        exclude_patterns: Optional[List[str]],
    ) -> None:
        """Copy files with given extensions, excluding patterns.

        Args:
            src_folder: Source directory root
            output_folder: Destination directory root
            file_extensions: Allowed file extensions
            exclude_patterns: Exclusion patterns (fnmatch format)

        Raises:
            FileNotFoundError: If source files disappear during copy
        """
        for in_file, out_file in iterate_files_with_creating_structure(
            src_folder, output_folder, supported_extensions=file_extensions
        ):
            relative_path = os.path.relpath(in_file, src_folder)

            if exclude_patterns:
                if any(fnmatch.fnmatch(relative_path, p) for p in exclude_patterns):
                    continue

            shutil.copyfile(in_file, out_file)

    def __call__(self) -> None:
        """Execute the copy operation."""
        self.dump_src(
            self.src_folder, self.output_folder, self.file_extensions, self.exclude_patterns
        )
