import fnmatch
import os
import shutil
from pathlib import Path
from typing import List, Optional, Union

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
        exclude_patterns (Optional[List[str]]): Unix-style exclusion patterns,
        e.g. ['*__init__.py', '*.pyc']

    Attributes:
        src_folder (str): Configured source directory path
        output_folder (str): Configured destination path
        file_extensions (List[str]): Active file extension filters
        exclude_patterns (List[str]): Active exclusion patterns
    """

    DEFAULT_FILE_EXTENSIONS: List[str] = [".py"]
    DEFAULT_EXCLUDE_PATTERNS: List[str] = []
    
    def __init__(
        self,
        src_folder: Union[str, Path],
        output_folder: Union[str, Path],
        file_extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        """Initialize copier with validation checks."""
        self.src_folder = Path(src_folder)
        self.output_folder = Path(output_folder)

        if not self.src_folder.is_dir() or not self.src_folder.exists():
            raise ValueError("Source folder must exist")

        if file_extensions is not None and len(file_extensions) == 0:
            raise ValueError("File extensions cannot be empty")
        elif file_extensions is None:
            file_extensions = self.DEFAULT_FILE_EXTENSIONS

        if exclude_patterns is None:
            exclude_patterns = self.DEFAULT_EXCLUDE_PATTERNS

        self.file_extensions = file_extensions
        self.exclude_patterns = exclude_patterns

    @staticmethod
    def _dump_src(
        src_folder: Path,
        output_folder: Path,
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
            relative_path = in_file.relative_to(src_folder)

            if exclude_patterns:
                if any(fnmatch.fnmatch(relative_path, p) for p in exclude_patterns):
                    continue

            shutil.copyfile(in_file, out_file)

    def dump_src(self) -> None:
        """Copy source code to output folder.

        Copies files from source directory to destination while:
        - Maintaining directory structure
        - Filtering by file extensions
        - Excluding files matching specified patterns

        Args:
            src_folder: Source directory to copy from
            output_folder: Destination directory to copy to
            file_extensions: Allowed file extensions
            exclude_patterns: Exclusion patterns (fnmatch format)
        """
        self._dump_src(
            self.src_folder, self.output_folder, 
            self.file_extensions, self.exclude_patterns
        )

    def __call__(self) -> None:
        """Execute the copy operation.

        This is a convenience method that calls dump_src().
        """
        self.dump_src()

