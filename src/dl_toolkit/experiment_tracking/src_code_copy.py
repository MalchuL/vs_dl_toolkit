import fnmatch
import os
import shutil
from pathlib import Path
from typing import List, Optional, Union

from dl_toolkit.utils.path_utils import iterate_files_with_creating_structure


class CopySrcCode:
    """Source code copier with extension filtering and pattern exclusion.

    Copies files from source directories to destination while:
    - Maintaining directory structure
    - Filtering by file extensions
    - Excluding files matching specified patterns

    Args:
        src_folders: Source directories to copy from. Can be a single path or list of paths.
        output_folder: Destination directory to copy to.
        file_extensions: Allowed file extensions (e.g. ['.py', '.txt']).
        exclude_patterns: Unix-style exclusion patterns,
            e.g. ['*__init__.py', '*.pyc'].

    Attributes:
        src_folders (List[Path]): Configured source directory paths.
        output_folder (Path): Configured destination path.
        file_extensions (List[str]): Active file extension filters.
        exclude_patterns (List[str]): Active exclusion patterns.
    """

    DEFAULT_FILE_EXTENSIONS: List[str] = [".py"]
    DEFAULT_EXCLUDE_PATTERNS: List[str] = []
    
    def __init__(
        self,
        src_folders: Union[str, Path, List[Union[str, Path]]],
        output_folder: Union[str, Path],
        file_extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        """Initialize copier with validation checks."""
        # Convert src_folders to list if single path provided
        if isinstance(src_folders, (str, Path)):
            src_folders = [src_folders]
            
        # Convert all paths to Path objects
        self.src_folders = [Path(src) for src in src_folders]
        self.output_folder = Path(output_folder)

        # Validate source folders
        for src_folder in self.src_folders:
            if not src_folder.is_dir() or not src_folder.exists():
                raise ValueError(f"Source folder must exist: {src_folder}")

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
            src_folder: Source directory root.
            output_folder: Destination directory root.
            file_extensions: Allowed file extensions.
            exclude_patterns: Exclusion patterns (fnmatch format).

        Raises:
            FileNotFoundError: If source files disappear during copy.
        """
        for in_file, out_file in iterate_files_with_creating_structure(
            src_folder, output_folder, supported_extensions=file_extensions
        ):
            relative_path = in_file.relative_to(src_folder)

            if exclude_patterns:
                if any(fnmatch.fnmatch(relative_path, p) for p in exclude_patterns):
                    continue

            # Create parent directories if they don't exist
            out_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(in_file, out_file)

    def dump_src(self) -> None:
        """Copy source code to output folder.

        Copies files from source directories to destination while:
        - Maintaining directory structure
        - Filtering by file extensions
        - Excluding files matching specified patterns
        - Handling multiple source directories
        """
        # Create output directory if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Process each source folder
        for src_folder in self.src_folders:
            # Create a subdirectory for each source folder to avoid conflicts
            folder_name = src_folder.name
            dest_folder = self.output_folder / folder_name

            self._dump_src(
                src_folder, dest_folder,
                self.file_extensions, self.exclude_patterns
            )

    def __call__(self) -> None:
        """Execute the copy operation.

        This is a convenience method that calls dump_src().
        """
        self.dump_src()

