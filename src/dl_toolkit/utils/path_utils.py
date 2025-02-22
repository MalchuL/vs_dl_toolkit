import os
from pathlib import Path
from typing import List

from natsort import natsorted
from tqdm import tqdm

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp")


def _fix_img_extensions(extensions: List[str]):
    """Fixes image extensions"""
    if len(extensions) == 0:
        raise ValueError("Extensions cannot be empty")
    result_extensions = []
    for ext in extensions:
        ext = ext.lower()
        if not ext.startswith("."):
            ext = "." + ext
        result_extensions.append(ext)
    result_extensions = list(set(result_extensions))
    return result_extensions


def iterate_files_with_creating_structure(
    in_folder: str, out_folder: str, supported_extensions: List[str] | None = None
):
    """Iterates over files and returns files with the same folder structure.

    Output folders are created automatically.

    Args:
        in_folder (str): Folder to iterate over.
        out_folder (str): Folder to save images.
        supported_extensions (list[str]): File extensions to include in the iteration.

    Yields:
        Iterator[tuple[str, str]]: An iterator yielding tuples of (filepath, output_path).
            The output path is automatically created when a file is encountered.
    """
    in_folder = Path(in_folder)
    out_folder = Path(out_folder)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)

    if supported_extensions is not None:
        supported_extensions = _fix_img_extensions(supported_extensions)

    files = in_folder.rglob(pattern="*")

    for file_path in tqdm(natsorted(files)):
        if not file_path.is_file():
            continue
        if supported_extensions is not None:
            if not file_path.suffix.lower() in supported_extensions:
                continue
        sub_path = os.path.relpath(file_path, in_folder)
        new_path = out_folder / sub_path
        if not os.path.exists(new_path.parent):
            new_path.parent.mkdir(exist_ok=True, parents=True)  # Create parent folder

        yield (file_path, new_path)


def iterate_files_recursively(in_folder, supported_extensions: List[str] | None = None):
    in_folder = Path(in_folder)
    files = in_folder.rglob(pattern="*")
    if supported_extensions is not None:
        supported_extensions = _fix_img_extensions(supported_extensions)
    for file_path in tqdm(natsorted(files)):
        if not file_path.is_file():
            continue
        if supported_extensions is not None:
            if not file_path.suffix.lower() in supported_extensions:
                continue
        yield file_path
