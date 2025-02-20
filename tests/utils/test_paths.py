import os

import pytest

from dl_toolkit.utils.path_utils import iterate_files_recursively, \
    iterate_files_with_creating_structure


@pytest.mark.parametrize("extensions", [None, ["CsV", ".txt"], [".csv", ".txt", ".CSV", ".TXT"]])
def test_iterate(data_dir, extensions):
    out_dir = os.path.join(data_dir, "structured_folder")
    counter = 0
    for path in iterate_files_recursively(out_dir, supported_extensions=extensions):
        assert os.path.exists(path)
        print(path)
        counter += 1
    assert counter == 4


@pytest.mark.parametrize("extensions", [None, ["CsV", ".txt"], [".csv", ".txt", ".CSV", ".TXT"]])
def test_iterate_with_out(data_dir, tmp_path, extensions):
    in_dir = os.path.join(data_dir, "structured_folder")
    out_dir = os.path.join(tmp_path, "structured_folder")
    counter = 0
    for in_path, out_path in iterate_files_with_creating_structure(in_dir, out_dir,
                                                                   supported_extensions=extensions):
        print(in_path)
        counter += 1
        with open(out_path, "w") as f:
            f.write(f"test \n {in_path} \n {counter}")
        assert os.path.exists(in_path)
        assert os.path.exists(out_path)
    assert counter == 4


def test_iterate_with_ext(data_dir):
    out_dir = os.path.join(data_dir, "structured_folder")
    counter = 0
    for path in iterate_files_recursively(out_dir, supported_extensions=["CsV"]):
        assert os.path.exists(path)
        counter += 1
    assert counter == 1


@pytest.mark.parametrize("extensions", [["CsV"], [".csv"]])
def test_iterate_with_out_with_ext(data_dir, tmp_path, extensions):
    in_dir = os.path.join(data_dir, "structured_folder")
    out_dir = os.path.join(tmp_path, "structured_folder")
    counter = 0
    for in_path, out_path in iterate_files_with_creating_structure(in_dir, out_dir, extensions):
        print(in_path)
        counter += 1
        with open(out_path, "w") as f:
            f.write(f"test \n {in_path} \n {counter}")
        assert os.path.exists(in_path)
        assert os.path.exists(out_path)
    print(out_path.parent)
    assert len(os.listdir(out_path.parent)) == 1  # Only single file without subfolders
    assert counter == 1
