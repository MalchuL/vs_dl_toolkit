import os

import pytest

DATA_FOLDER_NAME = "data"


@pytest.fixture
def data_dir() -> str:
    path = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(path, DATA_FOLDER_NAME)
    return data_path
