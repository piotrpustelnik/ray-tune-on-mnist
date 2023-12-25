import sys
from pathlib import Path

import pytest

# TODO: delete when package is built
sys.path.append(Path(__file__).parent.as_posix())


import pytest
from src import loaders


def test_data_downloaded_to_correct_folder(
    tmp_path: Path,
):
    # setup

    # act
    train, test = loaders.load_data(tmp_path)

    pass

    # assert


def test_train_and_test_sets_are_different():
    ...
