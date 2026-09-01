"""Checks for joint-training data filters."""

import pandas as pd

from build_joint_training_data import filter_training_rows


def test_filter_drops_reserved_and_unmapped_pipal_rows():
    rows = pd.DataFrame({
        "dataset": ["kadid10k", "pipal", "pipal", "pipal"],
        "distortion": ["1", "00", "06", "10"],
        "group": ["blur", "generative", None, None],
    })

    result = filter_training_rows(rows)

    assert result[["dataset", "distortion"]].values.tolist() == [
        ["kadid10k", "1"],
        ["pipal", "00"],
    ]
