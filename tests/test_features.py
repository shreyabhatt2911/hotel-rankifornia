import numpy as np
import pandas as pd
import pytest

from hotel_rankifornia.features import add_null_count, relevance_labels, select_feature_cols


def test_relevance_labels():
    df = pd.DataFrame({"click_bool": [0, 1, 1, 0], "booking_bool": [0, 0, 1, 1]})
    labels = relevance_labels(df)
    assert labels.tolist() == [0, 1, 5, 5]  # booking wins over click
    assert labels.dtype == np.int8


def test_null_count_counts_missing_flags():
    df = pd.DataFrame({
        "has_visitor_history": np.array([1, 0], dtype="int8"),
        "review_score_missing": np.array([0, 1], dtype="int8"),
        "loc_score2_missing": np.array([0, 1], dtype="int8"),
        "dist_missing": np.array([0, 0], dtype="int8"),
        "affinity_missing": np.array([0, 1], dtype="int8"),
        "no_hist_price": np.array([0, 1], dtype="int8"),
    })
    add_null_count(df)
    assert df["count_null"].tolist() == [0, 5]


def test_select_feature_cols_excludes_ids_and_label_and_checks_agreement():
    train = pd.DataFrame(columns=["srch_id", "prop_id", "f1", "f2", "relevance"])
    test = pd.DataFrame(columns=["srch_id", "prop_id", "f1", "f2"])
    assert select_feature_cols(train, test) == ["f1", "f2"]
    with pytest.raises(AssertionError):
        select_feature_cols(train.drop(columns="f2"), test)
