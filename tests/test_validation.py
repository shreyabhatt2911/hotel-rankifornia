import numpy as np
import pandas as pd
import pytest

from hotel_rankifornia.validation import ndcg_at_k, temporal_split

LOG2_3 = np.log2(3)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([1, 1, 1], [0, 1, 5], [0.1, 0.2, 0.3]) == pytest.approx(1.0)


def test_ndcg_worst_ranking_matches_hand_computation():
    # ranked order is relevance 0, 1, 5, so DCG = 0 + 1/log2(3) + 5/log2(4) and ideal = 5 + 1/log2(3)
    expected = (1 / LOG2_3 + 5 / 2) / (5 + 1 / LOG2_3)
    assert ndcg_at_k([1, 1, 1], [0, 1, 5], [0.3, 0.2, 0.1]) == pytest.approx(expected)


def test_ndcg_truncates_at_k():
    rel = [0, 0, 5]
    # with k=2 the booked item ranked last is outside the cut-off -> 0
    assert ndcg_at_k([1, 1, 1], rel, [3, 2, 1], k=2) == 0.0
    assert ndcg_at_k([1, 1, 1], rel, [3, 2, 1], k=3) > 0.0


def test_ndcg_skips_searches_without_relevance_and_singletons():
    srch = [1, 1, 2, 2, 3]
    rel = [0, 5, 0, 0, 5]  # search 2 has no relevant item, search 3 has a single row
    scores = [0.1, 0.9, 0.5, 0.4, 0.3]
    assert ndcg_at_k(srch, rel, scores) == pytest.approx(1.0)


def test_ndcg_handles_unsorted_rows():
    srch = np.array([2, 1, 2, 1])
    rel = np.array([0, 5, 5, 0])
    scores = np.array([0.1, 0.9, 0.8, 0.2])
    assert ndcg_at_k(srch, rel, scores) == pytest.approx(1.0)


def _dates(values):
    return pd.Series(pd.to_datetime(values), index=pd.Index(range(1, len(values) + 1), name="srch_id"))


def test_temporal_split_holds_out_latest_searches():
    # srch_id order is deliberately not chronological
    dates = _dates(["2013-03-05", "2013-01-01", "2013-05-09", "2013-02-02", "2013-04-04",
                    "2013-06-06", "2013-01-15", "2013-03-30", "2013-02-20", "2013-05-01"])
    train_ids, val_ids = temporal_split(dates, val_fraction=0.2)
    assert val_ids == {3, 6}  # the two latest searches
    assert train_ids == set(range(1, 11)) - {3, 6}


def test_temporal_split_is_disjoint_and_ties_go_to_validation():
    dates = _dates(["2013-01-01"] * 6 + ["2013-06-01"] * 4)  # 4 searches share the cutoff date
    train_ids, val_ids = temporal_split(dates, val_fraction=0.2)
    assert not train_ids & val_ids
    assert val_ids == {7, 8, 9, 10}
    assert train_ids == {1, 2, 3, 4, 5, 6}


def test_temporal_split_rejects_empty_validation_set():
    with pytest.raises(ValueError):
        temporal_split(_dates(["2013-01-01", "2013-01-02"]), val_fraction=0.2)
