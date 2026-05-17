import pandas as pd
import pytest

from hotel_rankifornia.blend import blend_submissions, load_submission, rrf_blend


def _sub(order_by_search: dict[int, list[int]]) -> pd.DataFrame:
    rows = [(s, p) for s, props in order_by_search.items() for p in props]
    df = pd.DataFrame(rows, columns=["srch_id", "prop_id"])
    df["rank"] = df.groupby("srch_id").cumcount() + 1
    return df


def _props(result: pd.DataFrame, srch_id: int) -> list[int]:
    return result.loc[result["srch_id"] == srch_id, "prop_id"].tolist()


def test_equal_weight_blend_follows_reciprocal_rank_fusion():
    a = _sub({1: [10, 11, 12]})
    b = _sub({1: [11, 12, 10]})
    # scores (k=60) 10 -> 1/61+1/63, 11 -> 1/62+1/61, 12 -> 1/63+1/62
    assert _props(rrf_blend([a, b]), 1) == [11, 10, 12]


def test_weights_shift_the_consensus_towards_the_heavier_submission():
    a = _sub({1: [10, 11, 12]})
    b = _sub({1: [12, 11, 10]})
    assert _props(rrf_blend([a, b], [0.7, 0.3]), 1) == [10, 11, 12]
    assert _props(rrf_blend([a, b], [0.3, 0.7]), 1) == [12, 11, 10]


def test_weights_are_normalised():
    a = _sub({1: [10, 11, 12], 2: [20, 21]})
    b = _sub({1: [11, 12, 10], 2: [21, 20]})
    pd.testing.assert_frame_equal(rrf_blend([a, b], [1, 3]), rrf_blend([a, b], [0.25, 0.75]))


def test_blend_keeps_searches_grouped_and_sorted():
    a = _sub({2: [20, 21], 1: [10, 11]})
    b = _sub({2: [21, 20], 1: [10, 11]})
    result = rrf_blend([a, b])
    assert result["srch_id"].tolist() == [1, 1, 2, 2]
    assert _props(result, 1) == [10, 11]


def test_blend_submissions_reads_files_in_weight_order(tmp_path):
    paths = {}
    for name, order in {"first": [10, 11, 12], "second": [12, 11, 10]}.items():
        path = tmp_path / f"{name}.csv"
        pd.DataFrame({"srch_id": [1, 1, 1], "prop_id": order}).to_csv(path, index=False)
        paths[name] = path
    assert list(load_submission(paths["first"])["rank"]) == [1, 2, 3]
    blended = blend_submissions(paths, {"first": 0.8, "second": 0.2})
    assert blended["prop_id"].tolist() == [10, 11, 12]


def test_blend_submissions_rejects_different_sizes(tmp_path):
    pd.DataFrame({"srch_id": [1, 1], "prop_id": [10, 11]}).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame({"srch_id": [1], "prop_id": [10]}).to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(ValueError):
        blend_submissions({"a": tmp_path / "a.csv", "b": tmp_path / "b.csv"}, {"a": 0.5, "b": 0.5})
