import numpy as np
import pandas as pd
import pytest

from hotel_rankifornia.ensemble import (
    ScoredModel,
    greedy_forward_selection,
    make_submission,
    rrf_terms,
    rrf_within_search,
)
from hotel_rankifornia.validation import ndcg_at_k


def test_rrf_terms_formula():
    assert rrf_terms(np.array([1.0, 2.0]), k=60) == pytest.approx([1 / 61, 1 / 62])
    assert rrf_terms(1, k=60, weight=0.5) == pytest.approx(0.5 / 61)


def test_rrf_within_search_ranks_each_search_independently():
    srch = [1, 1, 1, 2, 2]
    scores = [0.2, 0.9, 0.5, 3.0, 1.0]
    rrf = rrf_within_search(srch, scores, k=60)
    # search 1 ranks 3rd, 1st, 2nd and search 2 ranks 1st, 2nd
    assert rrf == pytest.approx([1 / 63, 1 / 61, 1 / 62, 1 / 61, 1 / 62])


def test_rrf_within_search_breaks_ties_by_row_order():
    rrf = rrf_within_search([1, 1, 1], [0.5, 0.5, 0.5], k=60)
    assert rrf == pytest.approx([1 / 61, 1 / 62, 1 / 63])


def test_rrf_is_scale_invariant():
    scores = np.array([0.3, 0.1, 0.8, 0.5])
    assert np.array_equal(rrf_within_search([1] * 4, scores), rrf_within_search([1] * 4, scores * 1000 - 7))


def test_make_submission_orders_by_descending_score():
    ids = pd.DataFrame({"srch_id": [2, 1, 1, 2], "prop_id": [20, 11, 12, 21], "extra": 0})
    sub = make_submission(ids, np.array([0.1, 0.2, 0.9, 0.5]))
    assert list(sub.columns) == ["srch_id", "prop_id"]
    assert sub.values.tolist() == [[1, 12], [1, 11], [2, 21], [2, 20]]


def _scored_models(rng, n_searches=40, per_search=8):
    srch = np.repeat(np.arange(n_searches), per_search)
    relevance = rng.choice([0, 0, 0, 1, 5], size=len(srch))
    signal = relevance + rng.normal(0, 2.0, len(srch))
    models = {}
    for name, noise in {"a": 1.0, "b": 1.0, "c": 6.0}.items():
        preds = signal + rng.normal(0, noise, len(srch))
        models[name] = ScoredModel(name, ndcg_at_k(srch, relevance, preds),
                                   rrf_within_search(srch, preds), best_iteration=100)
    return models, srch, relevance


def test_greedy_selection_starts_from_best_single_and_never_gets_worse():
    models, srch, relevance = _scored_models(np.random.default_rng(3))
    best_single = max(models.values(), key=lambda m: m.ndcg)
    selection = greedy_forward_selection(models, srch, relevance)
    assert selection.names[0] == best_single.name
    assert selection.ndcg >= best_single.ndcg
    assert len(set(selection.names)) == len(selection.names)
    blend = np.mean([models[n].rrf_preds for n in selection.names], axis=0)
    assert ndcg_at_k(srch, relevance, blend) == pytest.approx(selection.ndcg)


def test_greedy_selection_stops_when_nothing_improves():
    models, srch, relevance = _scored_models(np.random.default_rng(3))
    identical = {n: ScoredModel(n, models["a"].ndcg, models["a"].rrf_preds.copy(), 100) for n in "xyz"}
    assert len(greedy_forward_selection(identical, srch, relevance).names) == 1


def test_min_models_forces_a_larger_ensemble_within_tolerance():
    models, srch, relevance = _scored_models(np.random.default_rng(3))
    identical = {n: ScoredModel(n, models["a"].ndcg, models["a"].rrf_preds.copy(), 100) for n in "xyz"}
    selection = greedy_forward_selection(identical, srch, relevance, min_models=3)
    assert sorted(selection.names) == ["x", "y", "z"]
