"""Reciprocal Rank Fusion (RRF) and greedy forward model selection.

This is the single implementation of RRF used across the project, both within an
ensemble (fusing model scores inside each search) and by
``hotel_rankifornia.blend`` (fusing whole submissions).

RRF turns a ranking into ``weight / (k + rank)``. Averaging these values over
models gives a robust consensus ranking that ignores each model's score scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hotel_rankifornia.config import RRF_K
from hotel_rankifornia.validation import ndcg_at_k

logger = logging.getLogger(__name__)


def rrf_terms(ranks, k: int = RRF_K, weight: float = 1.0):
    """RRF contribution ``weight / (k + rank)`` (rank 1 = best)."""
    return weight / (k + ranks)


def rrf_within_search(srch_ids, scores, k: int = RRF_K) -> np.ndarray:
    """RRF value of every row, ranking each search's rows by descending ``scores``.

    Ties are broken by row order (``method="first"``), so the result is
    deterministic. Output is aligned with the input rows.
    """
    frame = pd.DataFrame({"srch_id": np.asarray(srch_ids), "score": np.asarray(scores)})
    frame["rank"] = frame.groupby("srch_id")["score"].rank(ascending=False, method="first")
    return rrf_terms(frame["rank"], k=k, weight=1.0).values


def make_submission(test_ids: pd.DataFrame, scores) -> pd.DataFrame:
    """Order each search's properties by descending ``scores``.

    Parameters
    ----------
    test_ids
        Frame with ``srch_id`` and ``prop_id`` (one row per test impression).
    scores
        Ensemble score per row of ``test_ids`` (higher = better).

    Returns
    -------
        ``srch_id, prop_id`` rows sorted by search and rank, which is the submission format.
    """
    sub = test_ids[["srch_id", "prop_id"]].copy()
    sub["score"] = scores
    return sub.sort_values(["srch_id", "score"], ascending=[True, False])[["srch_id", "prop_id"]]


@dataclass
class ScoredModel:
    """A validated candidate with its validation NDCG@5 and RRF-transformed predictions."""

    name: str
    ndcg: float
    rrf_preds: np.ndarray
    best_iteration: int


@dataclass
class Selection:
    names: list[str]
    ndcg: float


def greedy_forward_selection(
    models: dict[str, ScoredModel],
    val_srch_ids,
    val_relevance,
    min_models: int = 1,
    forced_tolerance: float = 0.0003,
) -> Selection:
    """Greedy forward selection of an equal-weight RRF ensemble.

    Start from the best single model, then repeatedly add the model that raises
    validation NDCG@5 of the averaged RRF predictions the most, and stop when no
    candidate improves it.

    If the greedy result has fewer than ``min_models`` members, the top
    ``min_models`` single models are used instead provided their ensemble is
    within ``forced_tolerance`` NDCG of the greedy score (ensembles are trusted
    to generalise better than one model). ``min_models=1`` disables this.
    """
    names = list(models.keys())
    by_score = sorted(models.items(), key=lambda kv: -kv[1].ndcg)
    best_single = by_score[0][0]
    selected = [best_single]
    selected_blend = models[best_single].rrf_preds.copy()
    best_score = models[best_single].ndcg
    logger.info("Step 0 %s -> %.6f", best_single, best_score)

    for step in range(1, len(names)):
        best_add_name = None
        best_add_score = best_score
        for name in names:
            if name in selected:
                continue
            n_sel = len(selected) + 1
            trial_blend = (selected_blend * len(selected) + models[name].rrf_preds) / n_sel
            trial_score = ndcg_at_k(val_srch_ids, val_relevance, trial_blend)
            if trial_score > best_add_score:
                best_add_score = trial_score
                best_add_name = name

        if best_add_name is None:
            logger.info("Step %d no improvement, stopping", step)
            break
        selected.append(best_add_name)
        selected_blend = np.mean([models[s].rrf_preds for s in selected], axis=0)
        best_score = best_add_score
        logger.info("Step %d +%s -> %.6f (%d models)", step, best_add_name, best_score, len(selected))

    if len(selected) < min_models:
        forced = [n for n, _ in by_score[:min_models]]
        forced_blend = np.mean([models[s].rrf_preds for s in forced], axis=0)
        forced_score = ndcg_at_k(val_srch_ids, val_relevance, forced_blend)
        logger.info("Greedy picked %d model(s), forced top-%d ensemble scores %.6f",
                    len(selected), min_models, forced_score)
        if forced_score >= best_score - forced_tolerance:
            selected, best_score = forced, forced_score
    return Selection(names=selected, ndcg=best_score)
