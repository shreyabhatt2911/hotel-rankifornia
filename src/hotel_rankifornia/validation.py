"""Temporal validation split and the NDCG@5 metric."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from hotel_rankifornia.config import NDCG_K, VAL_FRACTION


def temporal_split(
    search_dates: pd.Series,
    val_fraction: float = VAL_FRACTION,
) -> tuple[set[int], set[int]]:
    """Split searches into train / validation ids by date.

    ``srch_id`` is *not* chronological, so a split on the id would be a random
    split. Instead the searches are ordered by their timestamp and the latest
    ``val_fraction`` of them (``date >= cutoff``) become the validation set, and
    everything strictly earlier is the training set. All searches sharing the
    cutoff timestamp go to validation, so the two sets never overlap.

    Parameters
    ----------
    search_dates
        ``datetime64`` timestamp indexed by ``srch_id``.
    val_fraction
        Share of searches to hold out.

    Returns
    -------
        ``(train_ids, val_ids)`` as sets of ``srch_id``.
    """
    date_sorted = search_dates.sort_values()
    n_val = int(len(date_sorted) * val_fraction)
    if n_val < 1:
        raise ValueError("val_fraction too small, validation set would be empty")
    val_cutoff = date_sorted.iloc[-n_val]
    val_ids = set(date_sorted[date_sorted >= val_cutoff].index)
    train_ids = set(date_sorted[date_sorted < val_cutoff].index)
    return train_ids, val_ids


def ndcg_at_k(srch_ids, relevance, scores, k: int = NDCG_K) -> float:
    """Mean NDCG@k over searches (the competition metric).

    Searches with fewer than two results or without any relevant item are
    skipped, as in the original evaluation. Graded relevance is used directly as
    the gain (0 / 1 / 5).

    Parameters
    ----------
    srch_ids
        Search id per row.
    relevance
        True graded relevance per row.
    scores
        Predicted scores per row (higher = better).
    """
    srch = np.asarray(srch_ids)
    rel = np.asarray(relevance)
    pred = np.asarray(scores)
    order = np.argsort(srch, kind="stable")
    srch, rel, pred = srch[order], rel[order], pred[order]
    bounds = np.where(np.diff(srch))[0] + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(srch)]])
    per_search = []
    for s, e in zip(starts, ends):
        truth, predicted = rel[s:e], pred[s:e]
        if len(truth) < 2 or truth.max() == 0:
            continue
        per_search.append(ndcg_score([truth], [predicted], k=k))
    return float(np.mean(per_search))
