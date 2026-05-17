"""Weighted RRF blend of finished submissions (the final cross-variant blend).

Blending across *feature-set versions* (v4 unbiased vs. v3 improved / optuna)
is what added diversity in the project. Blending models trained on identical
features never did (rank correlations ~0.99).

Each submission lists, per search, the properties in ranked order. A property's
blended score is ``sum_i weight_i / (k + rank_i)`` where ``rank_i`` is its rank in
submission ``i``. Properties are then re-sorted by that score within each search.
The RRF transform is shared with ``hotel_rankifornia.ensemble``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from hotel_rankifornia.config import RRF_K
from hotel_rankifornia.ensemble import rrf_terms

logger = logging.getLogger(__name__)


def load_submission(path: Path) -> pd.DataFrame:
    """Read a submission and add each property's 1-based rank within its search.

    Submissions are stored in rank order, so the rank is the position within
    the search block.
    """
    df = pd.read_csv(path)
    df["rank"] = df.groupby("srch_id").cumcount() + 1
    return df


def rrf_blend(subs: Sequence[pd.DataFrame], weights: Sequence[float] | None = None,
              k: int = RRF_K) -> pd.DataFrame:
    """Weighted RRF blend of ``subs`` (frames from ``load_submission``).

    Weights are normalised to sum to one. A property missing from a submission
    is treated as ranked last (``max_rank + 1``) in it.

    Returns
    -------
        ``srch_id, prop_id`` rows ordered by search and descending blended score.
    """
    if weights is None:
        weights = [1.0] * len(subs)
    norm_weights = np.array(weights) / np.sum(weights)

    merged = subs[0][["srch_id", "prop_id", "rank"]].copy()
    merged.rename(columns={"rank": "rank_0"}, inplace=True)
    for i in range(1, len(subs)):
        other = subs[i][["srch_id", "prop_id", "rank"]].copy()
        other.rename(columns={"rank": f"rank_{i}"}, inplace=True)
        merged = merged.merge(other, on=["srch_id", "prop_id"], how="outer")

    merged["rrf_score"] = 0.0
    for i in range(len(subs)):
        rank_col = f"rank_{i}"
        max_rank = merged[rank_col].max()
        merged[rank_col] = merged[rank_col].fillna(max_rank + 1)
        merged["rrf_score"] += rrf_terms(merged[rank_col], k=k, weight=norm_weights[i])

    result = merged.sort_values(["srch_id", "rrf_score"], ascending=[True, False])
    return result[["srch_id", "prop_id"]]


def blend_submissions(paths: Mapping[str, Path], weights: Mapping[str, float], k: int = RRF_K) -> pd.DataFrame:
    """Blend the submission files ``paths[name]`` with ``weights[name]`` (in weight order)."""
    names = list(weights)
    subs = [load_submission(paths[name]) for name in names]
    sizes = {name: len(sub) for name, sub in zip(names, subs)}
    logger.info("Loaded submissions %s", sizes)
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Row count mismatch between submissions: {sizes}")
    return rrf_blend(subs, [weights[name] for name in names], k=k)
