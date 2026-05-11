"""LightGBM LambdaMART training, covering the validated fit and full-data retrain."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from hotel_rankifornia.config import (
    BOOKING_QUERY_WEIGHT,
    EARLY_STOPPING_ROUNDS,
    MAX_BOOST_ROUNDS,
    RELEVANCE_BOOKED,
    RETRAIN_MULTIPLIER,
    ModelConfig,
)

logger = logging.getLogger(__name__)

# LightGBM's per-iteration evaluation lines go through ``logging`` instead of stdout.
lgb.register_logger(logging.getLogger("lightgbm"))


@dataclass
class RankingData:
    """A set of searches in LightGBM ranking layout.

    Rows are sorted by ``srch_id`` so that each search is a contiguous block.

    Attributes
    ----------
    X
        ``(n_rows, n_features)`` float32 feature matrix.
    y
        ``(n_rows,)`` float32 graded relevance (0 / 1 / 5).
    group
        ``(n_searches,)`` number of rows in each search, in row order.
    srch_ids
        ``(n_rows,)`` search id per row.
    relevance
        ``(n_rows,)`` graded relevance in its stored dtype (for evaluation).
    weights
        ``(n_rows,)`` float32 per-row weights (``query_weights``).
    """

    X: np.ndarray
    y: np.ndarray
    group: np.ndarray
    srch_ids: np.ndarray
    relevance: np.ndarray
    weights: np.ndarray


def query_weights(df: pd.DataFrame) -> np.ndarray:
    """Per-row weight, ``BOOKING_QUERY_WEIGHT`` for searches with a booking, else 1."""
    has_booking = df.groupby("srch_id")["relevance"].max() >= RELEVANCE_BOOKED
    weight_by_search = has_booking.map({True: BOOKING_QUERY_WEIGHT, False: 1.0}).to_dict()
    return df["srch_id"].map(weight_by_search).values.astype("float32")


def make_ranking_data(df: pd.DataFrame, feature_cols: list[str]) -> RankingData:
    """Sort ``df`` by ``srch_id`` and convert it to ``RankingData``.

    The (unstable) default sort is kept deliberately because it defines the within-search
    row order the published models were trained with.
    """
    ordered = df.sort_values("srch_id").reset_index(drop=True)
    return RankingData(
        X=ordered[feature_cols].astype("float32").values,
        y=ordered["relevance"].values.astype("float32"),
        group=ordered.groupby("srch_id", sort=True)["prop_id"].count().values,
        srch_ids=ordered["srch_id"].values,
        relevance=ordered["relevance"].values,
        weights=query_weights(ordered),
    )


def split_train_val(
    train_fe: pd.DataFrame,
    train_ids: set[int],
    val_ids: set[int],
    feature_cols: list[str],
) -> tuple[RankingData, RankingData]:
    """Materialise the temporal split as two ``RankingData`` sets."""
    tr = make_ranking_data(train_fe[train_fe["srch_id"].isin(train_ids)], feature_cols)
    va = make_ranking_data(train_fe[train_fe["srch_id"].isin(val_ids)], feature_cols)
    logger.info("Train %s rows / %s searches, Val %s rows / %s searches",
                f"{len(tr.y):,}", f"{len(tr.group):,}", f"{len(va.y):,}", f"{len(va.group):,}")
    return tr, va


@dataclass
class ValidatedFit:
    best_iteration: int
    val_preds: np.ndarray


def fit_validated(cfg: ModelConfig, tr: RankingData, va: RankingData,
                  feature_cols: list[str]) -> ValidatedFit:
    """Train with early stopping on the validation set (NDCG@5).

    Returns the early-stopped iteration count and the validation predictions.
    """
    dtrain = lgb.Dataset(tr.X, label=tr.y, group=tr.group,
                         weight=tr.weights if cfg.weighted else None,
                         feature_name=feature_cols, free_raw_data=False)
    dval = lgb.Dataset(va.X, label=va.y, group=va.group,
                       feature_name=feature_cols, free_raw_data=False, reference=dtrain)
    model = lgb.train(
        cfg.lgb_params(), dtrain, num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(500)])
    return ValidatedFit(best_iteration=model.best_iteration, val_preds=model.predict(va.X))


def retrain_rounds(best_iteration: int, multiplier: float = RETRAIN_MULTIPLIER) -> int:
    """Boosting rounds for the full-data model (validation set is 20% => ~1.25x)."""
    return int(best_iteration * multiplier)


def fit_full_and_predict(cfg: ModelConfig, full: RankingData, X_test: np.ndarray,
                         feature_cols: list[str], num_rounds: int) -> np.ndarray:
    """Retrain on all training searches for ``num_rounds`` rounds and predict the test rows."""
    dfull = lgb.Dataset(full.X, label=full.y, group=full.group,
                        weight=full.weights if cfg.weighted else None,
                        feature_name=feature_cols, free_raw_data=False)
    model = lgb.train(cfg.lgb_params(), dfull, num_boost_round=num_rounds)
    return model.predict(X_test)
