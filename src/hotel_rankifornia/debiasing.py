"""Position-unbiased property rates ("v3" -> "v4" features, 139 -> 146 features).

Click and booking rates are contaminated by position bias. The hotel shown first
gets far more clicks regardless of quality. When ``random_bool == 1`` Expedia
showed the hotels in *random* order, so position has no systematic effect on
visibility and click/booking rates computed from that subset are unbiased
estimates of a property's appeal.

The unbiased rates use the same leak-free scheme as
``hotel_rankifornia.target_encoding`` (5-fold OOF on ``srch_id`` groups,
per-fold priors, averaged fold tables for test) but

* only random-order impressions contribute to the tables and to the priors, and
* the smoothing constant is lower (k=30) because random-order data is only ~30%
  of the training rows and the prior itself is unbiased.

The module also repairs ``price_vs_hist`` / ``price_discount`` for properties
without a historical price (``exp(0) = 1`` made them ~200 and ~-200).
"""

from __future__ import annotations

import gc
import logging

import numpy as np
import pandas as pd

from hotel_rankifornia.config import N_OOF_FOLDS
from hotel_rankifornia.target_encoding import oof_folds, smooth

logger = logging.getLogger(__name__)

SMOOTH_K_UNBIASED = 30
UNBIASED_COLUMNS = ("prop_click_rate_unbiased", "prop_booking_rate_unbiased")
LABEL_COLUMNS = ("srch_id", "prop_id", "click_bool", "booking_bool", "random_bool")

NEW_FEATURES = (
    "prop_click_rate_unbiased", "prop_booking_rate_unbiased",
    "click_rate_bias", "booking_rate_bias",
    "unbiased_click_rate_rank", "unbiased_booking_rate_rank",
    "unbiased_booking_vs_search_mean",
)


def _fit_unbiased_table(fold_train: pd.DataFrame) -> pd.DataFrame:
    """Smoothed click / booking rate per property from random-order rows only."""
    random_rows = fold_train[fold_train["random_bool"] == 1]
    prior_click = float(random_rows["click_bool"].mean())
    prior_book = float(random_rows["booking_bool"].mean())
    stats = random_rows.groupby("prop_id").agg(
        cl=("click_bool", "sum"), bk=("booking_bool", "sum"), n=("click_bool", "count"))
    stats["prop_click_rate_unbiased"] = smooth(
        stats["cl"], stats["n"], prior_click, SMOOTH_K_UNBIASED).astype("float32")
    stats["prop_booking_rate_unbiased"] = smooth(
        stats["bk"], stats["n"], prior_book, SMOOTH_K_UNBIASED).astype("float32")
    return stats[list(UNBIASED_COLUMNS)].copy()


def add_unbiased_rates(
    train_fe: pd.DataFrame,
    test_fe: pd.DataFrame,
    labels: pd.DataFrame,
    n_splits: int = N_OOF_FOLDS,
) -> None:
    """Add the two OOF unbiased rate columns to ``train_fe`` / ``test_fe`` (in place).

    Parameters
    ----------
    train_fe
        V3 train features, one row per raw train row (same order as ``labels``).
    test_fe
        V3 test features (needs ``prop_id``).
    labels
        Raw train ``srch_id``, ``prop_id``, ``click_bool``, ``booking_bool``,
        ``random_bool`` columns, aligned row-for-row with ``train_fe``.
    """
    if len(labels) != len(train_fe):
        raise ValueError(f"Row count mismatch: labels={len(labels)}, train_fe={len(train_fe)}")

    random_rows = labels[labels["random_bool"] == 1]
    global_click = float(random_rows["click_bool"].mean())
    global_book = float(random_rows["booking_bool"].mean())
    fills = dict(zip(UNBIASED_COLUMNS, (global_click, global_book)))

    for col in UNBIASED_COLUMNS:
        train_fe[col] = np.float32(np.nan)

    fold_tables = []
    for fold, (fit_idx, held_idx) in enumerate(oof_folds(labels["srch_id"].to_numpy(), n_splits)):
        table = _fit_unbiased_table(labels.iloc[fit_idx])
        held_props = labels["prop_id"].iloc[held_idx]
        for col in UNBIASED_COLUMNS:
            train_fe.loc[train_fe.index[held_idx], col] = held_props.map(table[col]).values
        fold_tables.append(table)
        gc.collect()
        logger.info("Unbiased OOF fold %d/%d done", fold + 1, n_splits)

    for col in UNBIASED_COLUMNS:
        train_fe[col] = train_fe[col].fillna(np.float32(fills[col]))

    avg_table = pd.concat(fold_tables).groupby(level=0).mean()
    for col in UNBIASED_COLUMNS:
        test_fe[col] = test_fe["prop_id"].map(avg_table[col]).values
    for col in UNBIASED_COLUMNS:
        test_fe[col] = test_fe[col].fillna(np.float32(fills[col])).astype("float32")


def _add_bias_features(df: pd.DataFrame) -> None:
    """Bias magnitude and within-search views of the unbiased rates (in place)."""
    # biased minus unbiased, i.e. how much position bias inflates this property's rates
    df["click_rate_bias"] = (df["prop_click_rate"] - df["prop_click_rate_unbiased"]).astype("float32")
    df["booking_rate_bias"] = (df["prop_booking_rate"] - df["prop_booking_rate_unbiased"]).astype("float32")

    grp = df.groupby("srch_id", sort=False)
    df["unbiased_click_rate_rank"] = grp["prop_click_rate_unbiased"].rank(
        ascending=False, method="min").astype("float32")
    df["unbiased_booking_rate_rank"] = grp["prop_booking_rate_unbiased"].rank(
        ascending=False, method="min").astype("float32")
    mu = grp["prop_booking_rate_unbiased"].transform("mean").astype("float32")
    df["unbiased_booking_vs_search_mean"] = (df["prop_booking_rate_unbiased"] - mu).astype("float32")


def fix_missing_history_price(df: pd.DataFrame) -> None:
    """Neutralise ``price_vs_hist`` / ``price_discount`` when there is no historical price.

    ``exp(prop_log_historical_price=0) = 1`` turned both into garbage (~200 / ~-200)
    for ~14% of rows. They are set to the neutral values 1.0 and 0.0 and the
    dependent within-search rank is recomputed (in place).
    """
    no_history = df["no_hist_price"] == 1
    df.loc[no_history, "price_vs_hist"] = np.float32(1.0)
    df.loc[no_history, "price_discount"] = np.float32(0.0)
    grp = df.groupby("srch_id", sort=False)
    df["price_vs_hist_rank"] = grp["price_vs_hist"].rank(ascending=True, method="min").astype("float32")


def add_unbiased_features(
    train_fe: pd.DataFrame,
    test_fe: pd.DataFrame,
    labels: pd.DataFrame,
    n_splits: int = N_OOF_FOLDS,
) -> None:
    """Upgrade v3 feature tables to v4 in place (adds 7 features, repairs 2)."""
    add_unbiased_rates(train_fe, test_fe, labels, n_splits)
    for df in (train_fe, test_fe):
        _add_bias_features(df)
    for df in (train_fe, test_fe):
        fix_missing_history_price(df)
