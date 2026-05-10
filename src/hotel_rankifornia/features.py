"""Base feature engineering (raw CSV -> the 139-feature "v3" table).

The pipeline has three parts, run in this order by ``build_base_features``.

1. leak-free OOF target encodings (``hotel_rankifornia.target_encoding``)
2. label-free lookup tables fitted on the training data (destination price and
   quality statistics, popularity counts, imputation constants)
3. ``engineer``, a per-row / per-search transformation that is applied
   identically to train and test.

Column order and dtypes are part of the contract. The model receives features
in ``feature_cols`` order and LightGBM's feature sub-sampling depends on it.

Data contract of the output tables
    * one row per (search, property) impression, same row order as the raw CSV
    * ``srch_id`` / ``prop_id`` identifiers (not features)
    * train additionally has ``relevance`` in {0, 1, 5} (last column)
    * every other column is a numeric model feature.
"""

from __future__ import annotations

import gc
import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hotel_rankifornia.config import RELEVANCE_BOOKED, RELEVANCE_CLICKED
from hotel_rankifornia.data import COMP_INV, COMP_PCT, COMP_RATE
from hotel_rankifornia.target_encoding import (
    KEY_COLUMNS,
    LABEL_COLUMNS,
    TEST_COLUMN_ORDER,
    TRAIN_COLUMN_ORDER,
    compute_oof_encodings,
    month_of,
)

logger = logging.getLogger(__name__)

NON_FEATURE_COLUMNS = {"srch_id", "prop_id", "relevance"}
LABEL_ONLY_COLUMNS = ["click_bool", "booking_bool", "gross_bookings_usd", "position", "relevance"]

# Encoded columns that ``engineer`` expects to find on its input.
_REQUIRED_INPUT_COLUMNS = list(TRAIN_COLUMN_ORDER) + [
    "dest_price_mean", "dest_price_median", "dest_price_std",
    "dest_avg_starrating", "dest_avg_review", "prop_srch_count", "dest_n_props",
]

EPS = 1e-6  # guards divisions


# ---------------------------------------------------------------------------
# Label-free lookup tables
# ---------------------------------------------------------------------------
@dataclass
class Lookups:
    """Statistics fitted on the training table and applied to train and test.

    None of these use labels, so fitting them on all training rows is safe.
    """

    dest_price: pd.DataFrame
    global_price_mean: np.float32
    global_price_median: np.float32
    global_price_std: np.float32
    dest_quality: pd.DataFrame
    global_star: np.float32
    global_review: np.float32
    prop_srch_count: pd.DataFrame
    dest_n_props: pd.DataFrame
    affinity_min: float
    dist_median_by_dest: pd.Series
    dist_global_median: float


def fit_lookups(train: pd.DataFrame) -> Lookups:
    """Fit destination / property statistics and imputation constants on ``train``."""
    dest_price = train.groupby("srch_destination_id", sort=False)["price_usd"].agg(
        dest_price_mean="mean", dest_price_median="median", dest_price_std="std"
    ).reset_index()
    dest_price["dest_price_mean"] = dest_price["dest_price_mean"].astype("float32")
    dest_price["dest_price_median"] = dest_price["dest_price_median"].astype("float32")
    dest_price["dest_price_std"] = dest_price["dest_price_std"].fillna(0).astype("float32")

    dest_quality = train.groupby("srch_destination_id", sort=False).agg(
        dest_avg_starrating=("prop_starrating", "mean"),
        dest_avg_review=("prop_review_score", "mean"),
    ).reset_index()
    dest_quality["dest_avg_starrating"] = dest_quality["dest_avg_starrating"].astype("float32")
    dest_quality["dest_avg_review"] = dest_quality["dest_avg_review"].astype("float32")

    prop_srch_count = train.groupby("prop_id")["srch_id"].nunique().reset_index()
    prop_srch_count.columns = ["prop_id", "prop_srch_count"]
    prop_srch_count["prop_srch_count"] = prop_srch_count["prop_srch_count"].astype("int32")

    dest_n_props = train.groupby("srch_destination_id")["prop_id"].nunique().reset_index()
    dest_n_props.columns = ["srch_destination_id", "dest_n_props"]
    dest_n_props["dest_n_props"] = dest_n_props["dest_n_props"].astype("int32")

    return Lookups(
        dest_price=dest_price,
        global_price_mean=np.float32(dest_price["dest_price_mean"].mean()),
        global_price_median=np.float32(dest_price["dest_price_median"].median()),
        global_price_std=np.float32(dest_price["dest_price_std"].mean()),
        dest_quality=dest_quality,
        global_star=np.float32(dest_quality["dest_avg_starrating"].mean()),
        global_review=np.float32(dest_quality["dest_avg_review"].mean()),
        prop_srch_count=prop_srch_count,
        dest_n_props=dest_n_props,
        affinity_min=float(train["srch_query_affinity_score"].min()),
        dist_median_by_dest=train.groupby("srch_destination_id")["orig_destination_distance"]
        .median().astype("float32"),
        dist_global_median=float(train["orig_destination_distance"].median()),
    )


def attach_lookups(df: pd.DataFrame, lk: Lookups) -> pd.DataFrame:
    """Join the fitted lookup tables onto ``df`` (row order is preserved)."""
    df = df.merge(lk.dest_price, on="srch_destination_id", how="left")
    df["dest_price_mean"] = df["dest_price_mean"].fillna(lk.global_price_mean)
    df["dest_price_median"] = df["dest_price_median"].fillna(lk.global_price_median)
    df["dest_price_std"] = df["dest_price_std"].fillna(lk.global_price_std)

    tmp = df[["srch_destination_id"]].merge(lk.dest_quality, on="srch_destination_id", how="left")
    df["dest_avg_starrating"] = tmp["dest_avg_starrating"].fillna(lk.global_star).values
    df["dest_avg_review"] = tmp["dest_avg_review"].fillna(lk.global_review).values

    tmp1 = df[["prop_id"]].merge(lk.prop_srch_count, on="prop_id", how="left")
    df["prop_srch_count"] = tmp1["prop_srch_count"].fillna(1).values.astype("int32")
    tmp2 = df[["srch_destination_id"]].merge(lk.dest_n_props, on="srch_destination_id", how="left")
    df["dest_n_props"] = tmp2["dest_n_props"].fillna(1).values.astype("int32")
    return df


def relevance_labels(train: pd.DataFrame) -> np.ndarray:
    """Graded relevance, 5 = booked, 1 = clicked only, 0 = neither (``int8``)."""
    return np.where(
        train["booking_bool"] == 1, RELEVANCE_BOOKED,
        np.where(train["click_bool"] == 1, RELEVANCE_CLICKED, 0),
    ).astype("int8")


# ---------------------------------------------------------------------------
# Row / search level feature engineering
# ---------------------------------------------------------------------------
def _add_datetime_features(df: pd.DataFrame) -> None:
    dt = pd.to_datetime(df["date_time"])
    df["month"] = dt.dt.month.astype("int8")
    df["day_of_week"] = dt.dt.dayofweek.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["is_weekend_search"] = (df["day_of_week"] >= 5).astype("int8")
    df["quarter"] = dt.dt.quarter.astype("int8")
    df.drop(columns=["date_time"], inplace=True)


def _add_missing_flags_and_impute(df: pd.DataFrame, lk: Lookups) -> None:
    df["has_visitor_history"] = df["visitor_hist_starrating"].notna().astype("int8")
    df["visitor_hist_starrating"] = df["visitor_hist_starrating"].fillna(0)
    df["visitor_hist_adr_usd"] = df["visitor_hist_adr_usd"].fillna(0)
    df["review_score_missing"] = df["prop_review_score"].isna().astype("int8")
    df["prop_review_score"] = df["prop_review_score"].fillna(0)
    df["loc_score2_missing"] = df["prop_location_score2"].isna().astype("int8")
    df["prop_location_score2"] = df["prop_location_score2"].fillna(0)
    df["dist_missing"] = df["orig_destination_distance"].isna().astype("int8")
    dist_fill = df["srch_destination_id"].map(lk.dist_median_by_dest).astype("float32")
    df["orig_destination_distance"] = (
        df["orig_destination_distance"].fillna(dist_fill).fillna(lk.dist_global_median).astype("float32"))
    df["affinity_missing"] = df["srch_query_affinity_score"].isna().astype("int8")
    df["srch_query_affinity_score"] = df["srch_query_affinity_score"].fillna(lk.affinity_min)
    df["no_hist_price"] = (df["prop_log_historical_price"] == 0).astype("int8")


def _add_competitor_features(df: pd.DataFrame) -> None:
    df[COMP_PCT] = df[COMP_PCT].fillna(0).astype("float32")
    rate = df[COMP_RATE].values.astype("float32")
    inv = df[COMP_INV].values
    pct = df[COMP_PCT].values

    # Rows without any competitor data are all-NaN, so nanmean/nanmax return NaN (filled with 0
    # below) and warn, which is expected.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rate_masked = np.where(rate == 0, np.nan, rate)
        rate_mean = np.nanmean(rate_masked, axis=1)
        pct_masked = np.where(pct == 0, np.nan, pct)
        pct_mean = np.nanmean(pct_masked, axis=1)
        pct_cheaper = np.where(rate == 1, pct, np.nan)
        max_pct_advantage = np.nanmax(pct_cheaper, axis=1)

    df["comp_rate_mean"] = rate_mean.astype("float32")
    df["comp_rate_mean"] = df["comp_rate_mean"].fillna(0).astype("float32")
    df["comp_cheaper_count"] = (rate == 1).sum(axis=1).astype("int8")
    df["comp_expensive_count"] = (rate == -1).sum(axis=1).astype("int8")
    df["comp_inv_advantage"] = (inv == 1).sum(axis=1).astype("int8")
    df["comp_data_coverage"] = (rate != 0).sum(axis=1).astype("int8")
    df["comp_pct_diff_mean"] = pct_mean.astype("float32")
    df["comp_pct_diff_mean"] = df["comp_pct_diff_mean"].fillna(0).astype("float32")

    # directional advantage features
    df["comp_max_pct_advantage"] = max_pct_advantage.astype("float32")
    df["comp_max_pct_advantage"] = df["comp_max_pct_advantage"].fillna(0).astype("float32")
    df["comp_dual_advantage"] = ((rate == 1) & (inv == 1)).sum(axis=1).astype("int8")
    df["comp_net_advantage"] = (df["comp_cheaper_count"] - df["comp_expensive_count"]).astype("int8")
    df["comp_weighted_advantage"] = (df["comp_cheaper_count"] * df["comp_pct_diff_mean"]).astype("float32")

    df.drop(columns=COMP_RATE + COMP_INV + COMP_PCT, inplace=True)


def _add_price_features(df: pd.DataFrame) -> None:
    hist = np.exp(df["prop_log_historical_price"].values).astype("float32")
    df["price_vs_hist"] = (df["price_usd"] / (hist + EPS)).astype("float32")
    df["price_discount"] = (hist - df["price_usd"]).astype("float32")
    df["log_price_usd"] = np.log1p(df["price_usd"]).astype("float32")
    df["price_per_night"] = (df["price_usd"] / (df["srch_length_of_stay"] + EPS)).astype("float32")
    df["log_booking_window"] = np.log1p(df["srch_booking_window"]).astype("float32")

    # fractional discount + market positioning
    df["prop_discount_rate"] = np.where(
        df["no_hist_price"] == 1, np.float32(0),
        ((hist - df["price_usd"].values) / (hist + EPS)).clip(-1, 1).astype("float32"))
    df["is_big_discount"] = (df["prop_discount_rate"] > 0.15).astype("int8")
    df["hist_price_to_dest_ratio"] = np.where(
        df["no_hist_price"] == 1, np.float32(1.0),
        (hist / (df["dest_price_median"].values + EPS)).astype("float32"))

    df["price_per_room_per_night"] = (df["price_usd"] / (
        df["srch_room_count"].clip(lower=1) * df["srch_length_of_stay"].clip(lower=1) + EPS)).astype("float32")


def _add_search_context_features(df: pd.DataFrame) -> None:
    df["total_travelers"] = (df["srch_adults_count"] + df["srch_children_count"]).astype("int8")
    df["is_family"] = (df["srch_children_count"] > 0).astype("int8")
    df["is_business"] = ((df["srch_children_count"] == 0) & (df["srch_adults_count"] <= 2)).astype("int8")
    df["is_long_stay"] = (df["srch_length_of_stay"] > 7).astype("int8")
    df["is_last_minute"] = (df["srch_booking_window"] <= 3).astype("int8")
    df["is_domestic"] = (df["visitor_location_country_id"] == df["prop_country_id"]).astype("int8")
    df["price_per_person"] = (df["price_usd"] / (df["total_travelers"] + EPS)).astype("float32")
    df["star_diff_from_hist"] = (df["prop_starrating"] - df["visitor_hist_starrating"]).abs().astype("float32")

    # visitor-history directional features
    has_hist = df["has_visitor_history"] == 1
    df["star_upgrade"] = np.int8(0)
    df["star_downgrade"] = np.int8(0)
    df.loc[has_hist, "star_upgrade"] = (
        df.loc[has_hist, "prop_starrating"] > df.loc[has_hist, "visitor_hist_starrating"]).astype("int8")
    df.loc[has_hist, "star_downgrade"] = (
        df.loc[has_hist, "prop_starrating"] < df.loc[has_hist, "visitor_hist_starrating"]).astype("int8")
    df["price_above_hist"] = np.float32(0)
    df.loc[has_hist, "price_above_hist"] = (
        df.loc[has_hist, "price_usd"] > df.loc[has_hist, "visitor_hist_adr_usd"]).astype("float32")

    # booking-window interactions
    df["booking_window_x_length"] = (df["srch_booking_window"] * df["srch_length_of_stay"]).astype("float32")
    df["price_per_night_x_window"] = (df["price_per_night"] * df["log_booking_window"]).astype("float32")


def _add_quality_and_interaction_features(df: pd.DataFrame) -> None:
    df["quality_score"] = (0.4 * df["prop_review_score"] + 0.3 * df["prop_starrating"]
                           + 0.2 * df["prop_location_score1"] + 0.1 * df["prop_location_score2"]).astype("float32")
    df["star_x_review"] = (df["prop_starrating"] * df["prop_review_score"]).astype("float32")
    df["review_per_dollar"] = (df["prop_review_score"] / (df["price_usd"] + EPS)).astype("float32")

    # star / review vs destination average
    df["star_vs_dest_mean"] = (df["prop_starrating"] - df["dest_avg_starrating"]).astype("float32")
    df["review_vs_dest_mean"] = (df["prop_review_score"] - df["dest_avg_review"]).astype("float32")

    # destination price features
    df["price_vs_dest_median"] = (df["price_usd"] - df["dest_price_median"]).astype("float32")
    df["price_dest_zscore"] = ((df["price_usd"] - df["dest_price_mean"]) / (df["dest_price_std"] + EPS)).astype("float32")

    # user-hotel affinity
    df["price_affinity"] = (df["price_usd"] / (df["visitor_hist_adr_usd"] + EPS)).clip(0, 10).astype("float32")
    df.loc[df["has_visitor_history"] == 0, "price_affinity"] = np.float32(1.0)
    df["price_hist_abs_diff"] = (df["price_usd"] - df["visitor_hist_adr_usd"]).abs().astype("float32")
    df["star_match_close"] = (df["star_diff_from_hist"] <= 1).astype("int8")

    # interactions
    df["brand_x_star"] = (df["prop_brand_bool"] * df["prop_starrating"]).astype("float32")
    df["loc1_x_booking_rate"] = (df["prop_location_score1"] * df["prop_booking_rate"]).astype("float32")
    df["promotion_x_price"] = (df["promotion_flag"] * df["price_usd"]).astype("float32")

    # property count features (log-transformed)
    df["log_prop_srch_count"] = np.log1p(df["prop_srch_count"]).astype("float32")
    df["log_dest_n_props"] = np.log1p(df["dest_n_props"]).astype("float32")


# (source column, ascending, output name) for within-search ranks
_WITHIN_SEARCH_RANKS = [
    ("price_usd", True, "price_rank"), ("prop_review_score", False, "review_rank"),
    ("prop_starrating", False, "star_rank"), ("prop_location_score1", False, "loc_score1_rank"),
    ("quality_score", False, "quality_score_rank"),
    ("prop_booking_rate", False, "prop_booking_rate_rank"),
    ("prop_click_rate", False, "prop_click_rate_rank"),
    ("orig_destination_distance", True, "dist_rank"),
    ("srch_query_affinity_score", False, "affinity_rank"),
    ("prop_month_click_rate", False, "prop_month_click_rate_rank"),
    ("prop_dest_booking_rate", False, "prop_dest_booking_rate_rank"),
    ("prop_location_score2", False, "loc_score2_rank"),
    ("prop_vcountry_booking_rate", False, "prop_vcountry_rate_rank"),
    ("prop_site_booking_rate", False, "prop_site_rate_rank"),
    ("prop_cvr", False, "prop_cvr_rank"),
    ("prop_avg_position", True, "prop_avg_position_rank"),  # lower position = better
]

# (source column, output name) pairs. Deviation from the search mean keeps magnitude, not just rank
_WITHIN_SEARCH_DEVIATIONS = [
    ("prop_location_score2", "loc_score2_vs_search_mean"),
    ("prop_location_score1", "loc_score1_vs_search_mean"),
    ("quality_score", "quality_vs_search_mean"),
    ("prop_booking_rate", "booking_rate_vs_search_mean"),
    ("prop_dest_booking_rate", "dest_booking_rate_vs_search_mean"),
    ("prop_avg_position", "avg_position_vs_search_mean"),
]

# (source column, ascending, output name) for reciprocal-rank features
_RECIPROCAL_RANKS = [
    ("prop_booking_rate", False, "rr_by_booking_rate"),
    ("quality_score", False, "rr_by_quality"),
    ("price_usd", True, "rr_by_price"),
    ("prop_avg_position", True, "rr_by_expedia_rank"),
]


def _add_within_search_features(df: pd.DataFrame) -> None:
    grp = df.groupby("srch_id", sort=False)

    for col, ascending, name in _WITHIN_SEARCH_RANKS:
        df[name] = grp[col].rank(ascending=ascending, method="min").astype("float32")

    pmean = grp["price_usd"].transform("mean").astype("float32")
    pstd = grp["price_usd"].transform("std").astype("float32")
    df["price_norm"] = ((df["price_usd"] - pmean) / (pstd + EPS)).astype("float32")
    df["price_ratio"] = (df["price_usd"] / (pmean + EPS)).astype("float32")

    df["n_props_in_search"] = grp["prop_id"].transform("count").astype("int16")

    df["price_percentile"] = grp["price_usd"].rank(pct=True).astype("float32")
    pmed = grp["price_usd"].transform("median").astype("float32")
    df["price_vs_median"] = (df["price_usd"] - pmed).astype("float32")
    df["price_median_ratio"] = (df["price_usd"] / (pmed + EPS)).astype("float32")

    df["price_per_star"] = (df["price_usd"] / (df["prop_starrating"] + 0.5)).astype("float32")
    grp2 = df.groupby("srch_id", sort=False)
    df["value_rank"] = grp2["price_per_star"].rank(ascending=True, method="min").astype("float32")

    rmean = grp2["prop_review_score"].transform("mean").astype("float32")
    df["review_vs_search_mean"] = (df["prop_review_score"] - rmean).astype("float32")

    df["price_vs_hist_rank"] = grp2["price_vs_hist"].rank(ascending=True, method="min").astype("float32")

    for base_col, new_col in _WITHIN_SEARCH_DEVIATIONS:
        mu = grp2[base_col].transform("mean").astype("float32")
        df[new_col] = (df[base_col] - mu).astype("float32")

    # search price elasticity
    df["srch_price_range"] = (grp2["price_usd"].transform("max") - grp2["price_usd"].transform("min")).astype("float32")
    sp_mean = grp2["price_usd"].transform("mean").astype("float32")
    sp_std = grp2["price_usd"].transform("std").astype("float32")
    df["srch_price_cv"] = (sp_std / (sp_mean + EPS)).astype("float32")
    df["price_in_bottom_quartile"] = (df["price_percentile"] <= 0.25).astype("int8")

    for col, ascending, name in _RECIPROCAL_RANKS:
        ranks = grp2[col].rank(ascending=ascending, method="first").astype("float32")
        df[name] = (1.0 / ranks).astype("float32")


def engineer(df: pd.DataFrame, lk: Lookups) -> pd.DataFrame:
    """Apply all feature engineering to a table that already has the encoded columns.

    Label columns must be removed beforehand. Returns a defragmented copy of ``df``
    (about a hundred columns are added one by one, which fragments the frame).
    """
    missing = [c for c in _REQUIRED_INPUT_COLUMNS if c not in df.columns]
    assert not missing, f"Missing: {missing}"

    with warnings.catch_warnings():
        # Columns are added one at a time and the frame is defragmented by the final copy.
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        _add_datetime_features(df)
        _add_missing_flags_and_impute(df, lk)
        _add_competitor_features(df)
        gc.collect()
        _add_price_features(df)
        _add_search_context_features(df)
        _add_quality_and_interaction_features(df)
        _add_within_search_features(df)
        return df.copy()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def add_null_count(df: pd.DataFrame) -> None:
    """Meta-feature counting the missing-value flags set for the row (in place)."""
    null_flags = ["review_score_missing", "loc_score2_missing", "dist_missing",
                  "affinity_missing", "no_hist_price"]
    # has_visitor_history == 1 means "not null", so it is inverted
    df["count_null"] = ((1 - df["has_visitor_history"]).astype("int8")
                        + df[null_flags].sum(axis=1).astype("int8"))


def select_feature_cols(train_fe: pd.DataFrame, test_fe: pd.DataFrame) -> list[str]:
    """Model feature columns (test-table order), checking that train and test agree."""
    feature_cols = [c for c in test_fe.columns if c not in NON_FEATURE_COLUMNS]
    train_feat = {c for c in train_fe.columns if c not in NON_FEATURE_COLUMNS}
    assert train_feat == set(feature_cols), (
        f"Feature mismatch! Only in train: {train_feat - set(feature_cols)}, "
        f"only in test: {set(feature_cols) - train_feat}")
    return feature_cols


def _encoding_inputs(df: pd.DataFrame, with_labels: bool) -> pd.DataFrame:
    cols = [c for c in KEY_COLUMNS if c != "_month"]
    if with_labels:
        cols = list(LABEL_COLUMNS) + cols
    inputs = df[cols].copy()
    inputs["_month"] = month_of(df["date_time"])
    return inputs


def build_base_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw train/test tables -> engineered feature tables (139 features).

    Parameters
    ----------
    train
        Raw training table as returned by ``hotel_rankifornia.data.load_train``.
    test
        Raw test table from ``hotel_rankifornia.data.load_test``.

    Returns
    -------
        ``(train_fe, test_fe)``. ``train_fe`` ends with the ``relevance`` label.
    """
    logger.info("Computing OOF target encodings")
    train_enc, test_enc = compute_oof_encodings(
        _encoding_inputs(train, with_labels=True), _encoding_inputs(test, with_labels=False))
    for col in TRAIN_COLUMN_ORDER:
        train[col] = train_enc[col].to_numpy()
    for col in TEST_COLUMN_ORDER:
        test[col] = test_enc[col].to_numpy()
    del train_enc, test_enc
    gc.collect()

    logger.info("Fitting lookup tables")
    lookups = fit_lookups(train)
    relevance = relevance_labels(train)
    train = attach_lookups(train, lookups)
    test = attach_lookups(test, lookups)

    logger.info("Engineering train features")
    train = train.drop(columns=LABEL_ONLY_COLUMNS, errors="ignore")
    train = engineer(train, lookups)
    train["relevance"] = relevance

    logger.info("Engineering test features")
    test = engineer(test, lookups)

    add_null_count(train)
    add_null_count(test)
    logger.info("Base features ready, train %s, test %s", train.shape, test.shape)
    return train, test
