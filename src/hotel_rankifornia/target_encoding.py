"""Out-of-fold (OOF) Bayesian target encoding without leakage.

Historical click / booking rates per property, destination, property x month,
... are the strongest features in the model, but they are computed from the
labels, so they must be built carefully.

* **Train rows** are encoded with lookup tables fitted on the *other* folds of
  a 5-fold ``GroupKFold`` (grouped by ``srch_id`` so a search never straddles
  train/validation).
* **Priors are per fold.** The Bayesian smoothing prior (global click rate,
  booking rate, ...) is computed from the training part of each fold only.
  Using a prior computed on all data would leak the validation labels into the
  encoding of those same rows.
* **Test rows** are encoded with the *average of the five fold tables*, so test
  features have the same noise level as the OOF train features (a table fitted
  on all data would be less noisy than what the model was trained on).

This is the "fixed" encoding of the project. The original first-pass version
(global priors, full-data test tables) is intentionally not reproduced.

The smoothed rate is ``(count + k * prior) / (total + k)``.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from hotel_rankifornia.config import N_OOF_FOLDS

logger = logging.getLogger(__name__)

AVG_POSITION_PRIOR_WEIGHT = 10  # pseudo-count pulling rare properties to the mean position


@dataclass(frozen=True)
class RateSpec:
    """One smoothed rate feature.

    ``kind`` selects the numerator, denominator and prior.

    * ``"click"`` uses clicks / impressions with the global click rate as prior
    * ``"book"`` uses bookings / impressions with the global booking rate as prior
    * ``"cvr"`` uses bookings / (clicks + 1) with the global booking-per-click rate as prior
    """

    column: str
    keys: tuple[str, ...]
    kind: str
    k: float


# Columns are listed in the order in which they appear in the *train* feature
# table of the published pipeline.
RATE_SPECS: tuple[RateSpec, ...] = (
    RateSpec("prop_click_rate", ("prop_id",), "click", 100),
    RateSpec("prop_booking_rate", ("prop_id",), "book", 100),
    RateSpec("dest_booking_rate", ("srch_destination_id",), "book", 100),
    RateSpec("prop_month_click_rate", ("prop_id", "_month"), "click", 20),
    RateSpec("prop_dest_booking_rate", ("prop_id", "srch_destination_id"), "book", 10),
    RateSpec("prop_vcountry_booking_rate", ("prop_id", "visitor_location_country_id"), "book", 50),
    RateSpec("prop_site_booking_rate", ("prop_id", "site_id"), "book", 50),
    RateSpec("prop_cvr", ("prop_id",), "cvr", 20),
    RateSpec("site_booking_rate", ("site_id",), "book", 200),
    RateSpec("prop_country_booking_rate", ("prop_country_id",), "book", 500),
)
AVG_POSITION_COLUMN = "prop_avg_position"
AVG_POSITION_KEYS = ("prop_id",)

# Column order of the encoded features in the train / test tables. The two
# differ (``prop_cvr``) for historical reasons. Both are kept so that the saved
# feature tables stay column-for-column identical to the published run, which
# matters because LightGBM's feature sampling depends on feature order.
TRAIN_COLUMN_ORDER: tuple[str, ...] = tuple(s.column for s in RATE_SPECS) + (AVG_POSITION_COLUMN,)
TEST_COLUMN_ORDER: tuple[str, ...] = (
    "prop_click_rate", "prop_booking_rate", "prop_cvr", "dest_booking_rate",
    "prop_month_click_rate", "prop_dest_booking_rate", "prop_vcountry_booking_rate",
    "prop_site_booking_rate", "site_booking_rate", "prop_country_booking_rate",
    AVG_POSITION_COLUMN,
)

# Columns the encoder reads (labels are only needed on the train side).
KEY_COLUMNS: tuple[str, ...] = (
    "prop_id", "srch_destination_id", "visitor_location_country_id",
    "site_id", "prop_country_id", "_month",
)
LABEL_COLUMNS: tuple[str, ...] = ("srch_id", "click_bool", "booking_bool", "position", "random_bool")


def month_of(date_time: pd.Series) -> pd.Series:
    """Calendar month of a ``date_time`` string column as ``int8``."""
    return pd.to_datetime(date_time).dt.month.astype("int8")


def smooth(count, total, prior: float, k: float):
    """Bayesian-smoothed rate that shrinks ``count / total`` towards ``prior`` with weight ``k``."""
    return (count + k * prior) / (total + k)


def oof_folds(groups, n_splits: int = N_OOF_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic ``GroupKFold`` splits (no shuffling), grouped by ``srch_id``."""
    dummy = np.zeros(len(groups))
    return list(GroupKFold(n_splits=n_splits).split(dummy, groups=groups))


@dataclass(frozen=True)
class Priors:
    click: float
    book: float
    cvr: float
    avg_position: float

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Priors":
        clicks = df["click_bool"].sum()
        return cls(
            click=float(df["click_bool"].mean()),
            book=float(df["booking_bool"].mean()),
            cvr=float(df["booking_bool"].sum() / max(clicks, 1)),
            avg_position=float(df.loc[df["random_bool"] == 0, "position"].mean()),
        )

    def for_kind(self, kind: str) -> float:
        return {"click": self.click, "book": self.book, "cvr": self.cvr}[kind]


def _fit_tables(fold_train: pd.DataFrame, priors: Priors) -> dict[str, pd.Series]:
    """Fit every smoothed lookup table on ``fold_train`` using ``priors``."""
    tables: dict[str, pd.Series] = {}
    stats_by_keys: dict[tuple[str, ...], pd.DataFrame] = {}

    for spec in RATE_SPECS:
        if spec.keys not in stats_by_keys:
            stats_by_keys[spec.keys] = fold_train.groupby(list(spec.keys)).agg(
                cl=("click_bool", "sum"), bk=("booking_bool", "sum"), n=("booking_bool", "count"))
        stats = stats_by_keys[spec.keys]
        prior = priors.for_kind(spec.kind)
        if spec.kind == "click":
            rate = smooth(stats["cl"], stats["n"], prior, spec.k)
        elif spec.kind == "book":
            rate = smooth(stats["bk"], stats["n"], prior, spec.k)
        else:  # "cvr"
            rate = smooth(stats["bk"], stats["cl"] + 1, prior, spec.k)
        tables[spec.column] = rate.astype("float32").rename(spec.column)

    # Expedia's own ranking, the mean result position over searches where the list was
    # ordered by Expedia (random_bool == 0), smoothed towards the global mean position.
    expedia_sorted = fold_train[fold_train["random_bool"] == 0]
    pos = expedia_sorted.groupby(list(AVG_POSITION_KEYS))["position"].agg(avg_pos="mean", n_pos="count")
    avg_pos = ((pos["avg_pos"] * pos["n_pos"] + priors.avg_position * AVG_POSITION_PRIOR_WEIGHT)
               / (pos["n_pos"] + AVG_POSITION_PRIOR_WEIGHT)).astype("float32")
    tables[AVG_POSITION_COLUMN] = avg_pos.rename(AVG_POSITION_COLUMN)
    return tables


def _keys_for(column: str) -> tuple[str, ...]:
    if column == AVG_POSITION_COLUMN:
        return AVG_POSITION_KEYS
    return next(s.keys for s in RATE_SPECS if s.column == column)


def _lookup(rows: pd.DataFrame, keys: tuple[str, ...], table: pd.Series) -> np.ndarray:
    """Look up ``table`` for every row of ``rows`` (NaN where the key is unseen)."""
    if len(keys) == 1:
        return rows[keys[0]].map(table).to_numpy()
    row_keys = rows[list(keys)].reset_index(drop=True)
    merged = row_keys.merge(table.reset_index(), on=list(keys), how="left")
    return merged[table.name].to_numpy()


def _average_tables(fold_tables: list[dict[str, pd.Series]], column: str) -> pd.Series:
    """Mean of one lookup table across folds (keys missing from a fold are ignored)."""
    stacked = pd.concat([tables[column] for tables in fold_tables])
    n_levels = stacked.index.nlevels
    return stacked.groupby(level=0 if n_levels == 1 else list(range(n_levels))).mean()


def compute_oof_encodings(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_splits: int = N_OOF_FOLDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leak-free target encodings for the train and test tables.

    Parameters
    ----------
    train
        Needs ``srch_id``, the label columns (``click_bool``,
        ``booking_bool``, ``position``, ``random_bool``) and the key columns
        (``prop_id``, ``srch_destination_id``, ``visitor_location_country_id``,
        ``site_id``, ``prop_country_id``, ``_month``). Row order defines the
        output row order.
    test
        Same key columns (no labels needed).
    n_splits
        Number of ``GroupKFold`` folds.

    Returns
    -------
        ``(train_enc, test_enc)`` as float32 frames (RangeIndex, same row order as the
        inputs) with columns in ``TRAIN_COLUMN_ORDER`` / ``TEST_COLUMN_ORDER``.
        Keys never seen in a training fold fall back to the global rate of the
        full training set.
    """
    n_rows = len(train)
    columns = list(TRAIN_COLUMN_ORDER)
    train_out = {col: np.full(n_rows, np.nan, dtype=np.float32) for col in columns}
    fold_tables: list[dict[str, pd.Series]] = []

    # Global priors are only used to fill keys that no training fold has seen.
    global_priors = Priors.from_frame(train)

    for fold, (fit_idx, held_idx) in enumerate(oof_folds(train["srch_id"].to_numpy(), n_splits)):
        fold_train = train.iloc[fit_idx]
        held_out = train.iloc[held_idx]
        tables = _fit_tables(fold_train, Priors.from_frame(fold_train))  # per-fold priors
        for col in columns:
            train_out[col][held_idx] = _lookup(held_out, _keys_for(col), tables[col])
        fold_tables.append(tables)
        del fold_train, held_out
        gc.collect()
        logger.info("OOF encoding fold %d/%d done (%s held-out rows)", fold + 1, n_splits, f"{len(held_idx):,}")

    fills = _fill_values(global_priors)
    train_enc = pd.DataFrame(
        {col: np.where(np.isnan(train_out[col]), np.float32(fills[col]), train_out[col])
         for col in TRAIN_COLUMN_ORDER})

    test_out = {}
    for col in TEST_COLUMN_ORDER:
        avg_table = _average_tables(fold_tables, col)
        values = _lookup(test, _keys_for(col), avg_table)
        test_out[col] = np.where(np.isnan(values), np.float32(fills[col]), values).astype("float32")
    test_enc = pd.DataFrame(test_out)
    return train_enc.astype("float32"), test_enc


def _fill_values(priors: Priors) -> dict[str, float]:
    """Fallback value per encoded column for keys unseen in every fold."""
    fills = {spec.column: priors.for_kind(spec.kind) for spec in RATE_SPECS}
    fills[AVG_POSITION_COLUMN] = priors.avg_position
    return fills
