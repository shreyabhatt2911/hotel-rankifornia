"""Loading the raw Expedia "Personalize Expedia Hotel Searches" CSV files."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

COMP_RATE = [f"comp{i}_rate" for i in range(1, 9)]
COMP_INV = [f"comp{i}_inv" for i in range(1, 9)]
COMP_PCT = [f"comp{i}_rate_percent_diff" for i in range(1, 9)]

# Memory-optimised dtypes (the raw CSVs are ~2.3 GB with default dtypes).
DTYPES_SHARED: dict[str, str] = {
    "srch_id": "int32", "site_id": "int8",
    "visitor_location_country_id": "int16", "prop_country_id": "int16",
    "prop_id": "int32", "srch_destination_id": "int32",
    "visitor_hist_starrating": "float32", "visitor_hist_adr_usd": "float32",
    "prop_starrating": "int8", "prop_review_score": "float32",
    "prop_brand_bool": "int8", "prop_location_score1": "float32",
    "prop_location_score2": "float32", "prop_log_historical_price": "float32",
    "price_usd": "float32", "promotion_flag": "int8",
    "srch_length_of_stay": "int8", "srch_booking_window": "int16",
    "srch_adults_count": "int8", "srch_children_count": "int8",
    "srch_room_count": "int8", "srch_saturday_night_bool": "int8",
    "srch_query_affinity_score": "float32",
    "orig_destination_distance": "float32", "random_bool": "int8",
}
for _i in range(1, 9):
    DTYPES_SHARED[f"comp{_i}_rate"] = "float32"
    DTYPES_SHARED[f"comp{_i}_inv"] = "float32"
    DTYPES_SHARED[f"comp{_i}_rate_percent_diff"] = "float32"

DTYPES_TRAIN: dict[str, str] = {
    **DTYPES_SHARED, "position": "int8", "click_bool": "int8",
    "booking_bool": "int8", "gross_bookings_usd": "float32",
}

# Accepted file names, tried in order (the competition ships ``training_set_*``
# and ``test_set_*``, plain ``train.csv`` / ``test.csv`` also work).
_RAW_FILE_PATTERNS = {
    "train": ("train.csv", "training_set*.csv"),
    "test": ("test.csv", "test_set*.csv"),
}


def find_raw_file(data_dir: Path, split: str) -> Path:
    """Locate the raw CSV for ``split`` (``"train"`` or ``"test"``) in ``data_dir``."""
    data_dir = Path(data_dir)
    for pattern in _RAW_FILE_PATTERNS[split]:
        matches = sorted(data_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No raw {split} CSV found in {data_dir} (looked for {_RAW_FILE_PATTERNS[split]})"
    )


def downcast_comp(df: pd.DataFrame) -> pd.DataFrame:
    """Competitor rate / availability columns are in {-1, 0, 1} (NaN -> 0) and are stored as int8."""
    cols = [f"comp{i}_{s}" for i in range(1, 9) for s in ("rate", "inv")]
    df[cols] = df[cols].fillna(0).astype("int8")
    return df


def load_train(data_dir: Path) -> pd.DataFrame:
    """Load the training CSV with compact dtypes, preserving row order."""
    path = find_raw_file(data_dir, "train")
    logger.info("Loading %s", path)
    train = downcast_comp(pd.read_csv(path, dtype=DTYPES_TRAIN))
    logger.info("Train %s, %.2f GB", train.shape, train.memory_usage(deep=True).sum() / 1e9)
    return train


def load_test(data_dir: Path) -> pd.DataFrame:
    """Load the test CSV with compact dtypes, preserving row order."""
    path = find_raw_file(data_dir, "test")
    logger.info("Loading %s", path)
    test = downcast_comp(pd.read_csv(path, dtype=DTYPES_SHARED))
    logger.info("Test %s, %.2f GB", test.shape, test.memory_usage(deep=True).sum() / 1e9)
    return test


def load_search_dates(data_dir: Path) -> pd.Series:
    """Timestamp of every training search (``srch_id`` -> ``datetime64``).

    Used for the temporal validation split. Reads only two columns.
    """
    path = find_raw_file(data_dir, "train")
    raw = pd.read_csv(path, usecols=["srch_id", "date_time"], dtype={"srch_id": "int32"})
    first = raw.groupby("srch_id")["date_time"].first()
    return pd.to_datetime(first)
