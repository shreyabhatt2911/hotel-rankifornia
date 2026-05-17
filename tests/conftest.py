"""Shared synthetic data for the unit tests (tiny, deterministic, no files)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

N_SEARCHES = 60
N_PROPS = 6


@pytest.fixture
def impressions() -> pd.DataFrame:
    """60 searches x 6 properties, every property shown in every search.

    Every key that the encoders group by has a single value except ``prop_id`` and
    ``srch_id``, so each lookup key is present in every training fold and the global
    fallback values are never used (which keeps the leakage tests exact).
    """
    rng = np.random.default_rng(0)
    srch_id = np.repeat(np.arange(1, N_SEARCHES + 1), N_PROPS)
    n = len(srch_id)
    return pd.DataFrame({
        "srch_id": srch_id.astype("int32"),
        "prop_id": np.tile(np.arange(100, 100 + N_PROPS), N_SEARCHES).astype("int32"),
        "srch_destination_id": np.full(n, 7, dtype="int32"),
        "visitor_location_country_id": np.full(n, 1, dtype="int16"),
        "site_id": np.full(n, 5, dtype="int8"),
        "prop_country_id": np.full(n, 3, dtype="int16"),
        "_month": np.full(n, 6, dtype="int8"),
        "click_bool": rng.integers(0, 2, n).astype("int8"),
        "booking_bool": (rng.random(n) < 0.15).astype("int8"),
        # random_bool is constant within a search, as in the raw data
        "random_bool": np.repeat(rng.integers(0, 2, N_SEARCHES), N_PROPS).astype("int8"),
        "position": np.tile(np.arange(1, N_PROPS + 1), N_SEARCHES).astype("int8"),
    })
