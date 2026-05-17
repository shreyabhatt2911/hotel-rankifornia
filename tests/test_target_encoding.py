import numpy as np
import pandas as pd

from hotel_rankifornia.target_encoding import (
    TEST_COLUMN_ORDER,
    TRAIN_COLUMN_ORDER,
    compute_oof_encodings,
    oof_folds,
    smooth,
)


def _encode(train: pd.DataFrame, test: pd.DataFrame):
    return compute_oof_encodings(train, test)


def test_smooth_shrinks_towards_prior():
    # no observations -> prior, many observations -> empirical rate
    assert smooth(0, 0, 0.3, k=100) == 0.3
    assert abs(smooth(500, 1000, 0.1, k=1) - 0.5) < 1e-2


def test_column_orders_cover_the_same_features(impressions):
    train_enc, test_enc = _encode(impressions, impressions)
    assert list(train_enc.columns) == list(TRAIN_COLUMN_ORDER)
    assert list(test_enc.columns) == list(TEST_COLUMN_ORDER)
    assert set(TRAIN_COLUMN_ORDER) == set(TEST_COLUMN_ORDER)
    assert (train_enc.dtypes == "float32").all() and (test_enc.dtypes == "float32").all()
    assert len(train_enc) == len(test_enc) == len(impressions)
    assert not train_enc.isna().any().any() and not test_enc.isna().any().any()


def test_oof_encoding_does_not_use_own_fold_labels(impressions):
    """Rows of a fold must be encoded without any information from that fold's labels.

    Rewriting every label of fold k (clicks, bookings and positions) must leave the
    encodings of fold k's own rows bit-for-bit unchanged, including the per-fold
    priors. It must change the encodings of the other folds, otherwise the test would
    prove nothing.
    """
    folds = oof_folds(impressions["srch_id"].to_numpy())
    held_idx = folds[0][1]
    baseline, _ = _encode(impressions, impressions)

    flipped = impressions.copy()
    flipped.loc[held_idx, "click_bool"] = 1 - flipped.loc[held_idx, "click_bool"]
    flipped.loc[held_idx, "booking_bool"] = 1 - flipped.loc[held_idx, "booking_bool"]
    flipped.loc[held_idx, "position"] = 7 - flipped.loc[held_idx, "position"]
    changed, _ = _encode(flipped, impressions)

    own = np.zeros(len(impressions), dtype=bool)
    own[held_idx] = True
    for col in TRAIN_COLUMN_ORDER:
        assert np.array_equal(baseline.loc[own, col], changed.loc[own, col]), f"{col} leaks its own labels"
    for col in ("prop_click_rate", "prop_booking_rate", "prop_avg_position"):
        assert not np.array_equal(baseline.loc[~own, col], changed.loc[~own, col]), f"{col} ignores other folds"


def test_test_encoding_is_average_of_fold_tables(impressions):
    """Test rows get the mean of the five per-fold smoothed rates, each with its own prior."""
    _, test_enc = _encode(impressions, impressions)
    prop = impressions["prop_id"].iloc[0]

    per_fold = []
    for fit_idx, _ in oof_folds(impressions["srch_id"].to_numpy()):
        fit = impressions.iloc[fit_idx]
        prior = fit["booking_bool"].mean()
        rows = fit[fit["prop_id"] == prop]
        per_fold.append((rows["booking_bool"].sum() + 100 * prior) / (len(rows) + 100))

    got = test_enc.loc[impressions["prop_id"] == prop, "prop_booking_rate"].to_numpy()
    assert np.allclose(got, np.mean(per_fold), atol=1e-6)


def test_oof_folds_keep_searches_together(impressions):
    for fit_idx, held_idx in oof_folds(impressions["srch_id"].to_numpy()):
        fit_searches = set(impressions["srch_id"].iloc[fit_idx])
        held_searches = set(impressions["srch_id"].iloc[held_idx])
        assert not fit_searches & held_searches
