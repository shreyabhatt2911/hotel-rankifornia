import numpy as np
import pandas as pd

from hotel_rankifornia.debiasing import (
    UNBIASED_COLUMNS,
    add_unbiased_rates,
    fix_missing_history_price,
)
from hotel_rankifornia.target_encoding import oof_folds


def _tables(impressions: pd.DataFrame):
    labels = impressions[["srch_id", "prop_id", "click_bool", "booking_bool", "random_bool"]].copy()
    train_fe = impressions[["srch_id", "prop_id"]].copy()
    test_fe = impressions[["srch_id", "prop_id"]].copy()
    add_unbiased_rates(train_fe, test_fe, labels)
    return train_fe, test_fe


def test_unbiased_rates_ignore_position_biased_impressions(impressions):
    """Only random_bool == 1 impressions may influence the unbiased rates."""
    base_train, base_test = _tables(impressions)

    tampered = impressions.copy()
    ordered = tampered["random_bool"] == 0
    tampered.loc[ordered, "click_bool"] = 1 - tampered.loc[ordered, "click_bool"]
    tampered.loc[ordered, "booking_bool"] = 1 - tampered.loc[ordered, "booking_bool"]
    new_train, new_test = _tables(tampered)

    for col in UNBIASED_COLUMNS:
        assert np.array_equal(base_train[col], new_train[col])
        assert np.array_equal(base_test[col], new_test[col])
        assert base_train[col].dtype == np.float32


def test_unbiased_rates_are_out_of_fold(impressions):
    """Flipping a fold's own labels must not change that fold's unbiased encodings."""
    base_train, _ = _tables(impressions)
    held_idx = oof_folds(impressions["srch_id"].to_numpy())[0][1]

    flipped = impressions.copy()
    flipped.loc[held_idx, "click_bool"] = 1 - flipped.loc[held_idx, "click_bool"]
    flipped.loc[held_idx, "booking_bool"] = 1 - flipped.loc[held_idx, "booking_bool"]
    new_train, _ = _tables(flipped)

    for col in UNBIASED_COLUMNS:
        assert np.array_equal(base_train[col].iloc[held_idx], new_train[col].iloc[held_idx])


def test_price_repair_neutralises_rows_without_history():
    df = pd.DataFrame({
        "srch_id": [1, 1, 1],
        "no_hist_price": [1, 0, 1],
        "price_vs_hist": np.array([200.0, 0.9, 150.0], dtype="float32"),
        "price_discount": np.array([-199.0, 12.0, -149.0], dtype="float32"),
    })
    fix_missing_history_price(df)
    assert df["price_vs_hist"].tolist() == [1.0, np.float32(0.9), 1.0]
    assert df["price_discount"].tolist() == [0.0, 12.0, 0.0]
    # the within-search rank is recomputed after the repair (ascending, ties share the min rank)
    assert df["price_vs_hist_rank"].tolist() == [2.0, 1.0, 2.0]
