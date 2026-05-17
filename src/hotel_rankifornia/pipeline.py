"""End-to-end stages used by the CLI scripts.

* ``build_features`` turns the raw CSVs into ``v3`` (+ ``v4``) feature tables on disk.
* ``run_ensemble`` runs the validated candidate search, greedy selection, full-data
  retrain and submission for one ensemble variant.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hotel_rankifornia import debiasing, features
from hotel_rankifornia.config import N_OOF_FOLDS, RETRAIN_MULTIPLIER, RRF_K, Paths, load_variant
from hotel_rankifornia.data import load_search_dates, load_test, load_train
from hotel_rankifornia.ensemble import (
    ScoredModel,
    Selection,
    greedy_forward_selection,
    make_submission,
    rrf_within_search,
)
from hotel_rankifornia.train import (
    fit_full_and_predict,
    fit_validated,
    make_ranking_data,
    retrain_rounds,
    split_train_val,
)
from hotel_rankifornia.validation import ndcg_at_k, temporal_split

logger = logging.getLogger(__name__)


def _save_feature_set(paths: Paths, feature_set: str, train_fe: pd.DataFrame, test_fe: pd.DataFrame) -> None:
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = features.select_feature_cols(train_fe, test_fe)
    with open(paths.feature_cols_path(feature_set), "w") as fh:
        json.dump(feature_cols, fh, indent=2)
    train_fe.to_parquet(paths.features_path("train", feature_set), index=False)
    test_fe.to_parquet(paths.features_path("test", feature_set), index=False)
    logger.info("Saved %s feature set with %d features -> %s", feature_set, len(feature_cols), paths.processed_dir)


def build_features(paths: Paths, feature_sets: tuple[str, ...] = ("v3", "v4")) -> None:
    """Build and save the requested feature tables.

    ``v3`` (139 features) is the base table with leak-free target encodings, and
    ``v4`` (146 features) adds position-unbiased property rates on top of it.
    Requesting ``v4`` also writes ``v3``.
    """
    unknown = set(feature_sets) - {"v3", "v4"}
    if unknown:
        raise ValueError(f"Unknown feature sets: {sorted(unknown)}")

    train = load_train(paths.data_dir)
    test = load_test(paths.data_dir)
    labels = train[list(debiasing.LABEL_COLUMNS)].copy()  # raw labels, needed again for v4

    train_fe, test_fe = features.build_base_features(train, test)
    del train, test
    gc.collect()
    _save_feature_set(paths, "v3", train_fe, test_fe)

    if "v4" in feature_sets:
        debiasing.add_unbiased_features(train_fe, test_fe, labels, N_OOF_FOLDS)
        _save_feature_set(paths, "v4", train_fe, test_fe)


def run_ensemble(paths: Paths, variant_name: str, reuse_selection: bool = False) -> Path:
    """Train one ensemble variant and write its submission.

    1. Temporal split (last 20% of searches by date) and validated training of
       every candidate with early stopping.
    2. Greedy forward selection of an RRF ensemble on the validation set.
    3. Retrain each selected model on all training searches for 1.25x its
       early-stopped iterations, then fuse test predictions with RRF.

    Parameters
    ----------
    paths
        Filesystem layout.
    variant_name
        ``unbiased``, ``improved`` or ``optuna`` (see ``configs/``).
    reuse_selection
        Skip steps 1-2 and reuse the selected models and
        iteration counts recorded in the variant config (from the original
        full search). Much faster, and step 3 is identical.

    Returns
    -------
        Path of the written submission CSV.
    """
    variant = load_variant(paths.config_dir, variant_name)
    with open(paths.feature_cols_path(variant.feature_set)) as fh:
        feature_cols = json.load(fh)
    train_fe = pd.read_parquet(paths.features_path("train", variant.feature_set))
    test_fe = pd.read_parquet(paths.features_path("test", variant.feature_set))
    logger.info("Variant %s on %s with %d features, train %s, test %s", variant_name,
                variant.feature_set, len(feature_cols), train_fe.shape, test_fe.shape)
    candidates = {c.name: c for c in variant.candidates}

    scored: dict[str, ScoredModel] = {}
    if reuse_selection:
        ref = variant.reference
        selection = Selection(names=list(ref["selected"]), ndcg=ref["ensemble_ndcg"])
        best_iters = {n: ref["individual_scores"][n]["iter"] for n in selection.names}
    else:
        train_ids, val_ids = temporal_split(load_search_dates(paths.data_dir))
        tr, va = split_train_val(train_fe, train_ids, val_ids, feature_cols)
        for cfg in variant.candidates:
            t0 = time.time()
            fit = fit_validated(cfg, tr, va, feature_cols)
            ndcg = ndcg_at_k(va.srch_ids, va.relevance, fit.val_preds)
            scored[cfg.name] = ScoredModel(cfg.name, ndcg, rrf_within_search(va.srch_ids, fit.val_preds),
                                           fit.best_iteration)
            logger.info("%s NDCG@5=%.6f iter=%d (%.0fs)", cfg.name, ndcg, fit.best_iteration, time.time() - t0)
        selection = greedy_forward_selection(scored, va.srch_ids, va.relevance, variant.min_models)
        best_iters = {n: scored[n].best_iteration for n in selection.names}
        logger.info("Ensemble %s -> %.6f", selection.names, selection.ndcg)
        del tr, va
        gc.collect()

    full = make_ranking_data(train_fe, feature_cols)
    X_test = test_fe[feature_cols].astype("float32").values
    del train_fe
    gc.collect()

    test_rrf = []
    for name in selection.names:
        rounds = retrain_rounds(best_iters[name])
        t0 = time.time()
        preds = fit_full_and_predict(candidates[name], full, X_test, feature_cols, rounds)
        test_rrf.append(rrf_within_search(test_fe["srch_id"].values, preds))
        logger.info("Retrained %s for %d rounds (%.0fs)", name, rounds, time.time() - t0)

    submission = make_submission(test_fe, np.mean(test_rrf, axis=0))
    paths.submissions_dir.mkdir(parents=True, exist_ok=True)
    out_path = paths.submissions_dir / f"submission_{variant_name}.csv"
    submission.to_csv(out_path, index=False)
    logger.info("Wrote %s (%s)", out_path, submission.shape)

    _write_report(paths, variant_name, variant.feature_set, len(feature_cols), selection, scored, candidates,
                  best_iters, reuse_selection)
    return out_path


def _write_report(paths, variant_name, feature_set, n_features, selection, scored, candidates, best_iters,
                  reused_selection) -> None:
    """Persist what was selected and how the candidates scored (JSON)."""
    report = {
        "variant": variant_name,
        "feature_set": feature_set,
        "n_features": n_features,
        "selection_source": "config reference (no candidate search)" if reused_selection else "candidate search",
        "blend_method": "RRF",
        "rrf_k": RRF_K,
        "retrain_multiplier": RETRAIN_MULTIPLIER,
        "selected": selection.names,
        "ensemble_ndcg": selection.ndcg,
        "retrain_iterations": {n: retrain_rounds(best_iters[n]) for n in selection.names},
        "individual_scores": {
            n: {"ndcg": m.ndcg, "iter": m.best_iteration, "weighted": candidates[n].weighted}
            for n, m in scored.items()
        },
    }
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = paths.reports_dir / f"ensemble_{variant_name}.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Wrote %s", report_path)
