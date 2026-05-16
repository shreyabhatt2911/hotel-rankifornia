"""Optuna search over LambdaMART hyperparameters (TPE, validation NDCG@5).

The best trials found here were copied into ``configs/ensemble_optuna.json`` as
candidate models (with several seeds each). See ``configs/optuna_best.json`` for
the recorded outcome of the original 60-trial search.

Requires the optional ``tune`` dependencies (``pip install -e .[tune]``).
"""

from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd

from hotel_rankifornia.config import (
    EARLY_STOPPING_ROUNDS,
    LGB_RUNTIME_PARAMS,
    MAX_BOOST_ROUNDS,
    SEED,
    Paths,
)
from hotel_rankifornia.data import load_search_dates
from hotel_rankifornia.train import RankingData, split_train_val
from hotel_rankifornia.validation import ndcg_at_k, temporal_split

logger = logging.getLogger(__name__)

N_TRIALS = 60
N_STARTUP_TRIALS = 15
TOP_K_REPORTED = 10

# Fixed part of every trial's LightGBM parameters.
BASE_PARAMS: dict[str, Any] = {
    "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5],
    "label_gain": [0, 1, 0, 0, 0, 5],
    "lambdarank_truncation_level": 12,
    "bagging_freq": 1,
    "feature_pre_filter": False,
    "seed": SEED,
}

# Known-good configurations evaluated first so the search starts from a strong baseline.
_OLD_500 = {"num_leaves": 500, "learning_rate": 0.01, "min_data_in_leaf": 20, "feature_fraction": 0.6,
            "bagging_fraction": 0.8, "lambda_l1": 0.25, "lambda_l2": 0.25, "min_gain_to_split": 0.0}
ENQUEUED_TRIALS: list[dict[str, Any]] = [
    {**_OLD_500, "weighted": False},
    {**_OLD_500, "weighted": True},
    {"num_leaves": 300, "learning_rate": 0.02, "min_data_in_leaf": 25, "feature_fraction": 0.7,
     "bagging_fraction": 0.8, "lambda_l1": 0.15, "lambda_l2": 0.15, "min_gain_to_split": 0.0,
     "weighted": False},
    {"num_leaves": 700, "learning_rate": 0.008, "min_data_in_leaf": 15, "feature_fraction": 0.6,
     "bagging_fraction": 0.8, "lambda_l1": 0.3, "lambda_l2": 0.3, "min_gain_to_split": 0.0,
     "weighted": False},
    {"num_leaves": 1000, "learning_rate": 0.005, "min_data_in_leaf": 10, "feature_fraction": 0.5,
     "bagging_fraction": 0.8, "lambda_l1": 0.4, "lambda_l2": 0.4, "min_gain_to_split": 0.0,
     "weighted": False},
]


@dataclass
class TrialRecord:
    trial: int
    ndcg: float
    best_iteration: int
    weighted: bool
    params: dict[str, Any]


def suggest_params(trial) -> tuple[dict[str, Any], bool]:
    """Sample one configuration from the search space."""
    params = {
        **BASE_PARAMS,
        "num_leaves": trial.suggest_int("num_leaves", 300, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.01, 2.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.01, 2.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.5),
    }
    weighted = trial.suggest_categorical("weighted", [True, False])
    return params, weighted


def run_search(tr: RankingData, va: RankingData, feature_cols: list[str],
               n_trials: int = N_TRIALS) -> tuple[Any, list[TrialRecord]]:
    """Run the TPE search and return the Optuna study and one record per trial."""
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    records: list[TrialRecord] = []

    def objective(trial) -> float:
        t0 = time.time()
        params, weighted = suggest_params(trial)
        # Datasets are built inside the trial so only one copy is alive at a time.
        dtrain = lgb.Dataset(tr.X, label=tr.y, group=tr.group, weight=tr.weights if weighted else None,
                             feature_name=feature_cols, free_raw_data=False)
        dval = lgb.Dataset(va.X, label=va.y, group=va.group, feature_name=feature_cols,
                           free_raw_data=False, reference=dtrain)
        model = lgb.train({**params, **LGB_RUNTIME_PARAMS}, dtrain, num_boost_round=MAX_BOOST_ROUNDS,
                          valid_sets=[dval], valid_names=["val"],
                          callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        ndcg = ndcg_at_k(va.srch_ids, va.relevance, model.predict(va.X))
        records.append(TrialRecord(trial.number, ndcg, model.best_iteration, weighted, params))
        del model, dtrain, dval
        gc.collect()
        logger.info("Trial %3d NDCG@5=%.6f leaves=%d lr=%.4f mdl=%d iter=%d weighted=%s (%.0fs)",
                    trial.number, ndcg, params["num_leaves"], params["learning_rate"],
                    params["min_data_in_leaf"], records[-1].best_iteration, weighted, time.time() - t0)
        return ndcg

    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=SEED, n_startup_trials=N_STARTUP_TRIALS),
                                study_name="lgb_lambdarank")
    for params in ENQUEUED_TRIALS:
        study.enqueue_trial(params)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study, records


def tune(paths: Paths, feature_set: str = "v3", n_trials: int = N_TRIALS) -> Path:
    """Search hyperparameters on the temporal split and write ``optuna_best.json``.

    Parameters
    ----------
    paths
        Filesystem layout (expects the feature tables from ``build_features``).
    feature_set
        Feature table to tune on (the search was run on ``v3``).
    n_trials
        Number of Optuna trials (each is one early-stopped LightGBM fit).
    """
    with open(paths.feature_cols_path(feature_set)) as fh:
        feature_cols = json.load(fh)
    train_fe = pd.read_parquet(paths.features_path("train", feature_set))
    train_ids, val_ids = temporal_split(load_search_dates(paths.data_dir))
    tr, va = split_train_val(train_fe, train_ids, val_ids, feature_cols)
    del train_fe
    gc.collect()

    study, records = run_search(tr, va, feature_cols, n_trials)
    ranked = sorted(records, key=lambda r: -r.ndcg)
    best = ranked[0]
    result = {
        "feature_set": feature_set,
        "n_trials": len(records),
        "best_trial": {"trial": best.trial, "ndcg": best.ndcg, "iter": best.best_iteration,
                       "weighted": best.weighted, "params": best.params},
        "top_trials": [{"trial": r.trial, "ndcg": r.ndcg, "leaves": r.params["num_leaves"],
                        "lr": r.params["learning_rate"]} for r in ranked[:TOP_K_REPORTED]],
    }
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = paths.reports_dir / "optuna_best.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info("Best trial #%d NDCG@5=%.6f -> %s", best.trial, best.ndcg, out_path)
    return out_path
