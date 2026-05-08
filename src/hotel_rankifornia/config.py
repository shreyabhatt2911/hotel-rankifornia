"""Paths, seeds, constants and config-file loading shared across the pipeline.

Everything that is a *choice* (hyperparameters, candidate models, blend
weights) lives in ``configs/*.json``. This module only holds project-wide
constants and the small loaders that read those files.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------
# Data / validation constants
# --------------------------------------------------------------------------
N_OOF_FOLDS = 5  # GroupKFold folds (grouped by srch_id) for target encoding
VAL_FRACTION = 0.20  # last 20% of searches (by date) form the validation set

RELEVANCE_BOOKED = 5
RELEVANCE_CLICKED = 1
RELEVANCE_NONE = 0
NDCG_K = 5

# --------------------------------------------------------------------------
# Training / ensembling constants
# --------------------------------------------------------------------------
MAX_BOOST_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 50
RETRAIN_MULTIPLIER = 1.25  # full-data rounds = early-stopped iterations * 1.25
BOOKING_QUERY_WEIGHT = 3.0  # weight of searches that contain a booking
RRF_K = 60  # reciprocal rank fusion constant

# Parameters added to every LightGBM config at runtime (not part of the search).
LGB_RUNTIME_PARAMS: dict[str, Any] = {"verbose": -1, "n_jobs": -1}

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. All locations can be overridden from the CLI."""

    data_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    output_dir: Path = Path("outputs")
    config_dir: Path = field(default=DEFAULT_CONFIG_DIR)

    @property
    def submissions_dir(self) -> Path:
        return self.output_dir / "submissions"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    def features_path(self, split: str, feature_set: str) -> Path:
        """Parquet file holding the engineered ``train``/``test`` features."""
        return self.processed_dir / f"{split}_fe_{feature_set}.parquet"

    def feature_cols_path(self, feature_set: str) -> Path:
        return self.processed_dir / f"feature_cols_{feature_set}.json"


def add_path_arguments(parser) -> None:
    """Register the shared ``--data-dir/--processed-dir/--output-dir`` options."""
    parser.add_argument("--data-dir", type=Path, default=Paths.data_dir,
                        help="directory with the raw Expedia CSV files")
    parser.add_argument("--processed-dir", type=Path, default=Paths.processed_dir,
                        help="directory for engineered feature parquet files")
    parser.add_argument("--output-dir", type=Path, default=Paths.output_dir,
                        help="directory for submissions and reports")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR,
                        help="directory with the JSON experiment configs")


def paths_from_args(args) -> Paths:
    return Paths(data_dir=args.data_dir, processed_dir=args.processed_dir,
                 output_dir=args.output_dir, config_dir=args.config_dir)


def configure_logging(level: int = logging.INFO) -> None:
    """Route library and script output through ``logging``."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
                        datefmt="%H:%M:%S")


def seed_everything(seed: int = SEED) -> None:
    """Seed Python and NumPy RNGs.

    LightGBM and Optuna receive their seeds explicitly through their own
    parameters (see ``configs/`` and ``hotel_rankifornia.tuning``).
    """
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------
# Experiment config files
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    """One LambdaMART candidate."""

    name: str
    params: dict[str, Any]
    weighted: bool = False

    def lgb_params(self) -> dict[str, Any]:
        return {**self.params, **LGB_RUNTIME_PARAMS}


@dataclass(frozen=True)
class VariantConfig:
    """An ensemble variant, made of a feature set, candidate models and selection rules."""

    name: str
    feature_set: str
    candidates: list[ModelConfig]
    min_models: int
    reference: dict[str, Any]  # selection/iterations recorded from the original run


def _read_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def load_variant(config_dir: Path, name: str) -> VariantConfig:
    """Load ``configs/ensemble_<name>.json``."""
    raw = _read_json(Path(config_dir) / f"ensemble_{name}.json")
    candidates = [
        ModelConfig(name=cname, params=spec["params"], weighted=spec.get("weighted", False))
        for cname, spec in raw["candidates"].items()
    ]
    return VariantConfig(
        name=name,
        feature_set=raw["feature_set"],
        candidates=candidates,
        min_models=raw.get("min_models", 1),
        reference=raw["reference"],
    )


def load_blend_config(config_dir: Path) -> dict[str, Any]:
    """Load ``configs/blend.json`` (RRF constant and per-variant weights)."""
    return _read_json(Path(config_dir) / "blend.json")
