#!/usr/bin/env python
"""Optuna hyperparameter search for the LambdaMART ranker (slow, ~9 h for 60 trials).

Requires the ``tune`` extra (``pip install -e .[tune]``). Writes
``<output-dir>/reports/optuna_best.json``.

Example
-------
    python scripts/tune.py --n-trials 60
"""

from __future__ import annotations

import argparse

from hotel_rankifornia.config import add_path_arguments, configure_logging, paths_from_args, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(parser)
    parser.add_argument("--feature-set", choices=["v3", "v4"], default="v3")
    parser.add_argument("--n-trials", type=int, default=60)
    args = parser.parse_args()

    configure_logging()
    seed_everything()
    from hotel_rankifornia.tuning import tune  # optional dependency, imported lazily

    tune(paths_from_args(args), feature_set=args.feature_set, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
