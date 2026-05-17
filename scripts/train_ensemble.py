#!/usr/bin/env python
"""Train one LambdaMART ensemble variant and write its submission.

Variants, configured in ``configs/ensemble_<variant>.json``
    unbiased  v4 features (position-unbiased rates), Optuna-derived + earlier params
    improved  v3 features, hand-tuned depth family + query-weighted model
    optuna    v3 features, top Optuna trials with seed diversity

By default the full candidate search runs (hours). ``--reuse-selection`` skips it
and retrains only the models/iterations recorded in the config.

Example
-------
    python scripts/train_ensemble.py --variant unbiased
"""

from __future__ import annotations

import argparse

from hotel_rankifornia.config import add_path_arguments, configure_logging, paths_from_args, seed_everything
from hotel_rankifornia.pipeline import run_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(parser)
    parser.add_argument("--variant", choices=["unbiased", "improved", "optuna"], required=True)
    parser.add_argument("--reuse-selection", action="store_true",
                        help="skip the candidate search and retrain the selection recorded in the config")
    args = parser.parse_args()

    configure_logging()
    seed_everything()
    run_ensemble(paths_from_args(args), args.variant, reuse_selection=args.reuse_selection)


if __name__ == "__main__":
    main()
