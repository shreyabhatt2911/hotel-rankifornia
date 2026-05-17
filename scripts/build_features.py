#!/usr/bin/env python
"""Build the engineered feature tables from the raw Expedia CSVs.

Writes ``train_fe_<set>.parquet``, ``test_fe_<set>.parquet`` and
``feature_cols_<set>.json`` for ``v3`` (139 features) and, by default, ``v4``
(146 features, adds position-unbiased rates).

Example
-------
    python scripts/build_features.py --data-dir data/raw --processed-dir data/processed
"""

from __future__ import annotations

import argparse

from hotel_rankifornia.config import add_path_arguments, configure_logging, paths_from_args
from hotel_rankifornia.pipeline import build_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(parser)
    parser.add_argument("--feature-set", choices=["v3", "v4"], default="v4",
                        help="highest feature set to build, v4 also writes v3 (default v4)")
    args = parser.parse_args()

    configure_logging()
    feature_sets = ("v3",) if args.feature_set == "v3" else ("v3", "v4")
    build_features(paths_from_args(args), feature_sets)


if __name__ == "__main__":
    main()
