#!/usr/bin/env python
"""Weighted RRF blend of the three variant submissions (the final submission).

Default weights in ``configs/blend.json`` are unbiased 0.50, optuna 0.25 and improved 0.25.
Inputs default to ``<output-dir>/submissions/submission_<variant>.csv``.

Example
-------
    python scripts/blend.py --output-dir outputs
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hotel_rankifornia.blend import blend_submissions
from hotel_rankifornia.config import add_path_arguments, configure_logging, load_blend_config, paths_from_args

logger = logging.getLogger("blend")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_arguments(parser)
    for variant in ("unbiased", "optuna", "improved"):
        parser.add_argument(f"--{variant}", type=Path, default=None,
                            help=f"{variant} submission CSV (default: <output-dir>/submissions/submission_{variant}.csv)")
    parser.add_argument("--out", type=Path, default=None,
                        help="blended submission path, defaults to <output-dir>/submissions/submission_blend.csv")
    args = parser.parse_args()

    configure_logging()
    paths = paths_from_args(args)
    config = load_blend_config(paths.config_dir)
    inputs = {v: getattr(args, v) or paths.submissions_dir / f"submission_{v}.csv" for v in config["weights"]}

    blended = blend_submissions(inputs, config["weights"], k=config["rrf_k"])
    out = args.out or paths.submissions_dir / "submission_blend.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    blended.to_csv(out, index=False)
    logger.info("Wrote %s (%s rows) with weights %s", out, f"{len(blended):,}", config["weights"])


if __name__ == "__main__":
    main()
