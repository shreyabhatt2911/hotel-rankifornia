#!/usr/bin/env bash
# Full pipeline, raw CSVs -> features -> three ensembles -> final blend.
#
# Usage
#   DATA_DIR=data/raw PROCESSED_DIR=data/processed OUTPUT_DIR=outputs bash scripts/run_all.sh
#
# Optional environment variables
#   REUSE_SELECTION=1   skip the candidate search and retrain the recorded selection (much faster)
#   RUN_TUNING=1        also run the Optuna search first (~9 hours, not needed to reproduce)
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/raw}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
COMMON=(--data-dir "$DATA_DIR" --processed-dir "$PROCESSED_DIR" --output-dir "$OUTPUT_DIR")

EXTRA=()
if [[ "${REUSE_SELECTION:-0}" == "1" ]]; then
  EXTRA+=(--reuse-selection)
fi

python scripts/build_features.py "${COMMON[@]}"

if [[ "${RUN_TUNING:-0}" == "1" ]]; then
  python scripts/tune.py "${COMMON[@]}"
fi

for variant in unbiased improved optuna; do
  python scripts/train_ensemble.py "${COMMON[@]}" --variant "$variant" ${EXTRA[@]+"${EXTRA[@]}"}
done

python scripts/blend.py "${COMMON[@]}"
