#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <rho_tag> [extra args...]"
    echo "Example: $0 95"
    exit 1
fi

RHO_TAG="$1"

shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT_PATH="${SCRIPT_DIR}/train.py"
DATA_PATH="${SCRIPT_DIR}/data/waterbirds_rho_gan/waterbirds_256_${RHO_TAG}"
OUTDIR="${SCRIPT_DIR}/output_waterbirds_${RHO_TAG}"

if [ ! -d "$DATA_PATH" ]; then
    echo "[ERROR] Data path not found: $DATA_PATH"
    echo "Run preprocess_waterbirds.py first."
    exit 1
fi

echo "=== Training rho=${RHO_TAG} ==="
echo "  data: $DATA_PATH"
echo "  outdir: $OUTDIR"

python "$TRAIN_SCRIPT_PATH" \
    --outdir="$OUTDIR" \
    --cfg=stylegan2 \
    --data="$DATA_PATH" \
    --gpus=1 \
    --batch=32 \
    --gamma=8 \
    --cond=1 \
    --kimg=3500 \
    --mirror=1 \
    --aug=ada \
    --dry-run=1 \
    "$@"