#!/bin/bash
set -euo pipefail

# Retrieve arguments or use defaults
DATA_DIR=${1:-"./data"}
MODEL_PATH=${2:-"./pickle/model.pkl"}
OUTPUT_PATH=${3:-"./output/predictions.csv"}

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "========================================"
echo "PMIE Hackathon Prediction Pipeline"
echo "========================================"
echo "DATA_DIR    : $DATA_DIR"
echo "MODEL_PATH  : $MODEL_PATH"
echo "OUTPUT_PATH : $OUTPUT_PATH"
echo ""

python src/predict_cli.py \
    "$DATA_DIR" \
    "$MODEL_PATH" \
    "$OUTPUT_PATH"

if [ ! -f "$OUTPUT_PATH" ]; then
    echo "ERROR: Prediction file was not generated."
    exit 1
fi

echo ""
echo "SUCCESS"
echo "Predictions written to:"
echo "$OUTPUT_PATH"