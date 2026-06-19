#!/bin/bash

# Retrieve arguments or use defaults
DATA_DIR=${1:-"./data"}
MODEL_PATH=${2:-"./pickle/model.pkl"}
OUTPUT_PATH=${3:-"./output/predictions.csv"}

# Determine python executable
PYTHON_EXE="python"
if [ -f "./venv/Scripts/python" ]; then
    PYTHON_EXE="./venv/Scripts/python"
elif [ -f "./venv/bin/python" ]; then
    PYTHON_EXE="./venv/bin/python"
fi

# Run predictor
$PYTHON_EXE src/predict_cli.py "$DATA_DIR" "$MODEL_PATH" "$OUTPUT_PATH"
