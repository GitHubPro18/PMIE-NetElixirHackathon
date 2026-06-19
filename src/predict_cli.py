import os
import sys
import pandas as pd
from predict import predict_all_horizons
from config import DATA_DIR, MODEL_PATH, PREDICTIONS_OUTPUT_FILE
from logger import get_logger

logger = get_logger(__name__)

def main():
    # Set default paths from config
    data_dir = DATA_DIR
    model_path = MODEL_PATH
    output_path = PREDICTIONS_OUTPUT_FILE
    
    # Read command line arguments
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_path = sys.argv[3]
        
    logger.info(f"PMIE CLI Predictor running with:")
    logger.info(f"  DATA_DIR:   {data_dir}")
    logger.info(f"  MODEL_PATH: {model_path}")
    logger.info(f"  OUTPUT_PATH:{output_path}")
    
    # Check inputs
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} does not exist.")
        sys.exit(1)
    if not os.path.exists(model_path):
        logger.error(f"Model path {model_path} does not exist. Please train the model first.")
        sys.exit(1)
        
    # Ensure output parent directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    try:
        # Run baseline forecast
        logger.info("Running forecasts...")
        preds_df = predict_all_horizons(data_dir, model_path, google_change=0.0, meta_change=0.0, bing_change=0.0)
        
        # Save to output path
        preds_df.to_csv(output_path, index=False)
        logger.info(f"Forecasts successfully generated and saved to {output_path}!")
        
    except Exception as e:
        logger.exception(f"Error during forecasting: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
