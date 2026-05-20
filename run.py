import argparse
import logging
import time
import sys
import json
import yaml
import pandas as pd
import numpy as np
import random

def setup_logging(log_file):

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  #for run.log file
            logging.StreamHandler(sys.stdout) #for terminal screen
        ]
    )
def write_metrics(output_path, status, version="unknown", error_message=None, 
                  rows_processed=None, metric_name=None, metric_value=None, 
                  latency_ms=None, seed=None):
    """
    Writes the structured JSON output for both success and error cases.
    """
    if status == "error":
        output_data = {
            "version": version,
            "status": "error",
            "error_message": error_message
        }
    else:
        output_data = {
            "version": version,
            "rows_processed": rows_processed,
            "metric": metric_name,
            "value": metric_value,
            "latency_ms": latency_ms,
            "seed": seed,
            "status": "success"
        }

    try:
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        # The prompt requires printing the final metrics JSON to stdout in Docker
        if status == "success":
            print(json.dumps(output_data, indent=4))
            
    except Exception as e:
        logging.error(f"CRITICAL: Failed to write metrics JSON to {output_path}: {e}")

def load_and_validate_config(config_path, output_path):
    """
    Loads YAML config, validates required fields, and enforces determinism.
    """
    logging.info("Attempting to load configuration...")
    
    # Safely load the YAML
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        msg = f"Failed to read config file at {config_path}: {e}"
        logging.error(msg)
        write_metrics(output_path, status="error", error_message=msg)
        sys.exit(1)

    # Validate the data structure
    if not isinstance(config, dict):
        msg = "Invalid config structure: YAML must contain a key-value mapping."
        logging.error(msg)
        write_metrics(output_path, status="error", error_message=msg)
        sys.exit(1)

    # Check for mandatory keys
    required_keys = ['seed', 'window', 'version']
    for key in required_keys:
        if key not in config:
            msg = f"Invalid config: Missing required key '{key}'"
            logging.error(msg)
            # Try to grab the version if it exists for the error output, otherwise default
            version = config.get('version', 'unknown') 
            write_metrics(output_path, status="error", version=version, error_message=msg)
            sys.exit(1)

    # Set determinism (Reproducibility requirement)
    seed_val = config['seed']
    random.seed(seed_val)
    np.random.seed(seed_val)
    
    logging.info(f"Config validated. Seed: {seed_val} | Window: {config['window']} | Version: {config['version']}")
    return config
def load_and_validate_data(data_path, output_path, config):
    """
    Loads the CSV dataset and cleanly catches any structural or missing file errors.
    """
    logging.info(f"Attempting to load dataset from {data_path}...")
    
    version = config.get('version', 'unknown')

    # Catch file-level and parsing errors
    try:
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.lower().str.strip()
    except FileNotFoundError:
        msg = f"Input file not found at path: {data_path}"
        logging.error(msg)
        write_metrics(output_path, status="error", version=version, error_message=msg)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        msg = f"Input file at {data_path} is completely empty."
        logging.error(msg)
        write_metrics(output_path, status="error", version=version, error_message=msg)
        sys.exit(1)
    except pd.errors.ParserError:
        msg = f"Invalid CSV format in file: {data_path}"
        logging.error(msg)
        write_metrics(output_path, status="error", version=version, error_message=msg)
        sys.exit(1)
    except Exception as e:
        msg = f"Unexpected error reading data: {e}"
        logging.error(msg)
        write_metrics(output_path, status="error", version=version, error_message=msg)
        sys.exit(1)

    # Validate that the necessary column actually exists
    if 'close' not in df.columns:
        msg = "Missing required column: 'close'."
        logging.error(msg)
        write_metrics(output_path, status="error", version=version, error_message=msg)
        sys.exit(1)

    logging.info(f"Dataset loaded successfully. Total rows: {len(df)}")
    return df
def process_data(df, config):
    """
    Computes the rolling mean on the 'close' column and generates a binary signal.
    """
    logging.info("Starting core processing...")
    window_size = config['window']

    # 1. Compute rolling mean
    df['rolling_mean'] = df['close'].rolling(window=window_size).mean()

    # 2. Handle the NaN boundaries consistently
    # We choose to drop the first (window - 1) rows where the mean cannot be calculated.
    initial_row_count = len(df)
    df = df.dropna(subset=['rolling_mean']).copy()
    dropped_rows = initial_row_count - len(df)
    logging.info(f"Dropped {dropped_rows} rows due to initial rolling window NaN values.")

    # 3. Generate the binary signal (1 if close > rolling_mean, else 0)
    df['signal'] = (df['close'] > df['rolling_mean']).astype(int)

    logging.info("Signal generation completed successfully.")
    return df

def main():
    # 1. Start the timer for latency tracking
    start_time = time.perf_counter()

    # 2. Parse command-line arguments 
    parser = argparse.ArgumentParser(description="MLOps Batch Job - Task 0")
    parser.add_argument('--input', required=True, help="Path to input CSV data")
    parser.add_argument('--config', required=True, help="Path to config YAML")
    parser.add_argument('--output', required=True, help="Path to output metrics JSON")
    parser.add_argument('--log-file', required=True, dest='log_file', help="Path to execution log file")
    
    args = parser.parse_args()

    # 3. Initialize logging using the path provided by the CLI
    setup_logging(args.log_file)
    logging.info("Job started. CLI arguments parsed.")
    logging.info(f"Target Input: {args.input}")
    logging.info(f"Target Config: {args.config}")

    #4. Load and validate configuration
    config = load_and_validate_config(args.config, args.output)
    # 5. Load and validate the CSV data
    df = load_and_validate_data(args.input, args.output, config)
    # 6. Core Processing
    processed_df = process_data(df, config)

    # 7. Calculate Final Metrics
    rows_processed = len(processed_df)
    
    # Calculate the mean of the signal column (this is our signal_rate)
    # We cast to float because NumPy floats can sometimes upset the standard JSON library
    signal_rate = float(processed_df['signal'].mean())
    
    # Stop the clock and convert to milliseconds
    end_time = time.perf_counter()
    latency_ms = int((end_time - start_time) * 1000)

    logging.info(f"Final Metrics - Rows: {rows_processed}, Signal Rate: {signal_rate:.4f}, Latency: {latency_ms}ms")

    # 8. Write Success Output
    write_metrics(
        output_path=args.output,
        status="success",
        version=config.get('version', 'unknown'),
        rows_processed=rows_processed,
        metric_name="signal_rate",
        metric_value=round(signal_rate, 4),
        latency_ms=latency_ms,
        seed=config.get('seed')
    )
    logging.info("Job completed successfully.")
    
    # Exit cleanly
    sys.exit(0)
if __name__ == "__main__":
    main()