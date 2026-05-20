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
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def write_metrics(output_path, status, version="unknown", error_message=None,
                  rows_processed=None, metric_name=None, metric_value=None,
                  latency_ms=None, seed=None):
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
        if status == "success":
            print(json.dumps(output_data, indent=4))
    except Exception as e:
        logging.error(f"failed to write metrics: {e}")

def load_and_validate_config(config_path, output_path):
    logging.info("loading config...")

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        msg = f"failed to read config: {e}"
        logging.error(msg)
        write_metrics(output_path, status="error", error_message=msg)
        sys.exit(1)

    if not isinstance(config, dict):
        msg = "invalid config structure"
        logging.error(msg)
        write_metrics(output_path, status="error", error_message=msg)
        sys.exit(1)

    # check required keys
    for key in ['seed', 'window', 'version']:
        if key not in config:
            msg = f"missing required key: '{key}'"
            logging.error(msg)
            write_metrics(output_path, status="error", version=config.get('version', 'unknown'), error_message=msg)
            sys.exit(1)

    # set seed for reproducibility
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    logging.info(f"config ok — seed: {config['seed']} | window: {config['window']} | version: {config['version']}")
    return config

def load_and_validate_data(data_path, output_path, config):
    logging.info(f"loading data from {data_path}...")
    version = config.get('version', 'unknown')

    try:
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.lower().str.strip()
    except FileNotFoundError:
        msg = f"file not found: {data_path}"
        logging.error(msg)
        write_metrics(output_path, status="error", version=version, error_message=msg)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        msg = f"empty file: {data_path}"
        logging.error(msg)
        write_metrics(output_path, status="error", version=version, error_message=msg)
        sys.exit(1)
    except pd.errors.ParserError:
        msg = f"invalid csv: {data_path}"
        logging.error(msg)
        write_metrics(output_path, status="error", version=version, error_message=msg)
        sys.exit(1)
    except Exception as e:
        msg = f"unexpected read error: {e}"
        logging.error(msg)
        write_metrics(output_path, status="error", version=version, error_message=msg)
        sys.exit(1)

    if 'close' not in df.columns:
        msg = "missing column: 'close'"
        logging.error(msg)
        write_metrics(output_path, status="error", version=version, error_message=msg)
        sys.exit(1)

    logging.info(f"data loaded — {len(df)} rows")
    return df

def process_data(df, config):
    logging.info("processing...")
    window_size = config['window']

    df['rolling_mean'] = df['close'].rolling(window=window_size).mean()

    # drop nan boundary rows
    initial = len(df)
    df = df.dropna(subset=['rolling_mean']).copy()
    logging.info(f"dropped {initial - len(df)} nan rows")

    # binary signal
    df['signal'] = (df['close'] > df['rolling_mean']).astype(int)

    logging.info("signal generation done")
    return df

def main():
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(description="MLOps Batch Job - Task 0")
    parser.add_argument('--input', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--log-file', required=True, dest='log_file')
    args = parser.parse_args()

    setup_logging(args.log_file)
    logging.info("job started")
    logging.info(f"input: {args.input} | config: {args.config}")

    config = load_and_validate_config(args.config, args.output)
    df = load_and_validate_data(args.input, args.output, config)
    processed_df = process_data(df, config)

    rows_processed = len(processed_df)
    signal_rate = float(processed_df['signal'].mean())

    end_time = time.perf_counter()
    latency_ms = int((end_time - start_time) * 1000)

    logging.info(f"rows: {rows_processed} | signal_rate: {signal_rate:.4f} | latency: {latency_ms}ms")

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
    logging.info("job complete")
    sys.exit(0)

if __name__ == "__main__":
    main()
