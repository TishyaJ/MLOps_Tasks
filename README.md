# MLOps Batch Job — Task 0

A minimal MLOps-style batch pipeline that loads OHLCV data, computes a rolling mean on `close`, generates a binary signal, and outputs structured metrics with full logging.

## Local Run

```bash
pip install -r requirements.txt

python run.py \
  --input data.csv \
  --config config.yaml \
  --output metrics.json \
  --log-file run.log
```

## Docker

```bash
docker build -t mlops-task .
docker run --rm mlops-task
```

The container runs the job on startup, prints metrics JSON to stdout, and exits 0 on success.

## Example Output — metrics.json

```json
{
    "version": "v1",
    "rows_processed": 9996,
    "metric": "signal_rate",
    "value": 0.4991,
    "latency_ms": 47,
    "seed": 42,
    "status": "success"
}
```

## Config (config.yaml)

| Key     | Value | Description                        |
|---------|-------|------------------------------------|
| seed    | 42    | RNG seed for reproducibility       |
| window  | 5     | Rolling mean window size           |
| version | v1    | Pipeline version tag               |

## Notes

- First `window - 1` rows are dropped (NaN rolling mean boundary).
- `rows_processed` reflects post-drop row count (9996 for window=5 on 10000 rows).
- Results are fully deterministic given the same config and data.
