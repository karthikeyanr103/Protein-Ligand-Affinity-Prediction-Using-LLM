from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from affinity.metrics import regression_metrics


def evaluate(artifact_directory: str) -> dict[str, float]:
    artifact = Path(artifact_directory)
    predictions = pd.read_csv(artifact / "test_predictions.csv")
    metrics = regression_metrics(
        predictions["label"].to_numpy(),
        predictions["prediction"].to_numpy(),
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Report held-out test metrics")
    parser.add_argument("--artifacts", default="/content/artifacts/affinity")
    args = parser.parse_args()
    evaluate(args.artifacts)


if __name__ == "__main__":
    main()
