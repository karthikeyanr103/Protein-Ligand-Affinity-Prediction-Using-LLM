from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from affinity.data import load_dataset
from affinity.metrics import regression_metrics
from affinity.model import AffinityRegressor
from affinity.pipeline import build_features, load_metadata, standardize_apply


def evaluate(artifact_directory: str, data_path: str = "") -> dict[str, float]:
    artifact = Path(artifact_directory)
    metadata = load_metadata(artifact / "metadata.json")
    if data_path:
        frame = load_dataset(data_path)
    else:
        frame = pd.read_csv(artifact / "splits.csv")
        frame = frame.loc[frame["split"].eq("test")].copy()
    features, _ = build_features(
        frame["protein_sequence"].tolist(),
        frame["compound_smiles"].tolist(),
        metadata.get("protein_embedding_path", ""),
        metadata.get("molecule_embedding_path", ""),
    )
    normalization = np.load(artifact / "normalization.npz")
    features = standardize_apply(features, normalization["mean"], normalization["scale"])
    model = AffinityRegressor(
        metadata["input_dim"],
        metadata["hidden_dims"],
        metadata["dropout"],
    )
    model.load_state_dict(torch.load(artifact / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(features)).numpy()
    metrics = regression_metrics(frame["label"].to_numpy(), predictions)
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--artifacts", default="artifacts/baseline")
    parser.add_argument("--data", default="")
    args = parser.parse_args()
    evaluate(args.artifacts, args.data)


if __name__ == "__main__":
    main()
