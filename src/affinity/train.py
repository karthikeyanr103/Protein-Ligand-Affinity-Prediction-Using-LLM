from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from affinity.config import load_config
from affinity.metrics import regression_metrics
from affinity.model import AffinityRegressor
from affinity.pipeline import save_metadata, standardize_apply, standardize_fit


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_split(directory: Path, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(directory / f"{split}.csv")
    features = np.load(directory / f"{split}_features.npz")["features"].astype(np.float32)
    if len(frame) != len(features):
        raise ValueError(f"{split} rows and feature rows do not match")
    return frame, features


def _predict(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(features).to(device)).cpu().numpy()


def train(config_path: str) -> dict[str, float]:
    config = load_config(config_path)
    _seed_everything(config.data.seed)
    dataset = Path(config.data.path)
    output = Path(config.output.directory)
    output.mkdir(parents=True, exist_ok=True)

    train_frame, train_raw = _load_split(dataset, "train")
    validation_frame, validation_raw = _load_split(dataset, "validation")
    test_frame, test_raw = _load_split(dataset, "test")
    train_features, mean, scale = standardize_fit(train_raw)
    validation_features = standardize_apply(validation_raw, mean, scale)
    test_features = standardize_apply(test_raw, mean, scale)
    train_targets = train_frame["label"].to_numpy(dtype=np.float32)
    validation_targets = validation_frame["label"].to_numpy(dtype=np.float32)
    test_targets = test_frame["label"].to_numpy(dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AffinityRegressor(
        input_dim=train_features.shape[1],
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_features),
            torch.from_numpy(train_targets),
        ),
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    best_rmse = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.training.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_features), batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_features)
        validation_predictions = _predict(model, validation_features, device)
        validation_metrics = regression_metrics(
            validation_targets,
            validation_predictions,
        )
        record = {
            "epoch": epoch,
            "train_mse": total_loss / len(train_features),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(record)
        print(json.dumps(record))
        if validation_metrics["rmse"] < best_rmse:
            best_rmse = validation_metrics["rmse"]
            stale_epochs = 0
            torch.save(model.state_dict(), output / "model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= config.training.patience:
                break

    model.load_state_dict(torch.load(output / "model.pt", map_location=device, weights_only=True))
    test_predictions = _predict(model, test_features, device)
    test_metrics = regression_metrics(test_targets, test_predictions)
    np.savez_compressed(output / "normalization.npz", mean=mean, scale=scale)
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    dataset_metadata = json.loads(
        (dataset / "dataset_metadata.json").read_text(encoding="utf-8")
    )
    save_metadata(
        output / "metadata.json",
        {
            "input_dim": train_features.shape[1],
            "hidden_dims": config.model.hidden_dims,
            "dropout": config.model.dropout,
            "features": dataset_metadata["features"],
            "split_strategy": dataset_metadata["split_strategy"],
            "test_metrics": test_metrics,
        },
    )
    pd.DataFrame(
        {
            "label": test_targets,
            "prediction": test_predictions,
        }
    ).to_csv(output / "test_predictions.csv", index=False)
    print(json.dumps({"test": test_metrics}, indent=2))
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the affinity regression head")
    parser.add_argument("--config", default="configs/colab.toml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
