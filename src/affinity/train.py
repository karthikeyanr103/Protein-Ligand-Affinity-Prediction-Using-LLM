from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from affinity.config import load_config
from affinity.data import assign_splits, load_dataset
from affinity.metrics import regression_metrics
from affinity.model import AffinityRegressor
from affinity.pipeline import build_features, save_metadata, standardize_apply, standardize_fit


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict(model: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(features).to(device)
        return model(tensor).cpu().numpy()


def train(config_path: str) -> dict[str, float]:
    config = load_config(config_path)
    _seed_everything(config.data.seed)
    output = Path(config.output.directory)
    output.mkdir(parents=True, exist_ok=True)

    frame = load_dataset(config.data.path)
    frame["split"] = assign_splits(
        frame,
        strategy=config.data.split_strategy,
        train_fraction=config.data.train_fraction,
        validation_fraction=config.data.validation_fraction,
        seed=config.data.seed,
    )
    if set(frame["split"]) != {"train", "validation", "test"}:
        raise ValueError("At least one split is empty; use more data or another seed")

    raw_features, feature_metadata = build_features(
        frame["protein_sequence"].tolist(),
        frame["compound_smiles"].tolist(),
        config.features.protein_embedding_path,
        config.features.molecule_embedding_path,
    )
    train_mask = frame["split"].eq("train").to_numpy()
    features = np.empty_like(raw_features)
    features[train_mask], mean, scale = standardize_fit(raw_features[train_mask])
    features[~train_mask] = standardize_apply(raw_features[~train_mask], mean, scale)
    targets = frame["label"].to_numpy(dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AffinityRegressor(
        input_dim=features.shape[1],
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(features[train_mask]),
            torch.from_numpy(targets[train_mask]),
        ),
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_mask = frame["split"].eq("validation").to_numpy()
    best_rmse = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_features), batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_features)
        predictions = _predict(model, features[validation_mask], device)
        validation_metrics = regression_metrics(targets[validation_mask], predictions)
        epoch_record = {
            "epoch": float(epoch),
            "train_mse": total_loss / int(train_mask.sum()),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record))

        if validation_metrics["rmse"] < best_rmse:
            best_rmse = validation_metrics["rmse"]
            stale_epochs = 0
            torch.save(model.state_dict(), output / "model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= config.training.patience:
                break

    model.load_state_dict(torch.load(output / "model.pt", map_location=device, weights_only=True))
    test_mask = frame["split"].eq("test").to_numpy()
    test_predictions = _predict(model, features[test_mask], device)
    test_metrics = regression_metrics(targets[test_mask], test_predictions)
    np.savez_compressed(output / "normalization.npz", mean=mean, scale=scale)
    frame.loc[:, ["protein_sequence", "compound_smiles", "label", "split"]].to_csv(
        output / "splits.csv", index=False
    )
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_metadata(
        output / "metadata.json",
        {
            "input_dim": features.shape[1],
            "hidden_dims": config.model.hidden_dims,
            "dropout": config.model.dropout,
            "split_strategy": config.data.split_strategy,
            "features": feature_metadata,
            "protein_embedding_path": config.features.protein_embedding_path,
            "molecule_embedding_path": config.features.molecule_embedding_path,
            "test_metrics": test_metrics,
        },
    )
    print(json.dumps({"test": test_metrics}, indent=2))
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an affinity regression model")
    parser.add_argument("--config", default="configs/baseline.toml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
