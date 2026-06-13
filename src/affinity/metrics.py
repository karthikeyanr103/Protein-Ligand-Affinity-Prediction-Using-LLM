from __future__ import annotations

import numpy as np


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    errors = predictions - targets
    mse = float(np.mean(errors**2))
    target_centered = targets - targets.mean()
    prediction_centered = predictions - predictions.mean()
    denominator = np.sqrt(np.sum(target_centered**2) * np.sum(prediction_centered**2))
    pearson = (
        float(np.sum(target_centered * prediction_centered) / denominator)
        if denominator
        else 0.0
    )
    ss_total = float(np.sum(target_centered**2))
    r2 = 1.0 - float(np.sum(errors**2)) / ss_total if ss_total else 0.0
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(mse)),
        "r2": r2,
        "pearson": pearson,
    }
