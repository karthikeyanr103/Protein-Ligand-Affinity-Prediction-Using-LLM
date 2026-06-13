import numpy as np

from affinity.metrics import regression_metrics


def test_perfect_predictions():
    values = np.array([2.0, 4.0, 6.0])
    metrics = regression_metrics(values, values)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["pearson"] == 1.0

