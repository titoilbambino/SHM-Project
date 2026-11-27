"""Recording-level regression metrics used in 001."""

import numpy as np


def regression_scores(measured, predicted):
    if measured.shape != predicted.shape:
        raise ValueError("Measured and predicted strains must have the same shape")
    residuals = (measured - predicted) ** 2
    return {
        "rmse": float(np.sqrt(np.mean(residuals))),
        "r2": float(1 - residuals.sum() / np.sum((measured - measured.mean()) ** 2)),
    }


def mean_scores(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in ("rmse", "r2")}
