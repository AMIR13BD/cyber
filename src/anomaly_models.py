"""Train and score anomaly-detection models with percentile thresholds."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from src.autoencoder import (
    TabularAutoencoder,
    fit_threshold,
    predict_from_errors,
    reconstruction_errors,
    train_autoencoder,
)
from src.metrics_utils import classification_report_dict
from src.preprocessing import RANDOM_STATE


def isolation_forest_scores(model: IsolationForest, x: np.ndarray) -> np.ndarray:
    """Higher score means more anomalous."""
    return -model.score_samples(x)


def train_isolation_forest_model(
    x_normal_train: np.ndarray,
    n_estimators: int = 200,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_normal_train)
    return model


def percentile_threshold(scores: np.ndarray, percentile: float) -> float:
    return float(np.percentile(scores, percentile))


def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    """Binary predictions from anomaly scores and a fixed threshold."""
    normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    y_pred = (scores >= threshold).astype(int)
    return classification_report_dict(y_true, y_pred, normalized)


def train_autoencoder_source(
    x_normal_train: np.ndarray,
    x_normal_val: np.ndarray,
    input_dim: int,
    epochs: int = 50,
    early_stopping: bool = False,
) -> tuple[TabularAutoencoder, dict]:
    """Train autoencoder using source-protocol settings."""
    meta: dict = {"epochs_requested": epochs, "early_stopping": early_stopping}
    model, _ = train_autoencoder(
        x_normal_train,
        input_dim=input_dim,
        epochs=epochs,
        x_val_normal=x_normal_val if early_stopping else None,
        patience=3,
    )
    meta["epochs_completed"] = epochs
    return model, meta


def autoencoder_validation_errors(model: TabularAutoencoder, x_normal_val: np.ndarray) -> np.ndarray:
    return reconstruction_errors(model, x_normal_val)


def autoencoder_test_errors(model: TabularAutoencoder, x_test: np.ndarray) -> np.ndarray:
    return reconstruction_errors(model, x_test)
