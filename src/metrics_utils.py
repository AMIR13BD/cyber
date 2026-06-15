"""Evaluation helpers for intrusion detection experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_report_dict(y_true, y_pred, y_score=None) -> dict[str, float | list[list[int]]]:
    """Compute standard classification metrics for binary intrusion detection."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: dict[str, float | list[list[int]]] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
    return metrics


def metrics_to_frame(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert nested metric dictionaries into a comparison table."""
    return pd.DataFrame(results).T.sort_values("f1", ascending=False)
