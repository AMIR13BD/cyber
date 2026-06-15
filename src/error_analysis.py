"""Per-attack-category error analysis for IDS evaluation."""

from __future__ import annotations

import pandas as pd


def category_error_summary(
    y_true,
    y_pred,
    categories: pd.Series,
) -> dict[str, dict[str, int]]:
    """Summarize false negatives and false positives by attack category."""
    frame = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "category": categories.values,
    })
    fn = (
        frame[(frame["y_true"] == 1) & (frame["y_pred"] == 0)]
        .groupby("category")
        .size()
        .sort_values(ascending=False)
    )
    fp = (
        frame[(frame["y_true"] == 0) & (frame["y_pred"] == 1)]
        .groupby("category")
        .size()
        .sort_values(ascending=False)
    )
    return {
        "false_negatives": fn.astype(int).to_dict(),
        "false_positives": fp.astype(int).to_dict(),
    }


def category_recall_table(
    y_true,
    y_pred,
    categories: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    """
    Per-category recall for attack categories.
    recall = TP_in_category / total_samples_in_category
    """
    frame = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "category": categories.values,
    })
    rows = []
    attack_frame = frame[frame["category"] != "normal"]
    for category, group in attack_frame.groupby("category"):
        total = len(group)
        true_positives = int(((group["y_true"] == 1) & (group["y_pred"] == 1)).sum())
        false_negatives = int(((group["y_true"] == 1) & (group["y_pred"] == 0)).sum())
        rows.append({
            "model": model_name,
            "category": category,
            "total_samples": total,
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "recall": true_positives / total if total else 0.0,
        })
    return pd.DataFrame(rows)
