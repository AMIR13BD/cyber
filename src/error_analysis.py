"""Per-attack-category error analysis for IDS evaluation."""

from __future__ import annotations

import pandas as pd


def category_error_summary(
    y_true,
    y_pred,
    categories: pd.Series,
) -> dict[str, dict[str, dict[str, int]]]:
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
