"""Distribution shift analysis between KDDTrain+ and KDDTest+."""

from __future__ import annotations

import pandas as pd
from scipy.stats import ks_2samp

from src.preprocessing import ATTACK_CATEGORIES


NUMERIC_SHIFT_FEATURES = [
    "duration", "src_bytes", "dst_bytes", "count", "srv_count", "serror_rate",
]


def prevalence_table(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Compare class and category prevalence between official splits."""
    rows = []
    for name, df in [("train", train_df), ("test", test_df)]:
        rows.append({
            "split": name,
            "rows": len(df),
            "attack_rate": df["is_attack"].mean(),
            "normal_rate": 1 - df["is_attack"].mean(),
        })
    base = pd.DataFrame(rows)

    category_rows = []
    for split_name, df in [("train", train_df), ("test", test_df)]:
        counts = df["category"].value_counts(normalize=True)
        for category, rate in counts.items():
            category_rows.append({
                "split": split_name,
                "category": category,
                "prevalence": rate,
                "count": int(df["category"].eq(category).sum()),
            })
    category_df = pd.DataFrame(category_rows)

    protocol_rows = []
    for split_name, df in [("train", train_df), ("test", test_df)]:
        counts = df["protocol_type"].value_counts(normalize=True)
        for protocol, rate in counts.items():
            protocol_rows.append({
                "split": split_name,
                "protocol_type": protocol,
                "prevalence": rate,
                "count": int(df["protocol_type"].eq(protocol).sum()),
            })
    protocol_df = pd.DataFrame(protocol_rows)

    ks_rows = []
    for feature in NUMERIC_SHIFT_FEATURES:
        stat, pvalue = ks_2samp(train_df[feature], test_df[feature])
        ks_rows.append({
            "feature": feature,
            "ks_statistic": stat,
            "ks_pvalue": pvalue,
            "train_mean": train_df[feature].mean(),
            "test_mean": test_df[feature].mean(),
            "train_median": train_df[feature].median(),
            "test_median": test_df[feature].median(),
        })
    ks_df = pd.DataFrame(ks_rows)

    summary = pd.concat(
        [
            base.assign(metric_type="binary_prevalence"),
            category_df.assign(metric_type="category_prevalence"),
            protocol_df.assign(metric_type="protocol_prevalence"),
            ks_df.assign(metric_type="numeric_ks"),
        ],
        ignore_index=True,
        sort=False,
    )
    return summary


def shift_explanation() -> str:
    return (
        "Train/test shift matters in IDS because models trained on one prevalence mix "
        "may look better or worse on another. NSL-KDD test has a higher attack rate "
        "than train, so accuracy and precision/recall trade-offs can change even when "
        "ranking quality stays similar."
    )
