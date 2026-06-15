"""Run NSL-KDD intrusion detection reproduction pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.autoencoder import train_and_evaluate_autoencoder
from src.error_analysis import category_error_summary
from src.metrics_utils import classification_report_dict, metrics_to_frame
from src.plotting import (
    plot_class_imbalance,
    plot_confusion_matrices,
    plot_metric_comparison,
    plot_reconstruction_error_distribution,
)
from src.preprocessing import RANDOM_STATE, build_feature_matrix, load_nslkdd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"


def train_supervised_models(x_train, y_train, x_test, y_test):
    """Train and evaluate supervised classifiers."""
    results = {}
    predictions = {}

    print("[1/4] Training Logistic Regression...")
    t0 = time.perf_counter()
    lr_model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    lr_model.fit(x_train, y_train)
    lr_pred = lr_model.predict(x_test)
    lr_score = lr_model.predict_proba(x_test)[:, 1]
    results["logistic_regression"] = classification_report_dict(y_test, lr_pred, lr_score)
    predictions["logistic_regression"] = lr_pred
    lr_time = time.perf_counter() - t0
    print(f"      Done in {lr_time:.1f}s")

    print("[2/4] Training Random Forest (100 trees)...")
    t0 = time.perf_counter()
    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_model.fit(x_train, y_train)
    rf_pred = rf_model.predict(x_test)
    rf_score = rf_model.predict_proba(x_test)[:, 1]
    results["random_forest"] = classification_report_dict(y_test, rf_pred, rf_score)
    predictions["random_forest"] = rf_pred
    rf_time = time.perf_counter() - t0
    print(f"      Done in {rf_time:.1f}s")

    return results, predictions, {"logistic_regression": lr_time, "random_forest": rf_time}


def train_isolation_forest(x_train_normal, x_test, y_test):
    """Train Isolation Forest on normal traffic only (unsupervised IDS baseline)."""
    print("[3/4] Training Isolation Forest on normal traffic...")
    t0 = time.perf_counter()
    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train_normal)
    raw_scores = -model.score_samples(x_test)
    threshold = np.percentile(-model.score_samples(x_train_normal), 95)
    y_pred = (raw_scores >= threshold).astype(int)
    score = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    metrics = classification_report_dict(y_test, y_pred, score)
    elapsed = time.perf_counter() - t0
    print(f"      Done in {elapsed:.1f}s")
    return metrics, y_pred, float(threshold), elapsed


def main() -> None:
    pipeline_start = time.perf_counter()
    step_times: dict[str, float] = {}

    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading NSL-KDD data...")
    t0 = time.perf_counter()
    train_df = load_nslkdd(str(DATA_DIR / "KDDTrain+.txt"))
    test_df = load_nslkdd(str(DATA_DIR / "KDDTest+.txt"))
    x_train, x_test, feature_cols, _ = build_feature_matrix(train_df, test_df)
    y_train = train_df["is_attack"].to_numpy()
    y_test = test_df["is_attack"].to_numpy()
    test_categories = test_df["category"]
    step_times["data_loading"] = time.perf_counter() - t0
    print(f"      Done in {step_times['data_loading']:.1f}s")

    plot_class_imbalance(train_df, FIG_DIR / "class_imbalance.png")

    x_train_fit, _, y_train_fit, _ = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    supervised_results, supervised_preds, supervised_times = train_supervised_models(
        x_train_fit, y_train_fit, x_test, y_test
    )
    step_times.update(supervised_times)

    normal_mask = y_train_fit == 0
    x_train_normal_all = x_train_fit[normal_mask]
    x_train_normal, x_val_normal = train_test_split(
        x_train_normal_all,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    iforest_results, iforest_pred, if_threshold, if_time = train_isolation_forest(
        x_train_normal, x_test, y_test
    )
    step_times["isolation_forest"] = if_time

    print("[4/4] Training PyTorch Autoencoder on normal traffic (max 20 epochs, early stopping)...")
    t0 = time.perf_counter()
    _, ae_pred, ae_score, ae_threshold, ae_errors = train_and_evaluate_autoencoder(
        x_train_normal,
        x_val_normal,
        x_test,
        y_test,
        input_dim=x_train.shape[1],
    )
    autoencoder_results = classification_report_dict(y_test, ae_pred, ae_score)
    step_times["autoencoder"] = time.perf_counter() - t0
    print(f"      Done in {step_times['autoencoder']:.1f}s")

    all_results = {
        **supervised_results,
        "isolation_forest": iforest_results,
        "autoencoder": autoencoder_results,
    }
    all_predictions = {
        **supervised_preds,
        "isolation_forest": iforest_pred,
        "autoencoder": ae_pred,
    }

    error_by_model = {
        name: category_error_summary(y_test, pred, test_categories)
        for name, pred in all_predictions.items()
    }

    summary = metrics_to_frame(
        {name: {k: v for k, v in metrics.items() if k != "confusion_matrix"} for name, metrics in all_results.items()}
    )
    summary.to_csv(RESULTS_DIR / "model_comparison.csv")

    plot_reconstruction_error_distribution(
        ae_errors,
        y_test,
        ae_threshold,
        FIG_DIR / "autoencoder_reconstruction_error.png",
    )
    plot_confusion_matrices(y_test, all_predictions, FIG_DIR / "confusion_matrices.png")
    plot_metric_comparison(summary, FIG_DIR / "model_metric_comparison.png")

    total_runtime = time.perf_counter() - pipeline_start
    runtime_summary = {
        **{k: round(v, 2) for k, v in step_times.items()},
        "total_seconds": round(total_runtime, 2),
    }

    payload = {
        "dataset": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_attack_rate": float(train_df["is_attack"].mean()),
            "test_attack_rate": float(test_df["is_attack"].mean()),
            "num_features_after_encoding": len(feature_cols),
        },
        "thresholds": {
            "autoencoder": ae_threshold,
            "isolation_forest": if_threshold,
        },
        "metrics": all_results,
        "error_by_category": error_by_model,
        "runtime_seconds": runtime_summary,
    }
    with open(RESULTS_DIR / "experiment_results.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("\n=== Model Comparison (test set) ===")
    print(summary.round(4).to_string())
    print("\n=== Runtime Summary (seconds) ===")
    for step, seconds in runtime_summary.items():
        print(f"  {step}: {seconds}")
    print(f"\nPipeline finished in {total_runtime:.1f}s ({total_runtime / 60:.1f} min)")


if __name__ == "__main__":
    main()
