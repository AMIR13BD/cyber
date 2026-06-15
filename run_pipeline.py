"""Run NSL-KDD intrusion detection reproduction pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.anomaly_models import (
    evaluate_scores,
    isolation_forest_scores,
    percentile_threshold,
    train_isolation_forest_model,
)
from src.autoencoder import (
    reconstruction_errors,
    train_and_evaluate_autoencoder,
    train_autoencoder,
)
from src.distribution_shift import prevalence_table
from src.error_analysis import category_recall_table
from src.metrics_utils import classification_report_dict, metrics_to_frame
from src.plotting import (
    plot_class_imbalance,
    plot_confusion_matrices,
    plot_distribution_shift,
    plot_metric_comparison,
    plot_reconstruction_error_distribution,
    plot_threshold_sensitivity,
)
from src.preprocessing import (
    RANDOM_STATE,
    build_preprocess_bundle,
    load_nslkdd,
    save_preprocessing_summary,
    split_normal_rows,
    transform_splits,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

SOURCE_AE_EPOCHS = 50
SOURCE_IF_TREES = 200
SOURCE_PERCENTILE = 99
SENSITIVITY_PERCENTILES = [95, 97, 99]
RF_TREES = 100


def train_supervised_models(x_train, y_train, x_test, y_test):
    """Train supervised models on all official KDDTrain+ rows."""
    results = {}
    predictions = {}
    timings = {}

    print("[supervised] Training Logistic Regression on full KDDTrain+...")
    t0 = time.perf_counter()
    lr_model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="liblinear",
    )
    lr_model.fit(x_train, y_train)
    lr_pred = lr_model.predict(x_test)
    lr_score = lr_model.predict_proba(x_test)[:, 1]
    results["logistic_regression"] = classification_report_dict(y_test, lr_pred, lr_score)
    predictions["logistic_regression"] = lr_pred
    timings["logistic_regression"] = time.perf_counter() - t0
    print(f"      Done in {timings['logistic_regression']:.1f}s")

    print(f"[supervised] Training Random Forest ({RF_TREES} trees) on full KDDTrain+...")
    t0 = time.perf_counter()
    rf_model = RandomForestClassifier(
        n_estimators=RF_TREES,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_model.fit(x_train, y_train)
    rf_pred = rf_model.predict(x_test)
    rf_score = rf_model.predict_proba(x_test)[:, 1]
    results["random_forest"] = classification_report_dict(y_test, rf_pred, rf_score)
    predictions["random_forest"] = rf_pred
    timings["random_forest"] = time.perf_counter() - t0
    print(f"      Done in {timings['random_forest']:.1f}s")

    return results, predictions, timings


def run_source_protocol(
    x_normal_train,
    x_normal_val,
    x_test,
    y_test,
    input_dim: int,
) -> tuple[dict, dict, dict, np.ndarray, np.ndarray, dict, dict]:
    """Reproduce source tutorial protocol: 50 AE epochs, 200 IF trees, 99th percentile."""
    print("[source_protocol] Training Isolation Forest (200 trees, normal_train only)...")
    t0 = time.perf_counter()
    iforest = train_isolation_forest_model(x_normal_train, n_estimators=SOURCE_IF_TREES)
    if_val_scores = isolation_forest_scores(iforest, x_normal_val)
    if_test_scores = isolation_forest_scores(iforest, x_test)
    if_threshold = percentile_threshold(if_val_scores, SOURCE_PERCENTILE)
    if_metrics = evaluate_scores(y_test, if_test_scores, if_threshold)
    if_time = time.perf_counter() - t0
    print(f"      Done in {if_time:.1f}s")

    print(f"[source_protocol] Training Autoencoder ({SOURCE_AE_EPOCHS} epochs, no early stopping)...")
    t0 = time.perf_counter()
    ae_model, ae_meta = train_autoencoder(
        x_normal_train,
        input_dim=input_dim,
        epochs=SOURCE_AE_EPOCHS,
        x_val_normal=None,
    )
    ae_val_errors = reconstruction_errors(ae_model, x_normal_val)
    ae_test_errors = reconstruction_errors(ae_model, x_test)
    ae_threshold = percentile_threshold(ae_val_errors, SOURCE_PERCENTILE)
    ae_metrics = evaluate_scores(y_test, ae_test_errors, ae_threshold)
    ae_time = time.perf_counter() - t0
    ae_meta["threshold_percentile"] = SOURCE_PERCENTILE
    ae_meta["threshold_value"] = ae_threshold
    print(f"      Done in {ae_time:.1f}s (epochs_completed={ae_meta['epochs_completed']})")

    metrics = {
        "autoencoder_source_protocol": ae_metrics,
        "isolation_forest_source_protocol": if_metrics,
    }
    thresholds = {
        "autoencoder": ae_threshold,
        "isolation_forest": if_threshold,
        "percentile": SOURCE_PERCENTILE,
    }
    timings = {"autoencoder": ae_time, "isolation_forest": if_time}
    return metrics, thresholds, timings, ae_test_errors, if_test_scores, ae_meta, {
        "ae_val_errors": ae_val_errors,
        "if_val_scores": if_val_scores,
        "ae_model": ae_model,
        "iforest": iforest,
    }


def threshold_sensitivity_rows(
    model_name: str,
    y_test: np.ndarray,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    percentiles: list[int],
) -> list[dict]:
    rows = []
    for percentile in percentiles:
        threshold = percentile_threshold(val_scores, percentile)
        metrics = evaluate_scores(y_test, test_scores, threshold)
        row = {
            "model": model_name,
            "percentile": percentile,
            "threshold": threshold,
        }
        row.update({k: v for k, v in metrics.items() if k != "confusion_matrix"})
        rows.append(row)
    return rows


def equal_fpr_threshold(val_scores: np.ndarray, target_fpr: float) -> float:
    """Choose threshold so flagged fraction on validation-normal equals target FPR."""
    return float(np.quantile(val_scores, 1 - target_fpr))


def run_feature_ablation(
    train_df,
    test_df,
    normal_train_df,
    normal_val_df,
    y_train,
    y_test,
) -> pd.DataFrame:
    configs = [
        ("A_full_features", False, False),
        ("B_remove_constants", True, False),
        ("C_remove_constants_and_redundant", True, True),
    ]
    rows = []
    for name, remove_constants, remove_correlated in configs:
        print(f"[ablation] {name}...")
        bundle = build_preprocess_bundle(
            train_df,
            normal_train_df,
            remove_constants=remove_constants,
            remove_correlated=remove_correlated,
        )
        matrices = transform_splits(bundle, train_df, test_df, normal_train_df, normal_val_df)
        input_dim = matrices["x_normal_train"].shape[1]

        rf = RandomForestClassifier(
            n_estimators=RF_TREES,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(matrices["x_train_supervised"], y_train)
        rf_pred = rf.predict(matrices["x_test_supervised"])
        rf_score = rf.predict_proba(matrices["x_test_supervised"])[:, 1]
        rf_metrics = classification_report_dict(y_test, rf_pred, rf_score)

        _, ae_pred, ae_score, _, _, _ = train_and_evaluate_autoencoder(
            matrices["x_normal_train"],
            matrices["x_normal_val"],
            matrices["x_test_anomaly"],
            y_test,
            input_dim=input_dim,
            epochs=20,
            percentile=SOURCE_PERCENTILE,
            early_stopping=False,
        )
        ae_metrics = classification_report_dict(y_test, ae_pred, ae_score)

        for model_name, metrics in [("random_forest", rf_metrics), ("autoencoder", ae_metrics)]:
            rows.append({
                "feature_set": name,
                "num_features": input_dim,
                "model": model_name,
                **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            })
    return pd.DataFrame(rows)


def main() -> None:
    pipeline_start = time.perf_counter()
    step_times: dict[str, float] = {}

    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading NSL-KDD and building strict preprocessing...")
    t0 = time.perf_counter()
    train_df = load_nslkdd(DATA_DIR / "KDDTrain+.txt")
    test_df = load_nslkdd(DATA_DIR / "KDDTest+.txt")
    normal_train_df, normal_val_df = split_normal_rows(train_df)

    bundle = build_preprocess_bundle(train_df, normal_train_df, remove_constants=True, remove_correlated=False)
    save_preprocessing_summary(bundle, RESULTS_DIR / "preprocessing_summary.json")
    matrices = transform_splits(bundle, train_df, test_df, normal_train_df, normal_val_df)

    y_train = train_df["is_attack"].to_numpy()
    y_test = test_df["is_attack"].to_numpy()
    test_categories = test_df["category"]
    input_dim = matrices["x_normal_train"].shape[1]
    step_times["data_preprocessing"] = time.perf_counter() - t0
    print(f"      Done in {step_times['data_preprocessing']:.1f}s ({input_dim} features)")

    plot_class_imbalance(train_df, FIG_DIR / "class_imbalance.png")

    shift_df = prevalence_table(train_df, test_df)
    shift_df.to_csv(RESULTS_DIR / "distribution_shift_summary.csv", index=False)
    plot_distribution_shift(shift_df, FIG_DIR / "train_test_shift.png")

    supervised_results, supervised_preds, supervised_times = train_supervised_models(
        matrices["x_train_supervised"],
        y_train,
        matrices["x_test_supervised"],
        y_test,
    )
    step_times.update(supervised_times)

    source_metrics, source_thresholds, source_times, ae_errors, if_scores, ae_meta, artifacts = run_source_protocol(
        matrices["x_normal_train"],
        matrices["x_normal_val"],
        matrices["x_test_anomaly"],
        y_test,
        input_dim,
    )
    step_times.update({f"source_{k}": v for k, v in source_times.items()})

    ae_pred = (ae_errors >= source_thresholds["autoencoder"]).astype(int)
    if_pred = (if_scores >= source_thresholds["isolation_forest"]).astype(int)

    print("[analysis] Threshold sensitivity (95/97/99 percentiles)...")
    sensitivity_rows = []
    sensitivity_rows.extend(
        threshold_sensitivity_rows(
            "autoencoder",
            y_test,
            artifacts["ae_val_errors"],
            ae_errors,
            SENSITIVITY_PERCENTILES,
        )
    )
    sensitivity_rows.extend(
        threshold_sensitivity_rows(
            "isolation_forest",
            y_test,
            artifacts["if_val_scores"],
            if_scores,
            SENSITIVITY_PERCENTILES,
        )
    )
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(RESULTS_DIR / "threshold_sensitivity.csv", index=False)
    plot_threshold_sensitivity(sensitivity_df, FIG_DIR)

    print("[analysis] Fair anomaly comparison at matched percentiles and equal FPR...")
    fair_rows = []
    for percentile in SENSITIVITY_PERCENTILES:
        ae_thr = percentile_threshold(artifacts["ae_val_errors"], percentile)
        if_thr = percentile_threshold(artifacts["if_val_scores"], percentile)
        for model_name, scores, thr in [
            ("autoencoder", ae_errors, ae_thr),
            ("isolation_forest", if_scores, if_thr),
        ]:
            metrics = evaluate_scores(y_test, scores, thr)
            fair_rows.append({
                "comparison_type": f"same_percentile_{percentile}",
                "model": model_name,
                "percentile": percentile,
                "threshold": thr,
                **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            })

    if_target_fpr = float((artifacts["if_val_scores"] >= source_thresholds["isolation_forest"]).mean())
    ae_equal_thr = equal_fpr_threshold(artifacts["ae_val_errors"], if_target_fpr)
    for model_name, scores, thr in [
        ("autoencoder_equal_fpr", ae_errors, ae_equal_thr),
        ("isolation_forest_reference_99", if_scores, source_thresholds["isolation_forest"]),
    ]:
        metrics = evaluate_scores(y_test, scores, thr)
        fair_rows.append({
            "comparison_type": "equal_fpr_on_normal_validation",
            "model": model_name,
            "percentile": SOURCE_PERCENTILE,
            "threshold": thr,
            "target_fpr_on_val": if_target_fpr,
            **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
        })
    fair_df = pd.DataFrame(fair_rows)
    fair_df.to_csv(RESULTS_DIR / "fair_anomaly_comparison.csv", index=False)

    print("[analysis] Feature ablation...")
    t0 = time.perf_counter()
    ablation_df = run_feature_ablation(
        train_df, test_df, normal_train_df, normal_val_df, y_train, y_test
    )
    ablation_df.to_csv(RESULTS_DIR / "feature_ablation.csv", index=False)
    step_times["feature_ablation"] = time.perf_counter() - t0

    all_results = {
        **supervised_results,
        **source_metrics,
    }
    all_predictions = {
        **supervised_preds,
        "autoencoder_source_protocol": ae_pred,
        "isolation_forest_source_protocol": if_pred,
    }

    recall_tables = [
        category_recall_table(y_test, pred, test_categories, name)
        for name, pred in all_predictions.items()
    ]
    category_recall_df = pd.concat(recall_tables, ignore_index=True)
    category_recall_df.to_csv(RESULTS_DIR / "category_recall_analysis.csv", index=False)

    summary = metrics_to_frame(
        {name: {k: v for k, v in metrics.items() if k != "confusion_matrix"} for name, metrics in all_results.items()}
    )
    summary.to_csv(RESULTS_DIR / "model_comparison.csv")
    pd.DataFrame(
        {k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"} for k, v in source_metrics.items()}
    ).T.to_csv(RESULTS_DIR / "source_protocol_model_comparison.csv")

    with open(RESULTS_DIR / "source_protocol_thresholds.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                **source_thresholds,
                "autoencoder_training": ae_meta,
                "calibration_split": "normal_validation_from_kddtrain",
                "isolation_forest_trees": SOURCE_IF_TREES,
                "autoencoder_epochs": SOURCE_AE_EPOCHS,
            },
            handle,
            indent=2,
        )

    plot_reconstruction_error_distribution(
        ae_errors,
        y_test,
        source_thresholds["autoencoder"],
        FIG_DIR / "autoencoder_reconstruction_error.png",
    )
    plot_confusion_matrices(y_test, all_predictions, FIG_DIR / "confusion_matrices.png")
    plot_metric_comparison(summary, FIG_DIR / "model_metric_comparison.png")

    total_runtime = time.perf_counter() - pipeline_start
    runtime_summary = {**{k: round(v, 2) for k, v in step_times.items()}, "total_seconds": round(total_runtime, 2)}

    payload = {
        "dataset": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "normal_train_rows": int(len(normal_train_df)),
            "normal_validation_rows": int(len(normal_val_df)),
            "train_attack_rate": float(train_df["is_attack"].mean()),
            "test_attack_rate": float(test_df["is_attack"].mean()),
            "num_features_after_encoding": input_dim,
        },
        "preprocessing": json.loads((RESULTS_DIR / "preprocessing_summary.json").read_text(encoding="utf-8")),
        "source_protocol_thresholds": source_thresholds,
        "metrics": all_results,
        "runtime_seconds": runtime_summary,
    }
    with open(RESULTS_DIR / "experiment_results.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("\n=== Model Comparison (test set) ===")
    print(summary.round(4).to_string())
    print("\n=== Source Protocol @ 99th percentile ===")
    print(pd.DataFrame(source_metrics).T.round(4).to_string())
    print("\n=== Runtime Summary (seconds) ===")
    for step, seconds in runtime_summary.items():
        print(f"  {step}: {seconds}")
    print(f"\nPipeline finished in {total_runtime:.1f}s ({total_runtime / 60:.1f} min)")


if __name__ == "__main__":
    main()
