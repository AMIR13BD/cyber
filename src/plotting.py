"""Plot helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def plot_reconstruction_error_distribution(
    errors: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """Plot normal vs attack reconstruction error distributions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    normal_errors = errors[y_true == 0]
    attack_errors = errors[y_true == 1]
    ax.hist(normal_errors, bins=60, alpha=0.65, label="Normal", color="#2ecc71", density=True)
    ax.hist(attack_errors, bins=60, alpha=0.65, label="Attack", color="#e74c3c", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold:.4f})")
    ax.set_title("Autoencoder Reconstruction Error Distribution")
    ax.set_xlabel("MSE Reconstruction Error")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_class_imbalance(train_df, output_path: Path) -> None:
    """Plot binary and multi-class attack prevalence."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    train_df["is_attack"].value_counts().plot(kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"])
    axes[0].set_title("Binary Class Balance (Train)")
    axes[0].set_xticklabels(["Normal", "Attack"], rotation=0)
    train_df["category"].value_counts().plot(kind="bar", ax=axes[1], color="steelblue")
    axes[1].set_title("Attack Category Distribution (Train)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_threshold_sensitivity(sensitivity_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot metric curves vs threshold percentile for anomaly models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["precision", "recall", "f1", "f2", "fpr"]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        for model, group in sensitivity_df.groupby("model"):
            ax.plot(group["percentile"], group[metric], marker="o", label=model)
        ax.set_title(f"{metric.upper()} vs Threshold Percentile")
        ax.set_xlabel("Percentile on normal_validation scores")
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0, 1.05)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"threshold_{metric}.png", dpi=150)
        plt.close(fig)


def plot_distribution_shift(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot attack-rate and key numeric shift indicators."""
    binary = summary_df[summary_df["metric_type"] == "binary_prevalence"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(binary["split"] + "_attack", binary["attack_rate"], color=["#3498db", "#e74c3c"])
    ax.set_title("Attack Prevalence Shift: KDDTrain+ vs KDDTest+")
    ax.set_ylabel("Attack rate")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Save confusion matrices for all models in one figure."""
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
        ax.set_title(name.replace("_", " ").title())
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metric_comparison(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart comparing F1 and recall across models."""
    plot_df = metrics_df[["f1", "recall", "precision"]].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", ax=ax, rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison on NSL-KDD Test Set")
    ax.set_ylabel("Score")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
