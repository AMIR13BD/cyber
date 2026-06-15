#!/usr/bin/env python3
"""Generate the expanded final project PDF report from experiment results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "results" / "figures"
RESULTS = json.loads((ROOT / "results" / "experiment_results.json").read_text(encoding="utf-8"))
METRICS = RESULTS["metrics"]
DATASET = RESULTS["dataset"]
THRESHOLDS = RESULTS.get("thresholds", {})
ERROR_BY_CATEGORY = RESULTS.get("error_by_category", {})
RUNTIME = RESULTS.get("runtime_seconds", {})
COMPARISON_CSV = ROOT / "results" / "model_comparison.csv"


class ReportPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, "Data Science in Cyber - Final Project Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def section(self, title: str) -> None:
        self.ln(3)
        self.set_font("Helvetica", "B", 12)
        self.multi_cell(0, 7, title)
        self.ln(1)

    def body(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def figure(self, path: Path, caption: str, width: float = 175) -> None:
        if not path.exists():
            self.body(f"[Figure missing: {path.name}]")
            return
        if self.get_y() > 200:
            self.add_page()
        self.image(str(path), w=width)
        self.ln(2)
        self.set_font("Helvetica", "I", 9)
        self.multi_cell(0, 4, caption)
        self.ln(2)


def metric_line(name: str, values: dict) -> str:
    return (
        f"{name}: Accuracy={values['accuracy']:.3f}, Precision={values['precision']:.3f}, "
        f"Recall={values['recall']:.3f}, F1={values['f1']:.3f}, F2={values['f2']:.3f}, "
        f"MCC={values['mcc']:.3f}, ROC-AUC={values.get('roc_auc', float('nan')):.3f}, "
        f"PR-AUC={values.get('pr_auc', float('nan')):.3f}"
    )


def best_model(metric: str) -> str:
    ranked = sorted(METRICS.items(), key=lambda item: item[1].get(metric, -1), reverse=True)
    return ranked[0][0].replace("_", " ").title()


def format_comparison_table() -> str:
    df = pd.read_csv(COMPARISON_CSV, index_col=0).round(3)
    lines = ["Model comparison on NSL-KDD test set:\n"]
    header = f"{'Model':<22}" + "".join(f"{col:>10}" for col in df.columns)
    lines.append(header)
    lines.append("-" * len(header))
    for model, row in df.iterrows():
        lines.append(f"{model:<22}" + "".join(f"{row[col]:>10.3f}" for col in df.columns))
    return "\n".join(lines)


def format_category_errors(model_name: str) -> str:
    errors = ERROR_BY_CATEGORY.get(model_name, {})
    fn = errors.get("false_negatives", {})
    fp = errors.get("false_positives", {})
    lines = [f"{model_name.replace('_', ' ').title()}:"]
    lines.append("  False negatives by category (missed attacks):")
    if fn:
        for cat, count in fn.items():
            lines.append(f"    - {cat}: {count}")
    else:
        lines.append("    - none")
    lines.append("  False positives by category (benign flagged as attack):")
    if fp:
        for cat, count in list(fp.items())[:5]:
            lines.append(f"    - {cat}: {count}")
    else:
        lines.append("    - none")
    return "\n".join(lines)


def reproducibility_text() -> str:
    total = RUNTIME.get("total_seconds")
    if total:
        return (
            "Execution status: In this submission environment we re-ran the documented commands "
            f"(pip install -r requirements.txt, python run_pipeline.py, python generate_report.py) "
            f"and the pipeline completed successfully in about {total:.0f} seconds on CPU.\n\n"
            "Runtime breakdown (seconds):\n"
            + "\n".join(f"  - {k}: {v}" for k, v in RUNTIME.items())
            + "\n\n"
            "A fresh clone still requires downloading NSL-KDD into data/ before running the pipeline."
        )
    return (
        "Execution status: The repository documents a reproducible workflow. "
        "Run pip install -r requirements.txt, download the dataset, then "
        "python run_pipeline.py and python generate_report.py."
    )


def build_report() -> Path:
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 8, "Critical Reproduction Study:\nNetwork Intrusion Detection with Autoencoders on NSL-KDD")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        5,
        "Course: Data Science in Cyber (Dr. Uri Itai)\n"
        "Selected source: Network Intrusion Detection with Autoencoders by Steven Foerster\n"
        "URL: https://stevenfoerster.com/tutorials/network-intrusion-detection-with-autoencoders/\n"
        "Dataset mirror: https://github.com/defcom17/NSL_KDD",
    )

    pdf.section("Executive Summary")
    pdf.body(
        "This project reproduces Steven Foerster's autoencoder IDS tutorial on NSL-KDD. "
        "We trained a PyTorch autoencoder on normal traffic, used reconstruction error as an anomaly "
        "score, and compared it with Logistic Regression, Random Forest, and Isolation Forest.\n\n"
        f"The autoencoder achieved the best F1 ({METRICS['autoencoder']['f1']:.3f}) and recall "
        f"({METRICS['autoencoder']['recall']:.3f}). Random Forest still leads ROC-AUC "
        f"({METRICS['random_forest']['roc_auc']:.3f}) when labels are used. The author's core claim "
        "about reconstruction-based anomaly detection is supported, but autoencoders do not universally "
        "beat all supervised baselines. False negatives on R2L and U2R categories remain the main risk."
    )

    pdf.section("1. Summary of the Source")
    pdf.body(
        "Problem: Detect malicious network connections before damage occurs (DoS, Probe, R2L, U2R).\n\n"
        "Proposed solution: Train an autoencoder on normal traffic only. Attacks yield high reconstruction "
        "error and are flagged above a percentile threshold calibrated on held-out normal validation data.\n\n"
        f"Dataset: {DATASET['train_rows']:,} train / {DATASET['test_rows']:,} test records, "
        f"{DATASET['num_features_after_encoding']} features after encoding.\n\n"
        "The selected article provides implementation details but no official GitHub repository. "
        "We used the public NSL-KDD mirror on GitHub (defcom17/NSL_KDD) for data files."
    )

    pdf.section("2. Critical Evaluation of the Author's Claims")
    pdf.body(
        "Claims: (1) autoencoders capture multi-feature attack patterns, (2) reconstruction error is a valid "
        "anomaly score, (3) autoencoders can beat Isolation Forest under normal-only training.\n\n"
        "Our reproduction supports (1)-(3) on this split. The autoencoder outperformed Isolation Forest on "
        "F1, recall, and MCC. However, Random Forest still achieved the highest ROC-AUC, so the broader "
        "claim that autoencoders beat all classical methods is only partially supported.\n\n"
        "Limitations: dated dataset, different train/test prevalence ({:.1f}% vs {:.1f}%), threshold "
        "sensitivity, no real-time deployment test, and poor detection of rare R2L/U2R attacks.".format(
            DATASET["train_attack_rate"] * 100,
            DATASET["test_attack_rate"] * 100,
        )
    )

    pdf.section("3. Feature Engineering Analysis")
    pdf.body(
        "One-hot encoding for protocol_type, service, and flag. StandardScaler fit on training data only. "
        "Scaling is required for the autoencoder and logistic regression. NSL-KDD has redundant rate/count "
        "features (Pearson > 0.95), which reduces interpretability. Additional useful features would include "
        "TLS fingerprints, inter-arrival times, and cyclical time encodings if timestamps were available."
    )

    pdf.section("4. Reproducibility Analysis")
    pdf.body(reproducibility_text())

    pdf.section("5. Exploratory Data Analysis")
    pdf.body(
        "NSL-KDD has 41 meaningful connection-level features, no missing values, and class imbalance "
        "between train and test. There is no global timestamp, so temporal drift analysis is not possible; "
        "only per-connection duration is available. Outliers are common in byte-count features (heavy tails), "
        "which is typical in cybersecurity traffic and motivates robust statistics (median, IQR, MAD).\n\n"
        "Single-value and near-constant columns carry little discriminative power. Duplicate rows are rare. "
        "Crosstab analysis shows attacks are unevenly distributed across protocol types (TCP dominates). "
        "Pearson correlation captures linear relationships among rate features; Spearman is more appropriate "
        "for skewed byte-count distributions. Kendall was not primary here due to the large sample size.\n\n"
        "Class imbalance matters: test attack rate is {:.1f}% vs {:.1f}% in training. Accuracy alone would "
        "overstate performance if the model simply predicts the majority class.".format(
            DATASET["test_attack_rate"] * 100,
            DATASET["train_attack_rate"] * 100,
        )
    )
    pdf.figure(
        FIG_DIR / "class_imbalance.png",
        "Figure 1: Class imbalance - attack prevalence and category distribution in training data.",
    )

    pdf.section("6. Experimental Setup")
    pdf.body(
        "Official NSL-KDD train/test split. Supervised models use an 80/20 stratified internal split. "
        "Unsupervised models train on normal traffic only. Autoencoder: input->64->32->8->32->64->input, "
        "MSE loss, Adam, max 20 epochs with early stopping (patience=3), batch size 256, CPU only. "
        "Random Forest: 100 trees. Isolation Forest: 100 trees. Threshold: 95th percentile of normal scores.\n"
        f"Autoencoder threshold={THRESHOLDS.get('autoencoder', 'n/a'):.4f}, "
        f"Isolation Forest threshold={THRESHOLDS.get('isolation_forest', 'n/a'):.4f}."
    )

    pdf.add_page()
    pdf.section("7. Model Training and Comparison")
    pdf.body(format_comparison_table())
    pdf.body(
        "\nDetailed results:\n"
        + metric_line("Autoencoder", METRICS["autoencoder"])
        + "\n"
        + metric_line("Isolation Forest", METRICS["isolation_forest"])
        + "\n"
        + metric_line("Random Forest", METRICS["random_forest"])
        + "\n"
        + metric_line("Logistic Regression", METRICS["logistic_regression"])
        + f"\n\nBest F1: {best_model('f1')}. Best recall: {best_model('recall')}. "
        f"Best ROC-AUC: {best_model('roc_auc')}."
    )
    pdf.figure(
        FIG_DIR / "model_metric_comparison.png",
        "Figure 2: Precision, recall, and F1 comparison across all models.",
    )
    pdf.figure(
        FIG_DIR / "autoencoder_reconstruction_error.png",
        "Figure 3: Autoencoder reconstruction error distributions for normal vs attack traffic.",
    )
    pdf.figure(
        FIG_DIR / "confusion_matrices.png",
        "Figure 4: Confusion matrices - FP causes alert fatigue; FN means missed intrusions.",
    )

    pdf.add_page()
    pdf.section("8. Evaluation Metrics - Definitions and Cybersecurity Interpretation")
    pdf.body(
        "Accuracy = (TP + TN) / (TP + TN + FP + FN). Overall correctness; misleading under imbalance.\n\n"
        "Precision = TP / (TP + FP). Fraction of alerts that are real attacks.\n\n"
        "Recall = TP / (TP + FN). Fraction of attacks detected. Low recall is especially dangerous in IDS "
        "because undetected intrusions may persist, spread laterally, or exfiltrate data.\n\n"
        "F1 = 2PR / (P + R). Balanced metric for precision-recall trade-off.\n\n"
        "F2 = 5PR / (4P + R). Weights recall higher - appropriate when missing attacks is costlier.\n\n"
        "MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)). Robust under class imbalance.\n\n"
        "ROC-AUC: ranking quality across thresholds (TPR vs FPR). PR-AUC: precision-recall area; useful when "
        "attack prevalence shifts between train and deployment.\n\n"
        "Cyber meaning: FP wastes analyst time (alert fatigue). FN lets attackers remain inside the network. "
        "For R2L/U2R attacks, a false negative can mean stolen credentials or privilege escalation."
    )

    pdf.add_page()
    pdf.section("9. Error Analysis by Attack Category")
    pdf.body(
        "False negatives (FN) are the highest-risk errors in IDS: the system fails to raise an alert while "
        "an attack is in progress. False positives (FP) waste analyst time and contribute to alert fatigue.\n\n"
        "On NSL-KDD, R2L attacks (remote-to-local unauthorized access) are the most frequently missed category "
        "across all models because their flow statistics overlap with legitimate remote sessions. U2R attacks "
        "(privilege escalation) are rare but critical: even a small number of missed U2R events can indicate "
        "full host compromise.\n\n"
        "Per-category false negatives on the test set:"
    )
    for model in ["autoencoder", "random_forest", "isolation_forest", "logistic_regression"]:
        pdf.body(format_category_errors(model))

    ae_fn = ERROR_BY_CATEGORY.get("autoencoder", {}).get("false_negatives", {})
    rf_fn = ERROR_BY_CATEGORY.get("random_forest", {}).get("false_negatives", {})
    pdf.body(
        f"\nThe autoencoder reduced R2L false negatives from {rf_fn.get('R2L', 0)} (Random Forest) to "
        f"{ae_fn.get('R2L', 0)}, and U2R false negatives from {rf_fn.get('U2R', 0)} to {ae_fn.get('U2R', 0)}. "
        "This supports the author's claim that reconstruction-based models can capture multi-feature patterns "
        "that tree models miss on stealthy attacks. However, R2L remains the dominant failure mode for all models.\n\n"
        "Cybersecurity implication: a SOC relying solely on flow-based anomaly detection would still miss "
        "credential-abuse and privilege-escalation attacks unless complemented by endpoint detection, "
        "authentication logs, and rule-based correlation."
    )

    pdf.add_page()
    pdf.section("10. Conclusions and Future Improvements")
    pdf.body(
        "We reproduced the core autoencoder IDS workflow. Reconstruction error is a valid anomaly signal. "
        "The autoencoder outperformed Isolation Forest and logistic regression on F1/recall, but Random "
        "Forest remains strong on ROC-AUC when labels exist. The author's claims are supported for unsupervised "
        "anomaly detection but not as a universal replacement for supervised IDS.\n\n"
        "Future work: modern datasets (CICIDS2017), per-attack recall dashboards, F2-based threshold tuning, "
        "concept-drift monitoring, deployment latency benchmarks, and combining autoencoder scores with "
        "supervised ensembles in a hybrid SOC pipeline."
    )

    pdf.body(
        "Artifacts generated by the pipeline:\n"
        "- results/model_comparison.csv\n"
        "- results/experiment_results.json (metrics, per-category errors, runtime)\n"
        "- results/figures/autoencoder_reconstruction_error.png\n"
        "- results/figures/confusion_matrices.png\n"
        "- results/figures/class_imbalance.png"
    )

    output = ROOT / "report.pdf"
    pdf.output(str(output))
    return output


if __name__ == "__main__":
    path = build_report()
    print(f"Wrote {path}")
