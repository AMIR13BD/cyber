#!/usr/bin/env python3
"""Generate the final project PDF report from experiment results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "results" / "figures"
RESULTS_DIR = ROOT / "results"
RESULTS = json.loads((RESULTS_DIR / "experiment_results.json").read_text(encoding="utf-8"))
METRICS = RESULTS["metrics"]
DATASET = RESULTS["dataset"]
PREPROCESSING = RESULTS.get("preprocessing", {})
RUNTIME = RESULTS.get("runtime_seconds", {})


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
        if self.get_y() > 190:
            self.add_page()
        self.image(str(path), w=width)
        self.ln(2)
        self.set_font("Helvetica", "I", 9)
        self.multi_cell(0, 4, caption)
        self.ln(2)

    def table_from_csv(self, path: Path, title: str, max_rows: int = 12) -> None:
        if not path.exists():
            self.body(f"[Table missing: {path.name}]")
            return
        df = pd.read_csv(path)
        if len(df) > max_rows:
            df = df.head(max_rows)
        self.body(f"{title}\n{df.round(3).to_string(index=False)}")


def load_source_protocol() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "source_protocol_model_comparison.csv", index_col=0)


def reproducibility_text() -> str:
    total = RUNTIME.get("total_seconds")
    if total:
        return (
            "In this submission environment we re-ran:\n"
            "  pip install -r requirements.txt\n"
            "  python run_pipeline.py\n"
            "  python generate_report.py\n"
            f"The pipeline completed successfully in about {total:.0f} seconds on CPU. "
            "A fresh clone still requires downloading KDDTrain+.txt and KDDTest+.txt into data/. "
            "Small numeric differences vs the original tutorial may occur because of package versions "
            "and random seeds, but the protocol (50 AE epochs, 200 IF trees, 99th percentile on the "
            "same normal_validation split) is matched."
        )
    return "See README for documented reproduction commands."


def build_report() -> Path:
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    source_df = load_source_protocol()
    ae_row = source_df.loc["autoencoder_source_protocol"]
    if_row = source_df.loc["isolation_forest_source_protocol"]

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 8, "Critical Reproduction Study:\nNetwork Intrusion Detection with Autoencoders on NSL-KDD")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        5,
        "Course: Data Science in Cyber (Dr. Uri Itai)\n"
        "Source: Network Intrusion Detection with Autoencoders - Steven Foerster\n"
        "https://stevenfoerster.com/tutorials/network-intrusion-detection-with-autoencoders/\n"
        "No official GitHub repo for the article. Dataset mirror: github.com/defcom17/NSL_KDD",
    )

    pdf.section("Executive Summary")
    pdf.body(
        "We reproduced the source protocol: PyTorch autoencoder trained on normal_train only (50 epochs), "
        "Isolation Forest with 200 trees on normal_train only, and both thresholds calibrated on the "
        "same normal_validation split at the 99th percentile. Strict preprocessing fits the anomaly "
        "scaler on normal_train only and removes constant features such as num_outbound_cmds.\n\n"
        f"At the source operating point (99th percentile), autoencoder F1={ae_row['f1']:.3f}, "
        f"recall={ae_row['recall']:.3f}; Isolation Forest F1={if_row['f1']:.3f}, "
        f"recall={if_row['recall']:.3f}. The tutorial claims a modest, dataset-dependent advantage for "
        "autoencoders over Isolation Forest - not universal superiority over all classical methods. "
        "Our threshold-sensitivity analysis shows higher recall at 95% is threshold-dependent and does "
        "not by itself prove better feature interaction learning."
    )

    pdf.section("1. Summary of the Source")
    pdf.body(
        "Problem: binary/normal-vs-attack intrusion detection on NSL-KDD connection records.\n"
        "Source protocol: 50 autoencoder epochs, MSE reconstruction error, 99th percentile threshold on "
        "held-out normal validation traffic, 200-tree Isolation Forest trained on the same normal_train, "
        "threshold also calibrated on normal_validation scores at the 99th percentile.\n"
        "Claim: autoencoders may outperform Isolation Forest on some attack types by learning feature "
        "relationships, but the advantage is modest and dataset-dependent."
    )

    pdf.section("2. Critical Evaluation")
    pdf.body(
        "Our earlier 95th-percentile result used a different precision-recall trade-off and is not the "
        "fair comparison to the source. The correct reproduction uses 99th percentile on normal_validation "
        "for both anomaly models.\n\n"
        "Threshold choice directly controls FPR/FNR: lower percentiles increase recall but also false "
        "alarms. We do NOT claim autoencoders capture patterns tree models miss unless shown under the "
        "same operating point. Lower raw false-negative counts at 95% may reflect threshold calibration, "
        "not proven feature-interaction superiority.\n\n"
        f"At 99%: Autoencoder F1={ae_row['f1']:.3f}, recall={ae_row['recall']:.3f}; "
        f"Isolation Forest F1={if_row['f1']:.3f}, recall={if_row['recall']:.3f}. "
        "At 95% both models achieve higher recall, but that is a different operating point. "
        "Supervised Random Forest remains competitive when labels are available."
    )

    pdf.section("3. Feature Engineering")
    pdf.body(
        f"Encoding fit on KDDTrain+ only. Constant columns removed: {PREPROCESSING.get('removed_constant_columns', [])}. "
        "Anomaly scaler fit on normal_train only - attack rows no longer influence anomaly preprocessing. "
        "Supervised scaler fit on all KDDTrain+ rows because labels are used. "
        "Feature ablation compares full features, constant removal, and redundant-feature removal."
    )

    pdf.section("4. Reproducibility")
    pdf.body(reproducibility_text())

    pdf.section("5. Exploratory Data Analysis and Distribution Shift")
    pdf.body(
        f"Train attack rate: {DATASET['train_attack_rate']*100:.1f}%. "
        f"Test attack rate: {DATASET['test_attack_rate']*100:.1f}%. "
        "This shift affects accuracy and precision/recall even when ranking quality is stable. "
        "No global timestamp exists; temporal drift analysis is not meaningful beyond connection duration."
    )
    pdf.figure(FIG_DIR / "class_imbalance.png", "Figure 1: Class and category imbalance in KDDTrain+.")
    pdf.figure(FIG_DIR / "train_test_shift.png", "Figure 2: Train vs test attack prevalence shift.")
    pdf.table_from_csv(RESULTS_DIR / "distribution_shift_summary.csv", "Table 1: Distribution shift summary (excerpt).", 8)

    pdf.add_page()
    pdf.section("6. Experimental Setup")
    pdf.body(
        f"normal_train rows: {DATASET.get('normal_train_rows', 'n/a')}, "
        f"normal_validation rows: {DATASET.get('normal_validation_rows', 'n/a')}. "
        "Supervised models trained on all KDDTrain+. Source-protocol anomaly models use normal_train / "
        "normal_validation split with 99th-percentile thresholds."
    )

    pdf.section("7. Source Protocol Results (99th percentile)")
    pdf.body(source_df.round(3).to_string())
    pdf.figure(
        FIG_DIR / "autoencoder_reconstruction_error.png",
        "Figure 3: Reconstruction error distribution with 99th-percentile threshold.",
    )

    pdf.section("8. Threshold Sensitivity")
    pdf.body(
        "Both anomaly models evaluated at 95th, 97th, and 99th percentiles on normal_validation. "
        "Cyber interpretation: lower percentiles increase recall (fewer missed attacks) but raise FPR "
        "(more alert fatigue)."
    )
    pdf.table_from_csv(RESULTS_DIR / "threshold_sensitivity.csv", "Table 2: Threshold sensitivity metrics.")
    pdf.figure(FIG_DIR / "threshold_recall.png", "Figure 4: Recall vs threshold percentile.")
    pdf.figure(FIG_DIR / "threshold_f1.png", "Figure 5: F1 vs threshold percentile.")
    pdf.figure(FIG_DIR / "threshold_fpr.png", "Figure 6: False positive rate vs threshold percentile.")

    pdf.add_page()
    pdf.section("9. Fair Anomaly Comparison")
    pdf.body(
        "Same-percentile comparison aligns operating points. Equal-FPR comparison calibrates the autoencoder "
        "threshold on normal_validation to match the Isolation Forest FPR at 99th percentile."
    )
    pdf.table_from_csv(RESULTS_DIR / "fair_anomaly_comparison.csv", "Table 3: Fair anomaly comparison.")

    pdf.section("10. Model Comparison (all models)")
    pdf.table_from_csv(RESULTS_DIR / "model_comparison.csv", "Table 4: Full model comparison.")
    pdf.figure(FIG_DIR / "confusion_matrices.png", "Figure 7: Confusion matrices.")
    pdf.figure(FIG_DIR / "model_metric_comparison.png", "Figure 8: Metric comparison bar chart.")

    pdf.section("11. Feature Ablation")
    pdf.body(
        "Constant features do not help and were removed in the main pipeline. Highly correlated features "
        "hurt interpretability; removing them may or may not change performance materially."
    )
    pdf.table_from_csv(RESULTS_DIR / "feature_ablation.csv", "Table 5: Feature ablation results.")

    pdf.add_page()
    pdf.section("12. Evaluation Metrics")
    pdf.body(
        "Accuracy = (TP+TN)/(TP+TN+FP+FN). Precision = TP/(TP+FP). Recall = TP/(TP+FN). "
        "F1 = 2PR/(P+R). F2 = 5PR/(4P+R). MCC = (TP*TN-FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)). "
        "FPR = FP/(FP+TN). FNR = FN/(FN+TP). ROC-AUC measures ranking; PR-AUC is useful under prevalence shift. "
        "In IDS, false negatives are often more dangerous than false positives because missed intrusions persist."
    )

    pdf.section("13. Error Analysis - Per-Category Recall")
    pdf.body(
        "Raw FN counts are misleading because categories have different sizes. Per-category recall = "
        "TP_in_category / total_category_samples. R2L and U2R have high security importance despite fewer samples."
    )
    pdf.table_from_csv(RESULTS_DIR / "category_recall_analysis.csv", "Table 6: Per-category recall (excerpt).", 20)

    pdf.section("14. Conclusions")
    pdf.body(
        "We fixed preprocessing leakage, matched the source protocol at 99th percentile, and added threshold "
        "sensitivity, fair comparison, ablation, and distribution-shift analysis. The author's claim is only "
        "partially supported: results depend on threshold and operating point. We do not overclaim feature-interaction "
        "superiority without equal-FPR evidence. Future work: modern datasets, per-attack dashboards, deployment latency."
    )

    output = ROOT / "report.pdf"
    pdf.output(str(output))
    return output


if __name__ == "__main__":
    path = build_report()
    print(f"Wrote {path}")
