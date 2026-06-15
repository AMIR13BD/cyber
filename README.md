# Data Science in Cyber - Final Project

**Topic:** Intrusion Detection with Autoencoders on NSL-KDD  
**Course:** Data Science in Cyber - Dr. Uri Itai

## Selected Source

- **Article:** [Network Intrusion Detection with Autoencoders](https://stevenfoerster.com/tutorials/network-intrusion-detection-with-autoencoders/) (Steven Foerster)

The selected article provides implementation details but **no official GitHub repository**.

**Dataset mirror used:** [defcom17/NSL_KDD](https://github.com/defcom17/NSL_KDD)

## Quick Start

```bash
pip install -r requirements.txt
python run_pipeline.py
python generate_report.py
```

## Download Data (fresh clone)

```bash
mkdir data
curl -L -o data/KDDTrain+.txt https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt
curl -L -o data/KDDTest+.txt https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt
```

## Source-Protocol Reproduction

The pipeline includes a `source_protocol` experiment that matches the tutorial:

| Setting | Value |
|---------|-------|
| Autoencoder epochs | 50 (no early stopping) |
| Isolation Forest trees | 200 |
| Threshold percentile | 99th on **normal_validation** |
| Anomaly scaler | fit on **normal_train** only |
| Constant features | removed (e.g. `num_outbound_cmds`) |

Results: `results/source_protocol_model_comparison.csv`, `results/source_protocol_thresholds.json`

Additional analyses: threshold sensitivity (95/97/99), fair comparison, feature ablation, distribution shift.

## Expected Runtime (CPU)

- `python run_pipeline.py`: about **3-6 minutes** (50 AE epochs + ablation + forests)
- `python generate_report.py`: a few seconds

Progress messages print before each stage. Runtime saved to `results/experiment_results.json`.

## Repository Structure

```
data/
notebooks/01_nsl_kdd_ids_analysis.ipynb
src/
results/
report.pdf
run_pipeline.py
generate_report.py
requirements.txt
```

## Note on Reproducibility

Small numeric differences vs the blog post may occur due to PyTorch/sklearn versions and random seed,
but the protocol (epochs, trees, percentile, calibration split, scaler fit) is intentionally matched.
