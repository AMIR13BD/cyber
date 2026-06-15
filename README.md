# Data Science in Cyber - Final Project

**Topic:** Intrusion Detection Systems (IDS) with Autoencoders  
**Dataset:** [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)  
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

Optional notebook:

```bash
jupyter notebook notebooks/01_nsl_kdd_ids_analysis.ipynb
```

## Download Data (fresh clone)

```bash
mkdir data
curl -L -o data/KDDTrain+.txt https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt
curl -L -o data/KDDTest+.txt https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt
```

## Expected Runtime (CPU)

On a typical laptop CPU (tested during submission prep):

- `python run_pipeline.py`: about **1-3 minutes** (autoencoder max 20 epochs with early stopping, 100-tree forests)
- `python generate_report.py`: a few seconds

Progress messages print before each model. A runtime summary is saved to `results/experiment_results.json`.

## Repository Structure

```
.
├── data/
├── notebooks/01_nsl_kdd_ids_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── autoencoder.py
│   ├── metrics_utils.py
│   ├── plotting.py
│   └── error_analysis.py
├── results/
│   ├── experiment_results.json
│   ├── model_comparison.csv
│   └── figures/
├── report.pdf
├── run_pipeline.py
├── generate_report.py
└── requirements.txt
```

## Models

| Model | Type | Training data |
|-------|------|---------------|
| Autoencoder | Unsupervised (PyTorch) | Normal traffic only |
| Isolation Forest | Unsupervised | Normal traffic only |
| Random Forest | Supervised (100 trees) | Labeled train split |
| Logistic Regression | Supervised | Labeled train split |

All outputs are written to `results/` by `python run_pipeline.py`.

## Submission Checklist

- [x] PDF report (`report.pdf`)
- [x] Python notebook
- [x] Supporting code (`src/`, `run_pipeline.py`)
- [x] README with links and execution instructions
