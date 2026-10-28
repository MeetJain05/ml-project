# Network Intrusion Detection using UNSW-NB15

PBL machine learning project. The goal is to classify each network flow as
**normal** or **attack**, using models written in NumPy rather than scikit-learn
classifiers.

Full write-up: [REPORT.md](REPORT.md)  
Viva notes and design choices: [PROJECT_PREPARATION.md](PROJECT_PREPARATION.md)

## Result (held-out test set)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| K-Means (unsupervised) | 0.540 | 0.919 | 0.356 | 0.513 | 0.752 |
| Logistic Regression | **0.887** | **0.980** | **0.851** | **0.911** | **0.976** |
| Gaussian Naive Bayes | 0.754 | 0.963 | 0.664 | 0.786 | 0.880 |

Logistic regression is the model I would actually use. K-Means is kept as a
baseline that does not need attack labels.

## Files

| File | Role |
|---|---|
| `preprocess.py` | Load CSVs, train/val split, scaling and one-hot encoding |
| `scratch_ml.py` | K-Means, logistic regression, Gaussian NB, PCA, metrics |
| `train.py` | Training, plots, evaluation, saves `artifacts/` |
| `feature_selection_experiment.py` | Extra experiment: rank features by correlation |
| `app.py` | Streamlit dashboard |
| `data/unsw_train.csv` | Official UNSW-NB15 training split (82,332 flows) |
| `data/unsw_test.csv` | Official testing split (175,341 flows) |

## Setup

Python 3.10+. From the project folder:

```powershell
python -m pip install -r requirements.txt
python train.py
streamlit run app.py
```

`train.py` writes models and figures into `artifacts/`. The dashboard reads
those files; it does not retrain.

Default data paths are already `data/unsw_train.csv` and `data/unsw_test.csv`.
To override:

```powershell
python train.py --train data/unsw_train.csv --test data/unsw_test.csv
```

## What the dashboard shows

1. Replay of stored test flows with a movable alert threshold
2. Test metrics, ROC, confusion matrices, recall by attack type
3. Class balance, feature plots, PCA
4. K-Means `k` sweep and logistic regression L2 sweep

The replay is a demo on saved rows, not a live packet sniffer.

## Models in short

- **K-Means:** fitted on normal training flows only. Score = distance to the
  nearest centroid. Alert if the distance is above the 95th percentile of
  normal training distances.
- **Logistic regression:** mini-batch gradient descent, optional L2. Default
  decision threshold 0.5. Selected L2 = 0 (lowest validation loss).
- **Gaussian NB:** class-conditional Gaussians, used as a simple baseline.
- **PCA:** 2-D plot only, not used for classification.

Preprocessing (fit on the training split only): median fill, `log1p` on
right-skewed non-negative columns, z-score, clip to [-5, 5], one-hot of
common `proto` / `service` / `state` values with an `other` bucket.

`id`, `attack_cat` and `label` are not used as input features.
