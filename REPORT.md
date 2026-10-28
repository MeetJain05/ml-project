# Network Intrusion Detection on UNSW-NB15

**Course:** Project-Based Learning (machine learning)  
**Topic:** Binary classification of network flows (normal vs attack)  
**Implementation:** Python, NumPy, pandas, matplotlib, Streamlit  
**Code:** `preprocess.py`, `scratch_ml.py`, `train.py`, `app.py`, `feature_selection_experiment.py`

---

## 1. Problem

A network produces many connection records (flows). Each flow has measurements such as duration, bytes sent/received, packet rate, protocol, service and connection state. Looking at every flow by hand is not practical. The job of an intrusion detector is to mark a flow as normal or attack so that an analyst can spend time on the suspicious ones.

I framed this as a **binary classification** problem:

- `label = 0` → normal traffic
- `label = 1` → attack traffic

The dataset also has `attack_cat` (Generic, Exploits, Fuzzers, DoS, and so on). I use that column only for analysis after prediction. It is **not** given to the model, because it would tell the model the answer.

Why this problem: it is a real security task, the data is mixed (numeric + categorical), class errors are not equal (a missed attack is different from a false alarm), and the same pipeline can be shown in a small web demo.

---

## 2. Dataset

I used the official partitioned **UNSW-NB15** CSVs:

| File | Rows | Normal | Attack | Attack share |
|---|---:|---:|---:|---:|
| `data/unsw_train.csv` | 82,332 | 37,000 | 45,332 | 55.1% |
| `data/unsw_test.csv` | 175,341 | 56,000 | 119,341 | 68.1% |

The test file is larger than the train file. That is how this public split is published; I did not merge them and re-split, because the assignment is to evaluate on a held-out set.

There are **45 columns**. After dropping `id`, `attack_cat` and `label`, the remaining fields are:

- **Numeric:** duration, packet and byte counts, TTL, load, jitter, TCP window, RTT, connection counts, FTP/HTTP flags, etc.
- **Categorical:** `proto`, `service`, `state`

Attack types in the two files:

| Category | Train | Test |
|---|---:|---:|
| Normal | 37,000 | 56,000 |
| Generic | 18,871 | 40,000 |
| Exploits | 11,132 | 33,393 |
| Fuzzers | 6,062 | 18,184 |
| DoS | 4,089 | 12,264 |
| Reconnaissance | 3,496 | 10,491 |
| Analysis | 677 | 2,000 |
| Backdoor | 583 | 1,746 |
| Shellcode | 378 | 1,133 |
| Worms | 44 | 130 |

Two dataset issues matter later:

1. **Shift between train and test.** The test set has a higher attack rate (68% vs 55%). A model that looks strong on training data can look worse on the official test split.
2. **Rare classes.** Worms has 44 training rows. A high recall on Worms is not as trustworthy as a high recall on Generic (40,000 test rows).

I split the training CSV 80/20 into train and validation (`seed = 42`):

| Split | Flows | Features after encoding |
|---|---:|---:|
| Train | 65,866 | 72 |
| Validation | 16,466 | 72 |
| Test (untouched) | 175,341 | 72 |

---

## 3. Methodology

### 3.1 Overall pipeline

1. Load the two CSVs.
2. Split the training CSV into train / validation.
3. Fit preprocessing on **train only**, then transform val and test.
4. Plot class balance, a few feature histograms, a correlation heatmap, and a 2-D PCA view.
5. Tune K-Means (`k`) on validation AUC.
6. Tune logistic regression (L2) on validation loss.
7. Fit Gaussian Naive Bayes (no extra hyperparameters besides variance smoothing).
8. Evaluate all three on train and test.
9. Break down recall by `attack_cat` on the test set.
10. Save models, metrics and figures under `artifacts/`.
11. Streamlit app loads those artifacts for demonstration.

`train.py` runs this end to end. Hyperparameters are chosen on validation, not on the test file.

### 3.2 Preprocessing (`preprocess.py`)

**Dropped columns:** `id` (row id), `attack_cat` (leak), `label` (target).

**Missing numbers:** filled with the **training median**. I used median instead of mean because byte/rate columns have heavy tails. Zero would be wrong for some fields where 0 is a real measurement.

**Skew:** if a numeric column is non-negative and skewness > 2, I apply `log1p(x) = log(1 + x)`. Bytes, packet counts and rates look like this. Without the log, a few huge flows pull means and distances around.

**Scaling:** z-score using training mean and standard deviation. K-Means is distance-based and logistic regression uses gradients; both care about scale. If a column has std 0, I set std to 1 so there is no divide-by-zero.

**Clipping:** after scaling, values are clipped to **[-5, 5]**. A handful of flows are extreme even after log + z-score and would otherwise dominate centroids.

**Categories:** one-hot of the most common values (top 10 protocols, top 13 services, top 8 states). Everything else, including values that only appear later, maps to `other`. That keeps the feature width fixed at 72 for training and for the dashboard.

All of these statistics (medians, means, which columns get log, which category levels are kept) come from the training split only.

### 3.3 Models (`scratch_ml.py`)

No scikit-learn / TensorFlow / PyTorch classifiers. NumPy is used for arrays and linear algebra.

**K-Means (unsupervised anomaly detector)**  
Fitted **only on normal training rows**. The idea is: learn a few clusters of “what normal looks like”, then score a new flow by Euclidean distance to the nearest centroid. Large distance → unusual → possible attack.

- Initialization: k-means++
- Empty cluster: that centroid is moved to the current worst-fitting point
- Default threshold: **95th percentile** of distances on the normal training rows (so about 5% of normal traffic would be flagged if the threshold were applied back on that same set)

I tried `k ∈ {2, 3, 4, 6, 8, 10, 12, 16}` and picked **k = 16** because it had the highest validation AUC (0.746), not because of the elbow plot alone. Inertia always falls as k grows; AUC is the quantity that matches the detection task.

**Logistic regression (supervised)**  
`p(attack) = sigmoid(x · w + b)`, trained with binary cross-entropy and mini-batch gradient descent (learning rate 0.1, 25 epochs, batch 512). L2 is applied to `w`, not to the bias. I swept `l2 ∈ {0, 1e-4, 1e-3, 0.01, 0.1, 1}` and kept **l2 = 0** (lowest validation loss). Default decision threshold is 0.5.

Sigmoid input is clipped to [-30, 30] so `exp` does not overflow.

**Gaussian Naive Bayes (supervised baseline)**  
For each class, each feature is treated as Gaussian. Prediction uses log-likelihoods, then a sigmoid of the log-odds so the output is a probability. Variance smoothing is 1e-3 because the data is already standardized; a tiny variance would otherwise blow up the likelihood.

**PCA**  
Covariance → eigenvectors → first two components. Used only to draw a scatter plot, not as a classifier.

### 3.4 Metrics

Accuracy is not enough here. The test set is 68% attack, so a dummy “always attack” rule already looks decent on accuracy and useless on false alarms.

| Metric | Meaning here |
|---|---|
| Precision | Of the alerts, how many were real attacks |
| Recall | Of the real attacks, how many were caught |
| F1 | Harmonic mean of the two |
| FPR | Fraction of normal flows that were flagged |
| AUC | Ranking quality across thresholds (area under ROC) |

ROC / AUC are implemented by sorting scores and accumulating TPR and FPR; AUC is trapezoidal area.

### 3.5 Extra experiment: feature ranking

`feature_selection_experiment.py` ranks the 72 preprocessed columns by absolute Pearson correlation with `label` on the training matrix, then retrains the three models on the top 10, 20, 30, 40, 50 and all 72 features. The count for each model is chosen by **validation AUC**. This was kept as a separate script so the main reported models stay the full 72-feature versions unless the experiment clearly wins.

---

## 4. Results

### 4.1 Main test set (72 features)

| Model | Accuracy | Precision | Recall | F1 | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|
| K-Means | 0.540 | 0.919 | 0.356 | 0.513 | 0.067 | 0.752 |
| Logistic regression | 0.887 | 0.980 | 0.851 | 0.911 | 0.037 | 0.976 |
| Gaussian NB | 0.754 | 0.963 | 0.664 | 0.786 | 0.055 | 0.880 |

Same models on the **training** split (for overfit checks):

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| K-Means | 0.585 | 0.876 | 0.287 | 0.432 | 0.748 |
| Logistic regression | 0.932 | 0.930 | 0.947 | 0.939 | 0.981 |
| Gaussian NB | 0.775 | 0.895 | 0.670 | 0.766 | 0.858 |

### 4.2 Hyperparameter sweeps

K-Means on normal traffic:

| k | Inertia | Train AUC | Val AUC |
|---:|---:|---:|---:|
| 2 | 713,321 | 0.450 | 0.451 |
| 4 | 519,017 | 0.625 | 0.624 |
| 8 | 358,739 | 0.741 | 0.738 |
| 10 | 303,706 | 0.694 | 0.690 |
| 12 | 282,734 | 0.728 | 0.725 |
| **16** | **216,319** | **0.748** | **0.746** |

k = 10 is worse than k = 8 on AUC even though inertia is lower. That is why I did not pick k from the elbow plot.

Logistic regression L2:

| L2 | Train acc | Val acc | Train loss | Val loss |
|---:|---:|---:|---:|---:|
| **0** | **0.932** | **0.928** | **0.188** | **0.191** |
| 0.0001 | 0.932 | 0.928 | 0.189 | 0.192 |
| 0.001 | 0.929 | 0.924 | 0.196 | 0.200 |
| 0.01 | 0.910 | 0.909 | 0.257 | 0.260 |
| 0.1 | 0.810 | 0.809 | 0.402 | 0.403 |
| 1.0 | 0.746 | 0.744 | 0.536 | 0.537 |

Train and validation stay close at every L2. Large L2 hurts both, which is underfitting, not a fix for overfitting.

### 4.3 Recall by attack type (test set)

| Attack | n | K-Means | LogReg | NB |
|---|---:|---:|---:|---:|
| Generic | 40,000 | 0.794 | 0.995 | 0.988 |
| Backdoor | 1,746 | 0.117 | 0.937 | 0.834 |
| DoS | 12,264 | 0.119 | 0.944 | 0.813 |
| Exploits | 33,393 | 0.195 | 0.917 | 0.437 |
| Worms | 130 | 0.331 | 0.923 | 0.115 |
| Reconnaissance | 10,491 | 0.045 | 0.887 | 0.514 |
| Shellcode | 1,133 | 0.006 | 0.861 | 0.509 |
| Analysis | 2,000 | 0.058 | 0.757 | 0.734 |
| Fuzzers | 18,184 | 0.105 | 0.330 | 0.342 |

### 4.4 Feature selection experiment

Top correlated transformed features (train): `state=INT`, `dload`, `sttl`, `ct_dst_sport_ltm`, `dmean`, `dbytes`, `proto=tcp`, `swin`, `dpkts`, `ct_src_dport_ltm`.

Validation-selected counts and **test** metrics:

| Model | Features kept | Test acc | Test F1 | Test AUC |
|---|---:|---:|---:|---:|
| K-Means | 50 | 0.541 | 0.513 | **0.812** |
| Logistic regression | 72 (all) | 0.887 | 0.911 | 0.976 |
| Gaussian NB | 72 (all) | 0.754 | 0.786 | 0.880 |

Cutting logistic regression down to 10 features drops test AUC from 0.976 to 0.924 and F1 from 0.911 to 0.850. At 50 features it is almost the same as 72 (AUC 0.975 vs 0.976). So the extra columns are not doing much harm, and I kept the full set in the main pipeline.

K-Means **does** benefit from dropping some dimensions: validation AUC is highest at 50 features, and test AUC rises from 0.752 (72 feats) to 0.812 (50 feats). Recall at the 95th-percentile threshold stays around 0.35 either way. I left the main dashboard K-Means on 72 features so all three models share the same preprocessor.

---

## 5. Insights

**Logistic regression is the only detector I would ship from this set.** AUC 0.976 means it ranks attacks above normal flows well. Precision 0.980 at threshold 0.5 means most alerts are real. Recall 0.851 means about 15% of attacks still get through at that threshold. The gap between train accuracy (0.932) and test accuracy (0.887) is mostly the train/test distribution difference (test is more attack-heavy and was collected as a separate official split), not a huge train–val overfit (val accuracy 0.928).

**K-Means is precise but blind to many attacks.** Precision ~0.92 with FPR ~0.07 looks like “when it alerts, it is usually right”. Recall 0.36 is the problem: most attacks sit close to some normal cluster. That is expected if you only model normal traffic and the attacks are not all outliers. Generic is the exception (recall 0.79) because a lot of Generic traffic is far from normal. Shellcode (0.006) and Reconnaissance (0.045) almost never trigger the distance rule.

**Fuzzers are the weak spot for the supervised models too.** LogReg recall 0.33, NB 0.34. Fuzzer traffic is designed to look like broken or noisy legitimate input, so it overlaps the normal region. A linear boundary and a Gaussian NB both struggle there.

**Naive Bayes is hurt by correlated features.** Packet counts, bytes, load and TCP window features move together (visible on the correlation heatmap). NB treats them as independent, so it over-counts evidence. It still beats K-Means on F1, but it is not close to logistic regression. On Exploits, NB recall is 0.44 vs 0.92 for logreg.

**L2 = 0 is not “I forgot regularisation”.** I ran the sweep. Every positive L2 made validation loss worse. With 65k rows and a linear model, 72 weights are not a high-capacity setup. Regularisation here just shrinks a model that was already not over-memorising.

**k-means++ and vectorised distances were necessary.** A naive triple loop over n × k × d would be too slow on tens of thousands of 72-D points. Distances are computed as `||x||² − 2xCᵀ + ||c||²`.

**Thresholds are a policy choice, not a math constant.** 0.5 for probabilities and the 95th percentile for K-Means are defaults. The dashboard slider is there because a SOC that hates false alarms would raise the threshold, and one that cannot miss Generic/DoS would lower it.

---

## 6. Challenges

**1. No library classifiers.** The assignment wanted models from scratch. That meant writing gradient descent, k-means++ , Gaussian likelihoods, ROC and AUC myself, and checking them against plots rather than against sklearn’s score.

**2. Scale and skew.** Raw `sload` / `sbytes` values are orders of magnitude apart. Early K-Means runs without log + standardisation collapsed to a few large-magnitude features. Clipping was added after seeing a small number of z-scores far outside ±5.

**3. Train vs test mismatch.** Using the official split is honest, but it is awkward: the “test” set is bigger and more attack-heavy. I had to be careful not to retune on test after seeing this. The validation split is carved from the training CSV only.

**4. Choosing k.** Elbow (inertia) and detection AUC do not pick the same k. I documented both plots and selected on AUC because the project is detection, not compression of normal traffic.

**5. Empty clusters.** With k-means++ this was rare, but the update step still has a fallback: if a cluster gets no points, its centroid jumps to the point with the largest min-distance.

**6. Numerical issues.** `log(p)` with p = 0, `exp` overflow in sigmoid, and zero variance in NB all showed up while implementing. Clipping probabilities and logits, and adding variance smoothing, fixed them.

**7. Rare attack types.** Worms (130 test rows) and Shellcode cannot support strong claims. I still report per-class recall, but I treat Generic / Exploits / Fuzzers / DoS as the ones that actually move the overall score.

**8. Demo vs production.** Streamlit needs a saved `model.pkl` and a sample of rows. The “live replay” tab is not packet capture. I kept it that way on purpose; wiring a live sniffer is a different project and would not change the ML results.

**9. Feature selection did not give a free win for the main model.** I hoped a smaller feature set would clean up K-Means and maybe help NB. It helped K-Means AUC, not its recall at the operating threshold, and logreg preferred keeping all columns. I did not silently swap the main pipeline over to 50 features.

---

## 7. Limitations

- UNSW-NB15 is a lab / benchmark mix, not this campus network.
- The train/val split is random, not time-ordered. If the real world is a time series, leakage of similar flows is possible.
- Binary labels hide that Fuzzers and Generic are very different problems.
- Default thresholds are not cost-tuned.
- Pickle is only for local artifacts. Do not load a `model.pkl` from someone else.
- I did not implement a non-linear model (tree / small neural net), so I cannot claim logistic regression is the best possible detector, only the best of the three I wrote.

---

## 8. Conclusion

The project takes UNSW-NB15 flow records, preprocesses them without leaking test statistics, and compares three NumPy models. Logistic regression is clearly best on the official test set (F1 0.911, AUC 0.976, FPR 0.037). K-Means shows what you get if attack labels are missing: few false alarms, many missed attacks. Gaussian NB is a fast baseline that is limited by independence assumptions.

The practical takeaway is that preprocessing and the choice of learning setup (labels vs normal-only) matter more than small hyperparameter tweaks. L2 and aggressive feature dropping did not beat a well-scaled linear model on this split. The remaining hard cases are attacks that look like normal traffic, especially Fuzzers.

---

## 9. How to reproduce

```powershell
python -m pip install -r requirements.txt
python train.py
streamlit run app.py
```

Optional:

```powershell
python feature_selection_experiment.py
```

Figures land in `artifacts/figures/`. Metrics are in `artifacts/results.json`. Setup detail is in `README.md`. Viva questions and the list of design choices are in `PROJECT_PREPARATION.md`.
