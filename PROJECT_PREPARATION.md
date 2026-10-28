# Project preparation (viva / presentation)

Use this file to prepare. The formal write-up is `REPORT.md`. Numbers below match the last `train.py` run in `artifacts/results.json`.

---

## 30-second intro

I built a network intrusion detector on UNSW-NB15. Each row is a flow. The model has to say normal or attack. I implemented K-Means, logistic regression and Gaussian Naive Bayes in NumPy, with preprocessing fitted only on the training split. Logistic regression is the best on the official test set: accuracy 0.887, F1 0.911, AUC 0.976. K-Means is an unsupervised baseline fitted on normal traffic only. There is a Streamlit app that replays stored flows and shows the plots.

If asked “from scratch”: the training loops, distances, probabilities, PCA and metrics are my code. pandas/NumPy/matplotlib/Streamlit are for dataframes, arrays, plots and the UI, not for `LogisticRegression.fit` from sklearn.

---

## Design decisions (why I did it this way)

### 1. Binary labels, not 10-class classification

The assignment is detection: is this flow bad? Multi-class would mix a rare class (Worms, 44 train rows) with Generic (18k). I still report recall per `attack_cat` so it is clear which attacks are easy.

### 2. Do not feed `attack_cat` or `id` into the model

`attack_cat` is almost the answer. `id` is just a row number. Using either would be leakage / noise.

### 3. Keep the official train/test files

I did not shuffle train+test together. The published test set is larger and more attack-heavy (68% vs 55%). That makes test scores a bit harder, which is fine.

### 4. Validation comes from the training CSV (20%), seed 42

Hyperparameters (`k`, L2) are chosen on validation. The test CSV is used once at the end. Seed 42 is only so the split and shuffles repeat.

### 5. Median impute, log1p, z-score, clip [-5, 5]

Median: outliers in bytes/rates.  
log1p: skew > 2 and min ≥ 0.  
z-score: K-Means distances and gradient steps.  
clip: a few z-scores were huge and yanked centroids.

### 6. One-hot with an `other` bucket (top 10 / 13 / 8)

There are many rare protocols. Keeping every level would make a wide, sparse matrix and break when the dashboard sees a new value. Unseen categories become `other`.

### 7. Three models on purpose

| Choice | Point |
|---|---|
| K-Means on **normal only** | What if you do not have attack labels? |
| Logistic regression | Simple supervised model with a linear boundary |
| Gaussian NB | Fast probabilistic baseline with strong assumptions |
| PCA | Picture of the 72-D data, not a detector |

If I had only submitted logistic regression, there would be nothing to compare.

### 8. K-Means threshold = 95th percentile of normal distances

That is a common anomaly cutoff: flag the most unusual 5% of “normal-like” distances. It is not magical. The app lets you move it.

### 9. Select `k` by validation AUC, not elbow

Inertia always drops as k grows. Detection AUC peaked at **k = 16**. k = 10 was worse than k = 8, which is a good viva example of why unsupervised loss ≠ task metric.

### 10. Select L2 by validation loss; result was 0

I still ran `{0, 1e-4, 1e-3, 0.01, 0.1, 1}`. Larger L2 pulled both train and val accuracy down (underfit). Train acc 0.932 vs val 0.928, so I was not looking at a wild overfit.

### 11. Mini-batch GD (batch 512, lr 0.1, 25 epochs)

Full-batch on 65k × 72 is slower and I wanted a learning curve per epoch. 25 epochs was enough for the loss to flatten.

### 12. Feature selection stayed in a side script

Ranking by |corr(label)| on train, then try top 10…72. Logistic regression wanted all 72. K-Means AUC improved at 50 features but recall at the operating threshold did not. I did not change the main `train.py` models so the report, pickle and dashboard stay consistent.

### 13. Streamlit, not a packet sniffer

The requirement was a working demo. Replay uses `artifacts/replay_sample.csv` (up to 10k test rows). Say this out loud if someone asks “is this live traffic?”

---

## Numbers to remember

**Test (main models, 72 features)**

- LogReg: acc **0.887**, prec **0.980**, rec **0.851**, F1 **0.911**, FPR **0.037**, AUC **0.976**
- NB: acc 0.754, F1 0.786, AUC 0.880
- K-Means: acc 0.540, prec 0.919, rec 0.356, F1 0.513, AUC 0.752

**Fit**

- LogReg train acc 0.932 / val 0.928 / train loss 0.188 / val loss 0.191
- best k = 16, best L2 = 0
- 65,866 / 16,466 / 175,341 flows, 72 features

**Hard class:** Fuzzers, logreg recall **0.33**  
**Easy class:** Generic, logreg recall **0.995**

---

## Likely questions

**What is the problem?**  
Classify a flow as normal or attack using flow features, not raw packets.

**Why UNSW-NB15?**  
Public, labelled, mixed feature types, several attack families, standard train/test files.

**Target variable?**  
`label`. 0 normal, 1 attack.

**Why not use `attack_cat` as a feature?**  
It names the attack. That is leakage. I only use it to compute per-type recall after prediction.

**What does from scratch mean?**  
K-Means, logreg, NB, PCA, ROC, AUC, confusion matrix are in `scratch_ml.py`. sklearn is not used for models.

**Walk through preprocessing.**  
Drop id / attack_cat / label → median fill → log1p on skewed non-negative columns → z-score → clip → one-hot with `other`. Fit on train split only.

**Why log1p?**  
`log(1+x)` handles zeros and compresses large byte/rate values.

**Why standardise?**  
Otherwise `sload` dominates Euclidean distance and the gradient.

**Why fit preproc on train only?**  
If test means leak into scaling, test scores are optimistic.

**Why did logistic regression win?**  
It uses labels, a linear boundary is enough after this preprocessing, and it does not assume independent features.

**Why is K-Means recall low?**  
Many attacks are close to some normal cluster. Distance to “normal” is a weak score except for obviously weird traffic (a lot of Generic).

**Why is NB worse than logreg?**  
Features are correlated (bytes, packets, windows). NB multiplies those as if they were independent.

**What is overfitting? Did you overfit?**  
Overfit = train much better than unseen data. Train vs val for logreg are close (0.932 vs 0.928). Test is lower (0.887) mainly because the official test split is different, not because val already diverged.

**What is underfitting?**  
Model too simple / too much L2. In my sweep, L2 = 1 drops acc to ~0.74 on both train and val.

**Why not only accuracy?**  
Test is 68% attack. Always-predict-attack would get 0.68 accuracy and 100% false alarm rate on normal. Precision, recall, FPR, AUC say more.

**Which metric matters in security?**  
Depends. Missed attack → recall. Analyst overload → precision / FPR. I report all of them and keep AUC as a threshold-free summary.

**Precision vs recall in one line.**  
Precision: “of my alerts, how many were attacks?” Recall: “of the attacks, how many did I catch?”

**What is AUC?**  
Area under TPR vs FPR as you slide the threshold. 1 = perfect ranking, 0.5 = random.

**What is the validation set for?**  
Pick k and L2. Test is not used for that.

**What is PCA doing?**  
2-D picture. First two principal axes of variance. Not the classifier.

**Is the dashboard live packets?**  
No. It scores saved test rows.

**What if a new protocol shows up?**  
It is encoded as `proto=other` (same for service/state).

**FPR of logreg is 0.037. What does that mean?**  
About 3.7% of genuine normal flows are flagged at threshold 0.5.

**Why k-means++?**  
Random init can put all centroids in one dense region. k-means++ spreads them using distance-biased sampling.

**Why mini-batch?**  
Faster than full-batch on 65k rows, noisier gradients which is acceptable here.

**Why clip the sigmoid?**  
`exp(± large)` overflows. Clip z to [-30, 30].

**Why variance smoothing in NB?**  
A class can have ~0 variance on a column after encoding. Adding 1e-3 avoids divide-by-zero.

**Did feature selection help?**  
A bit for K-Means AUC (0.75 → 0.81 at 50 features). Logreg is best with all 72. So the submitted models stay on 72 features.

**What would you do next?**  
Cost-based threshold, time-based split, look at logreg weights (which features push toward attack), maybe a small non-linear model, and a real flow exporter if this were production.

**Main limitations?**  
Benchmark data, random val split, Fuzzers still missed, replay is simulated, linear model only.

---

## Demo order (Streamlit)

1. Start: `streamlit run app.py` (need `artifacts/model.pkl` from `train.py`).
2. **Model performance:** show the test table, ROC, confusion matrices. Point at logreg vs K-Means recall.
3. **Per-attack plot:** Generic easy, Fuzzers hard.
4. **Dataset exploration:** class counts, PCA overlap.
5. **Overfitting:** k = 16, L2 = 0, learning curves almost on top of each other, L2 sweep going down.
6. **Live replay:** pick logistic regression, threshold 0.5, start replay. Then raise the threshold and say false alarms drop but you miss more attacks.

If something fails: usually `model.pkl` missing → run `python train.py` first.

---

## Easy mistakes to avoid while speaking

- Do not say the dashboard captures live packets.
- Do not say K-Means is “bad so I should not have used it”. It is the unsupervised control.
- Do not say L2 = 0 means regularisation is useless in general. It lost on **this** sweep.
- Do not quote Worms recall as a headline; n = 130.
- Do not claim sklearn was used for the classifiers.
- If asked for the loss: binary cross-entropy for logreg; inertia for K-Means; likelihood for NB.
- If asked for complexity: each GD epoch is O(n d); K-Means iteration is O(n k d) with the vectorised trick.

---

## File map if they open the code

| File | What to click |
|---|---|
| `preprocess.py` | `Preprocessor.fit` / `transform` |
| `scratch_ml.py` | `KMeans.score`, `LogisticRegression.fit`, `GaussianNB`, `roc_curve` |
| `train.py` | `tune_kmeans`, `tune_logreg`, evaluation loop |
| `app.py` | four tabs |
| `feature_selection_experiment.py` | correlation ranking |
| `artifacts/results.json` | exact numbers |
