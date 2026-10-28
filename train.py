"""Train the three detectors and write artifacts/ for the dashboard."""
import argparse, json, pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import load_csv, train_val_split, Preprocessor
from scratch_ml import (KMeans, LogisticRegression, GaussianNB, PCA,
                        prf, confusion, roc_curve, auc)

SEED = 42
ART = Path("artifacts")
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)

NAMES = ["K-Means (unsupervised)", "Logistic Regression", "Gaussian NB"]
COLORS = ["tab:green", "tab:blue", "tab:orange"]


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=130)
    plt.close()


def eda(df, X, y):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(["Normal", "Attack"], [(y == 0).sum(), (y == 1).sum()], color=["tab:green", "tab:red"])
    ax[0].set_title("Binary label distribution (train)")
    cats = df["attack_cat"].value_counts()
    ax[1].barh(cats.index[::-1], cats.values[::-1])
    ax[1].set_title("Traffic by category (train)")
    ax[1].set_xlabel("flows")
    savefig("eda_class_balance.png")

    feats = ["dur", "sbytes", "dbytes", "sload", "sttl", "ct_srv_src"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for a, f in zip(axes.ravel(), feats):
        v = np.log1p(df[f].clip(lower=0))
        a.hist(v[y == 0], bins=40, alpha=0.6, label="normal", color="tab:green", density=True)
        a.hist(v[y == 1], bins=40, alpha=0.6, label="attack", color="tab:red", density=True)
        a.set_title(f"log1p({f})")
    axes[0, 0].legend()
    savefig("eda_feature_dists.png")

    num = df.select_dtypes("number").drop(columns=["id", "label"])
    num = num.loc[:, num.std() > 0]
    corr_y = num.apply(lambda c: np.corrcoef(c, y)[0, 1]).abs().sort_values(ascending=False)
    top = list(corr_y.index[:15])
    cm = np.corrcoef(num[top].to_numpy().T)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(top)), top, rotation=90)
    plt.yticks(range(len(top)), top)
    plt.colorbar()
    plt.title("Correlation between top-15 label-correlated features")
    savefig("eda_correlation.png")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), min(6000, len(X)), replace=False)
    pca = PCA(2).fit(X)
    P = pca.transform(X[idx])
    plt.figure(figsize=(6.5, 5))
    plt.scatter(P[y[idx] == 0, 0], P[y[idx] == 0, 1], s=4, c="tab:green", alpha=0.4, label="normal")
    plt.scatter(P[y[idx] == 1, 0], P[y[idx] == 1, 1], s=4, c="tab:red", alpha=0.4, label="attack")
    plt.xlabel(f"PC1 ({pca.explained_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_[1]*100:.1f}% var)")
    plt.title("PCA projection of preprocessed flows")
    plt.legend(markerscale=3)
    savefig("eda_pca.png")


def tune_kmeans(X_norm, Xtr, ytr, Xva, yva):
    ks = [2, 3, 4, 6, 8, 10, 12, 16]
    rows, models = [], {}
    for k in ks:
        km = KMeans(k, seed=SEED).fit(X_norm)
        a_tr = auc(*roc_curve(ytr, km.score(Xtr)))
        a_va = auc(*roc_curve(yva, km.score(Xva)))
        rows.append({"k": k, "inertia": km.inertia_, "auc_train": a_tr, "auc_val": a_va})
        models[k] = km
        print(f"  k={k:2d}  inertia={km.inertia_:12.0f}  AUC train={a_tr:.4f}  val={a_va:.4f}")
    best_k = max(rows, key=lambda r: r["auc_val"])["k"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ks, [r["inertia"] for r in rows], "o-")
    ax[0].set_title("Elbow plot (normal traffic)"); ax[0].set_xlabel("k"); ax[0].set_ylabel("inertia")
    ax[1].plot(ks, [r["auc_train"] for r in rows], "o-", label="train")
    ax[1].plot(ks, [r["auc_val"] for r in rows], "s-", label="validation")
    ax[1].set_title("Anomaly-detection AUC vs k"); ax[1].set_xlabel("k"); ax[1].legend()
    savefig("kmeans_k_selection.png")
    return models[best_k], best_k, rows


def tune_logreg(Xtr, ytr, Xva, yva):
    l2s = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    rows, models = [], {}
    for l2 in l2s:
        m = LogisticRegression(lr=0.1, epochs=25, batch_size=512, l2=l2, seed=SEED)
        m.fit(Xtr, ytr, Xva, yva)
        r = {"l2": l2,
             "acc_train": prf(ytr, m.predict(Xtr))["accuracy"],
             "acc_val": prf(yva, m.predict(Xva))["accuracy"],
             "loss_train": m.train_loss[-1], "loss_val": m.val_loss[-1]}
        rows.append(r); models[l2] = m
        print(f"  l2={l2:<7} acc train={r['acc_train']:.4f} val={r['acc_val']:.4f}  "
              f"loss train={r['loss_train']:.4f} val={r['loss_val']:.4f}")
    best_l2 = min(rows, key=lambda r: r["loss_val"])["l2"]
    best = models[best_l2]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(best.train_loss, label="train"); ax[0].plot(best.val_loss, label="validation")
    ax[0].set_title(f"LogReg learning curve (l2={best_l2})"); ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("log loss"); ax[0].legend()
    xs = np.arange(len(l2s))
    ax[1].plot(xs, [r["acc_train"] for r in rows], "o-", label="train")
    ax[1].plot(xs, [r["acc_val"] for r in rows], "s-", label="validation")
    ax[1].set_xticks(xs); ax[1].set_xticklabels([str(v) for v in l2s])
    ax[1].set_xlabel("L2 strength"); ax[1].set_title("Regularisation sweep"); ax[1].legend()
    savefig("logreg_overfitting.png")
    return best, best_l2, rows


def summarize(y, score, pred):
    out = prf(y, pred)
    out["auc"] = auc(*roc_curve(y, score))
    return out


def main(args):
    print("Loading data...")
    train_full, test = load_csv(args.train), load_csv(args.test)
    train, val = train_val_split(train_full, 0.2, SEED)

    pre = Preprocessor().fit(train)
    Xtr, Xva, Xte = pre.transform(train), pre.transform(val), pre.transform(test)
    ytr, yva, yte = (d["label"].to_numpy() for d in (train, val, test))
    print(f"train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}")

    print("EDA plots...")
    eda(train, Xtr, ytr)

    print("K-Means (fit on normal flows only):")
    X_norm = Xtr[ytr == 0]
    km, best_k, k_rows = tune_kmeans(X_norm, Xtr, ytr, Xva, yva)
    d_norm = km.score(X_norm)
    km_thr = float(np.quantile(d_norm, 0.95))

    print("Logistic regression:")
    lr, best_l2, l2_rows = tune_logreg(Xtr, ytr, Xva, yva)

    print("Gaussian NB...")
    nb = GaussianNB().fit(Xtr, ytr)

    detectors = {
        NAMES[0]: (km.score, km_thr),
        NAMES[1]: (lr.predict_proba, 0.5),
        NAMES[2]: (nb.predict_proba, 0.5),
    }

    results = {"train": {}, "test": {}, "per_attack": {}}
    test_scores = {}
    for name, (fn, thr) in detectors.items():
        for split, X, y in (("train", Xtr, ytr), ("test", Xte, yte)):
            s = fn(X)
            results[split][name] = summarize(y, s, (s > thr).astype(int))
            if split == "test":
                test_scores[name] = s
        print(f"  {name:24s} test: " + ", ".join(f"{k}={v:.3f}" for k, v in results['test'][name].items()))

    cats = [c for c in sorted(test["attack_cat"].unique()) if c != "Normal"]
    for c in cats:
        mask = (test["attack_cat"] == c).to_numpy()
        results["per_attack"][c] = {n: float((test_scores[n][mask] > detectors[n][1]).mean())
                                    for n in NAMES}
        results["per_attack"][c]["n_flows"] = int(mask.sum())

    plt.figure(figsize=(6, 5))
    for n, col in zip(NAMES, COLORS):
        fpr, tpr = roc_curve(yte, test_scores[n])
        plt.plot(fpr, tpr, color=col, label=f"{n} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves on test set"); plt.legend(loc="lower right")
    savefig("roc_test.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for a, n in zip(axes, NAMES):
        tn, fp, fn_, tp = confusion(yte, (test_scores[n] > detectors[n][1]).astype(int))
        a.imshow([[tn, fp], [fn_, tp]], cmap="Blues")
        for (i, j), v in np.ndenumerate([[tn, fp], [fn_, tp]]):
            a.text(j, i, f"{v:,}", ha="center", va="center")
        a.set_xticks([0, 1]); a.set_xticklabels(["pred normal", "pred attack"])
        a.set_yticks([0, 1]); a.set_yticklabels(["normal", "attack"])
        a.set_title(n)
    savefig("confusion_matrices.png")

    plt.figure(figsize=(11, 4.5))
    w = 0.27
    for i, (n, col) in enumerate(zip(NAMES, COLORS)):
        plt.bar(np.arange(len(cats)) + (i - 1) * w, [results["per_attack"][c][n] for c in cats],
                w, label=n, color=col)
    plt.xticks(range(len(cats)), cats, rotation=30)
    plt.ylabel("detection rate (recall)"); plt.ylim(0, 1.05)
    plt.title("Per attack-type recall on test set"); plt.legend()
    savefig("per_attack_recall.png")

    results["k_sweep"], results["l2_sweep"] = k_rows, l2_rows
    results["best_k"], results["best_l2"] = best_k, best_l2
    results["sizes"] = {"train": len(train), "val": len(val), "test": len(test), "features": Xtr.shape[1]}
    with open(ART / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(ART / "model.pkl", "wb") as f:
        pickle.dump({"pre": pre, "km": km, "lr": lr, "nb": nb, "km_thr": km_thr,
                     "km_normal_dists": d_norm}, f)

    test.sample(n=min(10000, len(test)), random_state=SEED).to_csv(ART / "replay_sample.csv", index=False)
    print("Done. Artifacts saved in ./artifacts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/unsw_train.csv")
    ap.add_argument("--test", default="data/unsw_test.csv")
    main(ap.parse_args())
