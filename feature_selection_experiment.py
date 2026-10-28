"""
Rank preprocessed features by |corr| with the label (training split only),
then retrain the three models on the top-N features.
Does not change the main train.py pipeline.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from preprocess import load_csv, train_val_split, Preprocessor
from scratch_ml import KMeans, LogisticRegression, GaussianNB
from scratch_ml import auc, prf, roc_curve

SEED = 42
DEFAULT_COUNTS = (10, 20, 30, 40, 50, 72)
MODEL_NAMES = (
    "K-Means (unsupervised)",
    "Logistic Regression",
    "Gaussian NB",
)


def feature_scores(X, y):
    centered_x = X - X.mean(axis=0)
    centered_y = y.astype(float) - y.mean()
    numerator = np.abs(centered_x.T @ centered_y)
    denominator = np.sqrt((centered_x ** 2).sum(axis=0) * (centered_y ** 2).sum())
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def select_features(Xtr, ytr, Xva, Xte, names, count):
    scores = feature_scores(Xtr, ytr)
    order = np.argsort(-scores, kind="mergesort")
    chosen = order[:count]
    return (
        Xtr[:, chosen],
        Xva[:, chosen],
        Xte[:, chosen],
        [names[i] for i in chosen],
        scores[chosen],
    )


def binary_metrics(y, score, threshold):
    prediction = (score >= threshold).astype(int)
    result = prf(y, prediction)
    result["auc"] = auc(*roc_curve(y, score))
    return result


def evaluate_count(Xtr, ytr, Xva, yva, Xte, yte):
    normal = Xtr[ytr == 0]
    candidate_ks = (4, 8, 12, 16)
    k_models = []
    for k in candidate_ks:
        model = KMeans(k, seed=SEED).fit(normal)
        validation_auc = auc(*roc_curve(yva, model.score(Xva)))
        k_models.append((validation_auc, model, k))
    _, km, best_k = max(k_models, key=lambda row: row[0])
    normal_scores = km.score(normal)
    km_threshold = float(np.quantile(normal_scores, 0.95))

    l2_candidates = (0.0, 1e-4, 1e-3, 1e-2)
    lr_models = []
    for l2 in l2_candidates:
        model = LogisticRegression(
            lr=0.1, epochs=25, batch_size=512, l2=l2, seed=SEED
        ).fit(Xtr, ytr, Xva, yva)
        validation_loss = model.val_loss[-1]
        lr_models.append((validation_loss, model, l2))
    _, lr, best_l2 = min(lr_models, key=lambda row: row[0])

    nb = GaussianNB().fit(Xtr, ytr)
    detectors = {
        MODEL_NAMES[0]: (km.score, km_threshold),
        MODEL_NAMES[1]: (lr.predict_proba, 0.5),
        MODEL_NAMES[2]: (nb.predict_proba, 0.5),
    }

    metrics = {}
    for name, (score_fn, threshold) in detectors.items():
        metrics[name] = {
            "validation": binary_metrics(yva, score_fn(Xva), threshold),
            "test": binary_metrics(yte, score_fn(Xte), threshold),
        }
    return metrics, best_k, best_l2


def main(args):
    train_full = load_csv(args.train)
    test = load_csv(args.test)
    train, validation = train_val_split(train_full, 0.2, SEED)

    pre = Preprocessor().fit(train)
    Xtr, Xva, Xte = pre.transform(train), pre.transform(validation), pre.transform(test)
    ytr = train["label"].to_numpy()
    yva = validation["label"].to_numpy()
    yte = test["label"].to_numpy()
    counts = tuple(sorted(set(min(c, Xtr.shape[1]) for c in args.counts)))

    ranking = feature_scores(Xtr, ytr)
    order = np.argsort(-ranking, kind="mergesort")
    print(f"Training features before selection: {Xtr.shape[1]}")
    print(f"Feature counts evaluated: {', '.join(map(str, counts))}")
    print("Top 10 features by |corr| with label:")
    for index in order[:10]:
        print(f"  {pre.feature_names[index]:35s} score={ranking[index]:.4f}")

    experiment = {}
    for count in counts:
        selected = select_features(
            Xtr, ytr, Xva, Xte, pre.feature_names, count
        )
        Xtr_selected, Xva_selected, Xte_selected, names, scores = selected
        metrics, best_k, best_l2 = evaluate_count(
            Xtr_selected, ytr, Xva_selected, yva, Xte_selected, yte
        )
        experiment[str(count)] = {
            "selected_features": names,
            "selected_scores": [float(value) for value in scores],
            "best_k": best_k,
            "best_l2": best_l2,
            "metrics": metrics,
        }
        print(f"\nTop {count} features: best_k={best_k}, best_l2={best_l2}")
        for model_name in MODEL_NAMES:
            result = metrics[model_name]["test"]
            print(
                f"  {model_name:24s} "
                f"accuracy={result['accuracy']:.4f} "
                f"precision={result['precision']:.4f} "
                f"recall={result['recall']:.4f} "
                f"f1={result['f1']:.4f} "
                f"auc={result['auc']:.4f}"
            )

    selected_best = {}
    for model_name in MODEL_NAMES:
        selected_best[model_name] = max(
            (
                (values["metrics"][model_name]["validation"]["auc"], int(count), values)
                for count, values in experiment.items()
            ),
            key=lambda row: row[0],
        )

    print("\nValidation-selected feature counts and final test results:")
    for model_name, (_, count, values) in selected_best.items():
        result = values["metrics"][model_name]["test"]
        print(
            f"  {model_name:24s} features={count:2d} "
            f"accuracy={result['accuracy']:.4f} "
            f"precision={result['precision']:.4f} "
            f"recall={result['recall']:.4f} "
            f"f1={result['f1']:.4f} "
            f"auc={result['auc']:.4f}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "training-only absolute Pearson correlation ranking",
        "seed": SEED,
        "feature_counts": list(counts),
        "training_features_before_selection": int(Xtr.shape[1]),
        "selected_best": {
            name: {
                "feature_count": count,
                "validation_auc": float(validation_auc),
                "best_k": values["best_k"],
                "best_l2": values["best_l2"],
                "test_metrics": values["metrics"][name]["test"],
                "selected_features": values["selected_features"],
            }
            for name, (validation_auc, count, values) in selected_best.items()
        },
        "experiments": experiment,
    }
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"\nSaved detailed results to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/unsw_train.csv")
    parser.add_argument("--test", default="data/unsw_test.csv")
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_COUNTS),
    )
    parser.add_argument(
        "--output",
        default="artifacts/feature_selection_results.json",
    )
    main(parser.parse_args())
