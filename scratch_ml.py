"""
K-Means, logistic regression, Gaussian NB, PCA and metrics.
Written with NumPy only (no sklearn models).
"""
import numpy as np


def sq_dists(X, C):
    # ||x-c||^2 = ||x||^2 - 2x.c + ||c||^2  (avoids a huge 3-D array)
    d = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
    return np.maximum(d, 0)


class KMeans:
    def __init__(self, k=8, max_iter=100, tol=1e-4, seed=0):
        self.k, self.max_iter, self.tol, self.seed = k, max_iter, tol, seed

    def _init_plus_plus(self, X, rng):
        centroids = [X[rng.integers(len(X))]]
        d2 = sq_dists(X, centroids[0][None, :])[:, 0]
        for _ in range(1, self.k):
            nxt = X[rng.choice(len(X), p=d2 / d2.sum())]
            centroids.append(nxt)
            d2 = np.minimum(d2, sq_dists(X, nxt[None, :])[:, 0])
        return np.array(centroids)

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        C = self._init_plus_plus(X, rng)
        for it in range(self.max_iter):
            d = sq_dists(X, C)
            labels = d.argmin(1)
            new_C = C.copy()
            for j in range(self.k):
                members = X[labels == j]
                if len(members) > 0:
                    new_C[j] = members.mean(0)
                else:
                    new_C[j] = X[d.min(1).argmax()]
            shift = np.linalg.norm(new_C - C)
            C = new_C
            if shift < self.tol:
                break
        self.centroids = C
        self.n_iter_ = it + 1
        self.inertia_ = float(sq_dists(X, C).min(1).sum())
        return self

    def predict(self, X):
        return sq_dists(X, self.centroids).argmin(1)

    def score(self, X):
        return np.sqrt(sq_dists(X, self.centroids).min(1))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def bce(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class LogisticRegression:
    def __init__(self, lr=0.1, epochs=30, batch_size=512, l2=0.0, seed=0):
        self.lr, self.epochs, self.batch_size, self.l2, self.seed = lr, epochs, batch_size, l2, seed

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        self.train_loss, self.val_loss = [], []

        for _ in range(self.epochs):
            order = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = order[s:s + self.batch_size]
                err = sigmoid(X[idx] @ self.w + self.b) - y[idx]
                grad_w = X[idx].T @ err / len(idx) + self.l2 * self.w
                grad_b = err.mean()
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b
            self.train_loss.append(bce(y, self.predict_proba(X)))
            if X_val is not None:
                self.val_loss.append(bce(y_val, self.predict_proba(X_val)))
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, thr=0.5):
        return (self.predict_proba(X) >= thr).astype(int)


class GaussianNB:
    def __init__(self, var_smoothing=1e-3):
        self.eps = var_smoothing

    def fit(self, X, y):
        self.classes = np.array([0, 1])
        self.prior = np.array([(y == c).mean() for c in self.classes])
        self.mu = np.array([X[y == c].mean(0) for c in self.classes])
        self.var = np.array([X[y == c].var(0) + self.eps for c in self.classes])
        return self

    def _joint_log_lik(self, X):
        out = []
        for c in range(2):
            ll = -0.5 * (np.log(2 * np.pi * self.var[c]) + (X - self.mu[c]) ** 2 / self.var[c]).sum(1)
            out.append(np.log(self.prior[c]) + ll)
        return np.stack(out, axis=1)

    def predict_proba(self, X):
        jll = self._joint_log_lik(X)
        return sigmoid(jll[:, 1] - jll[:, 0])

    def predict(self, X, thr=0.5):
        return (self.predict_proba(X) >= thr).astype(int)


class PCA:
    def __init__(self, n_components=2):
        self.n = n_components

    def fit(self, X):
        self.mean = X.mean(0)
        cov = np.cov(X - self.mean, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1][:self.n]
        self.components = vecs[:, order]
        self.explained_ = vals[order] / vals.sum()
        return self

    def transform(self, X):
        return (X - self.mean) @ self.components


def confusion(y, pred):
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    return tn, fp, fn, tp


def prf(y, pred):
    tn, fp, fn, tp = confusion(y, pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
            "f1": f1, "fpr": fp / (fp + tn) if fp + tn else 0.0}


def roc_curve(y, score):
    order = np.argsort(-score, kind="mergesort")
    y_s, s_s = y[order], score[order]
    tps, fps = np.cumsum(y_s), np.cumsum(1 - y_s)
    keep = np.r_[np.where(np.diff(s_s))[0], len(y) - 1]
    tpr = np.r_[0, tps[keep] / tps[-1]]
    fpr = np.r_[0, fps[keep] / fps[-1]]
    return fpr, tpr


def auc(fpr, tpr):
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
