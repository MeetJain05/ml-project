"""Load UNSW-NB15 CSVs and turn mixed columns into a numeric matrix."""
import numpy as np
import pandas as pd

CAT_COLS = ["proto", "service", "state"]
NON_FEATURES = ["id", "attack_cat", "label"]
TOP_K = {"proto": 10, "service": 13, "state": 8}


def load_csv(path):
    df = pd.read_csv(path)
    df["attack_cat"] = df["attack_cat"].fillna("Normal").astype(str).str.strip()
    return df


def train_val_split(df, val_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_val = int(len(df) * val_frac)
    val = df.iloc[idx[:n_val]].reset_index(drop=True)
    train = df.iloc[idx[n_val:]].reset_index(drop=True)
    return train, val


class Preprocessor:
    def fit(self, df):
        self.num_cols = [c for c in df.columns if c not in CAT_COLS + NON_FEATURES]
        num = df[self.num_cols].apply(pd.to_numeric, errors="coerce")

        self.medians = num.median()
        num = num.fillna(self.medians)

        self.log_cols = [c for c in self.num_cols
                         if num[c].min() >= 0 and num[c].skew() > 2]
        num[self.log_cols] = np.log1p(num[self.log_cols])

        self.mean = num.mean().to_numpy()
        std = num.std().to_numpy()
        self.std = np.where((std == 0) | np.isnan(std), 1.0, std)

        self.categories = {c: [str(v) for v in df[c].value_counts().head(k).index]
                           for c, k in TOP_K.items()}

        self.feature_names = list(self.num_cols)
        for c in CAT_COLS:
            self.feature_names += [f"{c}={v}" for v in self.categories[c] + ["other"]]
        return self

    def transform(self, df):
        num = df[self.num_cols].apply(pd.to_numeric, errors="coerce").fillna(self.medians)
        num[self.log_cols] = np.log1p(num[self.log_cols].clip(lower=0))
        Z = (num.to_numpy() - self.mean) / self.std
        Z = np.clip(Z, -5, 5)

        parts = [Z]
        for c in CAT_COLS:
            vals = df[c].astype(str)
            vals = vals.where(vals.isin(self.categories[c]), "other")
            for cat in self.categories[c] + ["other"]:
                parts.append((vals == cat).to_numpy(dtype=float)[:, None])
        return np.hstack(parts)
