"""Streamlit dashboard. Run after train.py:  streamlit run app.py"""
import json, pickle, time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Network Intrusion Detection", layout="wide")
ART = Path(__file__).parent / "artifacts"
MODEL_LABELS = {"K-Means anomaly detector": "km",
                "Logistic Regression": "lr",
                "Gaussian Naive Bayes": "nb"}


@st.cache_resource
def load_all():
    with open(ART / "model.pkl", "rb") as f:
        art = pickle.load(f)
    with open(ART / "results.json") as f:
        res = json.load(f)
    flows = pd.read_csv(ART / "replay_sample.csv")
    X = art["pre"].transform(flows)
    scores = {"km": art["km"].score(X),
              "lr": art["lr"].predict_proba(X),
              "nb": art["nb"].predict_proba(X)}
    return art, res, flows, scores


if not (ART / "model.pkl").exists():
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

art, res, flows, scores = load_all()

st.title("Network Intrusion Detection on UNSW-NB15")
st.caption("K-Means vs logistic regression vs Gaussian NB (NumPy implementations).")

tab_live, tab_perf, tab_eda, tab_fit = st.tabs(
    ["Live replay", "Model performance", "Dataset exploration", "Overfitting analysis"])

with tab_live:
    c1, c2, c3 = st.columns(3)
    choice = c1.selectbox("Detector", list(MODEL_LABELS))
    key = MODEL_LABELS[choice]
    s = scores[key]

    if key == "km":
        pct = c2.slider("Alert threshold (percentile of normal-traffic distance)", 80.0, 99.9, 95.0)
        thr = float(np.quantile(art["km_normal_dists"], pct / 100))
        flags = s > thr
        c2.caption(f"distance threshold = {thr:.2f}")
    else:
        thr = c2.slider("Alert threshold (P(attack))", 0.05, 0.95, 0.5)
        flags = s >= thr

    n_flows = c3.slider("Flows to replay", 200, 5000, 1500, step=100)
    speed = st.select_slider("Replay speed", ["slow", "normal", "fast"], value="normal")
    delay = {"slow": 0.5, "normal": 0.2, "fast": 0.05}[speed]
    batch = 25

    if st.button("Start replay", type="primary"):
        order = np.random.default_rng(1).permutation(len(flows))[:n_flows]
        y_all = flows["label"].to_numpy()

        metric_ph, chart_ph, table_ph = st.empty(), st.empty(), st.empty()
        seen = alerts = caught = false_alarms = 0
        history = []

        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            seen += len(idx)
            alerts += int(flags[idx].sum())
            caught += int((flags[idx] & (y_all[idx] == 1)).sum())
            false_alarms += int((flags[idx] & (y_all[idx] == 0)).sum())
            history.append({"alerts": alerts, "true attacks seen": int(y_all[order[:seen]].sum())})

            with metric_ph.container():
                m = st.columns(4)
                m[0].metric("Flows processed", seen)
                m[1].metric("Alerts raised", alerts)
                m[2].metric("Attacks caught", caught)
                m[3].metric("False alarms", false_alarms)
            chart_ph.line_chart(pd.DataFrame(history))

            recent = order[max(0, start + batch - 12):start + batch]
            view = flows.iloc[recent][["proto", "service", "state", "dur", "sbytes", "dbytes", "attack_cat"]].copy()
            view["score"] = s[recent].round(3)
            view["status"] = np.where(flags[recent], "ALERT", "ok")
            styled = view.style.apply(
                lambda r: ["background-color:#ffd6d6;color:#000" if r["status"] == "ALERT" else "" for _ in r],
                axis=1)
            table_ph.dataframe(styled, use_container_width=True, hide_index=True)
            time.sleep(delay)
        st.success("Replay finished.")

with tab_perf:
    test_df = pd.DataFrame(res["test"]).T.round(4)
    train_df = pd.DataFrame(res["train"]).T.round(4)
    st.subheader("Test-set metrics")
    st.dataframe(test_df, use_container_width=True)
    st.subheader("Train-set metrics (for comparison)")
    st.dataframe(train_df, use_container_width=True)
    a, b = st.columns(2)
    a.image(str(ART / "figures/roc_test.png"), caption="ROC curves")
    b.image(str(ART / "figures/per_attack_recall.png"), caption="Per attack-type recall")
    st.image(str(ART / "figures/confusion_matrices.png"))

with tab_eda:
    st.write(f"Features after preprocessing: **{res['sizes']['features']}** | "
             f"train/val/test flows: {res['sizes']['train']:,} / {res['sizes']['val']:,} / {res['sizes']['test']:,}")
    a, b = st.columns(2)
    a.image(str(ART / "figures/eda_class_balance.png"))
    b.image(str(ART / "figures/eda_pca.png"))
    st.image(str(ART / "figures/eda_feature_dists.png"))
    st.image(str(ART / "figures/eda_correlation.png"))

with tab_fit:
    st.markdown(f"Best K-Means k = **{res['best_k']}**, best L2 for logistic regression = **{res['best_l2']}**.")
    a, b = st.columns(2)
    a.image(str(ART / "figures/kmeans_k_selection.png"))
    b.image(str(ART / "figures/logreg_overfitting.png"))
    st.subheader("L2 sweep")
    st.dataframe(pd.DataFrame(res["l2_sweep"]).round(4), use_container_width=True)
    st.subheader("k sweep")
    st.dataframe(pd.DataFrame(res["k_sweep"]).round(4), use_container_width=True)
