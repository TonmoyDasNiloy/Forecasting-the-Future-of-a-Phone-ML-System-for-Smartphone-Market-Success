"""
benchmark_research_models.py
============================
Standalone benchmark — tests every ML model mentioned across the cited research papers
against the actual project dataset (PMF failure-prediction task).

Models benchmarked (from the papers):
  1. Logistic Regression        — baseline across all papers
  2. Random Forest               — AI Agent Marketplace (88.9 %)
  3. SVM (RBF kernel)            — AI Agent Marketplace (87.5 %)
  4. DNN  64-32-16               — AI Agent Marketplace (91.2 %, 0.889 F1)
  5. PCA + DNN  64-32-16         — AI Agent Marketplace (PCA-driven dimensionality reduction)
  6. XGBoost (GBT)               — used in the production system (Pillar 1)

Task: binary classification — predict whether a phone is a market *failure*
      (bottom 25 % of a composite PMF score built from post-launch KPIs).
Features: launch-known specs only (18 numeric + 3 categorical), matching
          the no-leakage contract in config.py.

Output: benchmark_results.csv  (same directory as this file)

NOTE: This script is entirely self-contained. It does NOT import from or
      modify anything in src/.
"""

import time
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Config (mirrors config.py but inlined)
# ─────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / "global_smartphone_bi_v4_final.csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

NUM_SPECS = [
    "price_usd", "5g_or_not", "num_cores", "processor_speed", "battery_capacity",
    "fast_charging_available", "fast_charging", "ram_capacity", "internal_memory",
    "screen_size", "refresh_rate", "num_rear_cameras", "primary_camera_rear",
    "primary_camera_front", "extended_memory_available", "resolution_height",
    "resolution_width", "avg_rating",
]
CAT_SPECS = ["brand_name", "processor_brand", "os"]

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# ─────────────────────────────────────────────
# 1. Build PMF labels (same formula as pillar4_pmf.py)
# ─────────────────────────────────────────────
def _minmax(x, invert=False):
    r = (x - x.min()) / (x.max() - x.min() + 1e-9)
    return 1 - r if invert else r


def build_labels(df):
    g = df.groupby(["brand_name", "model"]).agg(
        avg_return   = ("return_rate_pct",          "mean"),
        avg_nps      = ("nps_score",                "mean"),
        avg_repeat   = ("repeat_purchase_rate_pct", "mean"),
        avg_rec      = ("recommendation_rate_pct",  "mean"),
        avg_warranty = ("warranty_claims_per_100",  "mean"),
    ).reset_index()

    slopes = {}
    for (b, m), sub in df.groupby(["brand_name", "model"]):
        if sub["months_since_launch"].nunique() >= 3:
            slopes[(b, m)] = np.polyfit(sub["months_since_launch"],
                                        sub["units_sold_monthly"], 1)[0]
        else:
            slopes[(b, m)] = 0.0
    g["sales_slope"] = g.apply(lambda r: slopes[(r["brand_name"], r["model"])], axis=1)

    g["pmf_score"] = (0.25 * _minmax(g["avg_nps"])
                    + 0.20 * _minmax(g["avg_repeat"])
                    + 0.20 * _minmax(g["avg_rec"])
                    + 0.15 * _minmax(g["avg_return"],   invert=True)
                    + 0.10 * _minmax(g["avg_warranty"],  invert=True)
                    + 0.10 * _minmax(g["sales_slope"]))
    cutoff = g["pmf_score"].quantile(0.25)
    g["failure"] = (g["pmf_score"] < cutoff).astype(int)
    return g[["brand_name", "model", "failure"]]


# ─────────────────────────────────────────────
# 2. Prepare feature matrix
# ─────────────────────────────────────────────
def prepare_data(df):
    labels = build_labels(df)
    agg = {s: "first" for s in NUM_SPECS}
    agg.update({c: "first" for c in ["processor_brand", "os"]})
    ph = (df.groupby(["brand_name", "model"])
            .agg(agg)
            .reset_index()
            .merge(labels, on=["brand_name", "model"]))

    X = ph[NUM_SPECS + CAT_SPECS].copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[CAT_SPECS] = enc.fit_transform(X[CAT_SPECS])
    y      = ph["failure"].values
    groups = (ph["brand_name"] + "__" + ph["model"]).values
    return X.values.astype(np.float32), y, groups


# ─────────────────────────────────────────────
# 3. DNN definition (paper architecture: 64-32-16)
# ─────────────────────────────────────────────
class DNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
            nn.Linear(64, 32),         nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.2),
            nn.Linear(32, 16),         nn.ReLU(),
            nn.Linear(16, 1),          nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_dnn(X_tr, y_tr, X_te, n_epochs=60, lr=1e-3, batch=256):
    scaler = StandardScaler().fit(X_tr)
    Xtr = torch.tensor(scaler.transform(X_tr), dtype=torch.float32)
    Xte = torch.tensor(scaler.transform(X_te), dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.float32)

    pos_w = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()], dtype=torch.float32)
    model = DNN(Xtr.shape[1])
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCELoss(weight=None)   # we handle imbalance via pos_weight in BCEWithLogitsLoss equiv
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # rebuild net without Sigmoid (BCEWithLogitsLoss expects raw logits)
    model.net[-1] = nn.Identity()

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)
    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(Xte)
        proba  = torch.sigmoid(logits).numpy()
    return proba, scaler


# ─────────────────────────────────────────────
# 4. Metrics helper
# ─────────────────────────────────────────────
def compute_metrics(y_true, proba, inf_ms):
    pred = (proba >= 0.5).astype(int)
    return {
        "Accuracy":       round(accuracy_score(y_true, pred),                   4),
        "F1_Score":       round(f1_score(y_true, pred, zero_division=0),        4),
        "Precision":      round(precision_score(y_true, pred, zero_division=0), 4),
        "Recall":         round(recall_score(y_true, pred, zero_division=0),    4),
        "AUC_ROC":        round(roc_auc_score(y_true, proba),                   4),
        "Inference_ms":   round(inf_ms, 2),
    }


# ─────────────────────────────────────────────
# 5. Main benchmark loop
# ─────────────────────────────────────────────
def main():
    print("Loading dataset ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {df.shape[0]:,} rows  |  {df.shape[1]} columns")

    print("Building features and labels ...")
    X, y, groups = prepare_data(df)
    n_features = X.shape[1]
    print(f"  Phone-level samples: {len(y)}  |  Failure rate: {y.mean():.1%}")

    # Group-aware train/test split (same as production)
    tr_idx, te_idx = next(
        GroupShuffleSplit(1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        .split(X, y, groups)
    )
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    scaler_std = StandardScaler().fit(X_tr)
    X_tr_sc    = scaler_std.transform(X_tr)
    X_te_sc    = scaler_std.transform(X_te)

    results = []

    # ── Model 1: Logistic Regression ─────────────────────────────────────
    print("\n[1/6] Logistic Regression ...")
    lr_clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    t0 = time.perf_counter()
    lr_clf.fit(X_tr_sc, y_tr)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    lr_proba = lr_clf.predict_proba(X_te_sc)[:, 1]
    inf_ms = (time.perf_counter() - t0) * 1000

    m = compute_metrics(y_te, lr_proba, inf_ms)
    m.update({"Model": "Logistic Regression", "Train_time_s": round(train_s, 2),
               "Paper_reported_acc": "84.7% (AI Agent Marketplace)"})
    results.append(m)
    print(f"   Accuracy={m['Accuracy']:.4f}  F1={m['F1_Score']:.4f}  AUC={m['AUC_ROC']:.4f}  Inf={inf_ms:.1f}ms")

    # ── Model 2: Random Forest ────────────────────────────────────────────
    print("[2/6] Random Forest ...")
    rf_clf = RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=3,
                                    class_weight="balanced", n_jobs=-1,
                                    random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    rf_clf.fit(X_tr, y_tr)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    rf_proba = rf_clf.predict_proba(X_te)[:, 1]
    inf_ms = (time.perf_counter() - t0) * 1000

    m = compute_metrics(y_te, rf_proba, inf_ms)
    m.update({"Model": "Random Forest", "Train_time_s": round(train_s, 2),
               "Paper_reported_acc": "88.9% (AI Agent Marketplace)"})
    results.append(m)
    print(f"   Accuracy={m['Accuracy']:.4f}  F1={m['F1_Score']:.4f}  AUC={m['AUC_ROC']:.4f}  Inf={inf_ms:.1f}ms")

    # ── Model 3: SVM (RBF) ────────────────────────────────────────────────
    print("[3/6] SVM (RBF kernel) ...")
    svm_clf = SVC(kernel="rbf", class_weight="balanced", probability=True,
                  random_state=RANDOM_STATE, C=1.0, gamma="scale")
    t0 = time.perf_counter()
    svm_clf.fit(X_tr_sc, y_tr)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    svm_proba = svm_clf.predict_proba(X_te_sc)[:, 1]
    inf_ms = (time.perf_counter() - t0) * 1000

    m = compute_metrics(y_te, svm_proba, inf_ms)
    m.update({"Model": "SVM (RBF)", "Train_time_s": round(train_s, 2),
               "Paper_reported_acc": "87.5% (AI Agent Marketplace)"})
    results.append(m)
    print(f"   Accuracy={m['Accuracy']:.4f}  F1={m['F1_Score']:.4f}  AUC={m['AUC_ROC']:.4f}  Inf={inf_ms:.1f}ms")

    # ── Model 4: DNN 64-32-16 ─────────────────────────────────────────────
    print("[4/6] DNN (64-32-16 architecture) ...")
    t0 = time.perf_counter()
    dnn_proba, _ = train_dnn(X_tr, y_tr, X_te, n_epochs=60)
    train_s = time.perf_counter() - t0

    # inference time only
    dnn_scaler = StandardScaler().fit(X_tr)
    X_te_t = torch.tensor(dnn_scaler.transform(X_te), dtype=torch.float32)
    dnn_model = DNN(n_features)
    dnn_model.net[-1] = nn.Identity()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = torch.sigmoid(dnn_model(X_te_t)).numpy()
    inf_ms = (time.perf_counter() - t0) * 1000

    m = compute_metrics(y_te, dnn_proba, inf_ms)
    m.update({"Model": "DNN (64-32-16)", "Train_time_s": round(train_s, 2),
               "Paper_reported_acc": "91.2% / F1=0.889 (AI Agent Marketplace)"})
    results.append(m)
    print(f"   Accuracy={m['Accuracy']:.4f}  F1={m['F1_Score']:.4f}  AUC={m['AUC_ROC']:.4f}  Inf={inf_ms:.1f}ms")

    # ── Model 5: PCA + DNN 64-32-16 ──────────────────────────────────────
    print("[5/6] PCA + DNN (64-32-16) ...")
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)  # keep 95% variance
    X_tr_pca = pca.fit_transform(X_tr_sc)
    X_te_pca = pca.transform(X_te_sc)
    print(f"   PCA reduced {n_features} -> {pca.n_components_} components (95% var)")

    t0 = time.perf_counter()
    pca_dnn_proba, _ = train_dnn(X_tr_pca.astype(np.float32),
                                  y_tr, X_te_pca.astype(np.float32), n_epochs=60)
    train_s = time.perf_counter() - t0

    X_te_pca_t = torch.tensor(
        StandardScaler().fit(X_tr_pca).transform(X_te_pca), dtype=torch.float32)
    pca_dnn_model = DNN(X_tr_pca.shape[1])
    pca_dnn_model.net[-1] = nn.Identity()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = torch.sigmoid(pca_dnn_model(X_te_pca_t)).numpy()
    inf_ms = (time.perf_counter() - t0) * 1000

    m = compute_metrics(y_te, pca_dnn_proba, inf_ms)
    m.update({"Model": "PCA + DNN (64-32-16)", "Train_time_s": round(train_s, 2),
               "Paper_reported_acc": "91.2% (AI Agent Marketplace — PCA pipeline)"})
    results.append(m)
    print(f"   Accuracy={m['Accuracy']:.4f}  F1={m['F1_Score']:.4f}  AUC={m['AUC_ROC']:.4f}  Inf={inf_ms:.1f}ms")

    # ── Model 6: XGBoost ──────────────────────────────────────────────────
    print("[6/6] XGBoost (Gradient-Boosted Decision Trees) ...")
    scale_pos = (y_tr == 0).sum() / (y_tr == 1).sum()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        n_jobs=-1, random_state=RANDOM_STATE,
        verbosity=0,
    )
    t0 = time.perf_counter()
    xgb_clf.fit(X_tr, y_tr,
                eval_set=[(X_te, y_te)],
                verbose=False)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    xgb_proba = xgb_clf.predict_proba(X_te)[:, 1]
    inf_ms = (time.perf_counter() - t0) * 1000

    m = compute_metrics(y_te, xgb_proba, inf_ms)
    m.update({"Model": "XGBoost (GBT)", "Train_time_s": round(train_s, 2),
               "Paper_reported_acc": "Production system (Pillar 1 & 4)"})
    results.append(m)
    print(f"   Accuracy={m['Accuracy']:.4f}  F1={m['F1_Score']:.4f}  AUC={m['AUC_ROC']:.4f}  Inf={inf_ms:.1f}ms")

    # ─────────────────────────────────────────────
    # 6. Save CSV report
    # ─────────────────────────────────────────────
    col_order = [
        "Model", "Accuracy", "F1_Score", "Precision", "Recall",
        "AUC_ROC", "Inference_ms", "Train_time_s", "Paper_reported_acc"
    ]
    report_df = pd.DataFrame(results)[col_order].sort_values("AUC_ROC", ascending=False)
    out_path  = Path(__file__).parent / "benchmark_results.csv"
    report_df.to_csv(out_path, index=False)
    print(f"\nReport saved -> {out_path}")

    # ─────────────────────────────────────────────
    # 7. Console summary table
    # ─────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Model':<25} {'Accuracy':>10} {'F1':>8} {'AUC':>8} {'Inf(ms)':>10} {'Train(s)':>10}")
    print("-" * 90)
    for _, row in report_df.iterrows():
        print(f"{row['Model']:<25} {row['Accuracy']:>10.4f} {row['F1_Score']:>8.4f} "
              f"{row['AUC_ROC']:>8.4f} {row['Inference_ms']:>10.1f} {row['Train_time_s']:>10.2f}")
    print("=" * 90)
    print(f"\nBest model by AUC: {report_df.iloc[0]['Model']}  "
          f"(AUC={report_df.iloc[0]['AUC_ROC']:.4f}, "
          f"Acc={report_df.iloc[0]['Accuracy']:.4f})")


if __name__ == "__main__":
    main()
