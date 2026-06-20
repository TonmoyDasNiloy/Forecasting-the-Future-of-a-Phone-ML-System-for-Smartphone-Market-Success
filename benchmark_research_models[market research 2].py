"""
benchmark_research_models.py  (v2 — full research coverage)
=============================================================
Tests EVERY model / approach cited across all research papers against the
actual project dataset (PMF failure-prediction, binary classification).

Papers covered
--------------
[1] AI Agent Marketplace          -> LR, RF, SVM, DNN 64-32-16, PCA+DNN
[2] AI in Marketing               -> Naive Bayes, KNN, Decision Tree (predictive analytics)
[3] AI-Driven Business Analytics  -> AdaBoost, Gradient Boosting, Extra Trees
[4] Leveraging AI (BizDev)        -> LightGBM (gradient boosting variant)
[5] MaRGen                        -> CatBoost (production-grade GBT; Multi-Agent LLM noted as N/A)
[6] AI Agents for Deep Sci.       -> ADAS-inspired AutoML stack (noted as N/A — NLP benchmarks only)
[7] Csaszar et al.                -> GPT-3.5 / GPT-4 zero-shot (noted as N/A — strategy task)
[8] AI Strategic Framework        -> GPT-4 LLM noted as N/A — conceptual framework

Fully trainable models (14)
----------------------------
 1  Logistic Regression
 2  Naive Bayes
 3  K-Nearest Neighbors
 4  Decision Tree
 5  SVM (RBF)
 6  AdaBoost
 7  Random Forest
 8  Extra Trees
 9  Gradient Boosting (sklearn GBM)
10  XGBoost (GBT)
11  LightGBM
12  CatBoost
13  DNN 64-32-16         (PyTorch)
14  PCA + DNN 64-32-16   (PyTorch)

Non-trainable on this task (4 rows, included in CSV as reference)
-------------------
15  PaperQA2             (retrieval-augmented LLM — scientific literature, not tabular)
16  Multi-Agent LLM      (MaRGen Researcher/Writer/Reviewer — consulting reports, not tabular)
17  GPT-3.5 / GPT-4      (entrepreneurial strategy scoring — not tabular classification)
18  Human Experts        (LitQA2 retrieval baseline — reference only)

Output: benchmark_results.csv
NOTE: zero modifications to src/.
"""

import sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection  import GroupShuffleSplit
from sklearn.preprocessing    import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.decomposition    import PCA
from sklearn.metrics          import (accuracy_score, f1_score, precision_score,
                                      recall_score, roc_auc_score)
# Classical models
from sklearn.linear_model     import LogisticRegression
from sklearn.naive_bayes      import GaussianNB
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.tree             import DecisionTreeClassifier
from sklearn.svm              import SVC
from sklearn.ensemble         import (AdaBoostClassifier, RandomForestClassifier,
                                      ExtraTreesClassifier, GradientBoostingClassifier)
import xgboost  as xgb
import lightgbm as lgb
import catboost as cbt

# Deep Learning
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Config (inlined — no src/ import)
# ──────────────────────────────────────────────────────────────
DATA_PATH    = Path(__file__).parent / "global_smartphone_bi_v4_final.csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

NUM_SPECS = [
    "price_usd","5g_or_not","num_cores","processor_speed","battery_capacity",
    "fast_charging_available","fast_charging","ram_capacity","internal_memory",
    "screen_size","refresh_rate","num_rear_cameras","primary_camera_rear",
    "primary_camera_front","extended_memory_available","resolution_height",
    "resolution_width","avg_rating",
]
CAT_SPECS = ["brand_name","processor_brand","os"]

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# ──────────────────────────────────────────────────────────────
# 1. Build PMF labels (identical formula to pillar4_pmf.py)
# ──────────────────────────────────────────────────────────────
def _minmax(x, invert=False):
    r = (x - x.min()) / (x.max() - x.min() + 1e-9)
    return 1 - r if invert else r

def build_labels(df):
    g = df.groupby(["brand_name","model"]).agg(
        avg_return   = ("return_rate_pct",          "mean"),
        avg_nps      = ("nps_score",                "mean"),
        avg_repeat   = ("repeat_purchase_rate_pct", "mean"),
        avg_rec      = ("recommendation_rate_pct",  "mean"),
        avg_warranty = ("warranty_claims_per_100",  "mean"),
    ).reset_index()
    slopes = {}
    for (b,m), sub in df.groupby(["brand_name","model"]):
        slopes[(b,m)] = (np.polyfit(sub["months_since_launch"],
                                    sub["units_sold_monthly"],1)[0]
                         if sub["months_since_launch"].nunique() >= 3 else 0.0)
    g["sales_slope"] = g.apply(lambda r: slopes[(r["brand_name"],r["model"])], axis=1)
    g["pmf_score"] = (0.25*_minmax(g["avg_nps"])
                    + 0.20*_minmax(g["avg_repeat"])
                    + 0.20*_minmax(g["avg_rec"])
                    + 0.15*_minmax(g["avg_return"],  invert=True)
                    + 0.10*_minmax(g["avg_warranty"], invert=True)
                    + 0.10*_minmax(g["sales_slope"]))
    g["failure"] = (g["pmf_score"] < g["pmf_score"].quantile(0.25)).astype(int)
    return g[["brand_name","model","failure"]]

def prepare_data(df):
    labels = build_labels(df)
    agg = {s:"first" for s in NUM_SPECS}
    agg.update({c:"first" for c in ["processor_brand","os"]})
    ph = (df.groupby(["brand_name","model"])
            .agg(agg).reset_index()
            .merge(labels, on=["brand_name","model"]))
    X = ph[NUM_SPECS + CAT_SPECS].copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[CAT_SPECS] = enc.fit_transform(X[CAT_SPECS])
    y      = ph["failure"].values
    groups = (ph["brand_name"]+"__"+ph["model"]).values
    return X.values.astype(np.float32), y, groups, enc


# ──────────────────────────────────────────────────────────────
# 2. DNN architecture (paper: 64-32-16)
# ──────────────────────────────────────────────────────────────
class DNN(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
            nn.Linear(64, 32),   nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.2),
            nn.Linear(32, 16),   nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

def train_dnn(X_tr, y_tr, X_te, n_epochs=60, lr=1e-3, batch=256):
    sc  = StandardScaler().fit(X_tr)
    Xtr = torch.tensor(sc.transform(X_tr), dtype=torch.float32)
    Xte = torch.tensor(sc.transform(X_te), dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.float32)
    pos_w = torch.tensor([(y_tr==0).sum()/(y_tr==1).sum()], dtype=torch.float32)
    model   = DNN(Xtr.shape[1])
    opt     = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    loader  = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)
    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(Xte)).numpy()
    return proba, sc, model


# ──────────────────────────────────────────────────────────────
# 3. Metrics
# ──────────────────────────────────────────────────────────────
def metrics(y_true, proba, inf_ms, train_s, model_name, paper_ref, paper_acc, paper_task):
    pred = (proba >= 0.5).astype(int)
    return {
        "Model":                model_name,
        "Category":             paper_ref,
        "Accuracy":             round(accuracy_score(y_true, pred),                    4),
        "F1_Score":             round(f1_score(y_true, pred, zero_division=0),         4),
        "Precision":            round(precision_score(y_true, pred, zero_division=0),  4),
        "Recall":               round(recall_score(y_true, pred, zero_division=0),     4),
        "AUC_ROC":              round(roc_auc_score(y_true, proba),                    4),
        "Inference_ms":         round(inf_ms, 2),
        "Train_time_s":         round(train_s, 2),
        "Paper_Reported_Metric": paper_acc,
        "Paper_Task":           paper_task,
        "Applicable_to_Dataset": "Yes",
    }


# ──────────────────────────────────────────────────────────────
# 4. Non-testable model reference rows
# ──────────────────────────────────────────────────────────────
NON_TESTABLE = [
    {
        "Model": "PaperQA2",
        "Category": "Tool-Augmented LLM [AI Agents for Deep Sci.]",
        "Accuracy": "N/A", "F1_Score": "N/A", "Precision": "N/A",
        "Recall": "N/A", "AUC_ROC": "N/A",
        "Inference_ms": "N/A", "Train_time_s": "N/A",
        "Paper_Reported_Metric": "85.2% Precision (LitQA2)",
        "Paper_Task": "Scientific literature retrieval — not tabular classification",
        "Applicable_to_Dataset": "No — domain mismatch (text retrieval vs tabular specs)",
    },
    {
        "Model": "Multi-Agent LLM (MaRGen: Researcher/Writer/Reviewer/Retriever)",
        "Category": "Multi-Agent LLM [MaRGen paper]",
        "Accuracy": "N/A", "F1_Score": "N/A", "Precision": "N/A",
        "Recall": "N/A", "AUC_ROC": "N/A",
        "Inference_ms": "N/A", "Train_time_s": "N/A",
        "Paper_Reported_Metric": "10/10 Clarity & Layout Score; 7 min at $1 cost",
        "Paper_Task": "Autonomous market consulting reports — not classification",
        "Applicable_to_Dataset": "No — generates text reports, not failure labels",
    },
    {
        "Model": "GPT-3.5 / GPT-4 (Csaszar et al.)",
        "Category": "LLM [Csaszar et al. 2024]",
        "Accuracy": "N/A", "F1_Score": "N/A", "Precision": "N/A",
        "Recall": "N/A", "AUC_ROC": "N/A",
        "Inference_ms": "N/A", "Train_time_s": "N/A",
        "Paper_Reported_Metric": "+0.14 sigma on investment rubric; +5pp acceptance rate; r=0.52 with human judges",
        "Paper_Task": "Entrepreneurial strategy plan evaluation — not tabular classification",
        "Applicable_to_Dataset": "No — requires unstructured strategy text as input",
    },
    {
        "Model": "ADAS-generated Agents (AI Agents paper)",
        "Category": "AutoML / Agentic [AI Agents for Deep Sci.]",
        "Accuracy": "N/A", "F1_Score": "N/A", "Precision": "N/A",
        "Recall": "N/A", "AUC_ROC": "N/A",
        "Inference_ms": "N/A", "Train_time_s": "N/A",
        "Paper_Reported_Metric": "Exceeds hand-designed models on DROP, MGSM, MMLU, GPQA NLP benchmarks",
        "Paper_Task": "NLP reasoning benchmarks — not tabular product classification",
        "Applicable_to_Dataset": "No — designed for NLP reasoning tasks",
    },
    {
        "Model": "Human Experts (LitQA2 baseline)",
        "Category": "Human Baseline [AI Agents for Deep Sci.]",
        "Accuracy": "N/A", "F1_Score": "N/A", "Precision": "N/A",
        "Recall": "N/A", "AUC_ROC": "N/A",
        "Inference_ms": "N/A", "Train_time_s": "N/A",
        "Paper_Reported_Metric": "73.8% Precision (LitQA2 scientific Q&A)",
        "Paper_Task": "Scientific literature Q&A — human reference only",
        "Applicable_to_Dataset": "No — human performance reference, not automatable",
    },
    {
        "Model": "Rejected Entrepreneurs (Csaszar et al.)",
        "Category": "Human Baseline [Csaszar et al. 2024]",
        "Accuracy": "N/A", "F1_Score": "N/A", "Precision": "N/A",
        "Recall": "N/A", "AUC_ROC": "N/A",
        "Inference_ms": "N/A", "Train_time_s": "N/A",
        "Paper_Reported_Metric": "7 pp lower acceptance probability than AI-assisted plans",
        "Paper_Task": "Accelerator acceptance — human reference baseline",
        "Applicable_to_Dataset": "No — human performance reference, not automatable",
    },
    {
        "Model": "LLM-Human Correlation (MaRGen)",
        "Category": "Statistical Metric [MaRGen paper]",
        "Accuracy": "N/A", "F1_Score": "N/A", "Precision": "N/A",
        "Recall": "N/A", "AUC_ROC": "N/A",
        "Inference_ms": "N/A", "Train_time_s": "N/A",
        "Paper_Reported_Metric": "r=0.60 (p<0.01) LLM vs human quality scores",
        "Paper_Task": "Validation metric (Pearson correlation), not a classifier",
        "Applicable_to_Dataset": "No — this is a measurement method, not a model",
    },
]


# ──────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  FULL RESEARCH MODEL BENCHMARK")
    print("=" * 70)
    print("Loading dataset ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {df.shape[0]:,} rows | {df.shape[1]} columns")

    print("Building features and labels ...")
    X, y, groups, enc = prepare_data(df)
    n_feat = X.shape[1]
    print(f"  Phone-level samples: {len(y)} | Failure rate: {y.mean():.1%}")

    tr, te = next(GroupShuffleSplit(1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
                  .split(X, y, groups))
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]
    scale_pos   = (y_tr == 0).sum() / (y_tr == 1).sum()

    sc     = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr)
    X_te_s = sc.transform(X_te)

    results = []

    # helper: fit sklearn prob classifier (with scaling flag)
    def run_sklearn(name, clf, use_scaled, category, paper_acc, paper_task):
        Xtr = X_tr_s if use_scaled else X_tr
        Xte = X_te_s if use_scaled else X_te
        t0 = time.perf_counter(); clf.fit(Xtr, y_tr); ts = time.perf_counter()-t0
        t1 = time.perf_counter(); prob = clf.predict_proba(Xte)[:,1]; im=(time.perf_counter()-t1)*1000
        m = metrics(y_te, prob, im, ts, name, category, paper_acc, paper_task)
        print(f"  Acc={m['Accuracy']:.4f}  F1={m['F1_Score']:.4f}  "
              f"AUC={m['AUC_ROC']:.4f}  Inf={im:.1f}ms  Train={ts:.2f}s")
        return m

    # ── 1. Logistic Regression ─────────────────────────────────────────────
    print("\n[ 1/14] Logistic Regression  [AI Agent Marketplace]")
    results.append(run_sklearn(
        "Logistic Regression",
        LogisticRegression(max_iter=2000, class_weight="balanced",
                           random_state=RANDOM_STATE, n_jobs=-1),
        use_scaled=True,
        category="Linear Model [AI Agent Marketplace]",
        paper_acc="84.7% Acc (AI Agent Marketplace)",
        paper_task="Agent success prediction (synthetic data)"))

    # ── 2. Naive Bayes ─────────────────────────────────────────────────────
    print("\n[ 2/14] Naive Bayes  [AI in Marketing / AI-Driven Business Analytics]")
    results.append(run_sklearn(
        "Naive Bayes (Gaussian)",
        GaussianNB(),
        use_scaled=True,
        category="Probabilistic [AI in Marketing / AI-Driven Analytics]",
        paper_acc="Cited as baseline in AI-Driven Business Analytics & AI in Marketing",
        paper_task="Customer segmentation and market prediction"))

    # ── 3. K-Nearest Neighbors ─────────────────────────────────────────────
    print("\n[ 3/14] K-Nearest Neighbors  [AI in Marketing / Leveraging AI]")
    results.append(run_sklearn(
        "K-Nearest Neighbors (k=7)",
        KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        use_scaled=True,
        category="Instance-Based [AI in Marketing / Leveraging AI BizDev]",
        paper_acc="Cited in market segmentation literature; no specific accuracy",
        paper_task="Similarity-based customer/product clustering"))

    # ── 4. Decision Tree ───────────────────────────────────────────────────
    print("\n[ 4/14] Decision Tree  [AI-Driven Business Analytics]")
    results.append(run_sklearn(
        "Decision Tree",
        DecisionTreeClassifier(max_depth=10, class_weight="balanced",
                               random_state=RANDOM_STATE),
        use_scaled=False,
        category="Tree [AI-Driven Business Analytics]",
        paper_acc="Interpretable baseline; no specific accuracy in papers",
        paper_task="Decision support in business analytics"))

    # ── 5. SVM (RBF) ───────────────────────────────────────────────────────
    print("\n[ 5/14] SVM (RBF)  [AI Agent Marketplace]")
    results.append(run_sklearn(
        "SVM (RBF)",
        SVC(kernel="rbf", class_weight="balanced", probability=True,
            random_state=RANDOM_STATE, C=1.0, gamma="scale"),
        use_scaled=True,
        category="Kernel Method [AI Agent Marketplace]",
        paper_acc="87.5% Acc (AI Agent Marketplace)",
        paper_task="Agent success prediction (synthetic data)"))

    # ── 6. AdaBoost ────────────────────────────────────────────────────────
    print("\n[ 6/14] AdaBoost  [AI-Driven Business Analytics]")
    results.append(run_sklearn(
        "AdaBoost",
        AdaBoostClassifier(n_estimators=200, learning_rate=0.5,
                           random_state=RANDOM_STATE),
        use_scaled=False,
        category="Boosting Ensemble [AI-Driven Business Analytics]",
        paper_acc="Cited in IBM Watson Health and ensemble analytics literature",
        paper_task="Business performance and anomaly detection"))

    # ── 7. Random Forest ───────────────────────────────────────────────────
    print("\n[ 7/14] Random Forest  [AI Agent Marketplace]")
    results.append(run_sklearn(
        "Random Forest",
        RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=3,
                               class_weight="balanced", n_jobs=-1,
                               random_state=RANDOM_STATE),
        use_scaled=False,
        category="Bagging Ensemble [AI Agent Marketplace]",
        paper_acc="88.9% Acc (AI Agent Marketplace)",
        paper_task="Agent success prediction (synthetic data)"))

    # ── 8. Extra Trees ─────────────────────────────────────────────────────
    print("\n[ 8/14] Extra Trees  [AI-Driven Business Analytics / Leveraging AI]")
    results.append(run_sklearn(
        "Extra Trees",
        ExtraTreesClassifier(n_estimators=500, max_depth=12, class_weight="balanced",
                             n_jobs=-1, random_state=RANDOM_STATE),
        use_scaled=False,
        category="Randomised Ensemble [AI-Driven Analytics / Leveraging AI BizDev]",
        paper_acc="Cited in AI-Driven Business Analytics ensemble methods",
        paper_task="Retail and financial prediction pipelines"))

    # ── 9. Gradient Boosting (sklearn) ─────────────────────────────────────
    print("\n[ 9/14] Gradient Boosting (sklearn GBM)  [AI-Driven Business Analytics]")
    results.append(run_sklearn(
        "Gradient Boosting (GBM)",
        GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, random_state=RANDOM_STATE),
        use_scaled=False,
        category="Gradient Boosting [AI-Driven Business Analytics]",
        paper_acc="Amazon/JPMorgan use-case GBT references in AI-Driven Business Analytics",
        paper_task="Fraud detection, demand forecasting, supply chain optimisation"))

    # ── 10. XGBoost ────────────────────────────────────────────────────────
    print("\n[10/14] XGBoost  [Production system — Pillar 1 & 4]")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="logloss", n_jobs=-1,
        random_state=RANDOM_STATE, verbosity=0)
    t0 = time.perf_counter()
    xgb_clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    ts = time.perf_counter()-t0
    t1 = time.perf_counter(); xgb_proba = xgb_clf.predict_proba(X_te)[:,1]
    im = (time.perf_counter()-t1)*1000
    m = metrics(y_te, xgb_proba, im, ts, "XGBoost (GBT)",
                "Gradient Boosted Trees [Production — Pillar 1 & 4]",
                "Production system; ~99% Seg Acc / 0.80 PMF Acc",
                "Market segment classification + Product-Market Fit prediction")
    print(f"  Acc={m['Accuracy']:.4f}  F1={m['F1_Score']:.4f}  "
          f"AUC={m['AUC_ROC']:.4f}  Inf={im:.1f}ms  Train={ts:.2f}s")
    results.append(m)

    # ── 11. LightGBM ───────────────────────────────────────────────────────
    print("\n[11/14] LightGBM  [Leveraging AI for Transformative Business Dev.]")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1)
    t0 = time.perf_counter(); lgb_clf.fit(X_tr, y_tr); ts = time.perf_counter()-t0
    t1 = time.perf_counter(); lgb_proba = lgb_clf.predict_proba(X_te)[:,1]
    im = (time.perf_counter()-t1)*1000
    m = metrics(y_te, lgb_proba, im, ts, "LightGBM",
                "Gradient Boosting [Leveraging AI BizDev]",
                "High-speed GBT cited in competitive intelligence pipelines",
                "Real-time market shift detection and trend forecasting")
    print(f"  Acc={m['Accuracy']:.4f}  F1={m['F1_Score']:.4f}  "
          f"AUC={m['AUC_ROC']:.4f}  Inf={im:.1f}ms  Train={ts:.2f}s")
    results.append(m)

    # ── 12. CatBoost ───────────────────────────────────────────────────────
    print("\n[12/14] CatBoost  [MaRGen / AI-Driven Business Analytics]")
    cb_clf = cbt.CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos,
        random_seed=RANDOM_STATE, verbose=False)
    t0 = time.perf_counter(); cb_clf.fit(X_tr, y_tr); ts = time.perf_counter()-t0
    t1 = time.perf_counter(); cb_proba = cb_clf.predict_proba(X_te)[:,1]
    im = (time.perf_counter()-t1)*1000
    m = metrics(y_te, cb_proba, im, ts, "CatBoost",
                "Gradient Boosting [MaRGen / AI-Driven Analytics]",
                "Production-grade GBT; cited for structured data in analytics",
                "Structured market/product data classification")
    print(f"  Acc={m['Accuracy']:.4f}  F1={m['F1_Score']:.4f}  "
          f"AUC={m['AUC_ROC']:.4f}  Inf={im:.1f}ms  Train={ts:.2f}s")
    results.append(m)

    # ── 13. DNN 64-32-16 ───────────────────────────────────────────────────
    print("\n[13/14] DNN (64-32-16)  [AI Agent Marketplace]")
    t0 = time.perf_counter()
    dnn_proba, dnn_sc, dnn_model = train_dnn(X_tr, y_tr, X_te, n_epochs=80)
    ts = time.perf_counter()-t0
    Xte_t = torch.tensor(dnn_sc.transform(X_te), dtype=torch.float32)
    t1 = time.perf_counter()
    with torch.no_grad(): _ = torch.sigmoid(dnn_model(Xte_t)).numpy()
    im = (time.perf_counter()-t1)*1000
    m = metrics(y_te, dnn_proba, im, ts, "DNN (64-32-16)",
                "Deep Learning [AI Agent Marketplace]",
                "91.2% Acc / 0.889 F1 (AI Agent Marketplace, synthetic data)",
                "Real-time agent success prediction (5,000 synthetic records)")
    print(f"  Acc={m['Accuracy']:.4f}  F1={m['F1_Score']:.4f}  "
          f"AUC={m['AUC_ROC']:.4f}  Inf={im:.1f}ms  Train={ts:.2f}s")
    results.append(m)

    # ── 14. PCA + DNN 64-32-16 ─────────────────────────────────────────────
    print("\n[14/14] PCA + DNN (64-32-16)  [AI Agent Marketplace — PCA pipeline]")
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_tr_pca = pca.fit_transform(X_tr_s).astype(np.float32)
    X_te_pca = pca.transform(X_te_s).astype(np.float32)
    print(f"  PCA: {n_feat} -> {pca.n_components_} components (95% variance)")
    t0 = time.perf_counter()
    pca_proba, pca_sc, pca_model = train_dnn(X_tr_pca, y_tr, X_te_pca, n_epochs=80)
    ts = time.perf_counter()-t0
    Xte_pca_t = torch.tensor(pca_sc.transform(X_te_pca), dtype=torch.float32)
    t1 = time.perf_counter()
    with torch.no_grad(): _ = torch.sigmoid(pca_model(Xte_pca_t)).numpy()
    im = (time.perf_counter()-t1)*1000
    m = metrics(y_te, pca_proba, im, ts, "PCA + DNN (64-32-16)",
                "PCA + Deep Learning [AI Agent Marketplace]",
                "91.2% Acc (AI Agent Marketplace — PCA dimensionality reduction)",
                "Agent success prediction with PCA feature extraction")
    print(f"  Acc={m['Accuracy']:.4f}  F1={m['F1_Score']:.4f}  "
          f"AUC={m['AUC_ROC']:.4f}  Inf={im:.1f}ms  Train={ts:.2f}s")
    results.append(m)

    # ──────────────────────────────────────────────────────────────
    # Build and save CSV
    # ──────────────────────────────────────────────────────────────
    col_order = [
        "Model","Category","Accuracy","F1_Score","Precision","Recall",
        "AUC_ROC","Inference_ms","Train_time_s",
        "Paper_Reported_Metric","Paper_Task","Applicable_to_Dataset"
    ]
    testable_df = (pd.DataFrame(results)[col_order]
                   .sort_values("AUC_ROC", ascending=False))

    non_test_df = pd.DataFrame(NON_TESTABLE)
    for c in col_order:
        if c not in non_test_df.columns:
            non_test_df[c] = "N/A"
    non_test_df = non_test_df[col_order]

    report = pd.concat([testable_df, non_test_df], ignore_index=True)
    out = Path(__file__).parent / "benchmark_results.csv"
    report.to_csv(out, index=False)
    print(f"\nReport saved -> {out}")

    # ──────────────────────────────────────────────────────────────
    # Console summary (testable models only)
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'#':<3} {'Model':<28} {'Acc':>8} {'F1':>7} {'Prec':>7} {'Rec':>7} "
          f"{'AUC':>7} {'Inf(ms)':>9} {'Train(s)':>9}")
    print("-" * 100)
    for rank, (_, row) in enumerate(testable_df.iterrows(), 1):
        print(f"{rank:<3} {row['Model']:<28} {row['Accuracy']:>8.4f} {row['F1_Score']:>7.4f} "
              f"{row['Precision']:>7.4f} {row['Recall']:>7.4f} {row['AUC_ROC']:>7.4f} "
              f"{row['Inference_ms']:>9.1f} {row['Train_time_s']:>9.2f}")
    print("=" * 100)
    top = testable_df.iloc[0]
    print(f"\nTop model by AUC:      {top['Model']}  (AUC={top['AUC_ROC']:.4f})")
    top_acc = testable_df.sort_values("Accuracy", ascending=False).iloc[0]
    print(f"Top model by Accuracy: {top_acc['Model']}  (Acc={top_acc['Accuracy']:.4f})")
    top_f1  = testable_df.sort_values("F1_Score", ascending=False).iloc[0]
    print(f"Top model by F1:       {top_f1['Model']}  (F1={top_f1['F1_Score']:.4f})")
    xgb_row = testable_df[testable_df["Model"] == "XGBoost (GBT)"].iloc[0]
    print(f"\nXGBoost rank by AUC:      "
          f"{list(testable_df['Model']).index('XGBoost (GBT)')+1} / {len(testable_df)}")
    print(f"XGBoost rank by Accuracy: "
          f"{list(testable_df.sort_values('Accuracy',ascending=False)['Model']).index('XGBoost (GBT)')+1} / {len(testable_df)}")
    print(f"XGBoost rank by F1:       "
          f"{list(testable_df.sort_values('F1_Score',ascending=False)['Model']).index('XGBoost (GBT)')+1} / {len(testable_df)}")

if __name__ == "__main__":
    main()
