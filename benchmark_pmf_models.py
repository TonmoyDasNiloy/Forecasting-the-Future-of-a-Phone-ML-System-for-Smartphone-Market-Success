"""
benchmark_pmf_models.py
========================
Standalone benchmark — tests every ML/DL model from the product-portfolio,
quality-control, and predictive-maintenance research papers against the
actual Pillar 4 PMF failure-prediction task.

Task: Binary classification — predict whether a phone is a market FAILURE
      (bottom 25% composite PMF score). 968 phone-level samples, 25% failure.
      Identical label formula, feature set, and group-split as pillar4_pmf.py.
      Zero imports from src/. Zero modifications to existing system.

Research papers covered
-----------------------
[Portfolio]  AI-Predicted Product Portfolio for Profit Maximization
             -> Naive Bayes (100%), OneR (>87.5%), REPTree (>87.5%), Neural Net (100%)
[DataDriv]   Data-Driven Product Management (mixed-methods, platforms)
             -> Google AutoML, Azure ML, IBM Watson — noted as N/A (platforms)
[RealTime]   AI-Driven Real-Time Intelligence (conceptual framework)
             -> Econometric forecasting, RL, RAG-LLM — noted as N/A (conceptual)
[GANFP]      GAN-FP: GANs for Failure Prediction
             -> InfoGAN, Consistency GAN, SMOTE, ADASYN — noted as N/A (augmentation)
[Survey]     Survey on AI for Quality Control & Predictive Maintenance (Industry 4.0)
             -> SVM, KNN, Naive Bayes, DT, RF (F1=0.98), MLP (F1=0.95),
                CNN, RNN/LSTM/GRU, GAN, Autoencoder, Transformer, XGBoost (F1=0.98), GRU (F1=0.98)
[EarlyFail]  AI-Based Early Failure Detection for Predictive Maintenance
             -> DT, LightGBM+PCA (93% acc — BEST), MLP (60-73% F1), CNN-1D (60-73% F1),
                Attention-LSTM-FCN+AdaBoost (hybrid), CNN+XGBoost (78% F1 hybrid)
[AugBI]      Augmenting Business Intelligence with AI (conceptual)
             -> AutoML, NLP, ML general — noted as N/A (platforms/conceptual)

Testable models (21)
--------------------
Classical:
  1  Naive Bayes              [Portfolio (100% acc), Survey]
  2  OneR (Decision Stump)    [Portfolio (>87.5% acc)]
  3  REPTree (Pruned DT)      [Portfolio (>87.5% acc)]
  4  Logistic Regression      [baseline / Survey]
  5  Decision Tree            [Survey, EarlyFail]
  6  KNN                      [Survey]
  7  SVM (RBF)                [Survey]

Ensemble / Boosting:
  8  AdaBoost                 [EarlyFail — used in hybrid]
  9  Gradient Boosting        [Survey ensembles]
 10  Random Forest (no balance) [Survey — comparison variant]
 11  Random Forest (balanced)   [Survey F1=0.98 quality; PRODUCTION]
 12  Extra Trees               [ensemble methods / Survey]
 13  XGBoost                   [Survey F1=0.98 maintenance, EarlyFail hybrid]
 14  LightGBM (+PCA)           [EarlyFail — 93% acc, BEST single model]
 15  CatBoost                  [structured-data GBT]

Deep Learning (tabular-mode, features treated as 1D sequence):
 16  MLP Neural Network        [Portfolio 100%, Survey F1=0.95, EarlyFail 60-73%]
 17  CNN 1D                    [Survey, EarlyFail 60-73%]
 18  LSTM                      [Survey, EarlyFail]
 19  GRU                       [Survey F1=0.98 maintenance]

Hybrid (two-stage feature extraction + classifier):
 20  CNN + XGBoost             [EarlyFail — 78% F1]
 21  LSTM-FCN + AdaBoost       [EarlyFail — Attention-LSTM-FCN hybrid]

Non-testable (8 reference rows):
  - GAN-FP / InfoGAN / Consistency GAN (generative augmentation, not a classifier)
  - SMOTE / ADASYN             (resampling techniques, not classifiers)
  - Autoencoder                (unsupervised representation, not a classifier)
  - RAG-based LLM              (advisory system — conceptual)
  - Google AutoML / Azure ML   (platforms)
  - Econometric + RL           (strategic forecasting — conceptual)
  - Mathematical Programming   (0-1 optimization — not ML)

Output: benchmark_pmf_results.csv
"""

import sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection  import GroupShuffleSplit
from sklearn.preprocessing    import OrdinalEncoder, StandardScaler
from sklearn.decomposition    import PCA
from sklearn.metrics          import (accuracy_score, f1_score, precision_score,
                                      recall_score, roc_auc_score, matthews_corrcoef)
# Classical / Ensemble
from sklearn.naive_bayes      import GaussianNB
from sklearn.linear_model     import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.svm              import SVC
from sklearn.ensemble         import (AdaBoostClassifier, GradientBoostingClassifier,
                                      RandomForestClassifier, ExtraTreesClassifier)
import xgboost  as xgb
import lightgbm as lgb
import catboost as cbt
from sklearn.neural_network   import MLPClassifier

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")
np.random.seed(42); torch.manual_seed(42)

# ──────────────────────────────────────────────────────────────
# Config (inlined — no src/ imports)
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


# ──────────────────────────────────────────────────────────────
# 1. Build PMF labels (identical to pillar4_pmf.py)
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
                         if sub["months_since_launch"].nunique()>=3 else 0.0)
    g["sales_slope"] = g.apply(lambda r: slopes[(r["brand_name"],r["model"])], axis=1)
    g["pmf_score"] = (0.25*_minmax(g["avg_nps"])
                    + 0.20*_minmax(g["avg_repeat"])
                    + 0.20*_minmax(g["avg_rec"])
                    + 0.15*_minmax(g["avg_return"],  invert=True)
                    + 0.10*_minmax(g["avg_warranty"], invert=True)
                    + 0.10*_minmax(g["sales_slope"]))
    g["failure"] = (g["pmf_score"] < g["pmf_score"].quantile(0.25)).astype(int)
    return g[["brand_name","model","failure"]]

def prepare(df):
    labels = build_labels(df)
    agg = {s:"first" for s in NUM_SPECS}
    agg.update({c:"first" for c in ["processor_brand","os"]})
    ph = (df.groupby(["brand_name","model"]).agg(agg)
            .reset_index().merge(labels, on=["brand_name","model"]))
    X = ph[NUM_SPECS + CAT_SPECS].copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[CAT_SPECS] = enc.fit_transform(X[CAT_SPECS])
    y      = ph["failure"].values
    groups = (ph["brand_name"]+"__"+ph["model"]).values
    tr, te = next(GroupShuffleSplit(1, test_size=TEST_SIZE,
                                    random_state=RANDOM_STATE).split(X, y, groups))
    return X.values.astype(np.float32), y, tr, te


# ──────────────────────────────────────────────────────────────
# 2. Metrics
# ──────────────────────────────────────────────────────────────
def metrics(y_true, proba, inf_ms, train_s, name, category, paper_ref, paper_task):
    pred = (proba >= 0.5).astype(int)
    return {
        "Model":              name,
        "Category":           category,
        "Accuracy":           round(accuracy_score(y_true, pred),                    4),
        "F1_Score":           round(f1_score(y_true, pred, zero_division=0),         4),
        "Precision":          round(precision_score(y_true, pred, zero_division=0),  4),
        "Recall":             round(recall_score(y_true, pred, zero_division=0),     4),
        "AUC_ROC":            round(roc_auc_score(y_true, proba),                    4),
        "MCC":                round(matthews_corrcoef(y_true, pred),                 4),
        "Inference_ms":       round(inf_ms, 2),
        "Train_time_s":       round(train_s, 2),
        "Paper_Reported_Metric": paper_ref,
        "Paper_Task":         paper_task,
        "Applicable":         "Yes",
    }


# ──────────────────────────────────────────────────────────────
# 3. PyTorch models (tabular: features treated as 1D sequence)
# ──────────────────────────────────────────────────────────────
class TabMLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in,128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),  nn.Dropout(0.3),
            nn.Linear(64,  32), nn.ReLU(),
            nn.Linear(32,   1),
        )
    def forward(self, x): return self.net(x).squeeze(1)

class TabCNN(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32,64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(nn.Linear(64*8, 32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)).flatten(1)).squeeze(1)

class TabLSTM(nn.Module):
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, 2, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))  # (batch, n_feat, 1) as sequence
        return self.head(out[:,-1,:]).squeeze(1)

class TabGRU(nn.Module):
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.gru  = nn.GRU(1, hidden, 2, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.gru(x.unsqueeze(-1))
        return self.head(out[:,-1,:]).squeeze(1)

# CNN feature extractor for CNN+XGBoost hybrid
class CNNExtractor(nn.Module):
    def __init__(self, n_in, embed=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,32,3,padding=1), nn.ReLU(),
            nn.Conv1d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.proj = nn.Linear(64*4, embed)
    def forward(self, x):
        return self.proj(self.conv(x.unsqueeze(1)).flatten(1))

# LSTM-FCN extractor for LSTM-FCN+AdaBoost hybrid
class LSTMFCNExtractor(nn.Module):
    def __init__(self, n_in, embed=32):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, 1, batch_first=True)
        self.fcn  = nn.Sequential(
            nn.Conv1d(1,32,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.proj = nn.Linear(32 + 32*4, embed)
    def forward(self, x):
        lstm_out,_ = self.lstm(x.unsqueeze(-1))
        lstm_feat  = lstm_out[:,-1,:]                       # (batch,32)
        fcn_feat   = self.fcn(x.unsqueeze(1)).flatten(1)    # (batch,32*4)
        return self.proj(torch.cat([lstm_feat, fcn_feat],1))


def _train_torch(model, X_tr_t, y_tr_t, pos_w, n_epochs=80, lr=1e-3, batch=32):
    opt     = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    loader  = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch, shuffle=True)
    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    return model

def run_torch(model_cls, n_feat, X_tr, y_tr, X_te, y_te,
              sc, pos_w, name, category, paper_ref, paper_task,
              n_epochs=80):
    Xtr = torch.tensor(sc.transform(X_tr), dtype=torch.float32)
    Xte = torch.tensor(sc.transform(X_te), dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.float32)
    t0  = time.perf_counter()
    m   = _train_torch(model_cls(n_feat), Xtr, ytr, pos_w, n_epochs=n_epochs)
    ts  = time.perf_counter()-t0
    m.eval()
    t1  = time.perf_counter()
    with torch.no_grad(): proba = torch.sigmoid(m(Xte)).numpy()
    inf = (time.perf_counter()-t1)*1000
    return metrics(y_te, proba, inf, ts, name, category, paper_ref, paper_task)


def run_hybrid(extractor_cls, second_clf, clf_name,
               n_feat, X_tr, y_tr, X_te, y_te, sc, pos_w,
               name, category, paper_ref, paper_task, n_epochs=60):
    Xtr = torch.tensor(sc.transform(X_tr), dtype=torch.float32)
    Xte = torch.tensor(sc.transform(X_te), dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.float32)

    # Phase 1: train extractor with classification head
    class ExtractorWithHead(nn.Module):
        def __init__(self, ext):
            super().__init__(); self.ext=ext; self.head=nn.Linear(32,1)
        def forward(self, x): return self.head(self.ext(x)).squeeze(1)

    ext    = extractor_cls(n_feat)
    ewh    = ExtractorWithHead(ext)
    t0     = time.perf_counter()
    ewh    = _train_torch(ewh, Xtr, ytr, pos_w, n_epochs=n_epochs, batch=32)
    ext    = ewh.ext; ext.eval()

    # Phase 2: extract features, train second classifier
    with torch.no_grad():
        feat_tr = ext(Xtr).numpy()
        feat_te = ext(Xte).numpy()
    second_clf.fit(feat_tr, y_tr)
    ts  = time.perf_counter()-t0

    t1  = time.perf_counter()
    if hasattr(second_clf,"predict_proba"):
        proba = second_clf.predict_proba(feat_te)[:,1]
    else:
        proba = second_clf.decision_function(feat_te)
        proba = (proba - proba.min())/(proba.max()-proba.min()+1e-9)
    inf = (time.perf_counter()-t1)*1000
    return metrics(y_te, proba, inf, ts, name, category, paper_ref, paper_task)


# ──────────────────────────────────────────────────────────────
# 4. Non-testable reference rows
# ──────────────────────────────────────────────────────────────
NON_TESTABLE = [
    {"Model":"GAN-FP (InfoGAN + Consistency GAN)",
     "Category":"Generative Model [GANFP paper]",
     "Accuracy":"N/A","F1_Score":"N/A","Precision":"N/A","Recall":"N/A",
     "AUC_ROC":"N/A","MCC":"N/A","Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"AUC>0.69, Failure-class F1>0.53 under 2% failure rate (APS/NASA datasets)",
     "Paper_Task":"GAN-based data synthesis for extreme imbalance (2% failure) — augmentation system, not a standalone classifier",
     "Applicable":"No — generative augmentation framework; our 25% failure rate does not need GAN synthesis; the InfoGAN+CGan architecture targets imbalance ratios of 1:50+"},
    {"Model":"SMOTE / ADASYN",
     "Category":"Resampling Technique [GANFP paper]",
     "Accuracy":"N/A","F1_Score":"N/A","Precision":"N/A","Recall":"N/A",
     "AUC_ROC":"N/A","MCC":"N/A","Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"SMOTE/ADASYN outperformed by GAN-FP on AUC and failure-class F1",
     "Paper_Task":"Synthetic minority oversampling — preprocessing technique, not a model; would be applied before a classifier",
     "Applicable":"No — resampling pre-processing step, not a standalone classifier; class_weight='balanced' in RF achieves same goal without synthetic data"},
    {"Model":"Autoencoder (Anomaly Detection)",
     "Category":"Unsupervised DL [Survey paper]",
     "Accuracy":"N/A","F1_Score":"N/A","Precision":"N/A","Recall":"N/A",
     "AUC_ROC":"N/A","MCC":"N/A","Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"Cited in Survey for anomaly/defect detection in manufacturing sensor data",
     "Paper_Task":"Unsupervised anomaly detection from reconstruction error — requires unlabeled normal-class data at training time; our task is supervised binary classification",
     "Applicable":"No — unsupervised reconstruction model; PMF failure label is known and supervised classification is appropriate"},
    {"Model":"RAG-based LLM Advisory System",
     "Category":"Conceptual LLM Framework [RealTime paper]",
     "Accuracy":"N/A","F1_Score":"N/A","Precision":"N/A","Recall":"N/A",
     "AUC_ROC":"N/A","MCC":"N/A","Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"Two-layer architecture for strategic agility — no quantitative performance metric reported",
     "Paper_Task":"Retrieval-augmented LLM for real-time strategic advisory — outputs textual strategy, not a failure probability",
     "Applicable":"No — conceptual advisory framework without empirical validation; outputs qualitative strategy, not binary failure prediction"},
    {"Model":"Google AutoML / Azure ML / IBM Watson",
     "Category":"AutoML Platform [DataDriv paper]",
     "Accuracy":"N/A","F1_Score":"N/A","Precision":"N/A","Recall":"N/A",
     "AUC_ROC":"N/A","MCC":"N/A","Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"85% real-time decision improvement; 78% customer insights gain; 70% feature prioritization",
     "Paper_Task":"Platform-level product management tools — survey-based metrics, not model accuracy on a specific dataset",
     "Applicable":"No — commercial SaaS platforms; reported metrics are survey-based organizational impact, not model-level accuracy"},
    {"Model":"Mathematical Programming (0-1 Optimization)",
     "Category":"Operations Research [Portfolio paper]",
     "Accuracy":"N/A","F1_Score":"N/A","Precision":"N/A","Recall":"N/A",
     "AUC_ROC":"N/A","MCC":"N/A","Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"Optimal portfolio profit=1631 (budget=800, risk=600); up to 1977 at higher budget",
     "Paper_Task":"Portfolio profit maximization via 0-1 integer programming — second-stage optimizer, not a failure predictor; requires classifier output as input",
     "Applicable":"No — optimization layer (takes classifier scores as input); not a standalone ML model for failure prediction"},
    {"Model":"Econometric Forecasting + RL (Strategic)",
     "Category":"Conceptual Framework [RealTime paper]",
     "Accuracy":"N/A","F1_Score":"N/A","Precision":"N/A","Recall":"N/A",
     "AUC_ROC":"N/A","MCC":"N/A","Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"Validated two-layer architecture — no empirical performance metrics reported",
     "Paper_Task":"Strategic agility via econometric macro-forecasting and RL for policy optimization — different task domain",
     "Applicable":"No — strategic planning system requiring macroeconomic data streams and RL simulation environment"},
    {"Model":"Transformer (Quality Control — Survey)",
     "Category":"Attention-Based DL [Survey paper]",
     "Accuracy":"N/A","F1_Score":"N/A","Precision":"N/A","Recall":"N/A",
     "AUC_ROC":"N/A","MCC":"N/A","Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"Cited in Survey for sequence/sensor-based quality control; no standalone accuracy reported",
     "Paper_Task":"Transformer for multi-variate sensor time-series in manufacturing defect detection — requires temporal sensor sequences",
     "Applicable":"No — designed for multi-variate temporal sensor streams; our phone dataset has static launch-time specs, not temporal sensor sequences"},
]


# ──────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  PMF FAILURE PREDICTION BENCHMARK (Pillar 4 Task)")
    print("=" * 72)
    df  = pd.read_csv(DATA_PATH)
    X, y, tr, te = prepare(df)
    X_tr, X_te   = X[tr], X[te]
    y_tr, y_te   = y[tr], y[te]
    n_feat        = X.shape[1]
    scale_pos     = float((y_tr==0).sum()/(y_tr==1).sum())
    pos_w         = torch.tensor([scale_pos], dtype=torch.float32)
    sc            = StandardScaler().fit(X_tr)
    X_tr_s        = sc.transform(X_tr)
    X_te_s        = sc.transform(X_te)
    print(f"  Phones: {len(y)} | Train: {len(tr)} | Test: {len(te)} | "
          f"Features: {n_feat} | Failure rate: {y.mean():.1%}")

    results = []

    def run_sk(name, clf, scaled, cat, ref, task, use_proba=True):
        Xtr = X_tr_s if scaled else X_tr
        Xte = X_te_s if scaled else X_te
        t0  = time.perf_counter(); clf.fit(Xtr, y_tr); ts = time.perf_counter()-t0
        t1  = time.perf_counter()
        if use_proba:
            prob = clf.predict_proba(Xte)[:,1]
        else:
            s = clf.decision_function(Xte)
            prob = (s-s.min())/(s.max()-s.min()+1e-9)
        inf = (time.perf_counter()-t1)*1000
        m   = metrics(y_te, prob, inf, ts, name, cat, ref, task)
        print(f"  Acc={m['Accuracy']:.4f}  F1={m['F1_Score']:.4f}  "
              f"AUC={m['AUC_ROC']:.4f}  MCC={m['MCC']:.4f}  "
              f"Inf={inf:.1f}ms  Train={ts:.2f}s")
        return m

    # ── Classical ────────────────────────────────────────────────────────
    print("\n[ 1/21] Naive Bayes  [Portfolio 100% acc, Survey]")
    results.append(run_sk("Naive Bayes",
        GaussianNB(), True,
        "Probabilistic [Portfolio / Survey]",
        "Portfolio: 100% accuracy on 16-attr product-market dataset; Survey: cited for quality control",
        "Product-market viability screening and manufacturing defect detection"))

    print("\n[ 2/21] OneR (Decision Stump — single best feature)  [Portfolio >87.5%]")
    results.append(run_sk("OneR (Decision Stump)",
        DecisionTreeClassifier(max_depth=1, class_weight="balanced",
                               random_state=RANDOM_STATE), False,
        "Single-Rule Classifier [Portfolio paper]",
        "Portfolio: OneR >87.5% accuracy on product-market screening (16 nonfinancial attributes)",
        "Simple rule-based product viability screening across Ansoff matrix quadrants"))

    print("\n[ 3/21] REPTree (Pruned Decision Tree)  [Portfolio >87.5%]")
    results.append(run_sk("REPTree (Pruned DT)",
        DecisionTreeClassifier(max_depth=6, ccp_alpha=0.005, class_weight="balanced",
                               random_state=RANDOM_STATE), False,
        "Pruned Tree [Portfolio paper]",
        "Portfolio: REPTree >87.5% accuracy; reduced-error pruning for product portfolio screening",
        "Product-market combination viability classification with overfitting control"))

    print("\n[ 4/21] Logistic Regression  [baseline / Survey]")
    results.append(run_sk("Logistic Regression",
        LogisticRegression(max_iter=2000, class_weight="balanced",
                           random_state=RANDOM_STATE, n_jobs=-1), True,
        "Linear [Survey baseline]",
        "Interpretable baseline cited across all survey papers",
        "Churn/failure baseline with feature-coefficient interpretability"))

    print("\n[ 5/21] Decision Tree  [Survey, EarlyFail]")
    results.append(run_sk("Decision Tree",
        DecisionTreeClassifier(max_depth=10, class_weight="balanced",
                               random_state=RANDOM_STATE), False,
        "Tree [Survey / EarlyFail]",
        "Survey: decision trees for quality control; EarlyFail: DT baseline vs LightGBM",
        "Interpretable failure prediction in manufacturing and product quality"))

    print("\n[ 6/21] KNN (k=7)  [Survey]")
    results.append(run_sk("KNN (k=7)",
        KNeighborsClassifier(n_neighbors=7, n_jobs=-1), True,
        "Instance-Based [Survey]",
        "Survey: KNN cited for manufacturing quality control and defect detection",
        "Spec-similarity-based failure prediction (neighbour voting)"))

    print("\n[ 7/21] SVM (RBF)  [Survey]")
    results.append(run_sk("SVM (RBF)",
        SVC(kernel="rbf", class_weight="balanced", probability=True,
            random_state=RANDOM_STATE, C=1.0, gamma="scale"), True,
        "Kernel Method [Survey]",
        "Survey: SVM cited for quality control and failure detection in Industry 4.0",
        "Manufacturing defect and failure classification via kernel boundary"))

    # ── Ensemble / Boosting ──────────────────────────────────────────────
    print("\n[ 8/21] AdaBoost  [EarlyFail hybrid component]")
    results.append(run_sk("AdaBoost",
        AdaBoostClassifier(n_estimators=200, learning_rate=0.5,
                           random_state=RANDOM_STATE), False,
        "Boosting [EarlyFail — hybrid component]",
        "EarlyFail: AdaBoost used in LSTM-FCN+AdaBoost hybrid for failure detection",
        "Adaptive boosting for rare failure detection in predictive maintenance"))

    print("\n[ 9/21] Gradient Boosting  [Survey ensembles]")
    results.append(run_sk("Gradient Boosting",
        GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, random_state=RANDOM_STATE), False,
        "Gradient Boosting [Survey]",
        "Survey: ensemble methods (GBT) for quality control and maintenance failure prediction",
        "Gradient-boosted failure risk classification for industrial maintenance"))

    print("\n[10/21] Random Forest (no class balancing)  [comparison variant]")
    results.append(run_sk("Random Forest (unbalanced)",
        RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=3,
                               n_jobs=-1, random_state=RANDOM_STATE), False,
        "Bagging Ensemble [Survey — unbalanced variant]",
        "Survey: RF F1=0.98 for quality control — unbalanced comparison",
        "Failure prediction without imbalance correction (25% failure rate)"))

    print("\n[11/21] Random Forest (class_weight='balanced') [Survey F1=0.98; PRODUCTION]")
    results.append(run_sk("Random Forest (balanced) [PRODUCTION]",
        RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=3,
                               class_weight="balanced", n_jobs=-1,
                               random_state=RANDOM_STATE), False,
        "Bagging Ensemble [Survey / EarlyFail — PRODUCTION Pillar 4]",
        "Survey: RF F1=0.98 quality control (Industry 4.0); pillar4_pmf.py production model",
        "Market failure prediction from launch specs — production model with balanced class weighting"))

    print("\n[12/21] Extra Trees  [Survey ensembles]")
    results.append(run_sk("Extra Trees",
        ExtraTreesClassifier(n_estimators=500, max_depth=12, class_weight="balanced",
                             n_jobs=-1, random_state=RANDOM_STATE), False,
        "Randomised Ensemble [Survey]",
        "Survey: ensemble diversity methods for quality control",
        "Randomised ensemble for robust failure classification"))

    print("\n[13/21] XGBoost  [Survey F1=0.98 maintenance, EarlyFail CNN+XGB hybrid]")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
        eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE, verbosity=0)
    t0 = time.perf_counter()
    xgb_clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    ts = time.perf_counter()-t0
    t1 = time.perf_counter(); xp = xgb_clf.predict_proba(X_te)[:,1]; inf=(time.perf_counter()-t1)*1000
    m = metrics(y_te, xp, inf, ts, "XGBoost",
                "Gradient Boosted Trees [Survey / EarlyFail hybrid]",
                "Survey: XGBoost F1=0.98 (maintenance); EarlyFail: CNN+XGBoost hybrid 78% F1",
                "Maintenance failure and product quality classification")
    print(f"  Acc={m['Accuracy']:.4f}  F1={m['F1_Score']:.4f}  AUC={m['AUC_ROC']:.4f}  "
          f"MCC={m['MCC']:.4f}  Inf={inf:.1f}ms  Train={ts:.2f}s")
    results.append(m)

    print("\n[14/21] LightGBM + PCA  [EarlyFail — 93% acc, BEST single model]")
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE).fit(X_tr_s)
    X_tr_pca = pca.transform(X_tr_s); X_te_pca = pca.transform(X_te_s)
    print(f"  PCA: {n_feat} -> {pca.n_components_} components (95% variance)")
    results.append(run_sk("LightGBM + PCA",
        lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                           subsample=0.8, scale_pos_weight=scale_pos,
                           n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1),
        False,   # uses PCA-transformed data directly
        "Gradient Boosting + PCA [EarlyFail — BEST single model]",
        "EarlyFail: LightGBM+PCA 93% accuracy and F1 on AI4I 2020 dataset (10k records, 339 failures)",
        "Early failure detection for predictive maintenance with dimensionality reduction"))
    # override Xtr/Xte with PCA version for this model only
    results[-1] = run_sk("LightGBM + PCA",
        lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                           subsample=0.8, scale_pos_weight=scale_pos,
                           n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1),
        True,  # but feed PCA data below via monkey-patch
        "Gradient Boosting + PCA [EarlyFail — BEST single model]",
        "EarlyFail: LightGBM+PCA 93% accuracy and F1 on AI4I 2020 dataset (10k records, 339 failures)",
        "Early failure detection for predictive maintenance with dimensionality reduction")
    # Re-run with actual PCA data:
    lgb_pca = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                  subsample=0.8, scale_pos_weight=scale_pos,
                                  n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1)
    t0=time.perf_counter(); lgb_pca.fit(X_tr_pca, y_tr); ts=time.perf_counter()-t0
    t1=time.perf_counter(); lp=lgb_pca.predict_proba(X_te_pca)[:,1]; inf=(time.perf_counter()-t1)*1000
    results[-1] = metrics(y_te, lp, inf, ts, "LightGBM + PCA",
                          "Gradient Boosting + PCA [EarlyFail — BEST single model]",
                          "EarlyFail: LightGBM+PCA 93% accuracy, F1=0.93 on AI4I 2020 (10k, 339 failures)",
                          "Early failure detection with PCA dimensionality reduction")
    print(f"  Acc={results[-1]['Accuracy']:.4f}  F1={results[-1]['F1_Score']:.4f}  "
          f"AUC={results[-1]['AUC_ROC']:.4f}  MCC={results[-1]['MCC']:.4f}  "
          f"Inf={inf:.1f}ms  Train={ts:.2f}s")

    print("\n[15/21] CatBoost  [structured-data GBT]")
    results.append(run_sk("CatBoost",
        cbt.CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                               scale_pos_weight=scale_pos,
                               random_seed=RANDOM_STATE, verbose=False), False,
        "Gradient Boosting [structured analytics]",
        "Production-grade GBT for structured tabular failure/quality prediction",
        "Tabular product failure classification"))

    # ── Deep Learning (tabular mode) ─────────────────────────────────────
    print("\n[16/21] MLP Neural Network  [Portfolio 100% acc, Survey F1=0.95, EarlyFail 60-73%]")
    results.append(run_torch(TabMLP, n_feat, X_tr, y_tr, X_te, y_te, sc, pos_w,
        "MLP Neural Network",
        "Deep Learning [Portfolio / Survey / EarlyFail]",
        "Portfolio: 2-hidden-layer NN 100% acc; Survey: MLP F1=0.95 quality control; EarlyFail: MLP 60-73% F1",
        "Product viability classification and manufacturing defect detection",
        n_epochs=100))
    print(f"  Acc={results[-1]['Accuracy']:.4f}  F1={results[-1]['F1_Score']:.4f}  "
          f"AUC={results[-1]['AUC_ROC']:.4f}  MCC={results[-1]['MCC']:.4f}  "
          f"Inf={results[-1]['Inference_ms']:.1f}ms  Train={results[-1]['Train_time_s']:.2f}s")

    print("\n[17/21] CNN 1D  [Survey, EarlyFail 60-73% F1]")
    results.append(run_torch(TabCNN, n_feat, X_tr, y_tr, X_te, y_te, sc, pos_w,
        "CNN 1D",
        "Convolutional NN [Survey / EarlyFail]",
        "EarlyFail: 1D-CNN 60-73% F1 on AI4I 2020; Survey: CNN for quality inspection",
        "Convolutional failure pattern detection from spec features",
        n_epochs=100))
    print(f"  Acc={results[-1]['Accuracy']:.4f}  F1={results[-1]['F1_Score']:.4f}  "
          f"AUC={results[-1]['AUC_ROC']:.4f}  MCC={results[-1]['MCC']:.4f}  "
          f"Inf={results[-1]['Inference_ms']:.1f}ms  Train={results[-1]['Train_time_s']:.2f}s")

    print("\n[18/21] LSTM  [Survey, EarlyFail]")
    results.append(run_torch(TabLSTM, n_feat, X_tr, y_tr, X_te, y_te, sc, pos_w,
        "LSTM",
        "Recurrent NN [Survey / EarlyFail]",
        "Survey: LSTM for sequential quality events; EarlyFail: LSTM in hybrid LSTM-FCN+AdaBoost",
        "Sequential failure pattern learning over spec feature sequence",
        n_epochs=100))
    print(f"  Acc={results[-1]['Accuracy']:.4f}  F1={results[-1]['F1_Score']:.4f}  "
          f"AUC={results[-1]['AUC_ROC']:.4f}  MCC={results[-1]['MCC']:.4f}  "
          f"Inf={results[-1]['Inference_ms']:.1f}ms  Train={results[-1]['Train_time_s']:.2f}s")

    print("\n[19/21] GRU  [Survey F1=0.98 maintenance]")
    results.append(run_torch(TabGRU, n_feat, X_tr, y_tr, X_te, y_te, sc, pos_w,
        "GRU",
        "Recurrent NN [Survey — F1=0.98 maintenance]",
        "Survey: GRU F1=0.98 for maintenance failure detection (alongside XGBoost)",
        "Gated recurrent failure detection over spec feature sequence",
        n_epochs=100))
    print(f"  Acc={results[-1]['Accuracy']:.4f}  F1={results[-1]['F1_Score']:.4f}  "
          f"AUC={results[-1]['AUC_ROC']:.4f}  MCC={results[-1]['MCC']:.4f}  "
          f"Inf={results[-1]['Inference_ms']:.1f}ms  Train={results[-1]['Train_time_s']:.2f}s")

    # ── Hybrid Models ─────────────────────────────────────────────────────
    print("\n[20/21] CNN + XGBoost hybrid  [EarlyFail 78% F1]")
    results.append(run_hybrid(
        CNNExtractor,
        xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                           scale_pos_weight=scale_pos,
                           eval_metric="logloss", n_jobs=-1,
                           random_state=RANDOM_STATE, verbosity=0),
        "XGBoost",
        n_feat, X_tr, y_tr, X_te, y_te, sc, pos_w,
        "CNN + XGBoost (Hybrid)",
        "Hybrid DL+GBT [EarlyFail]",
        "EarlyFail: CNN feature extractor + XGBoost classifier = 78% F1 on AI4I 2020",
        "Two-stage: CNN extracts spec representations, XGBoost classifies failure risk",
        n_epochs=60))
    print(f"  Acc={results[-1]['Accuracy']:.4f}  F1={results[-1]['F1_Score']:.4f}  "
          f"AUC={results[-1]['AUC_ROC']:.4f}  MCC={results[-1]['MCC']:.4f}  "
          f"Inf={results[-1]['Inference_ms']:.1f}ms  Train={results[-1]['Train_time_s']:.2f}s")

    print("\n[21/21] LSTM-FCN + AdaBoost hybrid  [EarlyFail Attention-LSTM-FCN]")
    results.append(run_hybrid(
        LSTMFCNExtractor,
        AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=RANDOM_STATE),
        "AdaBoost",
        n_feat, X_tr, y_tr, X_te, y_te, sc, pos_w,
        "LSTM-FCN + AdaBoost (Hybrid)",
        "Hybrid Recurrent+Conv+Boost [EarlyFail]",
        "EarlyFail: Attention-based LSTM-FCN + AdaBoost hybrid for failure detection",
        "Two-stage: LSTM-FCN extracts temporal-convolutional features, AdaBoost classifies",
        n_epochs=60))
    print(f"  Acc={results[-1]['Accuracy']:.4f}  F1={results[-1]['F1_Score']:.4f}  "
          f"AUC={results[-1]['AUC_ROC']:.4f}  MCC={results[-1]['MCC']:.4f}  "
          f"Inf={results[-1]['Inference_ms']:.1f}ms  Train={results[-1]['Train_time_s']:.2f}s")

    # ──────────────────────────────────────────────────────────────
    # Save CSV
    # ──────────────────────────────────────────────────────────────
    col_order = ["Model","Category","Accuracy","F1_Score","Precision","Recall",
                 "AUC_ROC","MCC","Inference_ms","Train_time_s",
                 "Paper_Reported_Metric","Paper_Task","Applicable"]
    testable_df = (pd.DataFrame(results)[col_order]
                   .sort_values("AUC_ROC", ascending=False).reset_index(drop=True))
    non_df = pd.DataFrame(NON_TESTABLE)
    for c in col_order:
        if c not in non_df.columns: non_df[c]="N/A"
    report = pd.concat([testable_df, non_df[col_order]], ignore_index=True)
    out = Path(__file__).parent / "benchmark_pmf_results.csv"
    report.to_csv(out, index=False)
    print(f"\nReport saved -> {out}")

    # ──────────────────────────────────────────────────────────────
    # Console summary
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 108)
    print(f"{'#':<3} {'Model':<36} {'Acc':>8} {'F1':>7} {'Prec':>7} "
          f"{'Rec':>7} {'AUC':>7} {'MCC':>7} {'Inf(ms)':>9} {'Tr(s)':>6}")
    print("-" * 108)
    for rank, row in testable_df.iterrows():
        flag = " <-- PROD" if "PRODUCTION" in row["Model"] else ""
        print(f"{rank+1:<3} {row['Model'][:36]:<36} {row['Accuracy']:>8.4f} "
              f"{row['F1_Score']:>7.4f} {row['Precision']:>7.4f} {row['Recall']:>7.4f} "
              f"{row['AUC_ROC']:>7.4f} {row['MCC']:>7.4f} "
              f"{row['Inference_ms']:>9.1f} {row['Train_time_s']:>6.2f}{flag}")
    print("=" * 108)
    prod = testable_df[testable_df["Model"].str.contains("PRODUCTION")].iloc[0]
    print(f"\nProduction RF (balanced): Acc={prod['Accuracy']:.4f}  F1={prod['F1_Score']:.4f}  "
          f"AUC={prod['AUC_ROC']:.4f}  MCC={prod['MCC']:.4f}")
    top = testable_df.iloc[0]
    print(f"Best by AUC: {top['Model']}  (AUC={top['AUC_ROC']:.4f})")
    rf_rank = testable_df[testable_df["Model"].str.contains("PRODUCTION")].index[0]+1
    print(f"RF (balanced) rank by AUC:      {rf_rank} / {len(testable_df)}")
    rf_r2 = list(testable_df.sort_values("F1_Score",ascending=False)["Model"]).index(
        [m for m in testable_df["Model"] if "PRODUCTION" in m][0])+1
    print(f"RF (balanced) rank by F1:       {rf_r2} / {len(testable_df)}")
    rf_r3 = list(testable_df.sort_values("MCC",ascending=False)["Model"]).index(
        [m for m in testable_df["Model"] if "PRODUCTION" in m][0])+1
    print(f"RF (balanced) rank by MCC:      {rf_r3} / {len(testable_df)}")

if __name__ == "__main__":
    main()
