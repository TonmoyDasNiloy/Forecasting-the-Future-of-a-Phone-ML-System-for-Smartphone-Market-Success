"""
benchmark_segmentation_models.py
=================================
Standalone benchmark — tests every ML/clustering model cited in the customer
segmentation research papers against the actual Pillar 3 task.

Task: Predict buyer demographic distributions (age, income, RFM) for a NEW
      phone from its launch-known specs, without any prior sales or transaction
      history. Evaluated at the phone-region level (~5,801 rows).

Three families of approaches are tested
----------------------------------------
A. KNN RETRIEVAL (production method — different k values)
   Find k most spec-similar phones in train set, average their demographics.

B. CLUSTERING-BASED (assign test phone to nearest centroid, use cluster demographics)
   Unsupervised methods: K-Means, K-Means++, Hierarchical, DBSCAN, Vector Quantization

C. DIRECT SUPERVISED REGRESSION (specs -> demographics)
   Supervised models that directly map phone specs to demographic percentages.
   NOTE: As documented in pillar3, direct regressors typically score R2 near 0
         because individual demographic columns have weak spec-level correlation.

Research papers covered
-----------------------
[CustProf]  Customer Profiling, Segmentation & Sales Prediction  -> Boosting Trees, SVM, NN, VQ clustering
[AIPow]     AI-Powered Consumer Segmentation (Theoretical)       -> K-Means, DBSCAN, RF, RL, NLP
[CLV]       AI-Driven CLV Forecasting with RFM                   -> K-Means++, Ridge, Lasso, RF, NN, Hierarchical
[DataDriv]  Data-Driven Customer Segmentation (Modern BI)        -> K-Means, DBSCAN, NN, recommenders
[AIBI]      AI-Driven BI Tools in Retail                         -> Linear Reg, DT, RNN
[CRM]       AI-Powered CRM Framework (UCI Online Retail II)      -> K-Means++, GB, Q-learning
[SME]       AI-Driven Personalization for SMEs                   -> K-Means, DT, SVM, NN, GB
[XCRM]      AI-Driven Predictive Analytics for CRM               -> XGBoost (98%), K-Means, BERT, CNN/RNN/LSTM
[Scalable]  Scalable AI Predictive Analytics Framework           -> DT, SVM, K-Means, Hierarchical
[PredChurn] Predictive Analytics for Customer Behavior           -> RF (92%), GB, XGBoost, LR

Testable models (24)
--------------------
KNN Retrieval Variants (5):
  1  KNN k=1
  2  KNN k=5
  3  KNN k=10  (PRODUCTION — region-aware, price-weighted)
  4  KNN k=15
  5  KNN k=20

Clustering-Based (6):
  6  K-Means (5 clusters)        [CustProf, AIPow, CLV, DataDriv, CRM, SME, Scalable]
  7  K-Means++ (5 clusters)      [CLV, CRM]
  8  K-Means++ (10 clusters)     [CLV — best inertia=9876, 90.4% accuracy]
  9  Hierarchical (10 clusters)  [CLV, Scalable]
 10  DBSCAN                      [AIPow, DataDriv]
 11  Vector Quantization (k=10)  [CustProf — VQ clustering]

Direct Supervised Regression (13):
 12  Linear Regression            [AIBI]
 13  Ridge Regression             [CLV]
 14  Lasso Regression             [CLV]
 15  Decision Tree                [AIBI, SME, Scalable]
 16  SVR (RBF)                    [CustProf, SME, Scalable]
 17  Random Forest                [AIPow, CLV, PredChurn]
 18  AdaBoost                     [ensemble literature]
 19  Gradient Boosting            [CustProf (MCC=0.469), CRM (84.3%), SME]
 20  XGBoost                      [XCRM (98% acc), PredChurn]
 21  LightGBM                     [general GBT literature]
 22  CatBoost                     [structured-data GBT]
 23  MLP Neural Network           [CustProf, CLV, DataDriv, SME]
 24  Extra Trees                  [ensemble methods]

Non-testable (5 reference rows):
  - BERT Sentiment Analysis     (NLP/text — no unstructured text in dataset)
  - Q-Learning / RL             (sequential campaign optimization, not regression)
  - Collaborative Filtering     (requires user-item interaction matrix)
  - NLP Chatbot                 (text/conversation model)
  - RNN (Time-Series)           (requires sequential transaction history)

Metrics: R2 per demographic block (age, income, RFM), matching pillar3_segmentation.py
Output:  benchmark_segmentation_results.csv
NOTE: Zero imports from src/. Zero modifications to existing system.
"""

import sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection    import GroupShuffleSplit
from sklearn.preprocessing      import OrdinalEncoder, StandardScaler
from sklearn.neighbors          import NearestNeighbors
from sklearn.metrics            import r2_score
# Clustering
from sklearn.cluster            import KMeans, AgglomerativeClustering, DBSCAN
# Regressors
from sklearn.linear_model       import LinearRegression, Ridge, Lasso
from sklearn.tree               import DecisionTreeRegressor
from sklearn.svm                import SVR
from sklearn.multioutput        import MultiOutputRegressor
from sklearn.ensemble           import (AdaBoostRegressor, RandomForestRegressor,
                                        ExtraTreesRegressor, GradientBoostingRegressor)
from sklearn.neural_network     import MLPRegressor
import xgboost  as xgb
import lightgbm as lgb
import catboost as cbt

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Config (inlined — no src/ imports)
# ──────────────────────────────────────────────────────────────
DATA_PATH    = Path(__file__).parent / "global_smartphone_bi_v4_final.csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

SIM_FEATS = ["price_usd","num_cores","processor_speed","battery_capacity","ram_capacity",
             "internal_memory","screen_size","refresh_rate","primary_camera_rear",
             "5g_or_not","resolution_height","resolution_width"]
PRICE_WEIGHT = 2.5          # mirrors pillar1/pillar3 weight scheme

AGE_COLS    = ["age_18_24_pct","age_25_34_pct","age_35_44_pct","age_45_54_pct","age_55_plus_pct"]
INCOME_COLS = ["income_low_pct","income_lower_mid_pct","income_mid_pct",
               "income_upper_mid_pct","income_high_pct"]
RFM_COLS    = ["rfm_recent_pct","rfm_champion_pct","rfm_loyal_pct",
               "rfm_at_risk_pct","rfm_hibernating_pct"]
DEMO_COLS   = AGE_COLS + INCOME_COLS + RFM_COLS

np.random.seed(RANDOM_STATE)


# ──────────────────────────────────────────────────────────────
# 1. Build phone-region table (mirrors p1._phone_table)
# ──────────────────────────────────────────────────────────────
def build_phone_table(df):
    num_specs = ["price_usd","5g_or_not","num_cores","processor_speed","battery_capacity",
                 "fast_charging_available","fast_charging","ram_capacity","internal_memory",
                 "screen_size","refresh_rate","num_rear_cameras","primary_camera_rear",
                 "primary_camera_front","extended_memory_available","resolution_height",
                 "resolution_width","avg_rating","market_segment"]
    agg = {s: "first" for s in num_specs}
    agg.update({c: "first" for c in ["processor_brand","os","market_segment"]})
    agg.update({d: "mean"  for d in DEMO_COLS})
    ph = df.groupby(["brand_name","model","region"]).agg(agg).reset_index()
    return ph


# ──────────────────────────────────────────────────────────────
# 2. Prepare scaled feature matrices (production-identical)
# ──────────────────────────────────────────────────────────────
def prepare(ph):
    ph_f = ph.copy()
    ph_f["price_usd"] = np.log1p(ph_f["price_usd"])          # log price
    groups = (ph["brand_name"] + "__" + ph["model"]).values
    tr_idx, te_idx = next(
        GroupShuffleSplit(1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        .split(ph_f, groups=groups)
    )
    scaler = StandardScaler().fit(ph_f[SIM_FEATS].iloc[tr_idx])
    w = np.ones(len(SIM_FEATS)); w[0] = PRICE_WEIGHT
    Xtr = scaler.transform(ph_f[SIM_FEATS].iloc[tr_idx]) * w
    Xte = scaler.transform(ph_f[SIM_FEATS].iloc[te_idx]) * w
    return Xtr, Xte, tr_idx, te_idx, ph["region"].values, groups


# ──────────────────────────────────────────────────────────────
# 3. Block R² helper (mirrors pillar3 evaluation)
# ──────────────────────────────────────────────────────────────
def block_r2(ph, te_idx, preds):
    """preds: dict[col] -> array(len(te_idx))"""
    age_r2    = float(np.mean([r2_score(ph[c].iloc[te_idx].values, preds[c]) for c in AGE_COLS]))
    income_r2 = float(np.mean([r2_score(ph[c].iloc[te_idx].values, preds[c]) for c in INCOME_COLS]))
    rfm_r2    = float(np.mean([r2_score(ph[c].iloc[te_idx].values, preds[c]) for c in RFM_COLS]))
    return age_r2, income_r2, rfm_r2


def row_entry(name, category, age_r2, income_r2, rfm_r2,
              inf_ms, train_s, paper_ref, paper_task, applicable="Yes"):
    mean_r2 = round(float(np.mean([age_r2, income_r2, rfm_r2])), 4)
    return {
        "Model":              name,
        "Category":           category,
        "R2_Age_block":       round(float(age_r2),    4),
        "R2_Income_block":    round(float(income_r2), 4),
        "R2_RFM_block":       round(float(rfm_r2),    4),
        "R2_Mean":            mean_r2,
        "Inference_ms":       round(inf_ms, 2),
        "Train_time_s":       round(train_s, 2),
        "Paper_Reported_Metric": paper_ref,
        "Paper_Task":         paper_task,
        "Applicable":         applicable,
    }


# ──────────────────────────────────────────────────────────────
# 4. KNN retrieval (production-identical: region-aware)
# ──────────────────────────────────────────────────────────────
def run_knn(ph, Xtr, Xte, tr_idx, te_idx, tr_region, te_region, k,
            name, category, paper_ref, paper_task):
    t0 = time.perf_counter()
    preds = {d: np.zeros(len(te_idx)) for d in DEMO_COLS}
    for i in range(len(te_idx)):
        m = tr_region == te_region[i]
        if m.sum() < k:
            m = np.ones(len(tr_region), dtype=bool)
        idx = np.where(m)[0]
        nn  = NearestNeighbors(n_neighbors=min(k, len(idx))).fit(Xtr[m])
        _, loc = nn.kneighbors(Xte[i:i+1])
        nbr = idx[loc[0]]
        for d in DEMO_COLS:
            preds[d][i] = ph[d].iloc[tr_idx].values[nbr].mean()
    inf_ms = (time.perf_counter()-t0)*1000
    age_r2, income_r2, rfm_r2 = block_r2(ph, te_idx, preds)
    print(f"  Age_R2={age_r2:>6.4f}  Inc_R2={income_r2:>6.4f}  "
          f"RFM_R2={rfm_r2:>6.4f}  Inf={inf_ms:.1f}ms")
    return row_entry(name, category, age_r2, income_r2, rfm_r2,
                     inf_ms, 0.0, paper_ref, paper_task)


# ──────────────────────────────────────────────────────────────
# 5. Clustering-based retrieval
# ──────────────────────────────────────────────────────────────
def run_clustering(ph, Xtr, Xte, tr_idx, te_idx, clusterer,
                   name, category, paper_ref, paper_task):
    t0 = time.perf_counter()
    labels_tr = clusterer.fit_predict(Xtr)
    ts = time.perf_counter()-t0

    # For each test point, find nearest training centroid
    # (works for all clustering methods — just find closest train neighbor per cluster)
    unique_labels = [l for l in np.unique(labels_tr) if l >= 0]  # exclude DBSCAN noise (-1)

    if len(unique_labels) == 0:
        # DBSCAN collapsed to all noise — fallback to global mean
        preds = {d: np.full(len(te_idx), ph[d].iloc[tr_idx].mean()) for d in DEMO_COLS}
    else:
        # Compute cluster centroids
        centroids = np.array([Xtr[labels_tr == l].mean(axis=0) for l in unique_labels])
        tr_demo   = {d: np.array([ph[d].iloc[tr_idx].values[labels_tr == l].mean()
                                   for l in unique_labels])
                     for d in DEMO_COLS}

        t1 = time.perf_counter()
        preds = {d: np.zeros(len(te_idx)) for d in DEMO_COLS}
        nn_centroids = NearestNeighbors(n_neighbors=1).fit(centroids)
        for i in range(len(te_idx)):
            _, loc = nn_centroids.kneighbors(Xte[i:i+1])
            c_idx = loc[0][0]
            for d in DEMO_COLS:
                preds[d][i] = tr_demo[d][c_idx]
        inf_ms = (time.perf_counter()-t1)*1000

    age_r2, income_r2, rfm_r2 = block_r2(ph, te_idx, preds)
    print(f"  Age_R2={age_r2:>6.4f}  Inc_R2={income_r2:>6.4f}  "
          f"RFM_R2={rfm_r2:>6.4f}  Inf={inf_ms:.1f}ms  Train={ts:.2f}s  "
          f"({len(unique_labels)} clusters)")
    return row_entry(name, category, age_r2, income_r2, rfm_r2,
                     inf_ms, ts, paper_ref, paper_task)


# ──────────────────────────────────────────────────────────────
# 6. Direct supervised multi-output regression
# ──────────────────────────────────────────────────────────────
def run_supervised(ph, Xtr, Xte, tr_idx, te_idx, Xtr_raw, Xte_raw, scaled,
                   reg, name, category, paper_ref, paper_task):
    Xtr_use = Xtr if scaled else Xtr_raw
    Xte_use = Xte if scaled else Xte_raw
    Y_tr = ph[DEMO_COLS].iloc[tr_idx].values
    t0 = time.perf_counter()
    reg.fit(Xtr_use, Y_tr)
    ts = time.perf_counter()-t0
    t1 = time.perf_counter()
    Y_pred = reg.predict(Xte_use)
    inf_ms = (time.perf_counter()-t1)*1000
    preds = {DEMO_COLS[j]: Y_pred[:, j] for j in range(len(DEMO_COLS))}
    age_r2, income_r2, rfm_r2 = block_r2(ph, te_idx, preds)
    print(f"  Age_R2={age_r2:>6.4f}  Inc_R2={income_r2:>6.4f}  "
          f"RFM_R2={rfm_r2:>6.4f}  Inf={inf_ms:.1f}ms  Train={ts:.2f}s")
    return row_entry(name, category, age_r2, income_r2, rfm_r2,
                     inf_ms, ts, paper_ref, paper_task)


# ──────────────────────────────────────────────────────────────
# 7. Non-testable reference rows
# ──────────────────────────────────────────────────────────────
NON_TESTABLE = [
    {"Model":"BERT Sentiment Analysis",
     "Category":"NLP / Transformer [XCRM paper]",
     "R2_Age_block":"N/A","R2_Income_block":"N/A","R2_RFM_block":"N/A","R2_Mean":"N/A",
     "Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"Customer sentiment extraction from reviews/feedback; used alongside XGBoost (98% acc)",
     "Paper_Task":"Unstructured text sentiment analysis for CRM — requires review text, not phone specs",
     "Applicable":"No — text modality; dataset has no customer review text"},
    {"Model":"Q-Learning / RL (Campaign Optimizer)",
     "Category":"Reinforcement Learning [CRM / AIPow papers]",
     "R2_Age_block":"N/A","R2_Income_block":"N/A","R2_RFM_block":"N/A","R2_Mean":"N/A",
     "Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"Dynamic personalization; 35% of Amazon sales from RL-driven recommendations",
     "Paper_Task":"Sequential campaign/offer optimization — needs reward signal from customer interactions",
     "Applicable":"No — RL policy model, not a demographic distribution predictor"},
    {"Model":"Collaborative Filtering (Recommender)",
     "Category":"Recommender System [SME / DataDriv papers]",
     "R2_Age_block":"N/A","R2_Income_block":"N/A","R2_RFM_block":"N/A","R2_Mean":"N/A",
     "Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"80% Netflix content discovered via CF; 35% Amazon sales uplift",
     "Paper_Task":"User-item recommendation — requires historical user-product interaction matrix",
     "Applicable":"No — no user-product interaction matrix available; cold-start new product"},
    {"Model":"NLP Chatbot (Automated Support)",
     "Category":"NLP / Conversational AI [XCRM paper]",
     "R2_Age_block":"N/A","R2_Income_block":"N/A","R2_RFM_block":"N/A","R2_Mean":"N/A",
     "Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"40% faster issue resolution; 10-15% CLV uplift from automated support",
     "Paper_Task":"Customer support automation via NLP — unrelated to demographic prediction",
     "Applicable":"No — conversational AI, not a segmentation or demographic model"},
    {"Model":"RNN (Time-Series Segmentation)",
     "Category":"Recurrent NN [AIBI paper]",
     "R2_Age_block":"N/A","R2_Income_block":"N/A","R2_RFM_block":"N/A","R2_Mean":"N/A",
     "Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"20-30% inventory improvement; 15-25% loyalty boost in AI-Driven BI Tools",
     "Paper_Task":"Sequential transaction history modeling — requires customer purchase sequences over time",
     "Applicable":"No — temporal sequence model needing transaction history; new product has none"},
]


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  CUSTOMER SEGMENTATION MODEL BENCHMARK (Pillar 3 Task)")
    print("=" * 72)

    print("Loading data and building phone-region table ...")
    df = pd.read_csv(DATA_PATH)
    ph = build_phone_table(df)
    print(f"  Phone-region rows: {len(ph)} | Unique phones: {ph.groupby(['brand_name','model']).ngroups}")

    Xtr, Xte, tr_idx, te_idx, region_all, groups = prepare(ph)
    tr_region = region_all[tr_idx]
    te_region = region_all[te_idx]
    print(f"  Train: {len(tr_idx)} | Test: {len(te_idx)} | Sim features: {Xtr.shape[1]}")

    # Raw (unscaled) versions for tree models
    Xtr_raw = ph[SIM_FEATS].iloc[tr_idx].values.astype(np.float32)
    Xte_raw = ph[SIM_FEATS].iloc[te_idx].values.astype(np.float32)

    results = []

    # ══════════════════════════════════════════════════════════
    # FAMILY A: KNN Retrieval (production method)
    # ══════════════════════════════════════════════════════════
    for k, tag in [(1,"sensitivity"), (5,"few"), (10,"PRODUCTION"), (15,"wider"), (20,"broadest")]:
        is_prod = " [PRODUCTION]" if k==10 else ""
        print(f"\n[KNN k={k:>2}]{is_prod}  [pillar3_segmentation.py + research papers]")
        results.append(run_knn(
            ph, Xtr, Xte, tr_idx, te_idx, tr_region, te_region, k,
            name=f"KNN Retrieval (k={k}){is_prod}",
            category="KNN Retrieval [Pillar 3 / all segmentation papers]",
            paper_ref=("Production method — region-aware, price-weighted (w=2.5); "
                       "pillar3: Age R2~0.68, Income R2~0.78, RFM R2~0.00" if k==10
                       else f"KNN sensitivity test (k={k})"),
            paper_task="Demographic distribution prediction via spec-similarity retrieval"))

    # ══════════════════════════════════════════════════════════
    # FAMILY B: Clustering-based
    # ══════════════════════════════════════════════════════════
    print("\n[ B1] K-Means (5 clusters)  [CustProf, AIPow, CLV, DataDriv, CRM, SME, Scalable]")
    results.append(run_clustering(
        ph, Xtr, Xte, tr_idx, te_idx,
        KMeans(n_clusters=5, init="random", random_state=RANDOM_STATE, n_init=10),
        "K-Means (5 clusters)", "Clustering [CustProf / AIPow / CLV / DataDriv / CRM / SME]",
        "Widely cited in all 7 segmentation papers; CRM Framework identified 5 optimal segments",
        "Customer segmentation for targeted promotions and retention"))

    print("\n[ B2] K-Means++ (5 clusters)  [CLV, CRM — 5 optimal segments]")
    results.append(run_clustering(
        ph, Xtr, Xte, tr_idx, te_idx,
        KMeans(n_clusters=5, init="k-means++", random_state=RANDOM_STATE, n_init=10),
        "K-Means++ (5 clusters)", "Clustering [CLV / CRM Framework]",
        "CRM: 5 segments (Champions/Loyal/At-Risk/Hibernating); CLV: K-means++ lowest inertia=9876",
        "Optimal segment identification for CLV-based retention strategies"))

    print("\n[ B3] K-Means++ (10 clusters)  [CLV — 90.4% prediction accuracy, 2.1s convergence]")
    results.append(run_clustering(
        ph, Xtr, Xte, tr_idx, te_idx,
        KMeans(n_clusters=10, init="k-means++", random_state=RANDOM_STATE, n_init=10),
        "K-Means++ (10 clusters)", "Clustering [CLV paper — headline result]",
        "CLV paper: K-means++ inertia=9876, convergence=2.1s, CLV prediction accuracy=90.4%",
        "CLV segment prediction (High 35%/Medium 45%/Low 20% segments)"))

    print("\n[ B4] Hierarchical Clustering (10 clusters)  [CLV, Scalable AI]")
    results.append(run_clustering(
        ph, Xtr, Xte, tr_idx, te_idx,
        AgglomerativeClustering(n_clusters=10),
        "Hierarchical Clustering (10)", "Clustering [CLV / Scalable AI]",
        "CLV & Scalable AI: hierarchical clustering for nested segment discovery",
        "Multi-level customer grouping for precision marketing"))

    print("\n[ B5] DBSCAN  [AIPow, DataDriv — density-based micro-segment discovery]")
    results.append(run_clustering(
        ph, Xtr, Xte, tr_idx, te_idx,
        DBSCAN(eps=1.5, min_samples=5),
        "DBSCAN", "Clustering [AIPow / DataDriv]",
        "AI-Powered Segmentation: DBSCAN for micro-segment discovery in real time",
        "Density-based segment discovery without pre-specified cluster count"))

    print("\n[ B6] Vector Quantization (k=10)  [CustProf — VQ clustering]")
    results.append(run_clustering(
        ph, Xtr, Xte, tr_idx, te_idx,
        KMeans(n_clusters=10, init="random", random_state=RANDOM_STATE+1, n_init=5,
               algorithm="lloyd"),   # VQ is Lloyd's algorithm = standard K-Means
        "Vector Quantization (k=10)", "Clustering [CustProf paper]",
        "Customer Profiling: VQ clustering for actionable segment derivation in direct marketing",
        "Customer grouping for targeted promotions and churn behaviour modeling"))

    # ══════════════════════════════════════════════════════════
    # FAMILY C: Direct Supervised Regression
    # NOTE: direct spec->demographics typically scores R2 near 0
    #       as documented in pillar3_segmentation.py
    # ══════════════════════════════════════════════════════════

    def sup(name, cat, reg, scaled, ref, task):
        print(f"\n{name}")
        results.append(run_supervised(
            ph, Xtr, Xte, tr_idx, te_idx, Xtr_raw, Xte_raw, scaled,
            MultiOutputRegressor(reg, n_jobs=-1), name, cat, ref, task))

    sup("[ C1] Linear Regression  [AIBI — 20-30% inventory improvement]",
        "Linear Model [AIBI]",
        LinearRegression(), True,
        "AI-Driven BI Tools: linear regression on historical sales and inventory data",
        "Spec-to-demographic direct regression (baseline)")

    sup("[ C2] Ridge Regression  [CLV paper]",
        "Regularised Linear [CLV]",
        Ridge(alpha=1.0), True,
        "CLV paper: Ridge alongside Lasso and RF for CLV prediction",
        "Regularised spec-to-CLV/demographic prediction")

    sup("[ C3] Lasso Regression  [CLV paper]",
        "Regularised Linear [CLV]",
        Lasso(alpha=0.01, max_iter=5000), True,
        "CLV paper: Lasso for sparse feature selection in CLV forecasting",
        "Feature-selecting spec-to-CLV prediction with L1 penalty")

    sup("[ C4] Decision Tree  [AIBI, SME, Scalable AI]",
        "Tree [AIBI / SME / Scalable AI]",
        DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE), False,
        "Cited in AIBI, AI-Driven SMEs, Scalable AI for customer behavior trees",
        "Interpretable spec-to-demographic segmentation tree")

    sup("[ C5] SVR (RBF)  [CustProf, SME, Scalable AI]",
        "Kernel Method [CustProf / SME / Scalable]",
        SVR(kernel="rbf", C=1.0, gamma="scale"), True,
        "CustProf: SVM alongside NN and Boosting Trees for sales/churn prediction",
        "Kernel-based demographic distribution regression")

    sup("[ C6] Random Forest  [AIPow, CLV, PredChurn — 92% churn accuracy]",
        "Bagging Ensemble [AIPow / CLV / PredChurn]",
        RandomForestRegressor(n_estimators=300, max_depth=10, n_jobs=-1,
                              random_state=RANDOM_STATE), False,
        "PredChurn: RF 92% accuracy (best); CLV: RF for CLV prediction; AIPow: RF for predictive analytics",
        "Demographic distribution prediction via random forest regression")

    sup("[ C7] AdaBoost  [ensemble literature]",
        "Boosting Ensemble [general analytics]",
        AdaBoostRegressor(n_estimators=100, random_state=RANDOM_STATE), False,
        "Ensemble boosting cited in AI-Driven Business Analytics papers",
        "Boosted demographic regression")

    sup("[ C8] Gradient Boosting  [CustProf MCC=0.469, CRM 84.3%, SME]",
        "Gradient Boosting [CustProf / CRM / SME]",
        GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                  random_state=RANDOM_STATE), False,
        "CustProf: Boosting Trees MCC=0.469 (best model); CRM: GB 84.3% (best vs LR 78.4%, RF 80.4%)",
        "Sales and churn prediction via gradient boosted trees")

    print("\n[ C9] XGBoost  [XCRM — 98% acc, PredChurn]")
    xgb_reg = MultiOutputRegressor(
        xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                         subsample=0.8, n_jobs=-1, random_state=RANDOM_STATE, verbosity=0),
        n_jobs=4)
    t0 = time.perf_counter()
    xgb_reg.fit(Xtr_raw, ph[DEMO_COLS].iloc[tr_idx].values); ts = time.perf_counter()-t0
    t1 = time.perf_counter(); Yp = xgb_reg.predict(Xte_raw); inf_ms=(time.perf_counter()-t1)*1000
    preds = {DEMO_COLS[j]: Yp[:,j] for j in range(len(DEMO_COLS))}
    a,inc,rfm = block_r2(ph, te_idx, preds)
    print(f"  Age_R2={a:>6.4f}  Inc_R2={inc:>6.4f}  RFM_R2={rfm:>6.4f}  Inf={inf_ms:.1f}ms  Train={ts:.2f}s")
    results.append(row_entry("XGBoost (Multi-Output Reg)",
                             "Gradient Boosted Trees [XCRM / PredChurn]",
                             a, inc, rfm, inf_ms, ts,
                             "XCRM: XGBoost 98% acc (churn), 96.8% F1 — best vs CNN/RNN/LSTM; PredChurn: cited",
                             "Churn prediction and customer behavior classification"))

    sup("[C10] LightGBM  [general GBT literature]",
        "Gradient Boosting [general analytics]",
        lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                          n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1), False,
        "Fast GBT cited in competitive intelligence and analytics pipelines",
        "High-speed demographic regression")

    print("\n[C11] CatBoost  [structured-data GBT]")
    cb_reg = MultiOutputRegressor(
        cbt.CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05,
                              random_seed=RANDOM_STATE, verbose=False), n_jobs=4)
    t0 = time.perf_counter()
    cb_reg.fit(Xtr_raw, ph[DEMO_COLS].iloc[tr_idx].values); ts = time.perf_counter()-t0
    t1 = time.perf_counter(); Yp = cb_reg.predict(Xte_raw); inf_ms=(time.perf_counter()-t1)*1000
    preds = {DEMO_COLS[j]: Yp[:,j] for j in range(len(DEMO_COLS))}
    a,inc,rfm = block_r2(ph, te_idx, preds)
    print(f"  Age_R2={a:>6.4f}  Inc_R2={inc:>6.4f}  RFM_R2={rfm:>6.4f}  Inf={inf_ms:.1f}ms  Train={ts:.2f}s")
    results.append(row_entry("CatBoost (Multi-Output Reg)",
                             "Gradient Boosted Trees [structured analytics]",
                             a, inc, rfm, inf_ms, ts,
                             "Structured-data GBT for tabular demographic regression",
                             "Tabular spec-to-demographic regression"))

    sup("[C12] MLP Neural Network  [CustProf, CLV, DataDriv, SME]",
        "Deep Learning [CustProf / CLV / DataDriv / SME]",
        MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=500, learning_rate="adaptive",
                     random_state=RANDOM_STATE, early_stopping=True), True,
        "CustProf: NN alongside SVM and Boosting Trees; CLV: NN for CLV prediction; SME: deep-learning recommenders",
        "Customer profile prediction and CLV forecasting via neural network")

    sup("[C13] Extra Trees  [ensemble methods]",
        "Randomised Ensemble [AI-Driven Analytics]",
        ExtraTreesRegressor(n_estimators=300, max_depth=10, n_jobs=-1,
                            random_state=RANDOM_STATE), False,
        "Randomised tree ensemble in AI-Driven Business Analytics literature",
        "Robust demographic prediction with high-variance spec data")

    # ──────────────────────────────────────────────────────────────
    # Save CSV
    # ──────────────────────────────────────────────────────────────
    col_order = ["Model","Category","R2_Age_block","R2_Income_block","R2_RFM_block","R2_Mean",
                 "Inference_ms","Train_time_s","Paper_Reported_Metric","Paper_Task","Applicable"]
    testable_df = (pd.DataFrame(results)[col_order]
                   .sort_values("R2_Mean", ascending=False).reset_index(drop=True))
    non_df = pd.DataFrame(NON_TESTABLE)
    for c in col_order:
        if c not in non_df.columns: non_df[c]="N/A"
    report = pd.concat([testable_df, non_df[col_order]], ignore_index=True)
    out = Path(__file__).parent / "benchmark_segmentation_results.csv"
    report.to_csv(out, index=False)
    print(f"\nReport saved -> {out}")

    # ──────────────────────────────────────────────────────────────
    # Console summary
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 104)
    print(f"{'#':<3} {'Model':<36} {'Age_R2':>8} {'Inc_R2':>8} {'RFM_R2':>8} "
          f"{'Mean_R2':>8} {'Inf(ms)':>9} {'Train(s)':>9}")
    print("-" * 104)
    for rank, row in testable_df.iterrows():
        flag = " <-- PRODUCTION" if "k=10)" in row["Model"] else ""
        print(f"{rank+1:<3} {row['Model'][:36]:<36} {row['R2_Age_block']:>8.4f} "
              f"{row['R2_Income_block']:>8.4f} {row['R2_RFM_block']:>8.4f} "
              f"{row['R2_Mean']:>8.4f} {row['Inference_ms']:>9.1f} "
              f"{row['Train_time_s']:>9.2f}{flag}")
    print("=" * 104)

    prod = testable_df[testable_df["Model"].str.contains("k=10")].iloc[0]
    print(f"\nProduction KNN (k=10): Age_R2={prod['R2_Age_block']:.4f}  "
          f"Income_R2={prod['R2_Income_block']:.4f}  RFM_R2={prod['R2_RFM_block']:.4f}  "
          f"Mean_R2={prod['R2_Mean']:.4f}")
    top = testable_df.iloc[0]
    print(f"Best model by Mean R2: {top['Model']}  (Mean_R2={top['R2_Mean']:.4f})")

    knn_rank = testable_df[testable_df["Model"].str.contains("k=10")].index[0] + 1
    print(f"\nProduction KNN rank (mean R2): {knn_rank} / {len(testable_df)}")

if __name__ == "__main__":
    main()
