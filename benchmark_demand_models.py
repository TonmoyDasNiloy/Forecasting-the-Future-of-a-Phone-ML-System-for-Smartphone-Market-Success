"""
benchmark_demand_models.py
==========================
Standalone benchmark — tests every ML/DL model cited across the demand-forecasting
research papers against the actual project dataset (Pillar 2 cold-start demand task).

Task: Regress log1p(units_sold_monthly) from launch-known specs + time context.
      Identical feature engineering and group-split as pillar2_demand.py.
      Zero imports from src/. Zero modifications to the existing system.

Research papers covered
-----------------------
[SRAS]      Revolutionizing Retail Analytics           -> GRU, LSTM, CNN, XGBoost, LR
[AIDF]      AI-Driven Demand Forecasting               -> Neural Network (MLP), RF, Gradient Boosting
[MARIOD]    Multi-Agent DRL (MARIOD)                   -> Transformer demand forecaster
[DeepAR]    DeepAR Probabilistic Forecasting           -> LSTM (DeepAR variant), noted as N/A for global multi-series
[USSup]     US Supply Chains AI Demand Forecasting     -> LR, ElasticNet, RF, XGBoost, MLP
[ERP]       AI-Driven Supply Chain Optimization (ERP)  -> ML demand forecasting (general)
[AIEnhInv]  AI Enhanced Inventory & Demand Forecasting -> LR, DT, RF, SVM, MLP, CNN, RNN/LSTM
[Churn]     AI Predictive Analytics for Retention      -> Self-attn NN, RL (noted as N/A)

Testable models (16)
--------------------
 1  Linear Regression        [SRAS, USSup, AIEnhInv]
 2  ElasticNet               [USSup]
 3  Decision Tree            [AIEnhInv]
 4  SVR (RBF)                [AIEnhInv]
 5  AdaBoost                 [AIDF / AI-Driven Analytics]
 6  Random Forest            [SRAS, AIDF, USSup, AIEnhInv]
 7  Extra Trees              [AI-Driven Analytics]
 8  Gradient Boosting (GBM)  [AIDF, AIEnhInv]
 9  XGBoost                  [SRAS — baseline vs GRU, USSup] — production model
10  LightGBM                 [general GBT literature]
11  CatBoost                 [structured-data GBT]
12  MLP (Neural Network)     [AIDF — MAE=1.23, RMSE=1.78; AIEnhInv — MAE=0.35]
13  LSTM                     [SRAS, DeepAR, AIEnhInv]
14  GRU                      [SRAS — outperforms XGBoost by 2.87-3.22% R²]
15  CNN 1D                   [SRAS, AIEnhInv]
16  Transformer (encoder)    [MARIOD]

Non-testable (7 reference rows in CSV)
---------------------------------------
- YOLOv8-X         (object detection — customer counting, not demand regression)
- BOT-SORT          (object tracking)
- ByteTrack         (object tracking)
- MARIOD / PPO      (multi-agent RL with IoT sensors — different input modality)
- DeepAR            (global probabilistic forecasting — requires 100s of related series)
- Self-Attn Churn   (classification task — customer churn, not demand)
- RL Dynamic Pricing (reinforcement learning for incentives, not regression)

Output: benchmark_demand_results.csv
"""

import sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection      import GroupShuffleSplit
from sklearn.preprocessing        import OrdinalEncoder, StandardScaler
from sklearn.metrics              import r2_score, mean_absolute_error
# Regressors
from sklearn.linear_model         import LinearRegression, ElasticNet
from sklearn.tree                 import DecisionTreeRegressor
from sklearn.svm                  import SVR
from sklearn.ensemble             import (AdaBoostRegressor, RandomForestRegressor,
                                          ExtraTreesRegressor, GradientBoostingRegressor)
from sklearn.neural_network       import MLPRegressor
import xgboost  as xgb
import lightgbm as lgb
import catboost as cbt

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Config (inlined — no src/ import)
# ──────────────────────────────────────────────────────────────
DATA_PATH    = Path(__file__).parent / "global_smartphone_bi_v4_final.csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
MAX_SEQ_LEN  = 24     # max months per phone-region in the dataset

NUM_SPECS = [
    "price_usd","5g_or_not","num_cores","processor_speed","battery_capacity",
    "fast_charging_available","fast_charging","ram_capacity","internal_memory",
    "screen_size","refresh_rate","num_rear_cameras","primary_camera_rear",
    "primary_camera_front","extended_memory_available","resolution_height",
    "resolution_width","avg_rating",
]
TIME_CONTEXT = ["months_since_launch","month_index"]
CAT_SPECS    = ["brand_name","processor_brand","os"]
REGION_CTX   = ["region"]
ENGINEERED   = ["perf_score","cam_total","pixels","battery_per_inch",
                 "ram_x_mem","price_per_ram","value_score"]

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# ──────────────────────────────────────────────────────────────
# Feature engineering (mirrors config.add_engineered_features)
# ──────────────────────────────────────────────────────────────
def add_engineered(d):
    d = d.copy()
    d["perf_score"]      = d["num_cores"] * d["processor_speed"]
    d["cam_total"]       = d["primary_camera_rear"] + d["primary_camera_front"]
    d["pixels"]          = d["resolution_height"] * d["resolution_width"]
    d["battery_per_inch"]= d["battery_capacity"] / d["screen_size"]
    d["ram_x_mem"]       = d["ram_capacity"] * d["internal_memory"]
    d["price_per_ram"]   = d["price_usd"] / (d["ram_capacity"] + 1)
    d["value_score"]     = (d["ram_capacity"] + d["battery_capacity"]/1000
                            + d["cam_total"]/10) / np.log1p(d["price_usd"])
    return d


# ──────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────
def prepare(df):
    d = add_engineered(df)
    num = NUM_SPECS + TIME_CONTEXT + ENGINEERED
    cat = CAT_SPECS + REGION_CTX
    y   = np.log1p(d["units_sold_monthly"].values)

    X = d[num + cat].copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[cat] = enc.fit_transform(X[cat])
    groups = (d["brand_name"] + "__" + d["model"]).values

    # group split by phone (brand+model)
    tr_idx, te_idx = next(
        GroupShuffleSplit(1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        .split(X, y, groups)
    )

    # leak-free brand demand prior (train fold only)
    brand_prior = pd.Series(y[tr_idx]).groupby(d["brand_name"].iloc[tr_idx].values).mean()
    gmean       = float(y[tr_idx].mean())
    bdp = d["brand_name"].map(brand_prior).fillna(gmean).values
    X.insert(0, "brand_demand_prior", bdp)

    feat_cols = ["brand_demand_prior"] + num  # cat already encoded in X

    Xf = X[feat_cols + cat].values.astype(np.float32)
    return Xf, y.astype(np.float32), groups, tr_idx, te_idx, d, feat_cols, cat


def build_sequences(d_orig, y, tr_groups_set, te_groups_set,
                    feat_cols_idx, groups, max_len=MAX_SEQ_LEN):
    """Build (n_phone_regions, max_len, n_feat) tensors for sequence models."""
    rows = []
    for name, sub in d_orig.groupby(["brand_name","model","region"]):
        gkey = f"{name[0]}__{name[1]}"
        split = "train" if gkey in tr_groups_set else ("test" if gkey in te_groups_set else None)
        if split is None:
            continue
        sub_s = sub.sort_values("months_since_launch")
        rows.append((split, sub_s.index.tolist()))

    def _make(idx_list, label):
        seqs, tgts, masks = [], [], []
        for split_label, idx in idx_list:
            if split_label != label:
                continue
            T = len(idx)
            feat = np.take(feat_cols_idx, idx, axis=0)   # (T, n_feat)
            tgt  = y[idx]                                  # (T,)
            pad_len = max_len - T
            if pad_len > 0:
                feat = np.vstack([feat, np.zeros((pad_len, feat.shape[1]))])
                tgt  = np.concatenate([tgt, np.zeros(pad_len)])
                mask = np.array([1]*T + [0]*pad_len, dtype=np.float32)
            else:
                feat = feat[:max_len]; tgt = tgt[:max_len]
                mask = np.ones(max_len, dtype=np.float32)
            seqs.append(feat); tgts.append(tgt); masks.append(mask)
        return (np.array(seqs, dtype=np.float32),
                np.array(tgts, dtype=np.float32),
                np.array(masks, dtype=np.float32))

    tr_seq, tr_tgt, tr_mask = _make(rows, "train")
    te_seq, te_tgt, te_mask = _make(rows, "test")
    return tr_seq, tr_tgt, tr_mask, te_seq, te_tgt, te_mask


# ──────────────────────────────────────────────────────────────
# Metrics helper
# ──────────────────────────────────────────────────────────────
def eval_metrics(y_true_log, y_pred_log, inf_ms, train_s,
                 model_name, category, paper_ref, paper_task):
    mae = float(mean_absolute_error(np.expm1(y_true_log), np.expm1(y_pred_log)))
    r2  = float(r2_score(y_true_log, y_pred_log))
    rmse= float(np.sqrt(np.mean((y_true_log - y_pred_log)**2)))
    return {
        "Model":              model_name,
        "Category":           category,
        "R2_log":             round(r2,  4),
        "MAE_units":          round(mae, 2),
        "RMSE_log":           round(rmse,4),
        "Inference_ms":       round(inf_ms, 2),
        "Train_time_s":       round(train_s, 2),
        "Paper_Reported_Metric": paper_ref,
        "Paper_Task":         paper_task,
        "Applicable":         "Yes",
    }


# ──────────────────────────────────────────────────────────────
# Deep-learning sequence models
# ──────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True,
                            dropout=dropout if layers>1 else 0)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out).squeeze(-1)          # (batch, seq)

class GRUModel(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, layers, batch_first=True,
                          dropout=dropout if layers>1 else 0)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out).squeeze(-1)

class CNN1DModel(nn.Module):
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        # x: (batch, seq, feat) -> (batch, feat, seq) for Conv1d
        out = self.conv(x.permute(0,2,1))          # (batch, hidden, seq)
        return self.head(out.permute(0,2,1)).squeeze(-1)

class TransformerModel(nn.Module):
    def __init__(self, n_feat, d_model=64, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        self.proj  = nn.Linear(n_feat, d_model)
        enc_layer  = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                                 dropout=dropout, batch_first=True)
        self.enc   = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head  = nn.Linear(d_model, 1)
    def forward(self, x):
        out = self.enc(self.proj(x))
        return self.head(out).squeeze(-1)


def train_seq_model(model, tr_seq, tr_tgt, tr_mask, te_seq, te_tgt, te_mask,
                    n_epochs=50, lr=1e-3, batch=64):
    sc   = StandardScaler()
    n, T, f = tr_seq.shape
    sc.fit(tr_seq.reshape(-1, f))
    tr_s = sc.transform(tr_seq.reshape(-1,f)).reshape(n,T,f).astype(np.float32)
    nt   = te_seq.shape[0]
    te_s = sc.transform(te_seq.reshape(-1,f)).reshape(nt,T,f).astype(np.float32)

    Xtr = torch.tensor(tr_s); ytr = torch.tensor(tr_tgt); mtr = torch.tensor(tr_mask)
    Xte = torch.tensor(te_s); yte = torch.tensor(te_tgt); mte = torch.tensor(te_mask)

    loader = DataLoader(TensorDataset(Xtr, ytr, mtr), batch_size=batch, shuffle=True)
    opt    = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    t0 = time.perf_counter()
    model.train()
    for _ in range(n_epochs):
        for xb, yb, mb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = ((pred - yb)**2 * mb).sum() / mb.sum().clamp(min=1)
            loss.backward(); opt.step()
    train_s = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        t1 = time.perf_counter()
        pred_te = model(Xte).numpy()
        inf_ms  = (time.perf_counter()-t1)*1000
    mask_np = mte.numpy().astype(bool)
    y_pred  = pred_te[mask_np]
    y_true  = yte.numpy()[mask_np]
    return y_true, y_pred, train_s, inf_ms


# ──────────────────────────────────────────────────────────────
# Non-testable reference rows
# ──────────────────────────────────────────────────────────────
NON_TESTABLE = [
    {"Model":"YOLOv8-X",
     "Category":"Object Detection [SRAS paper]",
     "R2_log":"N/A","MAE_units":"N/A","RMSE_log":"N/A",
     "Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"High mAP, Precision, Recall on MMPTRACK surveillance data",
     "Paper_Task":"Customer detection from surveillance video — not demand regression",
     "Applicable":"No — image/video model, no tabular sales input"},
    {"Model":"BOT-SORT",
     "Category":"Object Tracking [SRAS paper]",
     "R2_log":"N/A","MAE_units":"N/A","RMSE_log":"N/A",
     "Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"Superior MOTA tracking accuracy vs ByteTrack",
     "Paper_Task":"In-store customer tracking — not demand regression",
     "Applicable":"No — visual tracking algorithm, not a demand model"},
    {"Model":"ByteTrack",
     "Category":"Object Tracking [SRAS paper]",
     "R2_log":"N/A","MAE_units":"N/A","RMSE_log":"N/A",
     "Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"Higher FPS than BOT-SORT in real-time tracking",
     "Paper_Task":"High-FPS customer tracking — not demand regression",
     "Applicable":"No — visual tracking algorithm, not a demand model"},
    {"Model":"MARIOD / PPO (Multi-Agent DRL)",
     "Category":"Multi-Agent DRL [MARIOD paper]",
     "R2_log":"N/A","MAE_units":"N/A","RMSE_log":"N/A",
     "Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"MAPE -18.2%, Stockout -23.5%, Service Level 96.5% on Dunnhumby/Favorita/UCI",
     "Paper_Task":"Joint demand forecasting + inventory RL — requires RFID/shelf/environmental sensor streams and supply chain env simulation",
     "Applicable":"No — requires real-time IoT sensor streams and supply chain simulation environment"},
    {"Model":"DeepAR (Amazon)",
     "Category":"Probabilistic LSTM [DeepAR paper]",
     "R2_log":"N/A","MAE_units":"N/A","RMSE_log":"N/A",
     "Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"15-41% better median quantile loss vs traditional methods on 500K+ related series",
     "Paper_Task":"Global probabilistic forecasting across 100s of related time series — requires product family relatedness at scale",
     "Applicable":"No — requires a large corpus of related series for global parameter sharing; single cold-start product has no history"},
    {"Model":"Self-Attention Churn NN",
     "Category":"Attention-Based NN [AI Retention paper]",
     "R2_log":"N/A","MAE_units":"N/A","RMSE_log":"N/A",
     "Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"15-20% gain in churn prediction accuracy; 25-30% churn reduction",
     "Paper_Task":"Customer churn classification with eye-gaze and clickstream inputs — not demand regression",
     "Applicable":"No — classification task on behavioral signals, not tabular spec-based demand"},
    {"Model":"RL Dynamic Pricing / Incentive Model",
     "Category":"Reinforcement Learning [AI Retention paper]",
     "R2_log":"N/A","MAE_units":"N/A","RMSE_log":"N/A",
     "Inference_ms":"N/A","Train_time_s":"N/A",
     "Paper_Reported_Metric":"35% sales via personalized recommendations; 10-15% CLV uplift",
     "Paper_Task":"Adaptive incentive and discount assignment via RL — not demand quantity regression",
     "Applicable":"No — decision-making RL policy, not a demand forecast model"},
]


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  DEMAND FORECASTING MODEL BENCHMARK (Pillar 2 Task)")
    print("=" * 72)

    print("Loading and engineering features ...")
    df  = pd.read_csv(DATA_PATH)
    Xf, y, groups, tr_idx, te_idx, d_orig, feat_cols, cat_cols = prepare(df)
    n_feat = Xf.shape[1]
    print(f"  {len(y):,} rows | {n_feat} features | "
          f"Train: {len(tr_idx):,} rows | Test: {len(te_idx):,} rows")

    X_tr, X_te = Xf[tr_idx], Xf[te_idx]
    y_tr, y_te = y[tr_idx],  y[te_idx]
    sc = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

    tr_phones = set(groups[tr_idx])
    te_phones = set(groups[te_idx])

    # Build sequences for deep-learning sequence models
    print("Building phone-region sequences for LSTM/GRU/CNN/Transformer ...")
    tr_seq, tr_tgt, tr_mask, te_seq, te_tgt, te_mask = build_sequences(
        d_orig, y, tr_phones, te_phones, Xf, groups)
    print(f"  Train sequences: {tr_seq.shape[0]} | Test sequences: {te_seq.shape[0]} "
          f"| Seq length: {tr_seq.shape[1]} | Features: {tr_seq.shape[2]}")

    results = []

    def run_sklearn(name, reg, scaled, category, paper_ref, paper_task):
        Xtr = X_tr_s if scaled else X_tr
        Xte = X_te_s if scaled else X_te
        t0 = time.perf_counter(); reg.fit(Xtr, y_tr); ts = time.perf_counter()-t0
        t1 = time.perf_counter(); pred = reg.predict(Xte); im = (time.perf_counter()-t1)*1000
        m = eval_metrics(y_te, pred, im, ts, name, category, paper_ref, paper_task)
        print(f"  R2={m['R2_log']:>7.4f}  MAE={m['MAE_units']:>8.1f}  "
              f"RMSE={m['RMSE_log']:>6.4f}  Inf={im:.1f}ms  Train={ts:.2f}s")
        return m

    # ── 1. Linear Regression ─────────────────────────────────────────────
    print("\n[ 1/16] Linear Regression  [SRAS, USSup, AIEnhInv]")
    results.append(run_sklearn(
        "Linear Regression", LinearRegression(n_jobs=-1), scaled=True,
        category="Linear Model [SRAS / USSup / AIEnhInv]",
        paper_ref="Baseline in SRAS & US Supply Chains; outperformed by XGBoost/GRU",
        paper_task="Demand forecasting baseline across all retail papers"))

    # ── 2. ElasticNet ─────────────────────────────────────────────────────
    print("\n[ 2/16] ElasticNet  [USSup]")
    results.append(run_sklearn(
        "ElasticNet", ElasticNet(max_iter=5000, random_state=RANDOM_STATE), scaled=True,
        category="Regularised Linear [USSup]",
        paper_ref="RMSE=0.985, MAE=0.866, R2=0.003 (USSup best on CRM data)",
        paper_task="US logistics firm demand forecasting with PCA + MinMax scaling"))

    # ── 3. Decision Tree ──────────────────────────────────────────────────
    print("\n[ 3/16] Decision Tree  [AIEnhInv]")
    results.append(run_sklearn(
        "Decision Tree", DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
        scaled=False,
        category="Tree [AIEnhInv]",
        paper_ref="Cited in AI Enhanced Inventory; no specific RMSE reported",
        paper_task="Inventory replenishment decision trees"))

    # ── 4. SVR ────────────────────────────────────────────────────────────
    print("\n[ 4/16] SVR (RBF)  [AIEnhInv]")
    results.append(run_sklearn(
        "SVR (RBF)", SVR(kernel="rbf", C=1.0, gamma="scale"), scaled=True,
        category="Kernel Method [AIEnhInv]",
        paper_ref="Cited in AI Enhanced Inventory alongside MLP and LSTM",
        paper_task="Customer demand regression with RBF kernel"))

    # ── 5. AdaBoost ───────────────────────────────────────────────────────
    print("\n[ 5/16] AdaBoost  [AI-Driven Analytics literature]")
    results.append(run_sklearn(
        "AdaBoost", AdaBoostRegressor(n_estimators=200, learning_rate=0.5,
                                      random_state=RANDOM_STATE), scaled=False,
        category="Boosting Ensemble [AI-Driven Analytics]",
        paper_ref="Ensemble boosting cited across AI-Driven Business Analytics papers",
        paper_task="Demand and sales trend prediction"))

    # ── 6. Random Forest ──────────────────────────────────────────────────
    print("\n[ 6/16] Random Forest  [SRAS, AIDF, USSup, AIEnhInv]")
    results.append(run_sklearn(
        "Random Forest",
        RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=3,
                              n_jobs=-1, random_state=RANDOM_STATE), scaled=False,
        category="Bagging Ensemble [SRAS / AIDF / USSup / AIEnhInv]",
        paper_ref="Cited in 4 papers; outperformed by GRU in SRAS",
        paper_task="Sales forecasting and inventory demand across retail papers"))

    # ── 7. Extra Trees ────────────────────────────────────────────────────
    print("\n[ 7/16] Extra Trees  [AI-Driven Analytics]")
    results.append(run_sklearn(
        "Extra Trees",
        ExtraTreesRegressor(n_estimators=500, max_depth=12, n_jobs=-1,
                            random_state=RANDOM_STATE), scaled=False,
        category="Randomised Ensemble [AI-Driven Analytics]",
        paper_ref="Randomised tree ensemble in AI-Driven Business Analytics",
        paper_task="Robust demand prediction with high variance data"))

    # ── 8. Gradient Boosting (sklearn GBM) ───────────────────────────────
    print("\n[ 8/16] Gradient Boosting (sklearn GBM)  [AIDF, AIEnhInv]")
    results.append(run_sklearn(
        "Gradient Boosting (GBM)",
        GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, random_state=RANDOM_STATE), scaled=False,
        category="Gradient Boosting [AIDF / AIEnhInv]",
        paper_ref="AI-Driven Demand Forecasting: inventory reduction 30% with GBM ensemble",
        paper_task="Demand forecasting with seasonal and promotional signals"))

    # ── 9. XGBoost ────────────────────────────────────────────────────────
    print("\n[ 9/16] XGBoost  [SRAS — baseline vs GRU, USSup] — PRODUCTION MODEL")
    xgb_reg = xgb.XGBRegressor(
        n_estimators=700, max_depth=8, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.8, min_child_weight=3,
        n_jobs=-1, random_state=RANDOM_STATE, verbosity=0)
    t0 = time.perf_counter()
    xgb_reg.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    ts = time.perf_counter()-t0
    t1 = time.perf_counter(); xgb_pred = xgb_reg.predict(X_te)
    im = (time.perf_counter()-t1)*1000
    m = eval_metrics(y_te, xgb_pred, im, ts, "XGBoost (Regressor)",
                     "Gradient Boosted Trees [SRAS / USSup — PRODUCTION]",
                     "SRAS: XGBoost is baseline vs GRU; R2 gap 2.87-3.22%; USSup: cited with RF",
                     "Cold-start demand forecasting — production Pillar 2 model")
    print(f"  R2={m['R2_log']:>7.4f}  MAE={m['MAE_units']:>8.1f}  "
          f"RMSE={m['RMSE_log']:>6.4f}  Inf={im:.1f}ms  Train={ts:.2f}s")
    results.append(m)

    # ── 10. LightGBM ──────────────────────────────────────────────────────
    print("\n[10/16] LightGBM  [general gradient boosting literature]")
    lgb_reg = lgb.LGBMRegressor(
        n_estimators=700, max_depth=8, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.8,
        n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1)
    t0 = time.perf_counter(); lgb_reg.fit(X_tr, y_tr); ts = time.perf_counter()-t0
    t1 = time.perf_counter(); lgb_pred = lgb_reg.predict(X_te); im=(time.perf_counter()-t1)*1000
    m = eval_metrics(y_te, lgb_pred, im, ts, "LightGBM",
                     "Gradient Boosting [general analytics literature]",
                     "Fast GBT cited in competitive intelligence & supply chain analytics",
                     "High-speed demand forecasting for real-time pipelines")
    print(f"  R2={m['R2_log']:>7.4f}  MAE={m['MAE_units']:>8.1f}  "
          f"RMSE={m['RMSE_log']:>6.4f}  Inf={im:.1f}ms  Train={ts:.2f}s")
    results.append(m)

    # ── 11. CatBoost ──────────────────────────────────────────────────────
    print("\n[11/16] CatBoost  [structured-data GBT]")
    cb_reg = cbt.CatBoostRegressor(
        iterations=700, depth=8, learning_rate=0.03,
        random_seed=RANDOM_STATE, verbose=False)
    t0 = time.perf_counter(); cb_reg.fit(X_tr, y_tr); ts = time.perf_counter()-t0
    t1 = time.perf_counter(); cb_pred = cb_reg.predict(X_te); im=(time.perf_counter()-t1)*1000
    m = eval_metrics(y_te, cb_pred, im, ts, "CatBoost",
                     "Gradient Boosting [structured-data analytics]",
                     "Production-grade GBT cited in structured retail/supply analytics",
                     "Tabular demand forecasting with mixed feature types")
    print(f"  R2={m['R2_log']:>7.4f}  MAE={m['MAE_units']:>8.1f}  "
          f"RMSE={m['RMSE_log']:>6.4f}  Inf={im:.1f}ms  Train={ts:.2f}s")
    results.append(m)

    # ── 12. MLP (Neural Network) ──────────────────────────────────────────
    print("\n[12/16] MLP Neural Network  [AIDF — MAE=1.23, RMSE=1.78; AIEnhInv — MAE=0.35]")
    results.append(run_sklearn(
        "MLP (Neural Network)",
        MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500, learning_rate="adaptive",
                     random_state=RANDOM_STATE, early_stopping=True, n_iter_no_change=15),
        scaled=True,
        category="Deep Learning [AIDF / AIEnhInv]",
        paper_ref="AIDF: Neural Network MAE=1.23, RMSE=1.78 (best model); AIEnhInv: MAE=0.35, RMSE=0.45",
        paper_task="Demand forecasting with 5-yr sales history, seasonality, promotions (k-fold CV)"))

    # ── 13. LSTM ──────────────────────────────────────────────────────────
    print("\n[13/16] LSTM  [SRAS, DeepAR, AIEnhInv] — sequence mode over monthly demand")
    y_true_s, y_pred_s, ts, im = train_seq_model(
        LSTMModel(n_feat), tr_seq, tr_tgt, tr_mask, te_seq, te_tgt, te_mask, n_epochs=60)
    m = eval_metrics(y_true_s, y_pred_s, im, ts, "LSTM",
                     "Recurrent NN [SRAS / DeepAR / AIEnhInv]",
                     "SRAS: LSTM baseline vs GRU; DeepAR: LSTM outperforms trad. methods by 15-41%; AIEnhInv: MAE=0.35",
                     "Sequential demand forecasting over monthly sales time series (sequence mode)")
    print(f"  R2={m['R2_log']:>7.4f}  MAE={m['MAE_units']:>8.1f}  "
          f"RMSE={m['RMSE_log']:>6.4f}  Inf={im:.1f}ms  Train={ts:.2f}s")
    results.append(m)

    # ── 14. GRU ───────────────────────────────────────────────────────────
    print("\n[14/16] GRU  [SRAS — outperforms XGBoost by 2.87-3.22% R²]")
    y_true_g, y_pred_g, ts, im = train_seq_model(
        GRUModel(n_feat), tr_seq, tr_tgt, tr_mask, te_seq, te_tgt, te_mask, n_epochs=60)
    m = eval_metrics(y_true_g, y_pred_g, im, ts, "GRU",
                     "Recurrent NN [SRAS paper — headline model]",
                     "SRAS: GRU outperforms LR, XGBoost, CNN, LSTM by 2.87-3.22% R2 and up to 29.31% mAPE",
                     "Real-time demand forecasting from customer flow + sales data (sequence mode)")
    print(f"  R2={m['R2_log']:>7.4f}  MAE={m['MAE_units']:>8.1f}  "
          f"RMSE={m['RMSE_log']:>6.4f}  Inf={im:.1f}ms  Train={ts:.2f}s")
    results.append(m)

    # ── 15. CNN 1D ────────────────────────────────────────────────────────
    print("\n[15/16] CNN 1D  [SRAS, AIEnhInv]")
    y_true_c, y_pred_c, ts, im = train_seq_model(
        CNN1DModel(n_feat), tr_seq, tr_tgt, tr_mask, te_seq, te_tgt, te_mask, n_epochs=60)
    m = eval_metrics(y_true_c, y_pred_c, im, ts, "CNN 1D",
                     "Convolutional NN [SRAS / AIEnhInv]",
                     "SRAS: CNN baseline vs GRU; AIEnhInv: CNN cited with MAE=0.35, RMSE=0.45",
                     "Demand pattern extraction from sales time series (sequence mode)")
    print(f"  R2={m['R2_log']:>7.4f}  MAE={m['MAE_units']:>8.1f}  "
          f"RMSE={m['RMSE_log']:>6.4f}  Inf={im:.1f}ms  Train={ts:.2f}s")
    results.append(m)

    # ── 16. Transformer ───────────────────────────────────────────────────
    print("\n[16/16] Transformer  [MARIOD — cross-modal attention demand forecaster]")
    y_true_t, y_pred_t, ts, im = train_seq_model(
        TransformerModel(n_feat), tr_seq, tr_tgt, tr_mask, te_seq, te_tgt, te_mask,
        n_epochs=60, batch=32)
    m = eval_metrics(y_true_t, y_pred_t, im, ts, "Transformer (Encoder)",
                     "Attention-Based [MARIOD paper]",
                     "MARIOD: Transformer-based demand forecasting with cross-modal attention; MAPE -18.2%",
                     "Demand forecasting from heterogeneous sensor + sales data (sequence mode)")
    print(f"  R2={m['R2_log']:>7.4f}  MAE={m['MAE_units']:>8.1f}  "
          f"RMSE={m['RMSE_log']:>6.4f}  Inf={im:.1f}ms  Train={ts:.2f}s")
    results.append(m)

    # ──────────────────────────────────────────────────────────────
    # Build and save CSV
    # ──────────────────────────────────────────────────────────────
    col_order = ["Model","Category","R2_log","MAE_units","RMSE_log",
                 "Inference_ms","Train_time_s","Paper_Reported_Metric","Paper_Task","Applicable"]
    testable_df = (pd.DataFrame(results)[col_order]
                   .sort_values("R2_log", ascending=False))
    non_test_df = pd.DataFrame(NON_TESTABLE)
    for c in col_order:
        if c not in non_test_df.columns:
            non_test_df[c] = "N/A"
    report = pd.concat([testable_df, non_test_df[col_order]], ignore_index=True)
    out = Path(__file__).parent / "benchmark_demand_results.csv"
    report.to_csv(out, index=False)
    print(f"\nReport saved -> {out}")

    # ──────────────────────────────────────────────────────────────
    # Console summary
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print(f"{'#':<3} {'Model':<30} {'R2_log':>8} {'MAE_units':>10} {'RMSE':>8} "
          f"{'Inf(ms)':>9} {'Train(s)':>9}")
    print("-" * 96)
    for rank, (_, row) in enumerate(testable_df.iterrows(), 1):
        print(f"{rank:<3} {row['Model']:<30} {row['R2_log']:>8.4f} {row['MAE_units']:>10.1f} "
              f"{row['RMSE_log']:>8.4f} {row['Inference_ms']:>9.1f} {row['Train_time_s']:>9.2f}")
    print("=" * 96)
    top = testable_df.iloc[0]
    print(f"\nTop by R2:   {top['Model']}  (R2={top['R2_log']:.4f})")
    top_mae = testable_df.sort_values("MAE_units").iloc[0]
    print(f"Top by MAE:  {top_mae['Model']}  (MAE={top_mae['MAE_units']:.1f} units)")
    xgb_r = list(testable_df["Model"]).index("XGBoost (Regressor)") + 1
    print(f"\nXGBoost rank by R2:  {xgb_r} / {len(testable_df)}")
    xgb_r2 = list(testable_df.sort_values("MAE_units")["Model"]).index("XGBoost (Regressor)") + 1
    print(f"XGBoost rank by MAE: {xgb_r2} / {len(testable_df)}")

if __name__ == "__main__":
    main()
