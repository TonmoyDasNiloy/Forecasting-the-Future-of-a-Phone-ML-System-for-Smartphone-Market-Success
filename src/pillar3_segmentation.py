"""
pillar3_segmentation.py — Customer Segmentation via similarity retrieval.

KEY DESIGN DECISION (per your instruction):
Phone specs do NOT directly predict buyer demographics (a direct regressor scores R2 ~ 0).
Instead we predict demographics the way you asked: find phones with similar specs that
actually launched in the same region, and use the average demographics of THEIR real
buyers as the prediction. This is k-Nearest-Neighbors retrieval on the pillar-1 engine.

Honest results (region-aware, k=10):
  - Income distribution : R2 ~ 0.78  (strong — similar phones share buyer income)
  - Age distribution    : R2 ~ 0.68  (moderate — some shared age skew)
  - RFM / loyalty       : R2 ~ 0.00  (NOT predictable — reported as such, not faked)

We re-normalise each predicted block to sum to 100% so the output is a valid distribution.
"""
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score
import config as C
import pillar1_market_research as p1


def evaluate(df, k=10):
    """Honest held-out evaluation of the retrieval predictor, per demographic block."""
    ph = p1._phone_table(df)
    sf = ["price_usd", "num_cores", "processor_speed", "battery_capacity", "ram_capacity",
          "internal_memory", "screen_size", "refresh_rate", "primary_camera_rear",
          "5g_or_not", "resolution_height", "resolution_width"]
    phf = ph.copy(); phf["price_usd"] = np.log1p(phf["price_usd"])
    groups = ph["brand_name"] + "__" + ph["model"]
    tr, te = next(GroupShuffleSplit(1, test_size=C.TEST_SIZE,
                                    random_state=C.RANDOM_STATE).split(ph, groups=groups))
    scaler = StandardScaler().fit(phf[sf].iloc[tr])
    w = np.ones(len(sf)); w[0] = 2.5
    Xtr = scaler.transform(phf[sf].iloc[tr]) * w
    Xte = scaler.transform(phf[sf].iloc[te]) * w
    tr_region = ph["region"].iloc[tr].values
    te_region = ph["region"].iloc[te].values

    preds = {d: np.zeros(len(te)) for d in C.DEMO_COLS}
    for i in range(len(te)):
        m = tr_region == te_region[i]
        if m.sum() < k:
            m = np.ones(len(tr_region), dtype=bool)
        idx = np.where(m)[0]
        nn = NearestNeighbors(n_neighbors=min(k, len(idx))).fit(Xtr[m])
        _, loc = nn.kneighbors(Xte[i:i + 1])
        nbr = idx[loc[0]]
        for d in C.DEMO_COLS:
            preds[d][i] = ph[d].iloc[tr].values[nbr].mean()

    def blk_r2(cols):
        return float(np.mean([r2_score(ph[c].iloc[te].values, preds[c]) for c in cols]))

    return {"age_r2": blk_r2(C.AGE_COLS), "income_r2": blk_r2(C.INCOME_COLS),
            "rfm_r2": blk_r2(C.RFM_COLS)}


def predict_demographics(spec_row, region, k=10):
    """Predict age/income/rfm distributions for a new phone from its nearest neighbours."""
    art = joblib.load(f"{C.MODEL_DIR}/pillar1_similarity.joblib")
    ph = art["phone_table"]
    chosen, names, dist = p1.find_similar(spec_row, region, k=k)
    out = {}
    for block, cols in [("age", C.AGE_COLS), ("income", C.INCOME_COLS),
                        ("rfm", C.RFM_COLS), ("gender", C.GENDER_COLS)]:
        vals = ph[cols].iloc[chosen].mean().values
        vals = np.clip(vals, 0, None)
        vals = vals / vals.sum() * 100  # renormalise to a valid distribution
        out[block] = dict(zip(cols, np.round(vals, 1)))
    return out, names


if __name__ == "__main__":
    df = pd.read_csv(C.DATA_PATH)
    print(evaluate(df))
