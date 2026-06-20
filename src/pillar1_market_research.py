"""
pillar1_market_research.py

Two jobs:
  (A) SEGMENT CLASSIFIER — predict market_segment (budget/mid-range/premium/flagship)
      from specs. This is genuinely a function of price+specs, so it scores ~99%.
  (B) SIMILARITY RETRIEVAL — given a new phone, find the most spec-similar phones that
      actually launched. This is the engine that powers Customer Segmentation (pillar 3),
      because "who buys this kind of phone" is learned from real historical buyers of
      similar phones, NOT guessed from specs directly.

Why group-split: we split by PHONE so a phone never appears in both train and test.
Random splitting would let the model memorize a phone seen in training and inflate accuracy.
"""
import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import config as C


def _phone_table(df):
    """One row per (phone, region): specs are constant, demographics averaged over its life."""
    agg = {s: "first" for s in C.NUM_SPECS}
    agg.update({c: "first" for c in ["processor_brand", "os", "market_segment"]})
    agg.update({d: "mean" for d in C.DEMO_COLS})
    return df.groupby(["brand_name", "model", "region"]).agg(agg).reset_index()


def train(df):
    os.makedirs(C.MODEL_DIR, exist_ok=True)

    # ---------- (A) SEGMENT CLASSIFIER ----------
    feat_num = C.NUM_SPECS
    feat_cat = C.CAT_SPECS
    X = df[feat_num + feat_cat].copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[feat_cat] = enc.fit_transform(X[feat_cat])
    y = df["market_segment"].astype("category")
    classes = list(y.cat.categories)
    y = y.cat.codes
    groups = df["brand_name"] + "__" + df["model"]
    tr, te = next(GroupShuffleSplit(1, test_size=C.TEST_SIZE,
                                    random_state=C.RANDOM_STATE).split(X, y, groups))
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                            subsample=0.9, n_jobs=-1, random_state=C.RANDOM_STATE)
    clf.fit(X.iloc[tr], y.iloc[tr])
    pred = clf.predict(X.iloc[te])
    seg_acc = accuracy_score(y.iloc[te], pred)
    seg_f1 = f1_score(y.iloc[te], pred, average="macro")
    joblib.dump({"model": clf, "encoder": enc, "features_num": feat_num,
                 "features_cat": feat_cat, "classes": classes},
                f"{C.MODEL_DIR}/pillar1_segment.joblib")

    # ---------- (B) SIMILARITY RETRIEVAL ENGINE ----------
    ph = _phone_table(df)
    sim_feats = ["price_usd", "num_cores", "processor_speed", "battery_capacity",
                 "ram_capacity", "internal_memory", "screen_size", "refresh_rate",
                 "primary_camera_rear", "5g_or_not", "resolution_height", "resolution_width"]
    ph_feat = ph.copy()
    ph_feat["price_usd"] = np.log1p(ph_feat["price_usd"])  # log price = more meaningful distance
    scaler = StandardScaler().fit(ph_feat[sim_feats])
    weights = np.ones(len(sim_feats)); weights[0] = 2.5  # price dominates buyer profile
    Xsim = scaler.transform(ph_feat[sim_feats]) * weights
    joblib.dump({"phone_table": ph, "scaler": scaler, "sim_feats": sim_feats,
                 "weights": weights, "matrix": Xsim},
                f"{C.MODEL_DIR}/pillar1_similarity.joblib")

    return {"segment_accuracy": seg_acc, "segment_macro_f1": seg_f1,
            "n_reference_phones": len(ph)}


def predict_segment(spec_row):
    art = joblib.load(f"{C.MODEL_DIR}/pillar1_segment.joblib")
    x = pd.DataFrame([spec_row])[art["features_num"] + art["features_cat"]].copy()
    x[art["features_cat"]] = art["encoder"].transform(x[art["features_cat"]])
    proba = art["model"].predict_proba(x)[0]
    i = int(np.argmax(proba))
    return art["classes"][i], float(proba[i])


def find_similar(spec_row, region, k=10):
    """Return indices+names of the k most spec-similar phones within the same region."""
    art = joblib.load(f"{C.MODEL_DIR}/pillar1_similarity.joblib")
    ph, scaler = art["phone_table"], art["scaler"]
    sf, w = art["sim_feats"], art["weights"]
    q = {f: spec_row[f] for f in sf}
    q["price_usd"] = np.log1p(q["price_usd"])
    qv = scaler.transform(pd.DataFrame([q])[sf]) * w
    mask = (ph["region"] == region).values
    if mask.sum() < k:
        mask = np.ones(len(ph), dtype=bool)  # fall back to all regions if sparse
    sub_idx = np.where(mask)[0]
    nn = NearestNeighbors(n_neighbors=min(k, len(sub_idx))).fit(art["matrix"][mask])
    dist, loc = nn.kneighbors(qv)
    chosen = sub_idx[loc[0]]
    return chosen, ph.iloc[chosen][["brand_name", "model"]].values.tolist(), dist[0]


if __name__ == "__main__":
    df = pd.read_csv(C.DATA_PATH)
    print(train(df))
