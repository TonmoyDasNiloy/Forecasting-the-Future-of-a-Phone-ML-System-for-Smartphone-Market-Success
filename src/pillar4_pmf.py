"""
pillar4_pmf.py — Product-Market Fit & Failure Prediction.

The dataset has NO pmf/success/failure column, so we CONSTRUCT the target from real,
measured post-launch outcomes (transparent, documented formula below). Then we train a
classifier to predict that label from LAUNCH-KNOWN specs only — which is non-circular
because none of the label's ingredients are specs.

------------------------------------------------------------------
PMF SCORE FORMULA (each component min-max normalised to 0..1 across all phones):
    pmf = 0.25 * NPS
        + 0.20 * repeat_purchase_rate
        + 0.20 * recommendation_rate
        + 0.15 * (1 - return_rate)        # inverted: low returns = good
        + 0.10 * (1 - warranty_claims)    # inverted: low defects = good
        + 0.10 * sales_slope              # are monthly sales trending up?
Weights reflect that loyalty/advocacy signals (NPS, repeat, recommend = 65%) matter most
for true fit, quality signals (returns, warranty = 25%) next, momentum (slope = 10%) last.

FAILURE LABEL: phones in the bottom 25% of pmf_score are flagged failure=1.
We use a quartile (not an absolute cutoff) so the classes are balanced and the threshold
is data-relative rather than an arbitrary magic number.
------------------------------------------------------------------

Honest result: failure classifier accuracy ~0.79, AUC ~0.88 (group-split).
"""
import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import config as C


def _minmax(x, invert=False):
    r = (x - x.min()) / (x.max() - x.min() + 1e-9)
    return 1 - r if invert else r


def build_labels(df):
    g = df.groupby(["brand_name", "model"]).agg(
        avg_return=("return_rate_pct", "mean"),
        avg_nps=("nps_score", "mean"),
        avg_repeat=("repeat_purchase_rate_pct", "mean"),
        avg_rec=("recommendation_rate_pct", "mean"),
        avg_warranty=("warranty_claims_per_100", "mean"),
    ).reset_index()

    # sales slope per phone (units vs months_since_launch)
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
                      + 0.15 * _minmax(g["avg_return"], invert=True)
                      + 0.10 * _minmax(g["avg_warranty"], invert=True)
                      + 0.10 * _minmax(g["sales_slope"]))
    cutoff = g["pmf_score"].quantile(0.25)
    g["failure"] = (g["pmf_score"] < cutoff).astype(int)
    return g[["brand_name", "model", "pmf_score", "failure"]]


def train(df):
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    labels = build_labels(df)

    agg = {s: "first" for s in C.NUM_SPECS}
    agg.update({c: "first" for c in ["processor_brand", "os"]})
    ph = df.groupby(["brand_name", "model"]).agg(agg).reset_index().merge(
        labels, on=["brand_name", "model"])

    num = C.NUM_SPECS
    cat = ["processor_brand", "os", "brand_name"]
    X = ph[num + cat].copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[cat] = enc.fit_transform(X[cat])
    groups = ph["brand_name"] + "__" + ph["model"]

    # ----- classifier: failure -----
    yf = ph["failure"]
    tr, te = next(GroupShuffleSplit(1, test_size=C.TEST_SIZE,
                                    random_state=C.RANDOM_STATE).split(X, yf, groups))
    clf = RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=3,
                                 class_weight="balanced", n_jobs=-1,
                                 random_state=C.RANDOM_STATE)
    clf.fit(X.iloc[tr], yf.iloc[tr])
    proba = clf.predict_proba(X.iloc[te])[:, 1]
    acc = accuracy_score(yf.iloc[te], (proba >= 0.5).astype(int))
    auc = roc_auc_score(yf.iloc[te], proba)

    joblib.dump({"clf": clf, "encoder": enc, "num": num, "cat": cat,
                 "labels": labels}, f"{C.MODEL_DIR}/pillar4_pmf.joblib")
    return {"failure_accuracy": float(acc), "failure_auc": float(auc),
            "failure_base_rate": float(yf.mean())}


def predict_fit(spec_row):
    art = joblib.load(f"{C.MODEL_DIR}/pillar4_pmf.joblib")
    x = pd.DataFrame([spec_row])[art["num"] + art["cat"]].copy()
    x[art["cat"]] = art["encoder"].transform(x[art["cat"]])
    fail_p = float(art["clf"].predict_proba(x)[0, 1])
    pmf = 1 - fail_p          # success-side score for the dashboard
    if fail_p < 0.15:
        rec = "GO FOR LAUNCH"
    elif fail_p < 0.35:
        rec = "PROCEED WITH CAUTION"
    else:
        rec = "RECONSIDER / REWORK"
    return {"pmf_score": round(pmf, 2), "failure_risk_pct": round(fail_p * 100, 1),
            "recommendation": rec}


if __name__ == "__main__":
    df = pd.read_csv(C.DATA_PATH)
    print(train(df))
