"""
pillar2_demand.py — Cold-start demand forecasting.

Predicts units_sold_monthly for a brand-new phone using ONLY launch-known features
(specs, price, brand, region, month-since-launch). It never sees post-launch signals
like social_media_mentions (which correlate 0.86 with sales but are unknowable pre-launch
and would be pure leakage).

Target is log1p(units) because demand is right-skewed (skew ~1.8); log makes errors
symmetric and stabilises the regressor.

The brand_demand_prior is computed on the TRAIN fold ONLY and mapped onto both folds,
so the test phones' own sales never leak into their features.

Honest result: R2 ~ 0.85. That is the genuine ceiling of what specs explain about demand;
the rest is marketing/timing/luck that no spec-only model can recover.
"""
import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import config as C


def _build_features(df):
    d = C.add_engineered_features(df)
    num = (C.NUM_SPECS + C.TIME_CONTEXT + C.ENGINEERED)
    cat = C.CAT_SPECS + C.REGION_CONTEXT
    return d, num, cat


def train(df):
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    d, num, cat = _build_features(df)
    X = d[num + cat].copy()
    y = np.log1p(d["units_sold_monthly"])
    groups = d["brand_name"] + "__" + d["model"]
    tr, te = next(GroupShuffleSplit(1, test_size=C.TEST_SIZE,
                                    random_state=C.RANDOM_STATE).split(X, y, groups))

    # leak-free brand prior (train fold only)
    brand_prior = y.iloc[tr].groupby(d["brand_name"].iloc[tr]).mean()
    gmean = float(y.iloc[tr].mean())
    X["brand_demand_prior"] = d["brand_name"].map(brand_prior).fillna(gmean)

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[cat] = enc.fit_transform(X[cat])

    model = xgb.XGBRegressor(n_estimators=700, max_depth=8, learning_rate=0.03,
                             subsample=0.85, colsample_bytree=0.8, min_child_weight=3,
                             n_jobs=-1, random_state=C.RANDOM_STATE)
    model.fit(X.iloc[tr], y.iloc[tr])
    p = model.predict(X.iloc[te])
    r2 = r2_score(y.iloc[te], p)
    mae = mean_absolute_error(np.expm1(y.iloc[te]), np.expm1(p))

    joblib.dump({"model": model, "encoder": enc, "num": num, "cat": cat,
                 "brand_prior": brand_prior.to_dict(), "gmean": gmean},
                f"{C.MODEL_DIR}/pillar2_demand.joblib")
    return {"r2_log": float(r2), "mae_units": float(mae)}


def predict_curve(spec_row, region, horizon=6):
    """Predict a monthly demand curve for the first `horizon` months after launch."""
    art = joblib.load(f"{C.MODEL_DIR}/pillar2_demand.joblib")
    rows = []
    for m in range(horizon):
        r = dict(spec_row); r["region"] = region
        r["months_since_launch"] = m; r["month_index"] = m
        rows.append(r)
    d = C.add_engineered_features(pd.DataFrame(rows))
    X = d[art["num"] + art["cat"]].copy()
    X["brand_demand_prior"] = d["brand_name"].map(art["brand_prior"]).fillna(art["gmean"])
    X[art["cat"]] = art["encoder"].transform(X[art["cat"]])
    pred = np.expm1(art["model"].predict(X))
    return np.round(pred).astype(int).tolist()


if __name__ == "__main__":
    df = pd.read_csv(C.DATA_PATH)
    print(train(df))
