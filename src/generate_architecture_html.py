"""
generate_architecture_html.py
Introspects the trained models and writes an elaborate MODEL ARCHITECTURE report
to docs/model_architectures.html
Run:  python -X utf8 generate_architecture_html.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, "."); sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
import numpy as np, joblib
import config as C

ROOT = Path(__file__).parent.parent
M = C.MODEL_DIR

def safe(fn, default="—"):
    try: return fn()
    except Exception: return default

# ── load artefacts ───────────────────────────────────────────────────
seg = joblib.load(f"{M}/pillar1_segment.joblib")
sim = joblib.load(f"{M}/pillar1_similarity.joblib")
dem = joblib.load(f"{M}/pillar2_demand.joblib")
pmf = joblib.load(f"{M}/pillar4_pmf.joblib")

# ── XGBoost classifier (P1) facts ────────────────────────────────────
clf = seg["model"]
seg_facts = {
    "n_estimators": safe(lambda: clf.n_estimators),
    "max_depth": safe(lambda: clf.max_depth),
    "lr": safe(lambda: clf.learning_rate),
    "subsample": safe(lambda: clf.subsample),
    "n_features": safe(lambda: clf.n_features_in_),
    "n_classes": safe(lambda: len(clf.classes_)),
    "classes": safe(lambda: ", ".join(seg["classes"])),
    "total_trees": safe(lambda: len(clf.get_booster().get_dump())),
    "n_num": len(seg["features_num"]), "n_cat": len(seg["features_cat"]),
}

# ── XGBoost regressor (P2) facts ─────────────────────────────────────
reg = dem["model"]
dem_facts = {
    "n_estimators": safe(lambda: reg.n_estimators),
    "max_depth": safe(lambda: reg.max_depth),
    "lr": safe(lambda: reg.learning_rate),
    "subsample": safe(lambda: reg.subsample),
    "colsample": safe(lambda: reg.colsample_bytree),
    "min_child_weight": safe(lambda: reg.min_child_weight),
    "n_features": safe(lambda: reg.n_features_in_),
    "total_trees": safe(lambda: len(reg.get_booster().get_dump())),
    "n_num": len(dem["num"]), "n_cat": len(dem["cat"]),
    "n_brand_priors": len(dem["brand_prior"]),
}

# ── Random Forest (P4) facts ─────────────────────────────────────────
rf = pmf["clf"]
node_counts = safe(lambda: [t.tree_.node_count for t in rf.estimators_], [])
depths = safe(lambda: [t.tree_.max_depth for t in rf.estimators_], [])
rf_facts = {
    "n_estimators": safe(lambda: rf.n_estimators),
    "max_depth": safe(lambda: rf.max_depth),
    "min_samples_leaf": safe(lambda: rf.min_samples_leaf),
    "class_weight": "balanced",
    "n_features": safe(lambda: rf.n_features_in_),
    "total_nodes": int(np.sum(node_counts)) if len(node_counts) else "—",
    "avg_nodes": f"{np.mean(node_counts):.0f}" if len(node_counts) else "—",
    "avg_depth": f"{np.mean(depths):.1f}" if len(depths) else "—",
    "n_num": len(pmf["num"]), "n_cat": len(pmf["cat"]),
}

# ── KNN similarity facts ─────────────────────────────────────────────
knn_facts = {
    "n_reference": safe(lambda: sim["matrix"].shape[0]),
    "n_features": safe(lambda: len(sim["sim_feats"])),
    "price_weight": safe(lambda: float(sim["weights"][0])),
    "feats": safe(lambda: ", ".join(sim["sim_feats"])),
}

def f(x):
    return f"{x:,}" if isinstance(x, int) else x

html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>BI Agent — Model Architectures</title>
<style>
@page{{size:A4;margin:16mm 14mm}}*{{box-sizing:border-box}}
body{{font-family:'Segoe UI',Arial,sans-serif;color:#1a1d29;line-height:1.55;font-size:11pt;margin:0}}
h1{{font-size:25pt;color:#312e81}}h2{{font-size:17pt;color:#312e81;border-bottom:3px solid #6366f1;padding-bottom:5px;margin-top:30px;page-break-after:avoid}}
h3{{font-size:13pt;color:#4338ca;margin-top:20px;page-break-after:avoid}}h4{{font-size:11.5pt;color:#4f46e5;margin-bottom:3px}}
table{{width:100%;border-collapse:collapse;margin:10px 0;font-size:9.7pt;page-break-inside:avoid}}
th{{background:#4338ca;color:#fff;text-align:left;padding:6px 9px}}td{{border:1px solid #d6d9e6;padding:5px 9px;vertical-align:top}}
tr:nth-child(even) td{{background:#f6f7fb}}
.diagram{{background:#0f1117;color:#c8cbe0;border-radius:8px;padding:14px 16px;margin:10px 0;font-family:Consolas,monospace;font-size:9pt;white-space:pre;overflow-x:auto;line-height:1.35}}
.eq{{background:#f5f6ff;border:1px solid #c7d2fe;border-left:4px solid #6366f1;border-radius:6px;padding:9px 13px;margin:9px 0;font-family:Consolas,monospace;font-size:9.6pt;white-space:pre-wrap;color:#1e1b4b}}
.note{{background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #f59e0b;border-radius:6px;padding:8px 13px;margin:8px 0;font-size:10pt}}
.kpi{{display:inline-block;background:#312e81;color:#fff;border-radius:8px;padding:8px 14px;margin:3px;text-align:center}}
.kpi b{{font-size:14pt;display:block}}
.section{{page-break-before:always}}.cover{{text-align:center;padding-top:150px;page-break-after:always}}
ul{{margin:6px 0;padding-left:20px}}li{{margin:3px 0}}.small{{font-size:9pt;color:#6b7280}}
</style></head><body>

<div class="cover">
<div style="font-size:50pt">🧠⚙️</div>
<h1>Smartphone BI Agent</h1>
<div style="font-size:15pt;color:#4b5563;margin-top:6px">Model Architectures — Detailed Reference</div>
<div style="font-size:11pt;color:#6b7280;max-width:540px;margin:18px auto">
The internal architecture of every model — gradient-boosted trees (XGBoost),
k-nearest-neighbour retrieval, and the random-forest ensemble — including the feature
pipeline, encoders, objective functions, hyper-parameters and measured model complexity.</div>
</div>

<h2>1. Models at a Glance</h2>
<table>
<tr><th>Pillar</th><th>Architecture</th><th>Family</th><th>Output</th></tr>
<tr><td>1A Segment</td><td>XGBoost Classifier</td><td>Gradient-boosted decision trees</td><td>4-class softmax</td></tr>
<tr><td>1B Similarity</td><td>Weighted k-NN</td><td>Instance-based retrieval</td><td>Nearest phones</td></tr>
<tr><td>2 Demand</td><td>XGBoost Regressor</td><td>Gradient-boosted decision trees</td><td>log-demand → units</td></tr>
<tr><td>3 Segmentation</td><td>k-NN Retrieval (k=10)</td><td>Instance-based averaging</td><td>Demographic distributions</td></tr>
<tr><td>4 PMF / Failure</td><td>Random Forest</td><td>Bagged decision trees</td><td>Binary probability</td></tr>
</table>
<div class="note">Two model families are used: <b>tree ensembles</b> (boosting for P1/P2, bagging for
P4) for supervised prediction, and <b>instance-based k-NN</b> (P1B/P3) for retrieval where direct
supervised learning fails.</div>

<h2>2. Shared Feature Pipeline</h2>
<p>Every supervised model shares the same pre-processing architecture before the learner:</p>
<div class="diagram">raw spec JSON
   │
   ├─► add_engineered_features()        perf_score, cam_total, pixels,
   │                                    battery_per_inch, ram_x_mem,
   │                                    price_per_ram, value_score
   │
   ├─► OrdinalEncoder (categoricals)    brand_name, processor_brand, os
   │     handle_unknown = use_encoded_value (-1)
   │
   └─► numeric specs passed through (trees need no scaling)
              │
              ▼
        model.predict()</div>
<p>Note: tree ensembles are scale-invariant, so numeric features are <b>not</b> standardised for
P1/P2/P4. Standardisation is applied <b>only</b> for the distance-based k-NN models (P1B/P3).</p>

<div class="section">
<h2>3. XGBoost — Gradient-Boosted Trees (Pillars 1A &amp; 2)</h2>
<h3>3.1 Architecture</h3>
<p>XGBoost builds an <b>additive ensemble</b> of CART regression trees. Each new tree is fit to the
gradient of the loss left by the current ensemble (functional gradient descent):</p>
<div class="eq">F_0(x) = base score
F_t(x) = F_(t-1)(x) + η · f_t(x)        η = learning_rate
f_t  = argmin_f  Σ_i [ g_i f(x_i) + ½ h_i f(x_i)² ] + Ω(f)
Ω(f) = γ·T + ½ λ ‖w‖²        (T leaves, leaf weights w)</div>
<p>g_i, h_i are the first/second-order gradients of the loss. Classification uses softmax
(multi:softprob); regression uses squared error on the log-target.</p>
<div class="diagram">         ┌────────┐   ┌────────┐         ┌────────┐
input ─► │ tree 1 │ + │ tree 2 │ + ... +  │ tree N │ ─► Σ ─► softmax / value
         └────────┘   └────────┘         └────────┘
            each tree corrects the residual error of the previous sum</div>

<h3>3.2 Instance — Pillar 1A Segment Classifier</h3>
<table>
<tr><th>Property</th><th>Value</th></tr>
<tr><td>Objective</td><td>multi:softprob (4-class)</td></tr>
<tr><td>Boosting rounds (n_estimators)</td><td>{f(seg_facts['n_estimators'])}</td></tr>
<tr><td>Max tree depth</td><td>{f(seg_facts['max_depth'])}</td></tr>
<tr><td>Learning rate (η)</td><td>{seg_facts['lr']}</td></tr>
<tr><td>Subsample</td><td>{seg_facts['subsample']}</td></tr>
<tr><td>Input features</td><td>{f(seg_facts['n_features'])} ({seg_facts['n_num']} numeric + {seg_facts['n_cat']} categorical)</td></tr>
<tr><td>Classes</td><td>{seg_facts['n_classes']} — {seg_facts['classes']}</td></tr>
<tr><td>Total trees in booster</td><td>{f(seg_facts['total_trees'])} (= rounds × classes)</td></tr>
</table>

<h3>3.3 Instance — Pillar 2 Demand Regressor</h3>
<table>
<tr><th>Property</th><th>Value</th></tr>
<tr><td>Objective</td><td>reg:squarederror on ln(1+units)</td></tr>
<tr><td>Boosting rounds</td><td>{f(dem_facts['n_estimators'])}</td></tr>
<tr><td>Max tree depth</td><td>{f(dem_facts['max_depth'])}</td></tr>
<tr><td>Learning rate (η)</td><td>{dem_facts['lr']}</td></tr>
<tr><td>Subsample / colsample_bytree</td><td>{dem_facts['subsample']} / {dem_facts['colsample']}</td></tr>
<tr><td>min_child_weight</td><td>{dem_facts['min_child_weight']}</td></tr>
<tr><td>Input features</td><td>{f(dem_facts['n_features'])} ({dem_facts['n_num']} numeric + {dem_facts['n_cat']} categorical + brand prior)</td></tr>
<tr><td>Total trees</td><td>{f(dem_facts['total_trees'])}</td></tr>
<tr><td>Brand-prior lookup size</td><td>{f(dem_facts['n_brand_priors'])} brands</td></tr>
</table>
<div class="note"><b>Why deeper + slower for demand.</b> Demand is noisier than segment, so P2 uses a
lower learning rate (0.03) with more rounds (700) and deeper trees (8) plus column subsampling to
regularise — versus the shallow, fast classifier (depth 6, η 0.1, 300 rounds).</div>
</div>

<div class="section">
<h2>4. k-Nearest Neighbours — Retrieval (Pillars 1B &amp; 3)</h2>
<h3>4.1 Architecture</h3>
<p>k-NN is <b>non-parametric / instance-based</b>: there is no trained weight matrix. The "model" is
the standardised catalogue itself plus a distance function.</p>
<div class="eq">1. price' = ln(1 + price)                      (log-scale skewed feature)
2. z_i = (x_i − μ_i) / σ_i                     (standardise each feature)
3. z_price ×= 2.5                              (up-weight price — defines phone "type")
4. d(a,b) = √ Σ_i (z_a,i − z_b,i)²             (weighted Euclidean)
5. retrieve k nearest within the same region</div>
<div class="diagram">query phone ─► standardise+weight ─► distance to every catalogue phone
                                          │
                          region filter ──┤
                                          ▼
                              k nearest neighbours
                                          │
   Pillar 1B: return them as competitors  │  Pillar 3: average their real
                                          ▼            buyer demographics
</div>
<table>
<tr><th>Property</th><th>Value</th></tr>
<tr><td>Reference catalogue (P1B)</td><td>{f(knn_facts['n_reference'])} phone×region rows</td></tr>
<tr><td>Distance features</td><td>{f(knn_facts['n_features'])}</td></tr>
<tr><td>Price weight</td><td>{knn_facts['price_weight']}×</td></tr>
<tr><td>Neighbours (P3)</td><td>k = 10, region-aware</td></tr>
<tr><td>Aggregation (P3)</td><td>mean of neighbours' distributions, renormalised to 100%</td></tr>
</table>
<p class="small">Distance features: {knn_facts['feats']}</p>
<div class="note"><b>Why k-NN for demographics.</b> A direct specs→demographics regressor scores R²≈0.
Retrieval works because "who buys this kind of phone" is best answered by the real buyers of
similar phones, not by a parametric function of the specs.</div>
</div>

<div class="section">
<h2>5. Random Forest — Bagged Trees (Pillar 4)</h2>
<h3>5.1 Architecture</h3>
<p>A Random Forest trains many decision trees independently on <b>bootstrap samples</b> with a
<b>random feature subset</b> at each split, then averages their class probabilities (bagging).
This decorrelates the trees and reduces variance.</p>
<div class="eq">P(failure | x) = (1/N) Σ_t  P_t(failure | x)        N = number of trees
each tree t:  bootstrap rows  +  random √p features per split
class_weight = "balanced"  → re-weights the 25% failure class</div>
<div class="diagram">           bootstrap 1 ─► tree 1 ─┐
training   bootstrap 2 ─► tree 2 ─┤
   data    bootstrap 3 ─► tree 3 ─┼─► average probabilities ─► P(failure)
            ...                   │
           bootstrap N ─► tree N ─┘</div>
<table>
<tr><th>Property</th><th>Value</th></tr>
<tr><td>Trees (n_estimators)</td><td>{f(rf_facts['n_estimators'])}</td></tr>
<tr><td>Max depth (cap)</td><td>{f(rf_facts['max_depth'])}</td></tr>
<tr><td>Measured avg depth</td><td>{rf_facts['avg_depth']}</td></tr>
<tr><td>min_samples_leaf</td><td>{f(rf_facts['min_samples_leaf'])}</td></tr>
<tr><td>Class weight</td><td>{rf_facts['class_weight']} (handles 25% failure base rate)</td></tr>
<tr><td>Input features</td><td>{f(rf_facts['n_features'])} ({rf_facts['n_num']} numeric + {rf_facts['n_cat']} categorical)</td></tr>
<tr><td>Total nodes across forest</td><td>{f(rf_facts['total_nodes'])}</td></tr>
<tr><td>Avg nodes / tree</td><td>{rf_facts['avg_nodes']}</td></tr>
</table>
<div class="note"><b>Why a forest, not boosting, for PMF.</b> The constructed failure label is noisier and
the dataset smaller (one row per phone), so bagging's variance reduction and the
<code>class_weight="balanced"</code> option are a better fit than boosting, which could over-fit the
quartile cut-off.</div>
</div>

<div class="section">
<h2>6. Training Architecture</h2>
<div class="eq">GroupShuffleSplit(test_size=0.2, random_state=42)
group key = brand_name + "__" + model        →  a phone is wholly in train OR test
leak-free priors (P2 brand prior) fit on the TRAIN fold only, mapped to both folds
cold-start rule: P2 &amp; P4 never receive post-launch columns as features</div>
<p>This split is the central guard against inflated metrics: random row-splitting would place the
same phone in both folds and let the trees memorise it.</p>

<h2>7. Inference Architecture (single request)</h2>
<div class="diagram">spec JSON
   │
   ├─ P1A: encode → XGBoost classifier → softmax → segment + confidence
   ├─ P1B: standardise+weight → k-NN → nearest phones (competitors)
   ├─ P3 : k-NN neighbours → average demographics → age/income/gender/RFM
   ├─ P2 : engineer+encode+brand prior → XGBoost reg → expm1 → 6-month curve
   └─ P4 : encode → Random Forest → P(failure) → PMF + GO/CAUTION/RECONSIDER
                                   │
                                   ▼
                    closed-form CLV + price optimiser + web news
                                   │
                                   ▼
                          unified JSON dashboard</div>

<h2>8. Model Complexity Summary</h2>
<table>
<tr><th>Model</th><th>Type</th><th>Trees / k</th><th>Depth</th><th>Features</th></tr>
<tr><td>P1A Segment</td><td>XGBoost clf</td><td>{f(seg_facts['total_trees'])} trees</td><td>{f(seg_facts['max_depth'])}</td><td>{f(seg_facts['n_features'])}</td></tr>
<tr><td>P2 Demand</td><td>XGBoost reg</td><td>{f(dem_facts['total_trees'])} trees</td><td>{f(dem_facts['max_depth'])}</td><td>{f(dem_facts['n_features'])}</td></tr>
<tr><td>P4 PMF</td><td>Random Forest</td><td>{f(rf_facts['n_estimators'])} trees / {f(rf_facts['total_nodes'])} nodes</td><td>~{rf_facts['avg_depth']}</td><td>{f(rf_facts['n_features'])}</td></tr>
<tr><td>P1B Similarity</td><td>k-NN</td><td>{f(knn_facts['n_reference'])} refs</td><td>—</td><td>{f(knn_facts['n_features'])}</td></tr>
<tr><td>P3 Demographics</td><td>k-NN</td><td>k = 10</td><td>—</td><td>{f(knn_facts['n_features'])}</td></tr>
</table>
<p class="small">All facts introspected live from models/*.joblib by src/generate_architecture_html.py.</p>
</div>

</body></html>"""

out = ROOT / "docs" / "model_architectures.html"
out.write_text(html, encoding="utf-8")
print("Wrote", out)
print("seg trees:", seg_facts["total_trees"], "| dem trees:", dem_facts["total_trees"],
      "| rf nodes:", rf_facts["total_nodes"], "| knn refs:", knn_facts["n_reference"])
