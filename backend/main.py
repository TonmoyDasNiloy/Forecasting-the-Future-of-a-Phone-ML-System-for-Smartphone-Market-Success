"""
backend/main.py — FastAPI server that loads all 4 trained pillars and
exposes POST /api/analyze for the React frontend.

Run: uvicorn backend.main:app --reload  (from bi_agent root)
  or: cd backend && uvicorn main:app --reload
"""
import sys
from pathlib import Path

# Make src/ importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

app = FastAPI(title="BI Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PhoneSpec(BaseModel):
    brand_name: str
    model: str = "New Model"
    price_usd: float = Field(..., gt=0)
    region: str = "North America"
    five_g: int = Field(1, alias="5g_or_not")
    processor_brand: str = "snapdragon"
    num_cores: int = 8
    processor_speed: float = 2.4
    battery_capacity: int = 5000
    fast_charging_available: int = 1
    fast_charging: int = 25
    ram_capacity: int = 8
    internal_memory: int = 128
    screen_size: float = 6.5
    refresh_rate: int = 120
    num_rear_cameras: int = 3
    os: str = "android"
    primary_camera_rear: int = 50
    primary_camera_front: int = 16
    extended_memory_available: int = 1
    resolution_height: int = 2400
    resolution_width: int = 1080
    avg_rating: float = 7.5

    class Config:
        populate_by_name = True


def spec_to_dict(spec: PhoneSpec) -> dict:
    return {
        "brand_name": spec.brand_name,
        "model": spec.model,
        "price_usd": spec.price_usd,
        "5g_or_not": spec.five_g,
        "processor_brand": spec.processor_brand,
        "num_cores": spec.num_cores,
        "processor_speed": spec.processor_speed,
        "battery_capacity": spec.battery_capacity,
        "fast_charging_available": spec.fast_charging_available,
        "fast_charging": spec.fast_charging,
        "ram_capacity": spec.ram_capacity,
        "internal_memory": spec.internal_memory,
        "screen_size": spec.screen_size,
        "refresh_rate": spec.refresh_rate,
        "num_rear_cameras": spec.num_rear_cameras,
        "os": spec.os,
        "primary_camera_rear": spec.primary_camera_rear,
        "primary_camera_front": spec.primary_camera_front,
        "extended_memory_available": spec.extended_memory_available,
        "resolution_height": spec.resolution_height,
        "resolution_width": spec.resolution_width,
        "avg_rating": spec.avg_rating,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "message": "BI Agent API is running"}


@app.post("/api/analyze")
def analyze(spec: PhoneSpec, region: Optional[str] = None):
    target_region = region or spec.region
    try:
        from agent import analyze as run_analyze
        result = run_analyze(spec_to_dict(spec), region=target_region)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /api/competitors ─────────────────────────────────────────────────────────

# Fixed scale for normalising each spec to 0-100
_NORM = {
    "price_usd":           (50,   2500),
    "ram_capacity":        (2,    16),
    "battery_capacity":    (2000, 7000),
    "primary_camera_rear": (5,    250),
    "screen_size":         (4.5,  7.5),
    "refresh_rate":        (60,   144),
    "perf_score":          (4,    48),   # cores × speed
}

def _norm(val: float, lo: float, hi: float) -> float:
    return round(max(0, min(100, (val - lo) / (hi - lo) * 100)), 1)

@app.post("/api/competitors")
def competitor_analysis(spec: PhoneSpec, region: Optional[str] = None):
    """
    Compare the input phone against its top-5 nearest competitors.
    Returns radar chart data (normalised 0-100) and a text insight list.
    """
    try:
        d = spec_to_dict(spec)
        target_region = region or spec.region

        import pillar1_market_research as p1
        chosen, names, dists = p1.find_similar(d, target_region, k=5)
        art = __import__("joblib").load(f"{__import__('config').MODEL_DIR}/pillar1_similarity.joblib")
        ph  = art["phone_table"]

        axes = ["ram_capacity", "battery_capacity", "primary_camera_rear",
                "screen_size", "refresh_rate", "perf_score"]
        axis_labels = ["RAM", "Battery", "Camera", "Screen", "Refresh Rate", "Performance"]

        # Spec display labels for the UI
        SPEC_META = {
            "ram_capacity":         ("RAM",           "GB"),
            "battery_capacity":     ("Battery",       "mAh"),
            "primary_camera_rear":  ("Main Camera",   "MP"),
            "primary_camera_front": ("Front Camera",  "MP"),
            "screen_size":          ("Screen",        "in"),
            "refresh_rate":         ("Refresh Rate",  "Hz"),
            "num_cores":            ("Cores",         ""),
            "processor_speed":      ("CPU Speed",     "GHz"),
            "internal_memory":      ("Storage",       "GB"),
            "price_usd":            ("Price",         "USD"),
            "5g_or_not":            ("5G",            ""),
            "perf_score":           ("Performance",   ""),
        }

        import config as C
        import pandas as pd, numpy as np
        row = C.add_engineered_features(pd.DataFrame([d])).iloc[0]
        your_scores = [_norm(float(row.get(a, 0)), *_NORM[a]) for a in axes]

        # Actual spec values for your phone
        your_specs = {
            k: round(float(row.get(k, 0)), 2)
            for k in SPEC_META
        }

        def _clean_name(brand: str, model: str) -> str:
            if model.lower().startswith(brand.lower()):
                return model
            return f"{brand.title()} {model}"

        competitors = []
        for (brand, model), dist in zip(names[:3], dists[:3]):
            comp_row = ph[(ph["brand_name"] == brand) & (ph["model"] == model)]
            if comp_row.empty:
                continue
            comp_row = C.add_engineered_features(comp_row).iloc[0]
            scores = [_norm(float(comp_row.get(a, 0)), *_NORM[a]) for a in axes]
            # Actual spec values for this competitor
            comp_specs = {k: round(float(comp_row.get(k, 0)), 2) for k in SPEC_META}
            competitors.append({
                "name":     _clean_name(brand, model),
                "scores":   scores,
                "specs":    comp_specs,
                "price":    round(float(comp_row.get("price_usd", 0)), 0),
                "distance": round(float(dist), 3),
            })

        # Build insights using REAL spec differences (not normalised)
        INSIGHT_AXES = [
            ("ram_capacity",        "RAM",          "GB"),
            ("battery_capacity",    "Battery",      "mAh"),
            ("primary_camera_rear", "Main Camera",  "MP"),
            ("refresh_rate",        "Refresh Rate", "Hz"),
            ("screen_size",         "Screen Size",  "in"),
        ]
        insights = []
        if competitors:
            for field, label, unit in INSIGHT_AXES:
                your_val = float(row.get(field, 0))
                avg_val  = sum(float(c["specs"].get(field, 0)) for c in competitors) / len(competitors)
                if avg_val == 0:
                    continue
                pct = (your_val - avg_val) / avg_val * 100
                if pct >= 12:
                    insights.append({
                        "type": "win",
                        "text": f"Your {label}: {your_val:g}{unit} vs avg {avg_val:.1f}{unit} (+{pct:.0f}%)"
                    })
                elif pct <= -12:
                    insights.append({
                        "type": "loss",
                        "text": f"Your {label}: {your_val:g}{unit} vs avg {avg_val:.1f}{unit} ({pct:.0f}%)"
                    })

        spec_meta_list = [
            {"key": k, "label": v[0], "unit": v[1]}
            for k, v in SPEC_META.items()
        ]
        return {
            "axes": axis_labels,
            "your_phone": {
                "name": f"{spec.brand_name.title()} {spec.model}",
                "scores": your_scores,
                "specs": your_specs,
            },
            "competitors": competitors,
            "spec_meta": spec_meta_list,
            "insights": insights,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /api/price-optimize ───────────────────────────────────────────────────────

@app.post("/api/price-optimize")
def price_optimize(spec: PhoneSpec, region: Optional[str] = None):
    """
    Recommend a STABLE profit-maximising launch price anchored to COMPETITOR
    PRICING — not to the user's own input price.

    Why anchor to competitors: a phone's build cost and its realistic demand
    curve are set by the market it competes in, not by the number the seller
    types in. Earlier versions anchored cost + demand to the input price, so
    raising the input always pushed the recommendation higher (it never
    converged). Here the anchor is fixed by:

      market_price = median price of the nearest competitor phones
      spec_ratio   = this phone's spec strength vs those competitors (0.8–1.25)
      fair_price   = market_price * spec_ratio   (spec-justified market price)
      unit_cost    = fair_price * (1 - GROSS_MARGIN)
      demand(p)    = D0 * (1 - BETA * (p - fair_price)/fair_price)

    Because none of these depend on the input price, the recommendation is
    stable: it stays the same whatever price the user enters. The input price
    is used only to tell the user whether they are above/below the optimum.
    """
    try:
        import numpy as np
        import pandas as pd
        import config as C
        import pillar1_market_research as p1
        import pillar2_demand as p2
        import pillar4_pmf as p4

        d = spec_to_dict(spec)
        target_region = region or spec.region
        input_price = float(spec.price_usd)

        axes = ["ram_capacity", "battery_capacity", "primary_camera_rear",
                "screen_size", "refresh_rate", "perf_score"]

        # ---- competitor pricing (the stable market anchor) ----
        # IMPORTANT: competitors are matched on SPECS ONLY (price excluded), so the
        # competitor set — and therefore the whole recommendation — does NOT change
        # when the user edits their input price. This is what makes it converge.
        comp_prices: list = []
        comp_specscores: list = []
        comp_list: list = []
        try:
            art = __import__("joblib").load(f"{C.MODEL_DIR}/pillar1_similarity.joblib")
            ph  = art["phone_table"]

            SPEC_SIM_FEATS = [
                "num_cores", "processor_speed", "battery_capacity", "ram_capacity",
                "internal_memory", "screen_size", "refresh_rate", "primary_camera_rear",
                "5g_or_not", "resolution_height", "resolution_width",
            ]
            M  = ph[SPEC_SIM_FEATS].to_numpy(dtype=float)
            mu = M.mean(axis=0)
            sd = M.std(axis=0) + 1e-9
            Mn = (M - mu) / sd
            q  = np.array([[float(d.get(f, 0)) for f in SPEC_SIM_FEATS]])
            qn = (q - mu) / sd

            mask = (ph["region"] == target_region).to_numpy()
            if mask.sum() < 5:
                mask = np.ones(len(ph), dtype=bool)
            idx   = np.where(mask)[0]
            dists = np.sqrt(((Mn[mask] - qn) ** 2).sum(axis=1))
            order = np.argsort(dists)[:5]
            chosen = idx[order]

            for ci in chosen:
                crow = C.add_engineered_features(ph.iloc[[ci]]).iloc[0]
                cprice = float(crow.get("price_usd", 0))
                if cprice <= 0:
                    continue
                comp_prices.append(cprice)
                comp_specscores.append(float(np.mean([_norm(float(crow.get(a, 0)), *_NORM[a]) for a in axes])))
                brand, model = str(crow["brand_name"]), str(crow["model"])
                nm = model if model.lower().startswith(brand.lower()) else f"{brand.title()} {model}"
                comp_list.append({"name": nm, "price": round(cprice, 0)})
        except Exception:
            pass

        # this phone's spec strength
        yrow = C.add_engineered_features(pd.DataFrame([d])).iloc[0]
        your_specscore = float(np.mean([_norm(float(yrow.get(a, 0)), *_NORM[a]) for a in axes]))

        # ---- derive stable anchors ----
        if comp_prices:
            market_price = float(np.median(comp_prices))
            comp_min     = float(min(comp_prices))
            comp_max     = float(max(comp_prices))
            comp_avg_spec = float(np.mean(comp_specscores)) or 1.0
            spec_ratio   = max(0.8, min(1.25, your_specscore / comp_avg_spec)) if comp_avg_spec > 0 else 1.0
        else:
            # No competitors found — fall back to the input price as anchor.
            market_price = input_price
            comp_min = comp_max = input_price
            spec_ratio = 1.0

        fair_price   = market_price * spec_ratio
        GROSS_MARGIN = 0.40
        unit_cost    = fair_price * (1 - GROSS_MARGIN)
        BETA         = 1.3                                 # demand sensitivity (elastic)

        # demand anchored at the spec-justified fair price (single model call)
        anchor = dict(d); anchor["price_usd"] = fair_price
        base_demand = max(1.0, float(p2.predict_curve(anchor, target_region, horizon=1)[0]))

        # ---- sweep prices around the MARKET band (stable, input-independent) ----
        lo = market_price * 0.5
        hi = market_price * 1.6
        prices = np.linspace(lo, hi, 14)
        points = []
        for price in prices:
            price  = float(price)
            rel    = (price - fair_price) / fair_price
            factor = max(0.0, 1.0 - BETA * rel)
            demand = base_demand * factor
            revenue = price * demand
            profit  = (price - unit_cost) * demand

            test = dict(d); test["price_usd"] = price
            fit  = p4.predict_fit(test)

            points.append({
                "price":          round(price, 0),
                "demand":         int(round(demand)),
                "revenue":        int(round(revenue)),
                "profit":         int(round(profit)),
                "pmf_score":      fit["pmf_score"],
                "failure_risk":   fit["failure_risk_pct"],
                "recommendation": fit["recommendation"],
            })

        # objective: max profit weighted by PMF, among profitable points
        def objective(p):
            return p["profit"] * p["pmf_score"] if p["profit"] > 0 else -1e9

        best     = max(points, key=objective)
        peak_rev = max(points, key=lambda p: p["revenue"])

        return {
            "curve":              points,
            "recommended_price":  best["price"],
            "current_price":      round(input_price, 0),
            "peak_revenue_price": peak_rev["price"],
            "market_price":       round(market_price, 0),
            "competitor_min":     round(comp_min, 0),
            "competitor_max":     round(comp_max, 0),
            "fair_price":         round(fair_price, 0),
            "spec_ratio":         round(spec_ratio, 3),
            "competitors":        comp_list,
            "assumptions": {
                "gross_margin_pct":   int(GROSS_MARGIN * 100),
                "unit_cost":          round(unit_cost, 0),
                "demand_sensitivity": BETA,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── helpers for news search ───────────────────────────────────────────────────

def _source_domain(url: str) -> str:
    """Extract readable domain from a URL."""
    try:
        m = re.search(r"https?://(?:www\.)?([^/]+)", url)
        return m.group(1) if m else url
    except Exception:
        return url


def _ddg_search(query: str, max_results: int = 5) -> list:
    try:
        from ddgs import DDGS
        with DDGS() as d:
            return list(d.text(query, max_results=max_results))
    except Exception:
        return []


def _ddg_news(query: str, max_results: int = 5) -> list:
    try:
        from ddgs import DDGS
        with DDGS() as d:
            return list(d.news(query, max_results=max_results))
    except Exception:
        return []


def _build_articles(raw: list, tag: str, seen: set) -> list:
    out = []
    for r in raw:
        url = r.get("href") or r.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append({
            "title":   r.get("title", "").strip(),
            "url":     url,
            "snippet": (r.get("body") or r.get("excerpt") or "")[:280].strip(),
            "source":  r.get("source") or _source_domain(url),
            "date":    r.get("date") or r.get("published") or None,
            "tag":     tag,
        })
    return out


# ── /api/news ─────────────────────────────────────────────────────────────────

@app.post("/api/news")
def market_news(spec: PhoneSpec, region: Optional[str] = None):
    """
    Search the web for reviews/news in TWO buckets only:
      1. "Your Phone"  — the model itself + phones with very similar specs
      2. "Competitor"  — the exact competitor phones the system predicts (Pillar 1)
    There is no general "Market" bucket. All queries run in parallel.
    """
    try:
        d = spec_to_dict(spec)
        model = spec.model
        target_region = region or spec.region

        # Spec fingerprint — describes the phone for finding similar handsets
        ram   = int(spec.ram_capacity)
        fiveg = "5G" if spec.five_g else "4G"
        cam   = int(spec.primary_camera_rear)
        spec_desc = f"{ram}GB RAM {fiveg} {cam}MP smartphone"

        # ── YOUR PHONE + spec-similar handsets ──────────────────────────────
        q_model = f"{model} smartphone review specs"
        q_news  = f"{model} smartphone"
        q_sim   = f"{spec_desc} review comparison 2024 2025"   # phones similar to mine

        tasks: List[tuple] = [
            ("text", q_model, 3, "Your Phone"),
            ("news", q_news,  2, "Your Phone"),
            ("text", q_sim,   3, "Your Phone"),
        ]

        # ── PREDICTED COMPETITORS (same engine the dashboard uses) ──────────
        try:
            import pillar1_market_research as p1
            _, similar, _ = p1.find_similar(d, target_region, k=4)
            for b, m in similar:
                name = m if m.lower().startswith(str(b).lower()) else f"{str(b).title()} {m}"
                tasks.append(("text", f"{name} review specs", 2, "Competitor"))
                tasks.append(("news", name, 1, "Competitor"))
        except Exception:
            pass

        # ── Run all searches in parallel ────────────────────────────────────
        raw_results: List[tuple] = []

        def run_task(kind: str, query: str, n: int, tag: str):
            results = _ddg_news(query, n) if kind == "news" else _ddg_search(query, n)
            return results, tag, query

        with ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as pool:
            futures = {pool.submit(run_task, *t): t for t in tasks}
            for fut in as_completed(futures, timeout=15):
                try:
                    raw_results.append(fut.result())
                except Exception:
                    pass

        # ── Deduplicate and merge with slot caps (two buckets only) ─────────
        seen: set      = set()
        articles: list = []
        queries_used: List[str] = []
        CAPS   = {"Your Phone": 9, "Competitor": 11}
        counts = {"Your Phone": 0, "Competitor": 0}

        tag_order = {"Your Phone": 0, "Competitor": 1}
        raw_results.sort(key=lambda x: tag_order.get(x[1], 2))

        for raw, tag, query in raw_results:
            for art in _build_articles(raw, tag, seen):
                if counts.get(tag, 0) < CAPS.get(tag, 99):
                    articles.append(art)
                    counts[tag] = counts.get(tag, 0) + 1
            if query not in queries_used:
                queries_used.append(query)

        # Round-robin interleave so both buckets are visible early
        from itertools import zip_longest
        by_tag: dict = {"Your Phone": [], "Competitor": []}
        for a in articles:
            by_tag.get(a["tag"], by_tag["Your Phone"]).append(a)

        interleaved: list = []
        for group in zip_longest(*by_tag.values()):
            for item in group:
                if item is not None:
                    interleaved.append(item)

        return {
            "articles":     interleaved[:20],
            "queries_used": queries_used[:5],
            "distribution": {k: len(v) for k, v in by_tag.items()},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
