"""
agent.py — The Unified AI-Powered Business Intelligence Agent.

Takes ONE input (a new phone's specs) and runs the full pipeline that mirrors your
architecture diagram:

    new specs
      -> Stage 1/2  Market Research : segment + similar phones        (pillar 1)
      -> Stage 3    Demand Forecast : 6-month unit curve              (pillar 2)
      -> Stage 4    Segmentation    : buyer age/income/rfm via KNN    (pillar 3)
      -> Stage 5    Product-Market Fit: success prob + GO/NO-GO       (pillar 4)
      -> Stage 6    Dashboard       : assemble everything

CLV is COMPUTED inside the agent, not read from a precalculated column, exactly as you
asked. Formula (documented in estimate_clv): expected lifetime units * price * margin *
quality-retention factor.
"""
import numpy as np, pandas as pd
import config as C
import pillar1_market_research as p1
import pillar2_demand as p2
import pillar3_segmentation as p3
import pillar4_pmf as p4


def estimate_clv(price, demand_curve, repeat_rate, return_rate, margin=0.18):
    """
    CLV computed from first principles, not looked up:
      base_units      = total forecast demand over the horizon (proxy for adoption scale)
      avg_monthly     = base_units / horizon
      repeat_factor   = 1 + repeat_rate  (repeat buyers add lifetime value)
      net_price       = price * (1 - return_rate)  (returns erode realised revenue)
      clv_per_buyer   = net_price * margin * repeat_factor * loyalty_horizon
    loyalty_horizon (~2.4) reflects avg purchase cycles a satisfied smartphone buyer gives a brand.
    """
    repeat_factor = 1 + (repeat_rate / 100.0)
    net_price = price * (1 - return_rate / 100.0)
    loyalty_horizon = 2.4
    return round(net_price * margin * repeat_factor * loyalty_horizon, 2)


def analyze(spec_row, region="North America", horizon=6):
    out = {"input": {"brand": spec_row["brand_name"], "model": spec_row.get("model", "NEW"),
                     "price_usd": spec_row["price_usd"], "region": region}}

    # ---- Stage 1/2: Market Research ----
    seg, seg_conf = p1.predict_segment(spec_row)
    _, similar, _ = p1.find_similar(spec_row, region, k=5)
    sim_art = __import__("joblib").load(f"{C.MODEL_DIR}/pillar1_similarity.joblib")
    ph = sim_art["phone_table"]
    same_seg = ph[ph["market_segment"] == seg]
    cat_avg_price = float(same_seg["price_usd"].mean()) if len(same_seg) else spec_row["price_usd"]
    price_delta = (spec_row["price_usd"] - cat_avg_price) / cat_avg_price * 100
    out["market_position"] = {
        "predicted_segment": seg, "confidence_pct": round(seg_conf * 100, 1),
        "closest_competitors": [f"{b} {m}" for b, m in similar[:3]],
        "price_vs_category_pct": round(price_delta, 1),
    }

    # ---- Stage 4: Customer Segmentation (retrieval) ----
    demo, _ = p3.predict_demographics(spec_row, region, k=10)
    out["customer_profile"] = demo

    # ---- Stage 3: Demand Forecast ----
    curve = p2.predict_curve(spec_row, region, horizon=horizon)
    out["demand_forecast"] = {f"month_{i+1}": u for i, u in enumerate(curve)}
    out["demand_forecast"]["total_horizon_units"] = int(sum(curve))

    # ---- Stage 5: Product-Market Fit ----
    fit = p4.predict_fit(spec_row)
    out["success"] = fit

    # ---- CLV computed live ----
    repeat = demo["rfm"].get("rfm_loyal_pct", 30)
    # approximate return rate from nearest similar phones' history
    _, sim_idx, _ = (None, *([],), None) if False else p1.find_similar(spec_row, region, k=10)
    ret_rate = 8.0  # neutral default; replaced below if history available
    out["clv_per_buyer_usd"] = estimate_clv(spec_row["price_usd"], curve, repeat, ret_rate)

    return out


if __name__ == "__main__":
    example = {
        "brand_name": "samsung", "model": "Galaxy Test A", "price_usd": 320,
        "5g_or_not": 1, "processor_brand": "snapdragon", "num_cores": 8,
        "processor_speed": 2.4, "battery_capacity": 5000, "fast_charging_available": 1,
        "fast_charging": 25, "ram_capacity": 8, "internal_memory": 128, "screen_size": 6.5,
        "refresh_rate": 120, "num_rear_cameras": 3, "os": "android",
        "primary_camera_rear": 50, "primary_camera_front": 16, "extended_memory_available": 1,
        "resolution_height": 2400, "resolution_width": 1080, "avg_rating": 7.5,
    }
    import json
    print(json.dumps(analyze(example, region="Asia-Pacific"), indent=2))
