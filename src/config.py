"""
config.py — Single source of truth for the BI Agent.

The most important thing in this file is the FEATURE CONTRACT.
We split every column into:
  - KNOWN_AT_LAUNCH : things you know about a phone BEFORE it ships (specs, price, brand).
  - POST_LAUNCH     : things measured AFTER it ships (sales, mentions, returns, demographics).

The cold-start models (demand, PMF) are ONLY allowed to see KNOWN_AT_LAUNCH features.
This is what prevents "fake accuracy" / data leakage: we never let a model peek at
an outcome that co-occurs with the thing we're trying to predict.
"""

from pathlib import Path
_ROOT = Path(__file__).parent.parent
DATA_PATH = _ROOT / "global_smartphone_bi_v4_final.csv"
MODEL_DIR = str(_ROOT / "models")

# ---- Identity ----
ID_COLS = ["brand_name", "model"]

# ---- Numeric specs known at launch ----
NUM_SPECS = [
    "price_usd", "5g_or_not", "num_cores", "processor_speed", "battery_capacity",
    "fast_charging_available", "fast_charging", "ram_capacity", "internal_memory",
    "screen_size", "refresh_rate", "num_rear_cameras", "primary_camera_rear",
    "primary_camera_front", "extended_memory_available", "resolution_height",
    "resolution_width", "avg_rating",
]

# ---- Categorical specs known at launch ----
CAT_SPECS = ["brand_name", "processor_brand", "os"]

# ---- Context known at any point in time (legit for time-aware demand) ----
TIME_CONTEXT = ["months_since_launch", "month_index"]
REGION_CONTEXT = ["region"]
# market_segment & lifecycle are predicted by us (pillar 1) so treated as derived, not raw input.

# ---- POST-LAUNCH outcomes: NEVER feed these to cold-start predictors ----
POST_LAUNCH = [
    "social_media_mentions", "search_trend_score", "news_sentiment_score",
    "market_share_pct", "market_penetration_pct", "return_rate_pct",
    "warranty_claims_per_100", "nps_score", "recommendation_rate_pct",
    "repeat_purchase_rate_pct", "customer_lifetime_value_usd",
    "competitor_avg_price", "price_vs_competitor", "competitor_similarity",
    "units_sold_monthly",
]

# Demographic distribution blocks (predicted via similarity retrieval, pillar 3)
AGE_COLS = ["age_18_24_pct", "age_25_34_pct", "age_35_44_pct", "age_45_54_pct", "age_55_plus_pct"]
INCOME_COLS = ["income_low_pct", "income_lower_mid_pct", "income_mid_pct",
               "income_upper_mid_pct", "income_high_pct"]
RFM_COLS = ["rfm_recent_pct", "rfm_champion_pct", "rfm_loyal_pct",
            "rfm_at_risk_pct", "rfm_hibernating_pct"]
GENDER_COLS = ["gender_male_pct", "gender_female_pct", "gender_nonbinary_pct"]
DEMO_COLS = AGE_COLS + INCOME_COLS + RFM_COLS + GENDER_COLS

RANDOM_STATE = 42
TEST_SIZE = 0.2


def add_engineered_features(d):
    """Honest spec-derived features. All computable from launch specs alone."""
    d = d.copy()
    d["perf_score"] = d["num_cores"] * d["processor_speed"]
    d["cam_total"] = d["primary_camera_rear"] + d["primary_camera_front"]
    d["pixels"] = d["resolution_height"] * d["resolution_width"]
    d["battery_per_inch"] = d["battery_capacity"] / d["screen_size"]
    d["ram_x_mem"] = d["ram_capacity"] * d["internal_memory"]
    d["price_per_ram"] = d["price_usd"] / (d["ram_capacity"] + 1)
    d["value_score"] = (d["ram_capacity"] + d["battery_capacity"] / 1000
                        + d["cam_total"] / 10) / __import__("numpy").log1p(d["price_usd"])
    return d


ENGINEERED = ["perf_score", "cam_total", "pixels", "battery_per_inch",
              "ram_x_mem", "price_per_ram", "value_score"]
