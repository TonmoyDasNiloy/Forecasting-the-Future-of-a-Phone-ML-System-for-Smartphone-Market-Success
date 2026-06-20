"""
app.py — Streamlit dashboard for the BI Agent.
Run: streamlit run app.py  (from the bi_agent root directory)
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smartphone BI Agent",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* main bg */
  .stApp { background: #0f1117; }

  /* sidebar */
  [data-testid="stSidebar"] { background: #1a1d27; border-right: 1px solid #2a2d3e; }
  [data-testid="stSidebar"] .stMarkdown h3 { color: #7c8cf8; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: .1em; margin-bottom: 4px; }

  /* metric cards */
  .metric-card {
    background: #1e2130; border-radius: 12px; padding: 20px 18px;
    border: 1px solid #2a2d3e; text-align: center;
  }
  .metric-label { color: #8b8fa8; font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: .08em; margin-bottom: 6px; }
  .metric-value { color: #e8eaf6; font-size: 1.7rem; font-weight: 700; line-height: 1.1; }
  .metric-sub   { color: #8b8fa8; font-size: 0.78rem; margin-top: 4px; }

  /* recommendation banner */
  .banner-go        { background:#0d3b2e; border:1px solid #1db87a; border-radius:12px; padding:18px 24px; }
  .banner-caution   { background:#3b2d0d; border:1px solid #f0a500; border-radius:12px; padding:18px 24px; }
  .banner-reconsider{ background:#3b0d0d; border:1px solid #e03c3c; border-radius:12px; padding:18px 24px; }
  .banner-title  { font-size: 1.5rem; font-weight: 800; margin-bottom: 4px; }
  .banner-sub    { font-size: 0.88rem; color: #bbb; }

  /* section headers */
  .section-header {
    color: #7c8cf8; font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: .12em; border-bottom: 1px solid #2a2d3e;
    padding-bottom: 6px; margin-bottom: 12px;
  }

  /* run button */
  .stButton > button {
    width: 100%; background: linear-gradient(135deg,#4f6ef7,#7c3aed);
    color: white; border: none; border-radius: 10px; padding: 12px;
    font-weight: 700; font-size: 0.95rem; cursor: pointer;
    transition: opacity .2s;
  }
  .stButton > button:hover { opacity: .85; }

  /* hide streamlit chrome */
  #MainMenu { visibility: hidden; }
  footer     { visibility: hidden; }
  header     { visibility: hidden; }

  /* scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-thumb { background: #2a2d3e; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#c8cbe0",
    font_family="Inter, sans-serif",
)

REGIONS = ["Asia-Pacific", "North America", "Europe", "South Asia",
           "Middle East & Africa", "Latin America"]

BRANDS = ["samsung", "apple", "xiaomi", "oppo", "vivo", "realme",
          "oneplus", "motorola", "nokia", "huawei", "google", "sony",
          "honor", "tecno", "itel", "infinix", "gionee", "other"]

PROC_BRANDS = ["snapdragon", "mediatek", "apple", "exynos", "unisoc",
               "kirin", "dimensity", "helio", "other"]


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — input form
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📱 Phone Specs")
    st.markdown("Fill in the specs below and click **Analyze**.")
    st.markdown("---")

    # ---- Identity ----
    st.markdown("### Identity")
    brand   = st.selectbox("Brand", BRANDS, index=0)
    model   = st.text_input("Model name", value="Galaxy Test A")
    region  = st.selectbox("Target Region", REGIONS, index=0)

    st.markdown("### Pricing")
    price = st.number_input("Price (USD)", min_value=50, max_value=3000,
                            value=320, step=10)

    st.markdown("### Connectivity")
    five_g = st.toggle("5G Enabled", value=True)
    os_sel = st.selectbox("Operating System", ["android", "ios"], index=0)

    st.markdown("### Processor")
    proc_brand = st.selectbox("Processor Brand", PROC_BRANDS, index=0)
    num_cores  = st.select_slider("Cores", [2, 4, 6, 8, 10, 12], value=8)
    proc_speed = st.slider("Clock Speed (GHz)", 1.0, 4.0, 2.4, 0.1)

    st.markdown("### Memory")
    ram  = st.select_slider("RAM (GB)", [2, 3, 4, 6, 8, 12, 16], value=8)
    mem  = st.select_slider("Storage (GB)", [16, 32, 64, 128, 256, 512], value=128)
    ext  = st.toggle("Expandable Storage", value=True)

    st.markdown("### Battery")
    battery    = st.slider("Battery (mAh)", 2000, 7000, 5000, 100)
    fast_avail = st.toggle("Fast Charging", value=True)
    fast_w     = st.slider("Fast Charging (W)", 5, 120, 25, 5,
                            disabled=not fast_avail)

    st.markdown("### Display")
    screen   = st.slider("Screen Size (inches)", 4.5, 7.5, 6.5, 0.1)
    refresh  = st.select_slider("Refresh Rate (Hz)", [60, 90, 120, 144], value=120)
    res_h    = st.number_input("Resolution Height (px)", 720, 3200, 2400, 40)
    res_w    = st.number_input("Resolution Width (px)",  360, 1600, 1080, 40)

    st.markdown("### Camera")
    num_cams   = st.select_slider("Rear Cameras", [1, 2, 3, 4, 5], value=3)
    cam_rear   = st.slider("Main Camera (MP)", 5, 250, 50, 5)
    cam_front  = st.slider("Front Camera (MP)", 2, 64, 16, 2)

    st.markdown("### User Rating")
    avg_rating = st.slider("Avg Rating (1–10)", 1.0, 10.0, 7.5, 0.1)

    st.markdown("---")
    run = st.button("🔍  Analyze This Phone", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS — chart builders
# ══════════════════════════════════════════════════════════════════════════════
def demand_chart(forecast: dict):
    labels = [f"M{i}" for i in range(1, 7)]
    values = [forecast[f"month_{i}"] for i in range(1, 7)]
    colors = ["#4f6ef7"] * 6
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=values, textposition="outside", textfont_color="#c8cbe0",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=values, mode="lines+markers",
        line=dict(color="#7c8cf8", width=2, dash="dot"),
        marker=dict(size=6, color="#7c8cf8"),
        showlegend=False,
    ))
    fig.update_layout(
        **CHART_THEME,
        margin=dict(t=10, b=10, l=10, r=10),
        height=240,
        showlegend=False,
        yaxis=dict(gridcolor="#2a2d3e", zeroline=False),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        bargap=0.35,
    )
    return fig


def donut_chart(labels, values, colors, title=""):
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker_colors=colors,
        textinfo="percent",
        textfont_size=11,
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **CHART_THEME,
        margin=dict(t=30, b=10, l=10, r=10),
        height=230,
        showlegend=True,
        legend=dict(font_size=10, orientation="v",
                    x=1.02, y=0.5, bgcolor="rgba(0,0,0,0)"),
        annotations=[dict(text=f"<b>{title}</b>", x=0.5, y=0.5,
                          font_size=12, font_color="#e8eaf6",
                          showarrow=False)],
    )
    return fig


def pmf_gauge(score: float):
    color = "#1db87a" if score >= 0.65 else "#f0a500" if score >= 0.40 else "#e03c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number=dict(suffix="%", font=dict(size=32, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#8b8fa8",
                      tickfont=dict(color="#8b8fa8", size=10)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,  40], color="#2a0d0d"),
                dict(range=[40, 65], color="#2d2400"),
                dict(range=[65,100], color="#0d2d1e"),
            ],
            threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=65),
        ),
    ))
    fig.update_layout(**CHART_THEME, height=220,
                      margin=dict(t=20, b=10, l=30, r=30))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA — title + results
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📊 Smartphone BI Dashboard")
st.markdown("<p style='color:#8b8fa8;margin-top:-10px'>AI-powered market intelligence for new product launches</p>",
            unsafe_allow_html=True)
st.markdown("---")

if not run:
    st.markdown("""
    <div style="text-align:center; padding:80px 40px; color:#8b8fa8;">
      <div style="font-size:4rem">📱</div>
      <h2 style="color:#c8cbe0; margin-top:16px">Enter specs in the sidebar and click Analyze</h2>
      <p>The agent will predict market segment, demand curve, customer demographics,<br>
         product-market fit score, and customer lifetime value.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── build spec dict ───────────────────────────────────────────────────────────
spec = {
    "brand_name": brand,
    "model": model,
    "price_usd": float(price),
    "5g_or_not": int(five_g),
    "processor_brand": proc_brand,
    "num_cores": int(num_cores),
    "processor_speed": float(proc_speed),
    "battery_capacity": int(battery),
    "fast_charging_available": int(fast_avail),
    "fast_charging": int(fast_w) if fast_avail else 0,
    "ram_capacity": int(ram),
    "internal_memory": int(mem),
    "screen_size": float(screen),
    "refresh_rate": int(refresh),
    "num_rear_cameras": int(num_cams),
    "os": os_sel,
    "primary_camera_rear": int(cam_rear),
    "primary_camera_front": int(cam_front),
    "extended_memory_available": int(ext),
    "resolution_height": int(res_h),
    "resolution_width": int(res_w),
    "avg_rating": float(avg_rating),
}

# ── run inference ─────────────────────────────────────────────────────────────
with st.spinner("Running BI analysis across all four pillars…"):
    try:
        from agent import analyze
        result = analyze(spec, region=region)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.exception(e)
        st.stop()

mp   = result["market_position"]
df   = result["demand_forecast"]
cp   = result["customer_profile"]
suc  = result["success"]
clv  = result["clv_per_buyer_usd"]
rec  = suc["recommendation"]

# ══════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATION BANNER
# ══════════════════════════════════════════════════════════════════════════════
if "GO FOR" in rec:
    banner_cls, icon, color = "banner-go",         "✅", "#1db87a"
elif "CAUTION" in rec:
    banner_cls, icon, color = "banner-caution",    "⚠️", "#f0a500"
else:
    banner_cls, icon, color = "banner-reconsider", "🚫", "#e03c3c"

st.markdown(f"""
<div class="{banner_cls}">
  <div class="banner-title" style="color:{color}">{icon}  {rec}</div>
  <div class="banner-sub">
    {brand.capitalize()} {model} &nbsp;·&nbsp; ${price:,} &nbsp;·&nbsp; {region}
    &nbsp;·&nbsp; Failure risk: {suc['failure_risk_pct']}%
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ROW 1 — KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
seg_colors = {
    "budget": "#4fc3f7", "mid-range": "#7c8cf8",
    "premium": "#ab47bc", "flagship": "#f0a500",
}
seg_color = seg_colors.get(mp["predicted_segment"], "#7c8cf8")
price_delta = mp["price_vs_category_pct"]
price_sign  = "+" if price_delta >= 0 else ""
price_color = "#e03c3c" if price_delta > 15 else "#1db87a" if price_delta < -5 else "#f0a500"

c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    (c1, "Market Segment",   mp["predicted_segment"].upper(),
     f"Confidence: {mp['confidence_pct']}%",          seg_color),
    (c2, "PMF Score",        f"{suc['pmf_score']*100:.0f} / 100",
     "Product–Market Fit",                            color),
    (c3, "6-Month Demand",   f"{df['total_horizon_units']:,}",
     "Projected units",                               "#4fc3f7"),
    (c4, "CLV / Buyer",      f"${clv:,.0f}",
     "Customer lifetime value",                       "#1db87a"),
    (c5, "Price vs Market",  f"{price_sign}{price_delta:.1f}%",
     "vs. same-segment avg",                          price_color),
]

for col, label, val, sub, clr in cards:
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{clr}">{val}</div>
          <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ROW 2 — DEMAND FORECAST  +  COMPETITORS
# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="section-header">📈 6-Month Demand Forecast</div>',
                unsafe_allow_html=True)
    st.plotly_chart(demand_chart(df), use_container_width=True, config={"displayModeBar": False})

with right:
    st.markdown('<div class="section-header">🏆 Closest Competitors</div>',
                unsafe_allow_html=True)
    comps = mp["closest_competitors"]
    for i, c in enumerate(comps, 1):
        st.markdown(f"""
        <div style="background:#1e2130;border-radius:8px;padding:10px 14px;
                    margin-bottom:8px;border:1px solid #2a2d3e;font-size:0.85rem;color:#c8cbe0;">
          <span style="color:#7c8cf8;font-weight:700;">#{i}</span>&nbsp; {c}
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Monthly Breakdown</div>',
                unsafe_allow_html=True)
    cols_m = st.columns(6)
    for i in range(1, 7):
        with cols_m[i-1]:
            v = df[f"month_{i}"]
            st.markdown(f"""
            <div style="text-align:center;background:#1e2130;border-radius:8px;
                        padding:8px 4px;border:1px solid #2a2d3e;">
              <div style="color:#8b8fa8;font-size:0.65rem">M{i}</div>
              <div style="color:#4f6ef7;font-weight:700;font-size:0.95rem">{v}</div>
            </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ROW 3 — CUSTOMER PROFILE  (3 donuts)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">👥 Customer Profile</div>',
            unsafe_allow_html=True)

age_data = cp["age"]
inc_data = cp["income"]
rfm_data = cp["rfm"]

age_labels = ["18–24", "25–34", "35–44", "45–54", "55+"]
age_vals   = [age_data[k] for k in
              ["age_18_24_pct","age_25_34_pct","age_35_44_pct","age_45_54_pct","age_55_plus_pct"]]
age_colors = ["#4f6ef7","#7c8cf8","#ab47bc","#f06292","#4fc3f7"]

inc_labels = ["Low","Lower-mid","Mid","Upper-mid","High"]
inc_vals   = [inc_data[k] for k in
              ["income_low_pct","income_lower_mid_pct","income_mid_pct",
               "income_upper_mid_pct","income_high_pct"]]
inc_colors = ["#e03c3c","#f0a500","#ffd54f","#81c784","#1db87a"]

rfm_labels = ["Recent","Champion","Loyal","At-risk","Hibernating"]
rfm_vals   = [rfm_data[k] for k in
              ["rfm_recent_pct","rfm_champion_pct","rfm_loyal_pct",
               "rfm_at_risk_pct","rfm_hibernating_pct"]]
rfm_colors = ["#4fc3f7","#1db87a","#7c8cf8","#f0a500","#e03c3c"]

ca, cb, cc = st.columns(3)
with ca:
    st.markdown("<p style='text-align:center;color:#8b8fa8;font-size:0.75rem;text-transform:uppercase;letter-spacing:.08em'>Age Distribution</p>",
                unsafe_allow_html=True)
    st.plotly_chart(donut_chart(age_labels, age_vals, age_colors, "Age"),
                    use_container_width=True, config={"displayModeBar": False})

with cb:
    st.markdown("<p style='text-align:center;color:#8b8fa8;font-size:0.75rem;text-transform:uppercase;letter-spacing:.08em'>Income Distribution</p>",
                unsafe_allow_html=True)
    st.plotly_chart(donut_chart(inc_labels, inc_vals, inc_colors, "Income"),
                    use_container_width=True, config={"displayModeBar": False})

with cc:
    st.markdown("<p style='text-align:center;color:#8b8fa8;font-size:0.75rem;text-transform:uppercase;letter-spacing:.08em'>Customer Loyalty (RFM)</p>",
                unsafe_allow_html=True)
    st.plotly_chart(donut_chart(rfm_labels, rfm_vals, rfm_colors, "RFM"),
                    use_container_width=True, config={"displayModeBar": False})

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ROW 4 — PMF GAUGE  +  RAW JSON
# ══════════════════════════════════════════════════════════════════════════════
g_col, j_col = st.columns([1, 2])

with g_col:
    st.markdown('<div class="section-header">🎯 Product-Market Fit Score</div>',
                unsafe_allow_html=True)
    st.plotly_chart(pmf_gauge(suc["pmf_score"]),
                    use_container_width=True, config={"displayModeBar": False})
    st.markdown(f"""
    <div style="display:flex;justify-content:space-around;margin-top:4px">
      <div style="text-align:center">
        <div style="color:#8b8fa8;font-size:0.68rem;text-transform:uppercase">PMF Score</div>
        <div style="color:{color};font-size:1.1rem;font-weight:700">{suc['pmf_score']*100:.1f}%</div>
      </div>
      <div style="text-align:center">
        <div style="color:#8b8fa8;font-size:0.68rem;text-transform:uppercase">Failure Risk</div>
        <div style="color:#e03c3c;font-size:1.1rem;font-weight:700">{suc['failure_risk_pct']}%</div>
      </div>
      <div style="text-align:center">
        <div style="color:#8b8fa8;font-size:0.68rem;text-transform:uppercase">Base Rate</div>
        <div style="color:#8b8fa8;font-size:1.1rem;font-weight:700">25%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with j_col:
    st.markdown('<div class="section-header">🔩 Full Analysis JSON</div>',
                unsafe_allow_html=True)
    st.json(result, expanded=False)

st.markdown("<br><hr style='border-color:#2a2d3e'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#8b8fa8;font-size:0.72rem'>"
    "Smartphone BI Agent · Powered by XGBoost + Random Forest + KNN Retrieval · "
    "116,481 phone snapshots · Cold-start inference only"
    "</p>",
    unsafe_allow_html=True,
)
