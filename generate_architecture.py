"""
generate_architecture.py
Generates the full Smartphone BI Agent system architecture as PNG + PDF.
Run from bi_agent root:  python generate_architecture.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

OUTPUT = Path("docs/diagrams")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
WHITE  = '#ffffff'
DARK   = '#0f172a'
BORDER = '#e2e8f0'
GRAY   = '#64748b'

# Layer header colors  (dark)
LH = dict(user='#1e3a8a', fe='#3b0764', api='#0c4a6e',
          orch='#14532d', ml='#431407', store='#78350f',
          ext='#312e81', data='#1f2937', train='#1e3a8a')
# Layer tint colors  (light)
LT = dict(user='#eff6ff', fe='#faf5ff', api='#f0f9ff',
          orch='#f0fdf4', ml='#fff7ed', store='#fffbeb',
          ext='#eef2ff', data='#f8fafc', train='#eff6ff')
# Accent colors
AC = dict(blue='#3b82f6', violet='#8b5cf6', sky='#0ea5e9',
          green='#22c55e', amber='#f59e0b', pink='#ec4899',
          orange='#f97316', indigo='#6366f1', red='#ef4444')


# ── Drawing primitives ────────────────────────────────────────────────────────

def rbox(ax, cx, cy, w, h, rows, fc, ec, r=0.1, z=2, lw=1.4):
    """Rounded rect with list of {'text','fs','fw','col'} rows."""
    ax.add_patch(FancyBboxPatch(
        (cx-w/2, cy-h/2), w, h,
        boxstyle=f'round,pad=0,rounding_size={r}',
        fc=fc, ec=ec, lw=lw, zorder=z))
    if isinstance(rows, str):
        rows = [{'text': rows}]
    elif isinstance(rows, list) and rows and isinstance(rows[0], str):
        rows = [{'text': r_} for r_ in rows]
    n = len(rows)
    for i, row in enumerate(rows):
        txt = row.get('text', '')
        fs  = row.get('fs', 8.5)
        fw  = row.get('fw', 'normal')
        col = row.get('col', DARK)
        yp  = cy if n == 1 else cy + h/2 - (i+1) * h/(n+1)
        ax.text(cx, yp, txt, ha='center', va='center', fontsize=fs,
                fontweight=fw, color=col, zorder=z+1,
                multialignment='center', linespacing=1.3)


def layer_band(ax, x1, y1, x2, y2, title, badge, hc, tc, z=0):
    """Colored layer background with header bar."""
    BAR = 0.44
    ax.add_patch(FancyBboxPatch(
        (x1, y1), x2-x1, y2-y1,
        boxstyle='round,pad=0,rounding_size=0.18',
        fc=tc, ec=hc, lw=1.8, zorder=z))
    ax.add_patch(FancyBboxPatch(
        (x1, y2-BAR), x2-x1, BAR,
        boxstyle='round,pad=0,rounding_size=0.12',
        fc=hc, ec=hc, lw=0, zorder=z+1))
    ax.text(x1+0.28, y2-BAR/2, title,
            ha='left', va='center', fontsize=9.5, fontweight='bold',
            color=WHITE, zorder=z+2)
    ax.text(x2-0.28, y2-BAR/2, badge,
            ha='right', va='center', fontsize=7.8,
            color='#94a3b8', zorder=z+2)


def varr(ax, x, y1, y2, col=GRAY, label='', lw=1.4, bidir=False):
    style = '<->' if bidir else '->'
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle=style, color=col,
                                lw=lw, mutation_scale=13))
    if label:
        ax.text(x+0.14, (y1+y2)/2, label, ha='left', va='center',
                fontsize=7, color=col,
                bbox=dict(fc=WHITE, ec='none', pad=0.3))


def harr(ax, x1, x2, y, col=GRAY, label='', lw=1.2, bidir=False):
    style = '<->' if bidir else '->'
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle=style, color=col,
                                lw=lw, mutation_scale=12))
    if label:
        ax.text((x1+x2)/2, y+0.09, label, ha='center', va='bottom',
                fontsize=7, color=col,
                bbox=dict(fc=WHITE, ec='none', pad=0.3))


def mini_arr(ax, x1, y1, x2, y2, col):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=col,
                                lw=1.0, mutation_scale=9))


# ══════════════════════════════════════════════════════════════════════════════

def draw():
    W, H = 26, 16
    fig, ax = plt.subplots(figsize=(W, H))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis('off')

    M = 0.28   # margin

    # ══════════════════════════════════════════════════════════════════════════
    # TITLE BAR
    # ══════════════════════════════════════════════════════════════════════════
    ax.add_patch(FancyBboxPatch((0, 15.1), W, 0.9,
                                 boxstyle='square,pad=0', fc=DARK, ec=DARK, zorder=10))
    ax.text(W/2, 15.55, 'Smartphone BI Agent  —  Full System Architecture',
            ha='center', va='center', fontsize=18, fontweight='bold',
            color=WHITE, zorder=11)
    ax.text(W/2, 15.17,
            'Next.js Frontend  ·  FastAPI Backend  ·  ML Inference Pipeline  ·'
            '  Model Store  ·  Training Pipeline  ·  External APIs',
            ha='center', va='center', fontsize=8.5, color='#94a3b8', zorder=11)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 1  — USER / BROWSER
    # ══════════════════════════════════════════════════════════════════════════
    layer_band(ax, M, 13.85, W-M, 14.85,
               'USER LAYER', 'Web Browser — HTTP / JSON',
               LH['user'], LT['user'], z=1)

    rbox(ax, 7.5, 14.35, 5.8, 0.52,
         [{'text':'Business Analyst', 'fs':9.5, 'fw':'bold', 'col':LH['user']},
          {'text':'Enters phone specs and explores BI insights via browser', 'fs':7.5, 'col':GRAY}],
         fc=WHITE, ec=LH['user'])
    rbox(ax, 18.5, 14.35, 5.8, 0.52,
         [{'text':'Web Browser  (Chrome / Edge / Firefox)', 'fs':9, 'fw':'bold', 'col':LH['user']},
          {'text':'localhost:3000  —  REST over HTTP  —  JSON payloads', 'fs':7.5, 'col':GRAY}],
         fc=WHITE, ec=LH['user'])

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 2  — NEXT.JS FRONTEND
    # ══════════════════════════════════════════════════════════════════════════
    layer_band(ax, M, 10.95, W-M, 13.6,
               'PRESENTATION LAYER',
               'Next.js 16  ·  React 19  ·  TypeScript  ·  Tailwind CSS  ·  shadcn/ui  ·  Recharts  ·  port 3000',
               LH['fe'], LT['fe'], z=1)

    # Row A — core pages / containers
    fe_a = [
        (2.5,  'bi-agent-app.tsx\n(App Shell)', '#f5f3ff', '#7c3aed'),
        (7.0,  'specs-form.tsx\n(22-field PhoneSpec Form)', '#f5f3ff', '#7c3aed'),
        (13.0, 'dashboard.tsx\n(Main Result Panel)', '#f5f3ff', '#7c3aed'),
        (18.5, 'theme-provider.tsx\n+ logo.tsx', '#f5f3ff', '#7c3aed'),
        (23.5, 'app/page.tsx\n+ layout.tsx', '#f5f3ff', '#7c3aed'),
    ]
    for cx, text, fc, ec in fe_a:
        rbox(ax, cx, 12.98, 4.2, 0.72, text, fc=fc, ec=ec, r=0.1)

    # Row B — chart & display components
    fe_b = [
        (2.1,  'demand-chart.tsx\n(6-Month Bar Chart)', '#faf5ff', '#a78bfa'),
        (6.3,  'pmf-gauge.tsx\n(PMF Gauge + GO/NO-GO)', '#faf5ff', '#a78bfa'),
        (10.5, 'competitor-radar.tsx\n(Radar / Spider Chart)', '#faf5ff', '#a78bfa'),
        (14.7, 'price-optimizer.tsx\n(Profit Curve Chart)', '#faf5ff', '#a78bfa'),
        (18.9, 'donut-chart.tsx\n(Demographics Donut)', '#faf5ff', '#a78bfa'),
        (23.1, 'loyalty-diverging.tsx\n(RFM Diverging Bar)', '#faf5ff', '#a78bfa'),
    ]
    for cx, text, fc, ec in fe_b:
        rbox(ax, cx, 11.83, 3.9, 0.72, text, fc=fc, ec=ec, r=0.1)

    # Row C — panel components
    fe_c = [
        (4.0,  'news-panel.tsx\n(Article Cards + Tags)', '#ede9fe', '#6d28d9'),
        (10.5, 'ui/card · badge · tabs\n· button · slider · select', '#ede9fe', '#6d28d9'),
        (19.5, 'ui/separator · label\n(shadcn/ui base components)', '#ede9fe', '#6d28d9'),
    ]
    for cx, text, fc, ec in fe_c:
        rbox(ax, cx, 11.3, 6.5, 0.52, text, fc=fc, ec=ec, r=0.1)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 3  — FASTAPI BACKEND
    # ══════════════════════════════════════════════════════════════════════════
    layer_band(ax, M, 8.5, W-M, 10.7,
               'API GATEWAY  —  backend/main.py',
               'FastAPI (Python)  ·  Pydantic v2  ·  CORSMiddleware  ·  Uvicorn  ·  port 8000',
               LH['api'], LT['api'], z=1)

    apis = [
        (2.5,  'GET\n/api/health',           '#e0f2fe', '#0284c7', 'Returns server status'),
        (6.7,  'POST\n/api/analyze',          '#bae6fd', '#0369a1', 'Full BI analysis\nPhoneSpec -> JSON'),
        (11.0, 'POST\n/api/competitors',      '#e0f2fe', '#0284c7', 'k-NN radar data\n+ spec insights'),
        (15.3, 'POST\n/api/price-optimize',   '#e0f2fe', '#0284c7', '14-point sweep\nprofit curve'),
        (19.6, 'POST\n/api/news',             '#e0f2fe', '#0284c7', 'Parallel web search\nThreadPoolExecutor'),
        (23.5, 'Pydantic\nPhoneSpec',         '#dcfce7', '#16a34a', '22 validated\nfields'),
    ]
    for cx, method, fc, ec, desc in apis:
        rbox(ax, cx, 9.72, 3.9, 1.1,
             [{'text':method, 'fs':9.5, 'fw':'bold', 'col':ec},
              {'text':desc, 'fs':7.5, 'col':DARK}],
             fc=fc, ec=ec, r=0.12)

    # CORS badge
    rbox(ax, 13.0, 8.78, 8.0, 0.36,
         'CORSMiddleware  —  allow_origins: localhost:3000  ·  allow_methods: *  ·  allow_headers: *',
         fc='#f0fdf4', ec='#16a34a', r=0.07)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 4  — ORCHESTRATION
    # ══════════════════════════════════════════════════════════════════════════
    layer_band(ax, M, 7.3, W-M, 8.25,
               'ORCHESTRATION LAYER  —  src/agent.py',
               'Unified BI Pipeline  ·  Sequential + CLV',
               LH['orch'], LT['orch'], z=1)

    # Stages
    stage_data = [
        (1.9,  'analyze()\nentry', '#dcfce7', '#16a34a'),
        (4.9,  'Stage 1/2\nMarket Research', '#dbeafe', '#3b82f6'),
        (8.1,  'Stage 3\nSegmentation', '#d1fae5', '#059669'),
        (11.3, 'Stage 4\nDemand Forecast', '#dbeafe', '#3b82f6'),
        (14.5, 'Stage 5\nPMF', '#fce7f3', '#ec4899'),
        (17.7, 'Stage 6\nestimate_clv()', '#fef3c7', '#d97706'),
        (21.1, 'config.py\nadd_engineered_features()', '#f1f5f9', GRAY),
        (24.4, 'Result\nJSON', '#dcfce7', '#16a34a'),
    ]
    for cx, text, fc, ec in stage_data:
        rbox(ax, cx, 7.77, 2.9, 0.56, text, fc=fc, ec=ec, r=0.08)

    # Arrows between stages
    arrow_xs = [3.35, 6.55, 9.65, 12.85, 16.05, 19.2, 22.6]
    for ax_x in arrow_xs:
        mini_arr(ax, ax_x, 7.77, ax_x+0.22, 7.77, '#16a34a')

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 5  — ML INFERENCE
    # ══════════════════════════════════════════════════════════════════════════
    layer_band(ax, M, 4.3, W-M, 7.05,
               'ML INFERENCE LAYER',
               'scikit-learn  ·  XGBoost  ·  joblib  ·  numpy  ·  pandas  ·  CPU-only  ·  No GPU required',
               LH['ml'], LT['ml'], z=1)

    pillars = [
        (3.3, [
            {'text':'Pillar 1 — Market Research',        'fs':9.5,'fw':'bold','col':'#92400e'},
            {'text':'src/pillar1_market_research.py',     'fs':7.5,'col':GRAY},
            {'text':'── (A) SEGMENT CLASSIFIER ──',       'fs':7.5,'fw':'bold','col':'#3b82f6'},
            {'text':'XGBoostClassifier',                  'fs':8,  'col':DARK},
            {'text':'n_estimators=300  max_depth=6',      'fs':7.5,'col':DARK},
            {'text':'learning_rate=0.1  subsample=0.9',   'fs':7.5,'col':DARK},
            {'text':'Output: Budget/Mid/Premium/Flagship','fs':7.5,'col':DARK},
            {'text':'Accuracy ~99%  (group-split)',       'fs':7.5,'col':GRAY},
            {'text':'── (B) SIMILARITY ENGINE ──',        'fs':7.5,'fw':'bold','col':'#3b82f6'},
            {'text':'NearestNeighbors  k=5',              'fs':8,  'col':DARK},
            {'text':'StandardScaler-normalised specs',    'fs':7.5,'col':DARK},
            {'text':'Output: 5 similar phones + dist',   'fs':7.5,'col':DARK},
        ], '#dbeafe', '#3b82f6'),
        (9.1, [
            {'text':'Pillar 2 — Demand Forecast',         'fs':9.5,'fw':'bold','col':'#065f46'},
            {'text':'src/pillar2_demand.py',              'fs':7.5,'col':GRAY},
            {'text':'── DEMAND REGRESSOR ──',             'fs':7.5,'fw':'bold','col':'#059669'},
            {'text':'XGBoostRegressor',                   'fs':8,  'col':DARK},
            {'text':'Target: log1p(units_sold_monthly)',  'fs':7.5,'col':DARK},
            {'text':'Features: NUM_SPECS + TIME_CONTEXT', 'fs':7.5,'col':DARK},
            {'text':'         + ENGINEERED + REGION',     'fs':7.5,'col':DARK},
            {'text':'brand_demand_prior (train fold)',    'fs':7.5,'col':DARK},
            {'text':'Horizon: 6 months',                  'fs':7.5,'col':DARK},
            {'text':'Output: [month_1 .. month_6 units]','fs':7.5,'col':DARK},
            {'text':'R^2 approx 0.85  (group-split)',     'fs':7.5,'col':GRAY},
        ], '#d1fae5', '#059669'),
        (15.0, [
            {'text':'Pillar 3 — Segmentation',            'fs':9.5,'fw':'bold','col':'#831843'},
            {'text':'src/pillar3_segmentation.py',        'fs':7.5,'col':GRAY},
            {'text':'── k-NN RETRIEVAL ──',               'fs':7.5,'fw':'bold','col':'#ec4899'},
            {'text':'NearestNeighbors  k=10',             'fs':8,  'col':DARK},
            {'text':'Uses Pillar-1 phone_table',          'fs':7.5,'col':DARK},
            {'text':'No separate model file',             'fs':7.5,'col':GRAY},
            {'text':'Outputs (renormalised to 100%):',    'fs':7.5,'col':DARK},
            {'text':'Age: 18-24/25-34/35-44/45-54/55+',  'fs':7.5,'col':DARK},
            {'text':'Income: Low/LowerMid/Mid/High',      'fs':7.5,'col':DARK},
            {'text':'Gender: Male/Female/NonBinary',      'fs':7.5,'col':DARK},
            {'text':'RFM: Champion/Loyal/AtRisk/Hiber.',  'fs':7.5,'col':DARK},
        ], '#fce7f3', '#ec4899'),
        (20.9, [
            {'text':'Pillar 4 — PMF Classifier',          'fs':9.5,'fw':'bold','col':'#7c2d12'},
            {'text':'src/pillar4_pmf.py',                 'fs':7.5,'col':GRAY},
            {'text':'── PMF CLASSIFIER ──',               'fs':7.5,'fw':'bold','col':'#f97316'},
            {'text':'RandomForestClassifier',             'fs':8,  'col':DARK},
            {'text':'Label: bottom 25% pmf = failure',   'fs':7.5,'col':DARK},
            {'text':'PMF formula (6 signals, weighted):','fs':7.5,'col':DARK},
            {'text':'NPS(25%) + Repeat(20%) + Rec(20%)', 'fs':7.5,'col':DARK},
            {'text':'+ Return(15%) + Warranty(10%)',      'fs':7.5,'col':DARK},
            {'text':'+ Sales_slope(10%)',                 'fs':7.5,'col':DARK},
            {'text':'Output: pmf_score, failure_risk%',  'fs':7.5,'col':DARK},
            {'text':'Accuracy ~0.79  AUC ~0.88',         'fs':7.5,'col':GRAY},
        ], '#fff7ed', '#f97316'),
    ]
    for cx, rows, fc, ec in pillars:
        rbox(ax, cx, 5.67, 5.6, 2.54, rows, fc=fc, ec=ec, r=0.12)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 6  — MODEL STORE  +  EXTERNAL
    # ══════════════════════════════════════════════════════════════════════════
    # Model store (left)
    layer_band(ax, M, 2.65, 18.2, 4.05,
               'MODEL STORE  —  /models/',
               'joblib serialized  ·  18.9 MB total  ·  0 GPU  ·  CPU-only',
               LH['store'], LT['store'], z=1)

    store = [
        (2.5,  'pillar1_segment.joblib',    '878 KB',  'XGBoost Classifier\n(Segment Predictor)'),
        (6.6,  'pillar1_similarity.joblib', '2.7 MB',  'k-NN + phone_table\n(Similarity Engine)'),
        (10.7, 'pillar2_demand.joblib',     '9.7 MB',  'XGBoost Regressor\n(Demand Forecaster)'),
        (14.8, 'pillar4_pmf.joblib',        '5.2 MB',  'Random Forest\n(PMF Classifier)'),
    ]
    for cx, name, size, algo in store:
        rbox(ax, cx, 3.35, 3.8, 1.0,
             [{'text':name,  'fs':8,   'fw':'bold','col':LH['store']},
              {'text':size,  'fs':10,  'fw':'bold','col':'#92400e'},
              {'text':algo,  'fs':7.5, 'col':GRAY}],
             fc=WHITE, ec='#fbbf24', r=0.1)

    # External API (right)
    layer_band(ax, 18.5, 2.65, W-M, 4.05,
               'EXTERNAL', 'Web Search API',
               LH['ext'], LT['ext'], z=1)
    rbox(ax, 22.1, 3.35, 6.8, 1.0,
         [{'text':'DuckDuckGo Search API',    'fs':9,  'fw':'bold','col':LH['ext']},
          {'text':'ddgs  library',            'fs':8,  'col':DARK},
          {'text':'max 8 parallel threads',   'fs':7.5,'col':GRAY},
          {'text':'text() + news() endpoints','fs':7.5,'col':GRAY}],
         fc='#eef2ff', ec='#6366f1', r=0.1)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 7  — DATA  +  TRAINING PIPELINE
    # ══════════════════════════════════════════════════════════════════════════
    # Data store (left half)
    layer_band(ax, M, 0.45, 12.5, 2.4,
               'DATA LAYER',
               'CSV  ·  44,000 rows  ·  45 MB',
               LH['data'], LT['data'], z=1)
    rbox(ax, 6.4, 1.42, 11.5, 1.05,
         [{'text':'global_smartphone_bi_v4_final.csv',        'fs':9.5,'fw':'bold','col':DARK},
          {'text':'44,000 rows  ·  45 MB  ·  global smartphone dataset', 'fs':8,'col':GRAY},
          {'text':'22 launch specs (NUM_SPECS + CAT_SPECS)  +  POST_LAUNCH outcomes:', 'fs':7.5,'col':DARK},
          {'text':'NPS · repeat_rate · return_rate · units_sold · social_mentions · demographics', 'fs':7.5,'col':DARK}],
         fc=WHITE, ec='#374151', r=0.1)

    # Training pipeline (right half)
    layer_band(ax, 12.8, 0.45, W-M, 2.4,
               'TRAINING PIPELINE',
               'Offline — run once with:  python src/train_all.py',
               LH['train'], LT['train'], z=1)
    rbox(ax, 19.5, 1.42, 12.2, 1.05,
         [{'text':'src/train_all.py',                             'fs':9.5,'fw':'bold','col':LH['train']},
          {'text':'GroupShuffleSplit(test_size=0.20, random_state=42)', 'fs':8,'col':GRAY},
          {'text':'OrdinalEncoder (cat specs)  ·  StandardScaler (similarity feats)', 'fs':7.5,'col':DARK},
          {'text':'Trains: P1A Classifier  ·  P1B k-NN  ·  P2 Regressor  ·  P4 PMF  ->  saves .joblib', 'fs':7.5,'col':DARK}],
         fc=WHITE, ec='#3b82f6', r=0.1)

    # ══════════════════════════════════════════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════════════════════════════════════════
    ax.plot([M, W-M], [0.32, 0.32], color=BORDER, lw=0.8)
    ax.text(M+0.1, 0.18,
            'Smartphone BI Agent  —  System Architecture  —  2026',
            fontsize=7.5, color=GRAY)
    ax.text(W-M-0.1, 0.18,
            'CPU-only  ·  No GPU  ·  ~250 MB RAM at runtime  ·  18.9 MB model store  ·  Python + Node.js',
            fontsize=7.5, color=GRAY, ha='right')

    # ══════════════════════════════════════════════════════════════════════════
    # INTER-LAYER CONNECTION ARROWS
    # ══════════════════════════════════════════════════════════════════════════

    # User <-> Frontend  (bidirectional HTTP)
    varr(ax, 13.0, 13.6, 13.85, '#7c3aed', 'HTTP/JSON', lw=1.6, bidir=True)

    # Frontend -> API  (5 calls, one per endpoint column)
    for ep_x in [2.5, 6.7, 11.0, 15.3, 19.6]:
        varr(ax, ep_x, 10.7, 10.95, '#0369a1', lw=1.1)

    # API (analyze) -> Orchestration
    varr(ax, 6.7, 8.5, 7.3, '#14532d', 'calls analyze()', lw=1.6)

    # API (competitors, price-opt) -> pillars directly
    varr(ax, 11.0, 8.5, 7.3, '#0369a1', lw=1.0)
    varr(ax, 15.3, 8.5, 7.3, '#0369a1', lw=1.0)

    # API (news) <-> DuckDuckGo  (horizontal)
    harr(ax, 19.6+1.9, 22.1-3.4, 3.9, '#6366f1', 'search queries', lw=1.2, bidir=True)

    # Orchestration -> ML Pillars  (4 vertical arrows)
    for p_x in [3.3, 9.1, 15.0, 20.9]:
        varr(ax, p_x, 7.05, 7.3, '#7c2d12', lw=1.1)

    # ML Pillars -> Model Store  (load arrows)
    ml_to_store = [(3.3, 2.5), (3.3, 6.6), (9.1, 10.7), (20.9, 14.8)]
    for px, sx in ml_to_store:
        varr(ax, sx, 4.05, 4.3, '#f59e0b', lw=0.9)

    # Training -> Model Store  (produces artifacts)
    varr(ax, 17.0, 2.4, 2.65, '#3b82f6', 'trains & saves', lw=1.3)

    # Data -> Training
    harr(ax, 12.5, 12.8, 1.42, '#1e3a8a', 'reads CSV', lw=1.2)

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Generating system architecture diagram...")

    fig = draw()
    png_path = OUTPUT / 'system_architecture.png'
    fig.savefig(png_path, dpi=180, bbox_inches='tight', facecolor=WHITE)
    plt.close(fig)
    print(f"  PNG  -> {png_path.resolve()}")

    fig = draw()
    pdf_path = OUTPUT / 'system_architecture.pdf'
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight', facecolor=WHITE)
        d = pdf.infodict()
        d['Title']   = 'BI Agent - System Architecture'
        d['Subject'] = 'Full Stack Architecture'
    plt.close(fig)
    print(f"  PDF  -> {pdf_path.resolve()}")
    print("Done.")
