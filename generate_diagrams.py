"""
generate_diagrams.py
Creates all 5 UML diagrams as PNG + combined PDF.
Run from bi_agent root:  python generate_diagrams.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path

OUTPUT = Path("docs/diagrams")
OUTPUT.mkdir(parents=True, exist_ok=True)

# â”€â”€ Palette â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
GREEN  = '#22c55e'; BLUE   = '#3b82f6'; INDIGO = '#6366f1'
AMBER  = '#f59e0b'; PINK   = '#ec4899'; SKY    = '#0ea5e9'
VIOLET = '#8b5cf6'; ORANGE = '#f97316'; SLATE  = '#64748b'
RED    = '#ef4444'; WHITE  = '#ffffff'; DARK   = '#0f172a'
BORDER = '#cbd5e1'; LIGHT  = '#f8fafc'


# â”€â”€ Universal helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def new_fig(title, subtitle, figsize, xlim=10, ylim=14):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, xlim); ax.set_ylim(0, ylim)
    ax.axis('off')
    # Header
    hdr = FancyBboxPatch((0, ylim - 0.82), xlim, 0.82,
                          boxstyle='square,pad=0', fc=DARK, ec=DARK, zorder=5)
    ax.add_patch(hdr)
    ax.text(xlim/2, ylim - 0.35, title, ha='center', va='center',
            fontsize=15, fontweight='bold', color=WHITE, zorder=6)
    ax.text(xlim/2, ylim - 0.65, subtitle, ha='center', va='center',
            fontsize=8, color='#94a3b8', zorder=6)
    # Footer line
    ax.plot([0.2, xlim-0.2], [0.35, 0.35], color=BORDER, lw=0.8)
    ax.text(0.25, 0.18, 'Smartphone BI Agent  â€”  System Documentation 2026',
            fontsize=7, color='#94a3b8')
    return fig, ax


def rbox(ax, cx, cy, w, h, text, fc, ec=None, tc=WHITE, fs=9,
         bold=False, r=0.1, z=2):
    ec = ec or fc
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                        boxstyle=f'round,pad=0,rounding_size={r}',
                        fc=fc, ec=ec, lw=1.5, zorder=z)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            color=tc, fontweight='bold' if bold else 'normal',
            zorder=z+1, multialignment='center', linespacing=1.35)


def diamond(ax, cx, cy, w, h, text, fc=AMBER, tc=WHITE, fs=9):
    xs = [cx, cx+w/2, cx, cx-w/2, cx]
    ys = [cy+h/2, cy, cy-h/2, cy, cy+h/2]
    ax.fill(xs, ys, fc=fc, ec=fc, lw=1.5, zorder=2)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            color=tc, fontweight='bold', zorder=3, multialignment='center')


def dot(ax, cx, cy, r, fc):
    ax.add_patch(plt.Circle((cx, cy), r, fc=fc, ec=fc, zorder=3))


def arr(ax, x1, y1, x2, y2, label='', color=SLATE, lw=1.4, lside='r', fs=7.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=14))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        dx, dy = x2-x1, y2-y1
        L = max(1e-9, (dx**2+dy**2)**0.5)
        nx, ny = -dy/L * 0.18, dx/L * 0.18
        off = (nx, ny) if lside == 'r' else (-nx, -ny)
        ax.text(mx+off[0], my+off[1], label, ha='center', va='center',
                fontsize=fs, color=DARK,
                bbox=dict(fc=WHITE, ec='none', pad=0.8))


def darr(ax, x1, y1, x2, y2, label='', color='#94a3b8', lw=1.1, fs=7):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle='dashed', mutation_scale=11))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.12, label, ha='center', va='bottom', fontsize=fs,
                color=SLATE, style='italic',
                bbox=dict(fc=WHITE, ec='none', pad=0.5))


def grp(ax, x1, y1, x2, y2, label='', fc=LIGHT, ec=BORDER, z=0):
    p = FancyBboxPatch((x1, y1), x2-x1, y2-y1,
                        boxstyle='round,pad=0,rounding_size=0.25',
                        fc=fc, ec=ec, lw=1.5, zorder=z, linestyle='dashed')
    ax.add_patch(p)
    if label:
        ax.text(x1+0.18, y2-0.16, label, ha='left', va='top', fontsize=8,
                color=SLATE, fontweight='bold', zorder=z+1)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 1 â€” ACTIVITY DIAGRAM
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def d1_activity():
    fig, ax = new_fig('1. Activity Diagram',
                      'End-to-end user flow: spec input â†’ 6-stage BI pipeline â†’ dashboard',
                      (9, 16), xlim=10, ylim=15)

    X = 5.0
    # Node Y positions
    y = dict(start=14.2, form=13.35, sub=12.5, val=11.6, err=11.6,
             api=10.65, s1=9.75, s2=8.9, s3=8.05, s4=7.2, s5=6.35,
             dash=5.45, cho=4.5, r1=3.35, r2=3.35, r3=3.35, end=2.15)
    BW, BH = 4.2, 0.6

    # nodes
    dot(ax, X, y['start'], 0.27, GREEN)
    rbox(ax, X, y['form'],  BW, BH, 'Fill Phone Spec Form\n(brand Â· price Â· specs Â· region)',
         fc='#ede9fe', ec=INDIGO, tc=DARK, fs=9)
    rbox(ax, X, y['sub'],   BW*0.65, BH*0.85, 'Click Analyze',
         fc=INDIGO, tc=WHITE, bold=True)
    diamond(ax, X, y['val'], 2.6, 0.72, 'Valid\nInput?', fc=AMBER)
    rbox(ax, 8.35, y['err'], 2.7, BH, 'Show Validation\nError',
         fc='#fee2e2', ec=RED, tc=DARK, fs=8.5)

    rbox(ax, X, y['api'],   BW, BH, 'POST  /api/analyze',
         fc=INDIGO, tc=WHITE, bold=True)

    stages = [
        (y['s1'], '#dbeafe', BLUE,   'Stage 1/2 Â· Market Research\n(XGBoost Classifier + k-NN Similar Phones)'),
        (y['s2'], '#d1fae5', GREEN,  'Stage 3 Â· Customer Segmentation\n(k-NN Demographic Retrieval â€” Age / Income / RFM)'),
        (y['s3'], '#dbeafe', BLUE,   'Stage 4 Â· Demand Forecast\n(XGBoost Regressor â€” 6-Month Unit Curve)'),
        (y['s4'], '#fce7f3', PINK,   'Stage 5 Â· Product-Market Fit\n(Random Forest â€” PMF Score + GO / NO-GO)'),
        (y['s5'], '#fef3c7', AMBER,  'Stage 6 Â· Compute CLV\n(price Ã— margin Ã— repeat Ã— loyalty factor)'),
    ]
    for sy, fc, ec, text in stages:
        rbox(ax, X, sy, BW, BH, text, fc=fc, ec=ec, tc=DARK, fs=8.8)

    rbox(ax, X, y['dash'], BW, BH, 'Render Dashboard', fc=GREEN, tc=WHITE, bold=True)
    diamond(ax, X, y['cho'], 3.0, 0.78, 'Explore\nFurther?', fc=SLATE)

    subs = [(2.0,  'Competitor\nCompare\n/api/competitors',    '#ede9fe', VIOLET),
            (5.0,  'Price\nOptimize\n/api/price-optimize',    '#d1fae5', GREEN),
            (8.0,  'Market\nNews\n/api/news',                 '#fef3c7', AMBER)]
    for sx, text, fc, ec in subs:
        rbox(ax, sx, y['r1'], 2.5, 0.85, text, fc=fc, ec=ec, tc=DARK, fs=8.5)

    dot(ax, X, y['end'], 0.32, DARK)
    ax.add_patch(plt.Circle((X, y['end']), 0.42, fc='none', ec=DARK, lw=2, zorder=3))

    # arrows
    arr(ax, X, y['start']+0.27, X, y['form']+BH/2, color=GREEN)
    arr(ax, X, y['form']-BH/2,  X, y['sub']+0.26)
    arr(ax, X, y['sub']-0.26,   X, y['val']+0.36)
    arr(ax, X+1.3, y['val'],     8.35-1.35, y['err'],  label='No',  color=RED)
    # Error loops back to form (L-shaped)
    ax.plot([8.35, 9.2], [y['err'], y['err']], color=RED, lw=1.4)
    ax.plot([9.2, 9.2],  [y['err'], y['form']], color=RED, lw=1.4)
    ax.annotate('', xy=(X+BW/2, y['form']), xytext=(9.2, y['form']),
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.4, mutation_scale=13))

    arr(ax, X, y['val']-0.36,   X, y['api']+BH/2, label='Yes', color=GREEN)
    prev = y['api']
    for sy, *_ in stages:
        arr(ax, X, prev-BH/2, X, sy+BH/2)
        prev = sy
    arr(ax, X, prev-BH/2, X, y['dash']+BH/2)
    arr(ax, X, y['dash']-BH/2, X, y['cho']+0.39)

    arr(ax, X-1.5, y['cho'], 2.0, y['r1']+0.43, label='Compare', color=VIOLET, lside='l')
    arr(ax, X,     y['cho']-0.39, 5.0, y['r1']+0.43, label='Price',   color=GREEN)
    arr(ax, X+1.5, y['cho'], 8.0, y['r1']+0.43, label='News',    color=AMBER,  lside='r')

    # Done path â€” down from choice to end
    ax.plot([X+1.5, 9.5], [y['cho'], y['cho']], color=DARK, lw=1.4)
    ax.plot([9.5, 9.5],   [y['cho'], y['end']],  color=DARK, lw=1.4)
    ax.annotate('', xy=(X+0.44, y['end']), xytext=(9.5, y['end']),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=1.4, mutation_scale=13))
    ax.text(9.65, (y['cho']+y['end'])/2, 'Done', fontsize=8, color=DARK, va='center')

    fig.savefig(OUTPUT/'01_activity_diagram.png', dpi=180,
                bbox_inches='tight', facecolor=WHITE)
    plt.close(fig)
    print("  âœ“  1/5  Activity Diagram")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 2 â€” USE CASE DIAGRAM
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def actor(ax, cx, cy, label, color=DARK):
    ax.add_patch(plt.Circle((cx, cy+0.55), 0.22, fc='#f1f5f9', ec=color, lw=1.8, zorder=3))
    ax.plot([cx, cx],         [cy+0.33, cy-0.05], color=color, lw=1.8, zorder=3)
    ax.plot([cx-0.3, cx+0.3], [cy+0.18, cy+0.18], color=color, lw=1.8, zorder=3)
    ax.plot([cx, cx-0.28],    [cy-0.05, cy-0.52], color=color, lw=1.8, zorder=3)
    ax.plot([cx, cx+0.28],    [cy-0.05, cy-0.52], color=color, lw=1.8, zorder=3)
    ax.text(cx, cy-0.72, label, ha='center', va='top', fontsize=8.5,
            color=color, fontweight='bold', multialignment='center')


def ellipse_uc(ax, cx, cy, w, h, text, fc, ec, fs=8.5):
    ax.add_patch(mpatches.Ellipse((cx, cy), w, h, fc=fc, ec=ec, lw=1.8, zorder=2))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=DARK, zorder=3, multialignment='center')


def d2_usecase():
    fig, ax = new_fig('2. Use Case Diagram',
                      'Actors, use cases, and their relationships in the BI Agent system',
                      (15, 11), xlim=15, ylim=11)

    # System boundary
    grp(ax, 2.8, 0.6, 13.5, 10.2, '', fc='#f8fafc', ec='#94a3b8', z=0)
    ax.text(8.15, 10.35, 'Â«systemÂ»  BI Agent', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=SLATE)

    # Actors
    actor(ax, 0.8,  7.0, 'Business\nAnalyst', DARK)
    actor(ax, 14.2, 6.0, 'DuckDuckGo\nSearch API', BLUE)

    # Core use cases
    core = [(8.15, 9.1, 'Analyze New Phone'),
            (4.8,  7.4, 'Compare\nCompetitors'),
            (8.15, 7.4, 'Optimize\nLaunch Price'),
            (11.5, 7.4, 'Read\nMarket News')]
    for cx, cy, t in core:
        ellipse_uc(ax, cx, cy, 3.1, 0.85, t, '#ede9fe', VIOLET)

    # Sub use cases
    subs = [(3.5, 5.4, 'Predict\nMarket Segment'),
            (6.0, 4.5, 'Forecast\nDemand'),
            (8.15,5.4, 'Segment\nCustomers'),
            (10.3,4.5, 'Score PMF\n& GO/NO-GO'),
            (12.5,5.4, 'Calculate\nCLV')]
    for cx, cy, t in subs:
        ellipse_uc(ax, cx, cy, 2.8, 0.8, t, '#dbeafe', BLUE, fs=8)

    # Admin
    ellipse_uc(ax, 8.15, 2.7, 3.0, 0.82, 'Train ML Models', '#fef3c7', AMBER)

    # Actor â†’ Core
    for cx, cy, _ in core:
        ax.plot([1.1, cx-1.55], [7.0, cy], color='#475569', lw=1.3, zorder=1)
    # DDG â† News
    ax.annotate('', xy=(13.4, 6.4), xytext=(13.05, 7.4),
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.3, mutation_scale=12))

    # Include
    inc = [((8.15,8.68),(3.5, 5.8),'Â«includeÂ»'),
           ((8.15,8.68),(6.0, 4.9),'Â«includeÂ»'),
           ((8.15,8.68),(8.15,5.8),'Â«includeÂ»'),
           ((8.15,8.68),(10.3,4.9),'Â«includeÂ»'),
           ((8.15,8.68),(12.5,5.8),'Â«includeÂ»'),
           ((8.15,6.98),(6.0, 4.9),'Â«includeÂ»'),
           ((8.15,6.98),(10.3,4.9),'Â«includeÂ»')]
    for (x1,y1),(x2,y2),lbl in inc:
        darr(ax, x1, y1, x2, y2, lbl)

    # Enables (admin â†’ core)
    for cx, cy, _ in core:
        darr(ax, 8.15, 3.11, cx, cy-0.42, 'Â«enablesÂ»', color='#f59e0b', lw=0.9)

    # Legend
    items = [(VIOLET,'Core Use Case'), (BLUE,'Sub Use Case (include)'),
             (AMBER,'Admin (enables)')]
    for i,(col,lbl) in enumerate(items):
        bx = 3.5 + i*3.5
        ax.add_patch(mpatches.Ellipse((bx, 0.65), 0.55, 0.35, fc=col+'22', ec=col, lw=1.5))
        ax.text(bx+0.45, 0.65, lbl, va='center', fontsize=8, color=DARK)

    fig.savefig(OUTPUT/'02_use_case_diagram.png', dpi=180,
                bbox_inches='tight', facecolor=WHITE)
    plt.close(fig)
    print("  âœ“  2/5  Use Case Diagram")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 3 â€” CLASS DIAGRAM
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def classbox(ax, cx, cy, w, name, stereo, attrs, methods, hdr_fc, body_fc='#f8fafc'):
    LH = 0.245
    sh = 0.26 if stereo else 0
    th = 0.40
    ah = max(0.28, len(attrs)  * LH + 0.12)
    mh = max(0.28, len(methods)* LH + 0.12)
    total = sh + th + ah + mh
    top = cy + total/2
    y = top

    def blk(yb, ht, fc_, ec_=BORDER, top_round=False, bot_round=False):
        p = FancyBboxPatch((cx-w/2, yb), w, ht,
                            boxstyle='square,pad=0',
                            fc=fc_, ec=ec_, lw=1.2, zorder=2)
        ax.add_patch(p)
        return yb

    if stereo:
        blk(y-sh, sh, hdr_fc, hdr_fc)
        ax.text(cx, y-sh/2, f'Â«{stereo}Â»', ha='center', va='center',
                fontsize=7, color='#e2e8f0', style='italic', zorder=3)
        y -= sh

    blk(y-th, th, hdr_fc, hdr_fc)
    ax.text(cx, y-th/2, name, ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=WHITE, zorder=3)
    y -= th

    blk(y-ah, ah, body_fc)
    for i, a in enumerate(attrs):
        ax.text(cx-w/2+0.12, y-0.1-i*LH, a, ha='left', va='top',
                fontsize=6.8, color=DARK, family='monospace', zorder=3)
    y -= ah
    ax.plot([cx-w/2, cx+w/2], [y, y], color=BORDER, lw=0.8)

    blk(y-mh, mh, body_fc)
    for i, m in enumerate(methods):
        ax.text(cx-w/2+0.12, y-0.1-i*LH, m, ha='left', va='top',
                fontsize=6.8, color='#1d4ed8', family='monospace', zorder=3)

    # outer border
    rect = FancyBboxPatch((cx-w/2, cy-total/2), w, total,
                           boxstyle='square,pad=0',
                           fc='none', ec=BORDER, lw=1.4, zorder=4)
    ax.add_patch(rect)
    return cy - total/2


def d3_class():
    fig, ax = new_fig('3. Class Diagram',
                      'Modules, data models, and relationships across the BI Agent codebase',
                      (17, 12), xlim=17, ylim=12)

    W = 3.3

    # Row 0: FastAPIApp â€” top center
    classbox(ax, 8.5, 11.1, W+0.8, 'FastAPIApp', 'REST API',
             [],
             ['+health() â†’ dict', '+analyze(spec, region)',
              '+competitor_analysis(spec)', '+price_optimize(spec)',
              '+market_news(spec)', 'â€“spec_to_dict(spec)', 'â€“_ddg_search(q,n)'],
             '#6366f1', '#eef2ff')

    # Row 1: PhoneSpec | Agent | Config
    classbox(ax, 2.1, 8.4, W, 'PhoneSpec', 'Pydantic',
             ['+brand_name: str', '+price_usd: float', '+region: str',
              '+five_g: int', '+ram_capacity: int',
              '+battery_capacity: int', '+os: str'],
             [],
             '#0ea5e9', '#f0f9ff')

    classbox(ax, 8.5, 8.2, W, 'Agent', 'Orchestrator',
             [],
             ['+analyze(spec, region, horizon)',
              '+estimate_clv(price, curve, repeat)'],
             '#8b5cf6', '#faf5ff')

    classbox(ax, 14.9, 8.4, W, 'Config', 'Module',
             ['DATA_PATH: Path', 'MODEL_DIR: str', 'NUM_SPECS: List[str]',
              'CAT_SPECS: List[str]', 'POST_LAUNCH: List[str]'],
             ['+add_engineered_features(df)'],
             '#64748b', '#f8fafc')

    # Row 2: 4 Pillars
    pcfg = [(2.0,  5.2, 'Pillar1\nMarketResearch', '#3b82f6',
             ['Algorithm: XGBoost + k-NN'],
             ['+train(df)', '+predict_segment(spec)','+find_similar(spec, k)']),
            (6.0,  5.2, 'Pillar2\nDemandForecast', '#059669',
             ['Algorithm: XGBoost Regressor'],
             ['+train(df)', '+predict_curve(spec, region, horizon)']),
            (10.0, 5.2, 'Pillar3\nSegmentation', '#ec4899',
             ['Algorithm: k-NN  k=10'],
             ['+evaluate(df)', '+predict_demographics(spec, region)']),
            (14.0, 5.2, 'Pillar4\nPMF', '#f97316',
             ['Algorithm: Random Forest'],
             ['+train(df)', '+build_labels(df)', '+predict_fit(spec)'])]

    for cx, cy, nm, col, at, mt in pcfg:
        classbox(ax, cx, cy, W, nm, 'ML Pillar', at, mt, col)

    # Row 3: ModelStore
    classbox(ax, 8.5, 2.5, W+1.8, 'ModelStore', 'joblib Files',
             ['pillar1_segment.joblib    â€” 878 KB  (XGBoost Clf)',
              'pillar1_similarity.joblib â€” 2.7 MB  (k-NN + table)',
              'pillar2_demand.joblib     â€” 9.7 MB  (XGBoost Reg)',
              'pillar4_pmf.joblib        â€” 5.2 MB  (RandomForest)'],
             [],
             '#f59e0b', '#fffbeb')

    # â”€â”€ Relationships â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def rel(x1,y1,x2,y2,lbl='',col=SLATE,ls='-',lw=1.2,ms=11):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                    linestyle=ls, mutation_scale=ms))
        if lbl:
            mx,my = (x1+x2)/2,(y1+y2)/2
            ax.text(mx+0.1, my+0.1, lbl, fontsize=7, color=SLATE,
                    bbox=dict(fc=WHITE,ec='none',pad=0.5))

    # FastAPI â†” PhoneSpec / Agent / Config
    rel(7.1, 10.45, 2.1, 9.1, 'validates', BLUE)
    rel(8.5, 10.25, 8.5, 9.0, 'calls', VIOLET)
    rel(9.9, 10.45, 14.9, 9.1, 'reads', SLATE)

    # Agent â†’ Pillars
    for px in [2.0, 6.0, 10.0, 14.0]:
        rel(8.5, 7.6, px, 5.85, 'uses', '#94a3b8', lw=1.0)

    # FastAPI â†’ Pillar1, Pillar4 directly
    rel(7.3, 10.3, 2.0, 5.85, col='#c4b5fd', lw=0.9, ls='dashed')
    rel(9.7, 10.3, 14.0,5.85, col='#c4b5fd', lw=0.9, ls='dashed')

    # Pillars â†’ ModelStore
    for px in [2.0, 6.0, 10.0, 14.0]:
        rel(px, 4.1, 8.5, 3.05, '', '#f59e0b', lw=0.9)

    # Pillars â†’ Config
    for px in [2.0, 6.0, 10.0]:
        rel(px, 4.2, 14.5, 7.6, '', '#cbd5e1', lw=0.8)
    rel(14.0, 4.2, 14.9, 7.6, '', '#cbd5e1', lw=0.8)

    # Legend row
    legend = [('â”€â–¶  uses / calls', VIOLET), ('â”€â–¶  loads', '#f59e0b'),
              ('--â–¶ direct API ref', '#c4b5fd'), ('â”€â–¶  reads config', '#cbd5e1')]
    for i,(lbl,col) in enumerate(legend):
        bx = 1.0 + i*4.0
        ax.plot([bx, bx+0.7], [0.6, 0.6], color=col, lw=1.8)
        ax.text(bx+0.85, 0.6, lbl, va='center', fontsize=7.5, color=DARK)

    fig.savefig(OUTPUT/'03_class_diagram.png', dpi=180,
                bbox_inches='tight', facecolor=WHITE)
    plt.close(fig)
    print("  âœ“  3/5  Class Diagram")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 4 â€” SEQUENCE DIAGRAM
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def d4_sequence():
    fig, ax = new_fig('4. Sequence Diagram',
                      'Message flow for POST /api/analyze â€” user submission to dashboard render',
                      (18, 12), xlim=18, ylim=12)

    TOP, BOT = 11.5, 0.55
    lanes = [
        (1.1,  'ðŸ‘¤\nUser',          DARK,    '#f1f5f9'),
        (3.3,  'Next.js\nFrontend', SKY,     '#f0f9ff'),
        (5.5,  'FastAPI\nBackend',  INDIGO,  '#eef2ff'),
        (7.7,  'agent.py\nOrch.',   VIOLET,  '#faf5ff'),
        (9.7,  'Pillar 1\nMkt Res', BLUE,    '#eff6ff'),
        (11.5, 'Pillar 2\nDemand',  GREEN,   '#f0fdf4'),
        (13.3, 'Pillar 3\nSeg.',    PINK,    '#fdf2f8'),
        (15.1, 'Pillar 4\nPMF',     ORANGE,  '#fff7ed'),
        (17.0, 'Model\nFiles',      AMBER,   '#fffbeb'),
    ]
    for x, name, col, bg in lanes:
        p = FancyBboxPatch((x-0.7, TOP-0.5), 1.4, 0.65,
                            boxstyle='round,pad=0,rounding_size=0.1',
                            fc=bg, ec=col, lw=2, zorder=3)
        ax.add_patch(p)
        ax.text(x, TOP-0.18, name, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color=col, zorder=4,
                multialignment='center', linespacing=1.25)
        ax.plot([x, x], [TOP-0.5, BOT], color=col, lw=1,
                linestyle='dashed', alpha=0.45, zorder=1)

    def msg(x1, x2, y, text, col='#475569', ret=False):
        sty = '<-' if ret else '->'
        ls  = 'dashed' if ret else 'solid'
        lw  = 1.1 if ret else 1.5
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle=sty, color=col, lw=lw,
                                    linestyle=ls, mutation_scale=12))
        mx = (x1+x2)/2
        dy = -0.13 if ret else 0.13
        ax.text(mx, y+dy, text, ha='center', va='bottom' if not ret else 'top',
                fontsize=7.2, color=DARK,
                bbox=dict(fc=WHITE, ec='none', pad=0.4))

    def note(y, text, col='#fef9c3'):
        p = FancyBboxPatch((2.3, y-0.15), 5.0, 0.3,
                            boxstyle='round,pad=0,rounding_size=0.07',
                            fc=col, ec='#fde68a', lw=1, zorder=2)
        ax.add_patch(p)
        ax.text(4.8, y, text, ha='center', va='center',
                fontsize=7.5, color='#78350f', fontweight='bold', zorder=3)

    def act(x, y1, y2, fc='#e0e7ff', ec='#818cf8'):
        p = FancyBboxPatch((x-0.13, y2), 0.26, y1-y2,
                            boxstyle='square,pad=0', fc=fc, ec=ec, lw=1, zorder=2)
        ax.add_patch(p)

    # Activation bars
    act(3.3,  11.0, 0.7,  '#e0f2fe', '#38bdf8')
    act(5.5,  10.7, 0.7,  '#eef2ff', '#818cf8')
    act(7.7,  10.4, 1.1,  '#faf5ff', '#a78bfa')

    # Messages
    msg(1.1, 3.3,  10.95, 'Enter phone specs + region')
    msg(3.3, 5.5,  10.7,  'POST /api/analyze {JSON}',         INDIGO)
    msg(5.5, 7.7,  10.4,  'analyze(spec_dict, region)',        VIOLET)

    note(9.95, 'â–¶  Stage 1/2 â€” Market Research')
    msg(7.7, 9.7,   9.65, 'predict_segment(spec_row)',         BLUE)
    msg(9.7, 17.0,  9.35, 'load pillar1_segment.joblib',       AMBER)
    msg(17.0,9.7,   9.05, 'XGBoost Clf  (878 KB)',             AMBER, ret=True)
    msg(9.7, 7.7,   8.75, 'segment Â· confidence',              BLUE,  ret=True)
    msg(7.7, 9.7,   8.45, 'find_similar(spec_row, k=5)',       BLUE)
    msg(9.7, 17.0,  8.15, 'load pillar1_similarity.joblib',    AMBER)
    msg(17.0,9.7,   7.85, 'k-NN + phone_table  (2.7 MB)',      AMBER, ret=True)
    msg(9.7, 7.7,   7.55, '5 similar phone names',             BLUE,  ret=True)

    note(7.2, 'â–¶  Stage 3 â€” Customer Segmentation')
    msg(7.7, 13.3,  6.9,  'predict_demographics(spec_row, k=10)', PINK)
    msg(13.3,17.0,  6.6,  'load pillar1_similarity.joblib',    AMBER)
    msg(17.0,13.3,  6.3,  'phone_table',                       AMBER, ret=True)
    msg(13.3,7.7,   6.0,  'age Â· income Â· gender Â· RFM',       PINK,  ret=True)

    note(5.65, 'â–¶  Stage 4 â€” Demand Forecast')
    msg(7.7, 11.5,  5.35, 'predict_curve(spec_row, horizon=6)', GREEN)
    msg(11.5,17.0,  5.05, 'load pillar2_demand.joblib',        AMBER)
    msg(17.0,11.5,  4.75, 'XGBoost Reg  (9.7 MB)',             AMBER, ret=True)
    msg(11.5,7.7,   4.45, 'month_1 â€¦ month_6 unit curve',      GREEN, ret=True)

    note(4.1, 'â–¶  Stage 5 â€” Product-Market Fit')
    msg(7.7, 15.1,  3.8,  'predict_fit(spec_row)',              ORANGE)
    msg(15.1,17.0,  3.5,  'load pillar4_pmf.joblib',           AMBER)
    msg(17.0,15.1,  3.2,  'Random Forest  (5.2 MB)',            AMBER, ret=True)
    msg(15.1,7.7,   2.9,  'pmf_score Â· GO/NO-GO',              ORANGE, ret=True)

    note(2.55, 'â–¶  Stage 6 â€” CLV computed internally in agent.py')
    msg(7.7, 5.5,   2.1,  'full result JSON',                   VIOLET, ret=True)
    msg(5.5, 3.3,   1.7,  '200 OK + payload',                  INDIGO, ret=True)
    msg(3.3, 1.1,   1.3,  'Render Dashboard ðŸ–¥',               SKY,    ret=True)

    fig.savefig(OUTPUT/'04_sequence_diagram.png', dpi=180,
                bbox_inches='tight', facecolor=WHITE)
    plt.close(fig)
    print("  âœ“  4/5  Sequence Diagram")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 5 â€” DATA FLOW DIAGRAM
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def cyl(ax, cx, cy, w, h, text, fc='#fef3c7', ec='#fbbf24', fs=8):
    eh = h * 0.22
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2+eh/2), w, h-eh,
                                 boxstyle='square,pad=0', fc=fc, ec=ec, lw=1.4, zorder=2))
    ax.add_patch(mpatches.Ellipse((cx, cy+h/2-eh/2), w, eh, fc=fc, ec=ec, lw=1.4, zorder=3))
    ax.add_patch(mpatches.Ellipse((cx, cy-h/2+eh/2), w, eh, fc=fc, ec=ec, lw=1.4, zorder=3))
    ax.text(cx, cy+0.04, text, ha='center', va='center', fontsize=fs,
            color=DARK, fontweight='bold', zorder=4, multialignment='center')


def d5_dfd():
    fig, ax = new_fig('5. Data Flow Diagram',
                      'Data movement from raw CSV and user input through ML models to dashboard outputs',
                      (17, 13), xlim=17, ylim=13)

    # â”€â”€ Row 0 â€” External sources â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    grp(ax, 0.2, 11.2, 7.0, 12.8, 'External Data Sources',
        fc='#faf5ff', ec='#c4b5fd', z=0)
    rbox(ax, 2.0,  12.3, 2.8, 0.58, 'ðŸ“‚ CSV Dataset\n44 K rows Â· 45 MB',
         fc='#fef3c7', ec=AMBER,  tc=DARK, fs=8.5)
    rbox(ax, 5.3,  12.3, 2.8, 0.58, 'ðŸ‘¤ User Input\n22 Spec Fields',
         fc='#ede9fe', ec=VIOLET, tc=DARK, fs=8.5)
    rbox(ax, 1.4,  11.62, 2.2, 0.52, 'ðŸŒ DuckDuckGo',
         fc='#dbeafe', ec=BLUE,   tc=DARK, fs=8.5)

    # â”€â”€ Row 1 â€” Training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    grp(ax, 7.3, 11.0, 16.8, 12.9, 'Offline Training Phase',
        fc='#fff7ed', ec='#fb923c', z=0)
    rbox(ax, 10.0, 12.3, 3.2, 0.6,  'âš™ train_all.py\nGroupShuffleSplit 80/20',
         fc='#fed7aa', ec=ORANGE, tc=DARK, fs=8.5, bold=True)
    cyl(ax, 13.4, 12.55, 2.1, 0.52, 'segment.joblib\n878 KB')
    cyl(ax, 15.8, 12.55, 2.1, 0.52, 'similarity.joblib\n2.7 MB')
    cyl(ax, 13.4, 11.8,  2.1, 0.52, 'demand.joblib\n9.7 MB')
    cyl(ax, 15.8, 11.8,  2.1, 0.52, 'pmf.joblib\n5.2 MB')
    arr(ax, 6.7, 12.3, 8.4, 12.3, 'raw rows', ORANGE)
    for mx, my in [(13.4,12.55),(15.8,12.55),(13.4,11.8),(15.8,11.8)]:
        arr(ax, 11.6, 12.3, mx-1.05, my, color='#fbbf24', lw=1.1)

    # â”€â”€ Row 2 â€” Inference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    grp(ax, 0.2, 6.8, 16.8, 10.5, 'Online Inference Phase',
        fc='#f0f9ff', ec='#38bdf8', z=0)

    rbox(ax, 3.5, 10.0, 3.6, 0.65, 'ðŸ”Œ FastAPI Backend\nport 8000',
         fc='#dbeafe', ec=BLUE, tc=DARK, fs=9, bold=True)

    pcols = [
        (2.0,  8.25, 'P1\nMarket Res.', '#bfdbfe', BLUE,
         [(13.4,12.55),(15.8,12.55)]),
        (5.8,  8.25, 'P2\nDemand',      '#bbf7d0', GREEN,
         [(13.4,11.8)]),
        (9.6,  8.25, 'P3\nSegment',     '#fce7f3', PINK,
         [(15.8,12.55)]),
        (13.4, 8.25, 'P4\nPMF',         '#fed7aa', ORANGE,
         [(15.8,11.8)]),
    ]
    for cx, cy, text, fc, ec, models in pcols:
        rbox(ax, cx, cy, 2.8, 0.9, text, fc=fc, ec=ec, tc=DARK, fs=9)
        arr(ax, 3.5, 9.68, cx, 8.7, color=ec, lw=1.2)
        for mx, my in models:
            ax.annotate('', xy=(cx, cy+0.45), xytext=(mx, my-0.28),
                        arrowprops=dict(arrowstyle='->', color='#fbbf24',
                                        lw=0.9, linestyle='dotted', mutation_scale=8))

    # User â†’ API
    arr(ax, 5.3, 12.01, 5.3, 10.33, label='spec JSON', color=VIOLET, lw=1.5)
    ax.plot([5.3, 5.3], [10.33, 10.33], color=VIOLET, lw=0.5)
    ax.annotate('', xy=(5.27, 10.33), xytext=(5.3, 10.33),
                arrowprops=dict(arrowstyle='->', color=VIOLET, lw=1.5, mutation_scale=12))
    arr(ax, 5.3, 11.31, 5.1, 10.33, color=VIOLET, lw=1.5)

    # CLV engine
    rbox(ax, 7.7, 7.2, 3.4, 0.62, 'ðŸ’° CLV Engine\nprice Ã— margin Ã— repeat Ã— loyalty',
         fc='#fef9c3', ec='#eab308', tc=DARK, fs=8.5)
    for cx, cy, *_ in pcols:
        arr(ax, cx, cy-0.45, 7.7, 7.52, color='#94a3b8', lw=0.9)
    arr(ax, 7.7, 6.89, 3.5, 9.68, label='full JSON', color=VIOLET, lw=1.3)

    # â”€â”€ Row 3 â€” Extra endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    grp(ax, 0.2, 4.0, 16.8, 6.5, 'Additional Endpoints',
        fc='#f8fafc', ec='#94a3b8', z=0)
    eps = [(3.2, 5.25, '/api/competitors\nRadar Chart + Insights'),
           (8.5, 5.25, '/api/price-optimize\n14-Point Price Sweep'),
           (13.8,5.25, '/api/news\nParallel Web Search')]
    for cx, cy, text in eps:
        rbox(ax, cx, cy, 3.8, 0.7, text, fc='#f1f5f9', ec=SLATE, tc=DARK, fs=8.5)
        arr(ax, 3.5, 9.68, cx, 5.6, color=SLATE, lw=1.0)
    arr(ax, 12.0, 5.25, 1.4, 11.36, label='queries', color=BLUE, lw=1.0)

    # â”€â”€ Row 4 â€” Frontend outputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    grp(ax, 0.2, 0.5, 16.8, 3.5, 'Next.js Frontend â€” port 3000',
        fc='#f0fdf4', ec='#4ade80', z=0)
    outs = [(1.4, 2.0, 'Market\nPosition'),
            (3.4, 2.0, 'Demand\nChart'),
            (5.4, 2.0, 'Customer\nProfile'),
            (7.4, 2.0, 'PMF Score\nGO/NO-GO'),
            (9.6, 2.0, 'Competitor\nRadar'),
            (11.8,2.0, 'Price\nOptimizer'),
            (14.0,2.0, 'News\nFeed')]
    for cx, cy, text in outs:
        rbox(ax, cx, cy, 1.85, 0.72, text, fc='#dcfce7', ec='#4ade80', tc=DARK, fs=8.5)

    # Endpoint â†’ outputs
    src_map = {0:3.2, 1:3.2, 2:3.2, 3:3.2, 4:3.2, 5:8.5, 6:13.8}
    for i, (cx, cy, _) in enumerate(outs):
        arr(ax, src_map.get(i,3.2), 4.9, cx, 2.36, color='#4ade80', lw=0.9)

    # Legend
    leg = [('#fef3c7', AMBER,  'Data Store'),
           ('#dbeafe', BLUE,   'External Entity'),
           ('#dbeafe', BLUE,   'Process / API'),
           ('#dcfce7', GREEN,  'Output Component')]
    for i, (fc, ec, lbl) in enumerate(leg):
        bx = 1.5 + i*4.0
        rbox(ax, bx, 0.72, 1.3, 0.3, '', fc=fc, ec=ec, tc=DARK, fs=7, z=1)
        ax.text(bx+0.85, 0.72, lbl, va='center', fontsize=7.5, color=DARK)

    fig.savefig(OUTPUT/'05_data_flow_diagram.png', dpi=180,
                bbox_inches='tight', facecolor=WHITE)
    plt.close(fig)
    print("  âœ“  5/5  Data Flow Diagram")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Compile PDF
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def compile_pdf():
    pdf_path = OUTPUT / 'BI_Agent_All_Diagrams.pdf'
    files = [OUTPUT / f'0{i+1}_{n}.png'
             for i, n in enumerate(['activity_diagram','use_case_diagram',
                                     'class_diagram','sequence_diagram',
                                     'data_flow_diagram'])]
    with PdfPages(pdf_path) as pdf:
        for f in files:
            img = mpimg.imread(f)
            h, w = img.shape[:2]
            fig_w = min(16, w/100)
            fig_h = fig_w * h / w
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            fig.patch.set_facecolor(WHITE)
            ax.imshow(img); ax.axis('off')
            plt.tight_layout(pad=0)
            pdf.savefig(fig, bbox_inches='tight', facecolor=WHITE)
            plt.close(fig)
        info = pdf.infodict()
        info['Title']   = 'BI Agent â€” System Diagrams'
        info['Author']  = 'Smartphone BI Agent Project'
        info['Subject'] = 'UML & Architecture Diagrams'
    print(f"\n  âœ“  PDF compiled â†’ {pdf_path.resolve()}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == '__main__':
    print("\nGenerating BI Agent diagrams...\n")
    d1_activity()
    d2_usecase()
    d3_class()
    d4_sequence()
    d5_dfd()
    compile_pdf()
    print(f"\nAll files saved to: {OUTPUT.resolve()}\n")
