"use client"

import { useState, useEffect } from "react"
import { BIAgentLogo } from "@/components/logo"
import { SpecsForm } from "@/components/specs-form"
import { ThemeProvider, useTheme } from "@/components/theme-provider"
import { analyzePhone, fetchMarketNews, fetchCompetitors, fetchPriceOptimize } from "@/lib/api"
import type { PhoneSpec, AnalysisResult, NewsResult, CompetitorResult, PriceOptimizeResult } from "@/lib/types"
import {
  MarketSummary, DemandSection, CustomerSection, AffordabilitySection,
  LoyaltySection, CompetitorSection, PriceSection, PMFSection, NewsSection,
} from "@/components/dashboard"


/* ── Siri Orb ─────────────────────────────────────────────────── */
function SiriOrb({ size = 220, active = false }: { size?: number; active?: boolean }) {
  const orb     = Math.round(size * 0.73)
  const ring    = Math.round(size * 0.78)
  const spin    = active ? 2.4 : 6
  const spinRev = active ? 3.6 : 9
  const pulse   = active ? 1.8 : 4

  return (
    <div className="relative flex items-center justify-center" style={{ height: size, width: size }}>
      <span className="absolute rounded-full" style={{
        height: ring, width: ring, border: "3px solid rgba(129,140,248,0.55)",
        animation: "siri-ripple 3.2s ease-out infinite",
      }} />
      <span className="absolute rounded-full" style={{
        height: ring, width: ring, border: "3px solid rgba(192,132,252,0.55)",
        animation: "siri-ripple 3.2s ease-out infinite", animationDelay: "1.6s",
      }} />
      <div className="relative" style={{ height: orb, width: orb, animation: `siri-pulse ${pulse}s ease-in-out infinite` }}>
        <div className="absolute inset-0 overflow-hidden rounded-full"
          style={{ boxShadow: "0 0 75px rgba(99,102,241,0.7), 0 0 32px rgba(168,85,247,0.55)" }}>
          <div className="absolute" style={{
            inset: -45,
            background: "conic-gradient(from 0deg, #6366f1, #38bdf8, #a855f7, #ec4899, #6366f1)",
            filter: "blur(16px)", animation: `siri-spin ${spin}s linear infinite`,
          }} />
          <div className="absolute" style={{
            inset: -35,
            background: "conic-gradient(from 180deg, #22d3ee, #818cf8, #c084fc, transparent, #22d3ee)",
            filter: "blur(18px)", mixBlendMode: "screen", animation: `siri-spin-rev ${spinRev}s linear infinite`,
          }} />
          <div className="absolute inset-0 rounded-full" style={{
            background: "radial-gradient(circle at 35% 28%, rgba(255,255,255,0.42), transparent 55%)",
          }} />
        </div>
      </div>
    </div>
  )
}

/* ── Loading Screen ───────────────────────────────────────────── */
function LoadingScreen() {
  const stages = [
    "Researching market position…",
    "Forecasting 6-month demand…",
    "Profiling customer segments…",
    "Scoring product-market fit…",
    "Fetching live news & competitors…",
  ]
  const [i, setI] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setI(p => (p + 1) % stages.length), 1400)
    return () => clearInterval(t)
  }, [stages.length])

  return (
    <div className="flex h-full flex-col items-center justify-center gap-7" style={{ minHeight: "60vh", transform: "translateY(-20px)" }}>
      <SiriOrb size={170} active />
      <p key={i} className="text-sm font-semibold text-foreground" style={{ animation: "siri-fade-in 0.5s ease" }}>
        {stages[i]}
      </p>
      <div className="h-1 w-52 overflow-hidden rounded-full bg-muted">
        <div className="h-full w-full rounded-full" style={{
          background: "linear-gradient(90deg, transparent, #6366f1, #a855f7, transparent)",
          backgroundSize: "200% 100%",
          animation: "siri-shimmer 1.2s linear infinite",
        }} />
      </div>
    </div>
  )
}

/* ── Theme Toggle (sidebar icon-only) ────────────────────────── */
function ThemeToggle() {
  const { theme, toggle } = useTheme()
  const isDark = theme === "dark"
  return (
    <button
      onClick={toggle}
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
      className="neo-sm neo-hover rounded-xl w-12 h-10 flex items-center justify-center"
    >
      {isDark ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="text-amber-400" style={{ animation: "theme-icon-in 0.4s cubic-bezier(0.34,1.56,0.64,1) both" }}>
          <circle cx="12" cy="12" r="4"/>
          <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/>
        </svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="text-indigo-400" style={{ animation: "theme-icon-in 0.4s cubic-bezier(0.34,1.56,0.64,1) both" }}>
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
        </svg>
      )}
    </button>
  )
}

/* ── Download Button (sidebar icon-only) ─────────────────────── */
function DownloadButton({ result }: { result: AnalysisResult | null }) {
  const [saving, setSaving] = useState(false)

  function handleDownload() {
    if (!result) return
    setSaving(true)
    const mp  = result.market_position
    const df  = result.demand_forecast
    const suc = result.success
    const cp  = result.customer_profile

    const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>BI Report — ${result.input.brand} ${result.input.model}</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #fff; color: #0f1117; padding: 40px; max-width: 900px; margin: auto; }
  h1 { font-size: 28px; font-weight: 900; color: #4f46e5; margin-bottom: 4px; }
  h2 { font-size: 13px; color: #64748b; font-weight: 400; margin-bottom: 32px; letter-spacing:.04em; text-transform:uppercase; }
  h3 { font-size: 11px; color: #6366f1; font-weight: 700; letter-spacing:.12em; text-transform:uppercase;
       border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin: 28px 0 14px; }
  .grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-bottom: 8px; }
  .card { background: #f8f9fc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px; }
  .label { font-size: 10px; color: #94a3b8; text-transform: uppercase; letter-spacing:.1em; margin-bottom: 4px; }
  .value { font-size: 22px; font-weight: 900; color: #0f1117; }
  .value.green { color: #16a34a; } .value.red { color: #dc2626; }
  .value.indigo{ color: #4f46e5; } .value.amber { color: #d97706; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; padding: 8px 12px; background: #f1f5f9; color: #64748b;
       font-size: 10px; text-transform: uppercase; letter-spacing:.08em; }
  td { padding: 8px 12px; border-bottom: 1px solid #f1f5f9; }
  .rec { display:inline-block; padding:8px 20px; border-radius:100px; font-weight:800; font-size:15px;
         background:${suc.recommendation.includes("GO FOR") ? "#dcfce7" : suc.recommendation.includes("CAUTION") ? "#fef9c3" : "#fee2e2"};
         color:${suc.recommendation.includes("GO FOR") ? "#15803d" : suc.recommendation.includes("CAUTION") ? "#b45309" : "#dc2626"}; }
  .footer { margin-top: 40px; padding-top: 12px; border-top: 1px solid #e2e8f0;
            font-size: 10px; color: #94a3b8; text-align: center; }
  @media print { body { padding: 20px; } }
</style>
</head>
<body>
  <h1>Smartphone BI Agent Report</h1>
  <h2>${result.input.brand.toUpperCase()} ${result.input.model} &nbsp;·&nbsp; $${result.input.price_usd.toLocaleString()} &nbsp;·&nbsp; ${result.input.region}</h2>
  <div style="margin-bottom:24px"><span class="rec">${suc.recommendation}</span></div>
  <h3>Key Metrics</h3>
  <div class="grid">
    <div class="card"><div class="label">Market Segment</div><div class="value indigo">${mp.predicted_segment.toUpperCase()}</div></div>
    <div class="card"><div class="label">PMF Score</div><div class="value ${suc.pmf_score >= 0.65 ? "green" : suc.pmf_score >= 0.4 ? "amber" : "red"}">${Math.round(suc.pmf_score * 100)}%</div></div>
    <div class="card"><div class="label">Failure Risk</div><div class="value red">${suc.failure_risk_pct}%</div></div>
    <div class="card"><div class="label">6-Month Demand</div><div class="value">${df.total_horizon_units.toLocaleString()} units</div></div>
    <div class="card"><div class="label">CLV / Buyer</div><div class="value green">$${result.clv_per_buyer_usd.toLocaleString()}</div></div>
    <div class="card"><div class="label">vs Category Price</div><div class="value ${mp.price_vs_category_pct > 0 ? "red" : "green"}">${mp.price_vs_category_pct > 0 ? "+" : ""}${mp.price_vs_category_pct.toFixed(1)}%</div></div>
  </div>
  <h3>Demand Forecast (Monthly)</h3>
  <table>
    <tr>${[1,2,3,4,5,6].map(m => `<th>Month ${m}</th>`).join("")}</tr>
    <tr>${[1,2,3,4,5,6].map(m => `<td><strong>${(df as unknown as Record<string,unknown>)[`month_${m}`]}</strong></td>`).join("")}</tr>
  </table>
  <h3>Customer Profile — Age</h3>
  <table>
    <tr><th>18–24</th><th>25–34</th><th>35–44</th><th>45–54</th><th>55+</th></tr>
    <tr>${Object.values(cp.age).map(v => `<td>${(v as number).toFixed(1)}%</td>`).join("")}</tr>
  </table>
  <h3>Closest Competitors</h3>
  <table>
    <tr><th>#</th><th>Phone</th></tr>
    ${mp.closest_competitors.map((c, i) => `<tr><td>${i + 1}</td><td>${c}</td></tr>`).join("")}
  </table>
  <div class="footer">Generated by Smartphone BI Agent &nbsp;·&nbsp; XGBoost + Random Forest + KNN Retrieval &nbsp;·&nbsp; 116,481 records</div>
</body>
</html>`

    const blob = new Blob([html], { type: "text/html" })
    const url  = URL.createObjectURL(blob)
    const win  = window.open(url, "_blank")
    if (win) {
      win.onload = () => { win.print(); setSaving(false) }
    } else {
      const a = document.createElement("a")
      a.href = url; a.download = `bi-report-${result.input.model.replace(/\s+/g, "-")}.html`
      a.click(); setSaving(false)
    }
    setTimeout(() => URL.revokeObjectURL(url), 10000)
  }

  return (
    <button
      onClick={handleDownload}
      disabled={!result || saving}
      title={result ? "Download PDF report" : "Run analysis first"}
      className="neo-sm neo-hover rounded-xl w-12 h-10 flex items-center justify-center text-muted-foreground disabled:opacity-40 disabled:cursor-not-allowed"
    >
      {saving ? (
        <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
        </svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
      )}
    </button>
  )
}

/* ── Root component ───────────────────────────────────────────── */
export function BiAgentApp() {
  return (
    <ThemeProvider>
      <BiAgentAppInner />
    </ThemeProvider>
  )
}

/* ── Inner app (state lives here) ────────────────────────────── */
function BiAgentAppInner() {
  const [result, setResult]               = useState<AnalysisResult | null>(null)
  const [news, setNews]                   = useState<NewsResult | null>(null)
  const [competitors, setCompetitors]     = useState<CompetitorResult | null>(null)
  const [priceData, setPriceData]         = useState<PriceOptimizeResult | null>(null)
  const [loading, setLoading]             = useState(false)
  const [newsLoading, setNewsLoading]     = useState(false)
  const [error, setError]                 = useState<string | null>(null)
  const [newsError, setNewsError]         = useState<string | null>(null)
  const [drawerOpen, setDrawerOpen]       = useState(true)

  async function handleSubmit(spec: PhoneSpec) {
    setLoading(true)
    setError(null)
    setNews(null)
    setNewsError(null)
    setCompetitors(null)
    setPriceData(null)

    let analysisData: AnalysisResult
    try {
      analysisData = await analyzePhone(spec)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed")
      setLoading(false)
      return
    }
    setResult(analysisData)
    setLoading(false)

    setNewsLoading(true)
    fetchMarketNews(spec)
      .then(setNews)
      .catch((e) => setNewsError(e instanceof Error ? e.message : "News fetch failed"))
      .finally(() => setNewsLoading(false))

    fetchCompetitors(spec).then(setCompetitors).catch(() => {})
    fetchPriceOptimize(spec).then(setPriceData).catch(() => {})
  }

  return (
    <div className="h-screen overflow-hidden bg-background">

      {/* ── Specs Panel (docked, shifts main content) ─────── */}
      <div
        className="fixed top-0 bottom-0 left-18 z-40 w-85 bg-card overflow-hidden"
        style={{
          transform: drawerOpen ? "translateX(0)" : "translateX(calc(-100% - 72px))",
          transition: "transform 0.32s cubic-bezier(0.4,0,0.2,1)",
          borderRight: "1px solid var(--border)",
          boxShadow: "4px 0 20px var(--neo-shadow-dark)",
        }}
      >
        {/* Panel header */}
        <div className="sticky top-0 z-10 bg-card flex items-center justify-between px-4 py-3"
          style={{ borderBottom: "1px solid var(--border)" }}>
          <span className="text-[11px] font-black uppercase tracking-widest text-primary">Phone Specifications</span>
          <button
            onClick={() => setDrawerOpen(false)}
            title="Collapse specs panel"
            className="neo-sm rounded-lg w-7 h-7 flex items-center justify-center text-muted-foreground hover:text-foreground text-xs transition-colors"
          >✕</button>
        </div>
        {/* Scrollable form */}
        <div className="overflow-y-auto h-[calc(100%-49px)]">
          <SpecsForm onSubmit={handleSubmit} loading={loading} />
        </div>
      </div>

      {/* ── Sidebar Navigation ─────────────────────────────── */}
      <aside
        className="fixed top-0 left-0 bottom-0 z-50 w-18 bg-card flex flex-col items-center py-3 gap-1"
        style={{
          boxShadow: "5px 0 20px var(--neo-shadow-dark), -2px 0 8px var(--neo-shadow-light)",
        }}
      >
        {/* Logo */}
        <div className="mb-2 mt-1">
          <BIAgentLogo size={34} />
        </div>

        {/* Analyze trigger button */}
        <button
          onClick={() => setDrawerOpen(v => !v)}
          title="Enter phone specifications"
          className="w-12 h-12 rounded-2xl flex flex-col items-center justify-center gap-0.5 transition-all mb-1 select-none"
          style={{
            background: "linear-gradient(135deg, #6366f1, #a855f7)",
            boxShadow: drawerOpen
              ? "inset 3px 3px 8px rgba(0,0,0,0.3), inset -2px -2px 5px rgba(255,255,255,0.1), 0 0 20px var(--neo-glow)"
              : "7px 7px 18px var(--neo-shadow-dark), -5px -5px 14px var(--neo-shadow-light), 0 0 20px var(--neo-glow)",
            animation: !result ? "neo-pulse 2.5s ease-in-out infinite" : "none",
            transition: "box-shadow 0.2s ease, transform 0.15s ease",
          }}
        >
          {drawerOpen ? (
            /* collapse icon */
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M15 18l-6-6 6-6"/>
            </svg>
          ) : (
            /* expand / specs icon */
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 18l6-6-6-6"/>
            </svg>
          )}
          <span className="text-[8px] font-black text-white/90 leading-none tracking-wider">
            {drawerOpen ? "HIDE" : "SPECS"}
          </span>
        </button>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Divider */}
        <div className="w-9 h-px mb-1" style={{ background: "var(--border)" }} />

        {/* Theme + Download */}
        <ThemeToggle />
        <DownloadButton result={result} />
        <div className="h-1" />
      </aside>

      {/* ── Main Content (shifts right when specs panel is docked) ── */}
      <div
        className="flex flex-col h-screen"
        style={{
          marginLeft: drawerOpen ? 72 + 340 : 72,
          transition: "margin-left 0.32s cubic-bezier(0.4,0,0.2,1)",
        }}
      >

        {/* Top bar */}
        <header
          className="shrink-0 flex items-center justify-between px-5 py-2.5 bg-card z-10"
          style={{
            boxShadow: "0 4px 16px var(--neo-shadow-dark), 0 -1px 6px var(--neo-shadow-light)",
          }}
        >
          <div className="flex items-center gap-3">
            <div className="flex flex-col leading-tight">
              <div className="flex items-center gap-1.5">
                <span className="font-black text-sm tracking-tight text-foreground">Product Launch</span>
                <span
                  className="font-black text-sm tracking-tight"
                  style={{
                    background: "linear-gradient(90deg, #6366f1, #a855f7, #6366f1)",
                    backgroundSize: "200% auto",
                    WebkitBackgroundClip: "text",
                    WebkitTextFillColor: "transparent",
                    animation: "gradient-shift 3s linear infinite",
                  }}
                >Intelligence</span>
                <span className="font-black text-sm tracking-tight text-foreground">Dashboard</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className={`h-1.5 w-1.5 rounded-full ${
                  loading ? "bg-amber-400 animate-pulse"
                  : result  ? "bg-emerald-500"
                  : "bg-muted-foreground"
                }`} />
                <span className="text-[10px] text-muted-foreground">
                  {loading ? "Analyzing…" : result ? `${result.input.brand} ${result.input.model}` : "Enter specs to begin"}
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {newsLoading && (
              <span className="neo-sm flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] text-muted-foreground">
                <svg className="h-3 w-3 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                </svg>
                Fetching news…
              </span>
            )}
            {result && (
              <span className="neo-sm rounded-full px-3 py-1 text-[11px] font-semibold text-emerald-600 dark:text-emerald-400">
                📱 {result.input.model}
              </span>
            )}
          </div>
        </header>

        {/* Scrollable section content */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-5">

            {/* Error banner */}
            {error && (
              <div className="mb-4 neo-flat rounded-2xl px-4 py-3 text-sm text-red-400" style={{ border: "1px solid rgba(239,68,68,0.2)" }}>
                <strong>Error:</strong> {error}
              </div>
            )}

            {/* Empty state */}
            {!result && !loading && !error && (
              <div className="flex flex-col items-center justify-center" style={{ minHeight: "calc(100vh - 160px)" }}>
                <div style={{ transform: "translateY(-36px)" }}>
                  <SiriOrb />
                </div>
                <div className="text-center" style={{ marginTop: -28, animation: "siri-fade-in 1s ease" }}>
                  <h2 className="text-xl font-bold text-foreground" style={{ animation: "siri-text-glow 3s ease-in-out infinite" }}>
                    Enter specs and click Analyze
                  </h2>
                  <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground leading-relaxed">
                    Predicts market segment, 6-month demand, customer demographics,
                    PMF score, CLV, and fetches live competitor news.
                  </p>
                  <button
                    onClick={() => setDrawerOpen(true)}
                    className="mt-5 rounded-2xl px-6 py-2.5 text-sm font-bold text-white transition-transform hover:scale-105"
                    style={{
                      background: "linear-gradient(135deg, #6366f1, #a855f7)",
                      boxShadow: "7px 7px 18px var(--neo-shadow-dark), -5px -5px 14px var(--neo-shadow-light), 0 0 24px var(--neo-glow)",
                      animation: "neo-pulse 2.5s ease-in-out infinite",
                    }}
                  >
                    + Enter Phone Specs
                  </button>
                </div>
              </div>
            )}

            {/* Loading */}
            {loading && <LoadingScreen />}

            {/* All sections — single scrollable page, user-defined sequence */}
            {result && !loading && (
              <div className="space-y-10" style={{ animation: "fade-up 0.3s ease" }}>
                <DemandSection       result={result} />
                <CustomerSection     result={result} />
                <AffordabilitySection result={result} />
                <LoyaltySection      result={result} />
                <CompetitorSection   result={result} competitors={competitors} />
                <PriceSection        priceData={priceData} result={result} />
                <PMFSection          result={result} />
                <NewsSection         news={news} newsLoading={newsLoading} newsError={newsError} />
              </div>
            )}

            {/* Footer */}
            {result && !loading && (
              <p className="mt-6 pb-2 text-center text-[10px] text-muted-foreground">
                Smartphone BI Agent &nbsp;·&nbsp; XGBoost + Random Forest + KNN Retrieval &nbsp;·&nbsp; 116,481 snapshots
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
