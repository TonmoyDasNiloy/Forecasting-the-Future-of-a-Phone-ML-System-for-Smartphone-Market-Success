"use client"

import { cn } from "@/lib/utils"
import type { AnalysisResult, NewsResult, CompetitorResult, PriceOptimizeResult } from "@/lib/types"
import { DemandChart }      from "@/components/charts/demand-chart"
import { DonutChart }       from "@/components/charts/donut-chart"
import { LoyaltyDiverging } from "@/components/charts/loyalty-diverging"
import { PMFGauge }         from "@/components/charts/pmf-gauge"
import { NewsPanel }        from "@/components/news-panel"
import { CompetitorRadar }  from "@/components/charts/competitor-radar"
import { PriceOptimizer }   from "@/components/charts/price-optimizer"

/* ── Segment colour map ──────────────────────────────────────── */
const SEG_COLORS: Record<string, string> = {
  budget:      "text-sky-400 border-sky-500/30 bg-sky-500/10",
  "mid-range": "text-indigo-400 border-indigo-500/30 bg-indigo-500/10",
  premium:     "text-purple-400 border-purple-500/30 bg-purple-500/10",
  flagship:    "text-amber-400 border-amber-500/30 bg-amber-500/10",
}

/* ── Neo card primitive ──────────────────────────────────────── */
function NC({
  children, className, depth = "raised", style,
}: {
  children: React.ReactNode
  className?: string
  depth?: "raised" | "flat" | "inset" | "sm"
  style?: React.CSSProperties
}) {
  const shadow = { raised: "neo-raised", flat: "neo-flat", inset: "neo-inset", sm: "neo-sm" }[depth]
  return (
    <div className={cn("rounded-2xl bg-card p-4", shadow, className)} style={style}>
      {children}
    </div>
  )
}

/* ── Small label ─────────────────────────────────────────────── */
function Label({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-1">
      {children}
    </p>
  )
}

/* ── Section header ──────────────────────────────────────────── */
function SectionHeader({ emoji, title, sub }: { emoji: string; title: string; sub?: string }) {
  return (
    <div className="flex items-center gap-3 mb-5">
      <span className="text-2xl">{emoji}</span>
      <div>
        <h2 className="text-base font-black tracking-tight text-foreground leading-tight">{title}</h2>
        {sub && <p className="text-[11px] text-muted-foreground mt-0.5">{sub}</p>}
      </div>
    </div>
  )
}

/* ── Shared demographic helpers ──────────────────────────────── */
function buildDemoData(cp: AnalysisResult["customer_profile"]) {
  const ageData = [
    { name: "18–24", value: cp.age.age_18_24_pct,   color: "#4f6ef7" },
    { name: "25–34", value: cp.age.age_25_34_pct,   color: "#818cf8" },
    { name: "35–44", value: cp.age.age_35_44_pct,   color: "#a78bfa" },
    { name: "45–54", value: cp.age.age_45_54_pct,   color: "#f472b6" },
    { name: "55+",   value: cp.age.age_55_plus_pct, color: "#38bdf8" },
  ]
  const incomeData = [
    { name: "Low",        value: cp.income.income_low_pct,       color: "#ef4444" },
    { name: "Lower-mid",  value: cp.income.income_lower_mid_pct, color: "#f59e0b" },
    { name: "Mid",        value: cp.income.income_mid_pct,       color: "#facc15" },
    { name: "Upper-mid",  value: cp.income.income_upper_mid_pct, color: "#86efac" },
    { name: "High",       value: cp.income.income_high_pct,      color: "#22c55e" },
  ]
  const rfmData = [
    { name: "Recent",      value: cp.rfm.rfm_recent_pct,      color: "#38bdf8" },
    { name: "Champion",    value: cp.rfm.rfm_champion_pct,    color: "#22c55e" },
    { name: "Loyal",       value: cp.rfm.rfm_loyal_pct,       color: "#818cf8" },
    { name: "At-risk",     value: cp.rfm.rfm_at_risk_pct,     color: "#f59e0b" },
    { name: "Hibernating", value: cp.rfm.rfm_hibernating_pct, color: "#ef4444" },
  ]
  const genderData = [
    { name: "Male",       value: cp.gender.gender_male_pct,      color: "#4f6ef7" },
    { name: "Female",     value: cp.gender.gender_female_pct,    color: "#f472b6" },
    { name: "Non-binary", value: cp.gender.gender_nonbinary_pct, color: "#a78bfa" },
  ]
  const topOf = (arr: { name: string; value: number }[]) =>
    arr.reduce((a, b) => (b.value > a.value ? b : a), arr[0])

  return {
    ageData, incomeData, rfmData, genderData,
    topAge:    topOf(ageData),
    topIncome: topOf(incomeData),
    topRfm:    topOf(rfmData),
    topGender: topOf(genderData),
  }
}

/* ══════════════════════════════════════════════════════════════
   MARKET SUMMARY — verdict banner + KPI strip (top of page)
══════════════════════════════════════════════════════════════ */
export function MarketSummary({ result }: { result: AnalysisResult }) {
  const { input, market_position: mp, demand_forecast: df, success: suc, clv_per_buyer_usd: clv } = result

  const isGo      = suc.recommendation.includes("GO FOR")
  const isCaution = suc.recommendation.includes("CAUTION")
  const icon      = isGo ? "✅" : isCaution ? "⚠️" : "🚫"
  const recClass  = isGo
    ? "bg-emerald-950/60 border-emerald-600/40 text-emerald-300"
    : isCaution
    ? "bg-amber-950/60 border-amber-600/40 text-amber-300"
    : "bg-red-950/60 border-red-600/40 text-red-300"

  const priceDelta = mp.price_vs_category_pct
  const priceSign  = priceDelta >= 0 ? "+" : ""
  const priceColor = priceDelta > 15 ? "text-red-400" : priceDelta < -5 ? "text-emerald-400" : "text-amber-400"
  const pmfColor   = suc.pmf_score >= 0.65 ? "text-emerald-400" : suc.pmf_score >= 0.4 ? "text-amber-400" : "text-red-400"

  const kpis = [
    { label: "Segment",        value: mp.predicted_segment.toUpperCase(), color: SEG_COLORS[mp.predicted_segment]?.split(" ")[0] ?? "text-primary" },
    { label: "Confidence",     value: `${mp.confidence_pct}%`,            color: "text-primary" },
    { label: "PMF Score",      value: `${Math.round(suc.pmf_score * 100)}/100`, color: pmfColor },
    { label: "Failure Risk",   value: `${suc.failure_risk_pct}%`,         color: suc.failure_risk_pct <= 25 ? "text-emerald-400" : "text-red-400" },
    { label: "6-Mo Demand",    value: df.total_horizon_units.toLocaleString(), color: "text-sky-400" },
    { label: "CLV / Buyer",    value: `$${clv.toLocaleString()}`,         color: "text-emerald-400" },
    { label: "Price vs Mkt",   value: `${priceSign}${priceDelta.toFixed(1)}%`, color: priceColor },
  ]

  return (
    <div className="space-y-4">
      {/* Verdict banner */}
      <div className={cn("neo-raised rounded-2xl border p-5", recClass)}>
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-2xl">{icon}</span>
              <span className="text-xl font-black tracking-tight">{suc.recommendation}</span>
            </div>
            <p className="text-sm opacity-80">
              {input.model} &nbsp;·&nbsp; ${input.price_usd.toLocaleString()} &nbsp;·&nbsp; {input.region}
            </p>
          </div>
          <span className={cn(
            "shrink-0 rounded-full border px-3 py-1 text-xs font-bold uppercase tracking-wider",
            SEG_COLORS[mp.predicted_segment] ?? ""
          )}>
            {mp.predicted_segment}
          </span>
        </div>
      </div>

      {/* KPI strip */}
      <div className="grid grid-cols-7 gap-3">
        {kpis.map(k => (
          <NC key={k.label} depth="inset" className="p-3 text-center">
            <p className="text-[9px] font-bold uppercase tracking-widest text-muted-foreground">{k.label}</p>
            <p className={cn("text-sm font-black mt-1 leading-none", k.color)}>{k.value}</p>
          </NC>
        ))}
      </div>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════
   1. DEMAND FORECAST
══════════════════════════════════════════════════════════════ */
export function DemandSection({ result }: { result: AnalysisResult }) {
  const { demand_forecast: df, clv_per_buyer_usd: clv } = result

  return (
    <div>
      <SectionHeader emoji="📈" title="Demand Forecast"
        sub="6-month XGBoost regression · log1p target · monthly breakdown" />

      <div className="grid grid-cols-3 gap-4">
        {/* Chart — full width */}
        <NC depth="raised" className="col-span-3">
          <Label>6-Month Projected Demand</Label>
          <DemandChart forecast={df} />
        </NC>

        {/* Monthly tiles */}
        <div className="col-span-3 grid grid-cols-6 gap-2">
          {[1, 2, 3, 4, 5, 6].map(m => (
            <NC key={m} depth="inset" className="p-3 text-center">
              <p className="text-[10px] text-muted-foreground font-semibold">M{m}</p>
              <p className="text-sm font-black text-primary mt-0.5">
                {(df as unknown as Record<string, unknown>)[`month_${m}`] as string}
              </p>
              <p className="text-[9px] text-muted-foreground mt-0.5">units</p>
            </NC>
          ))}
        </div>

        {/* Total — 2 cols */}
        <NC depth="raised" className="col-span-2 flex flex-col gap-1">
          <Label>Total 6-Month Demand</Label>
          <p className="text-4xl font-black leading-none text-sky-400">
            {df.total_horizon_units.toLocaleString()}
          </p>
          <p className="text-xs text-muted-foreground mt-1">projected units across 6 months</p>
          <div className="mt-3 h-1.5 rounded-full overflow-hidden neo-inset">
            <div className="h-full rounded-full"
              style={{ width: "100%", background: "linear-gradient(90deg,#38bdf8,#6366f1)" }} />
          </div>
        </NC>

        {/* CLV */}
        <NC depth="raised" className="flex flex-col gap-1">
          <Label>CLV / Buyer</Label>
          <p className="text-4xl font-black leading-none text-emerald-400">${clv.toLocaleString()}</p>
          <p className="text-xs text-muted-foreground mt-1">customer lifetime value</p>
          <p className="text-[10px] text-muted-foreground mt-auto leading-relaxed">
            Net price × margin × repeat factor × 2.4-yr horizon
          </p>
        </NC>
      </div>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════
   2. CUSTOMER PROFILE — persona card + 3 donuts
══════════════════════════════════════════════════════════════ */
export function CustomerSection({ result }: { result: AnalysisResult }) {
  const { input, customer_profile: cp } = result
  const { ageData, incomeData, genderData, topAge, topIncome, topGender, topRfm } = buildDemoData(cp)

  const incomeWord: Record<string, string> = {
    "Low": "budget-conscious", "Lower-mid": "value-seeking", "Mid": "mid-income",
    "Upper-mid": "well-off", "High": "premium",
  }
  const rfmWord: Record<string, string> = {
    "Recent": "new to the brand", "Champion": "highly engaged", "Loyal": "loyal repeat buyers",
    "At-risk": "slipping away", "Hibernating": "long-lapsed",
  }

  return (
    <div>
      <SectionHeader emoji="👥" title="Customer Profile"
        sub="KNN similarity-derived demographics from real comparable buyers" />

      <div className="grid grid-cols-3 gap-4">
        {/* Persona card — full width */}
        <NC depth="raised" className="col-span-3">
          <div className="flex items-start gap-4">
            <div className="neo-inset flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl text-2xl">
              🧑‍💼
            </div>
            <div>
              <Label>Most Likely Buyer</Label>
              <p className="text-sm text-foreground leading-relaxed">
                A <span className="font-bold">{topAge.name}</span> year-old,{" "}
                <span className="font-bold">{incomeWord[topIncome.name] ?? topIncome.name.toLowerCase()}</span> shopper
                who is typically <span className="font-bold">{rfmWord[topRfm.name] ?? topRfm.name.toLowerCase()}</span>.{" "}
                Leans <span className="font-bold">{topGender.name}</span> ({topGender.value.toFixed(0)}%).
              </p>
              <p className="text-xs text-muted-foreground mt-1.5 leading-relaxed">
                Estimated from real buyers of the most spec-similar phones in {input.region}.
              </p>
            </div>
          </div>
        </NC>

        {/* Age donut */}
        <NC depth="raised">
          <Label>Age Groups</Label>
          <p className="text-[11px] text-muted-foreground mb-2">How old your buyers are</p>
          <DonutChart data={ageData} centerValue={`${topAge.value.toFixed(0)}%`} centerLabel={`${topAge.name} yrs`} />
          <p className="mt-2 text-center text-[11px] text-muted-foreground">
            Largest: <span className="font-bold text-foreground">{topAge.name}</span>
          </p>
        </NC>

        {/* Income donut */}
        <NC depth="raised">
          <Label>Income Level</Label>
          <p className="text-[11px] text-muted-foreground mb-2">Spending power of buyers</p>
          <DonutChart data={incomeData} centerValue={`${topIncome.value.toFixed(0)}%`} centerLabel={topIncome.name} />
          <p className="mt-2 text-center text-[11px] text-muted-foreground">
            Largest: <span className="font-bold text-foreground">{topIncome.name}</span>
          </p>
        </NC>

        {/* Gender donut */}
        <NC depth="raised">
          <Label>Gender Split</Label>
          <p className="text-[11px] text-muted-foreground mb-2">Who tends to buy this phone</p>
          <DonutChart data={genderData} centerValue={`${topGender.value.toFixed(0)}%`} centerLabel={topGender.name} />
          <p className="mt-2 text-center text-[11px] text-muted-foreground">
            Leans: <span className="font-bold text-foreground">{topGender.name}</span>
          </p>
        </NC>

      </div>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════
   3. AFFORDABILITY FEED
══════════════════════════════════════════════════════════════ */
export function AffordabilitySection({ result }: { result: AnalysisResult }) {
  const { input, customer_profile: cp } = result
  const price = input.price_usd

  const incomeTiers = [
    { name: "Low",       pct: cp.income.income_low_pct,       ceiling: 180 },
    { name: "Lower-mid", pct: cp.income.income_lower_mid_pct, ceiling: 350 },
    { name: "Mid",       pct: cp.income.income_mid_pct,       ceiling: 650 },
    { name: "Upper-mid", pct: cp.income.income_upper_mid_pct, ceiling: 1100 },
    { name: "High",      pct: cp.income.income_high_pct,      ceiling: 99999 },
  ]
  const comfortablePct = Math.round(
    incomeTiers.filter(t => price <= t.ceiling).reduce((s, t) => s + t.pct, 0)
  )
  const stretchPct = 100 - comfortablePct
  const affordTone  = comfortablePct >= 70 ? "good" : comfortablePct >= 45 ? "warn" : "bad"
  const affordLabel = affordTone === "good" ? "Well-matched price"
    : affordTone === "warn" ? "Priced at the edge" : "Priced above audience"
  const affordNote  = affordTone === "good"
    ? `${comfortablePct}% of likely buyers can comfortably afford this price.`
    : affordTone === "warn"
    ? `Only ${comfortablePct}% comfortably afford it; ${stretchPct}% may find it a stretch.`
    : `${stretchPct}% of likely buyers may find this price hard to justify.`

  return (
    <div>
      <SectionHeader emoji="💵" title="Affordability Feed"
        sub={`Does your $${price.toLocaleString()} price match what your predicted buyers can afford?`} />

      <div className="grid grid-cols-3 gap-4">
        {/* Verdict */}
        <NC depth="raised" className="col-span-3">
          <div className={cn("neo-inset rounded-xl px-4 py-3 mb-4",
            affordTone === "good" ? "border border-emerald-500/30"
            : affordTone === "warn" ? "border border-amber-500/30"
            : "border border-red-500/30")}>
            <p className={cn("text-base font-black",
              affordTone === "good" ? "text-emerald-600 dark:text-emerald-400"
              : affordTone === "warn" ? "text-amber-600 dark:text-amber-400"
              : "text-red-600 dark:text-red-400")}>
              {affordTone === "good" ? "✅" : affordTone === "warn" ? "⚠️" : "🚫"} {affordLabel}
            </p>
            <p className="text-sm text-muted-foreground mt-1">{affordNote}</p>
          </div>

          {/* Stacked bar */}
          <Label>Comfortable vs Stretch split</Label>
          <div className="neo-inset h-9 rounded-xl overflow-hidden flex mb-4">
            <div className="h-full bg-emerald-500/80 flex items-center justify-center transition-all"
              style={{ width: `${comfortablePct}%` }}>
              {comfortablePct >= 15 && (
                <span className="text-[12px] font-black text-white">{comfortablePct}% can afford</span>
              )}
            </div>
            <div className="h-full bg-red-500/70 flex items-center justify-center"
              style={{ width: `${stretchPct}%` }}>
              {stretchPct >= 15 && (
                <span className="text-[12px] font-black text-white">{stretchPct}% a stretch</span>
              )}
            </div>
          </div>

          {/* Per-tier breakdown */}
          <div className="grid grid-cols-5 gap-3">
            {incomeTiers.map(t => {
              const ok = price <= t.ceiling
              return (
                <NC key={t.name} depth="inset"
                  className={cn("p-3 text-center", ok ? "border border-emerald-500/25" : "border border-red-500/25")}>
                  <p className="text-[10px] text-muted-foreground">{t.name}</p>
                  <p className="text-lg font-black text-foreground">{t.pct.toFixed(0)}%</p>
                  <p className={cn("text-[10px] font-bold mt-0.5",
                    ok ? "text-emerald-500 dark:text-emerald-400" : "text-red-500 dark:text-red-400")}>
                    {ok ? "✓ affords" : "✗ stretch"}
                  </p>
                  <p className="text-[9px] text-muted-foreground mt-1">≤ ${t.ceiling === 99999 ? "any" : t.ceiling}</p>
                </NC>
              )
            })}
          </div>

          <p className="mt-3 text-[10px] text-muted-foreground leading-relaxed">
            A tier &quot;affords&quot; when your price sits within its typical phone budget ceiling. Income mix is predicted from spec-similar phones; tier budgets are indicative.
          </p>
        </NC>
      </div>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════
   4. BUYER LOYALTY
══════════════════════════════════════════════════════════════ */
export function LoyaltySection({ result }: { result: AnalysisResult }) {
  const { customer_profile: cp } = result
  const { rfmData, topRfm } = buildDemoData(cp)

  const engaged    = cp.rfm.rfm_champion_pct + cp.rfm.rfm_loyal_pct + cp.rfm.rfm_recent_pct
  const atRisk     = cp.rfm.rfm_at_risk_pct + cp.rfm.rfm_hibernating_pct
  const engagedTone = engaged >= 60 ? "good" : engaged >= 40 ? "warn" : "bad"

  return (
    <div>
      <SectionHeader emoji="🔁" title="Buyer Loyalty"
        sub="RFM segmentation — engaged champions vs at-risk & hibernating buyers" />

      <div className="grid grid-cols-3 gap-4">
        {/* Summary KPIs */}
        <NC depth="raised">
          <Label>Engaged Buyers</Label>
          <p className={cn("text-4xl font-black leading-none",
            engagedTone === "good" ? "text-emerald-400" : engagedTone === "warn" ? "text-amber-400" : "text-red-400")}>
            {engaged.toFixed(0)}%
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Champions + Loyal + Recent
          </p>
          <div className="mt-3 neo-inset h-1.5 rounded-full overflow-hidden">
            <div className="h-full rounded-full" style={{
              width: `${engaged}%`,
              background: engagedTone === "good"
                ? "linear-gradient(90deg,#22c55e,#86efac)"
                : engagedTone === "warn"
                ? "linear-gradient(90deg,#f59e0b,#facc15)"
                : "linear-gradient(90deg,#ef4444,#f87171)",
            }} />
          </div>
        </NC>

        <NC depth="raised">
          <Label>At-Risk Buyers</Label>
          <p className={cn("text-4xl font-black leading-none",
            atRisk <= 25 ? "text-emerald-400" : atRisk <= 45 ? "text-amber-400" : "text-red-400")}>
            {atRisk.toFixed(0)}%
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            At-risk + Hibernating
          </p>
          <div className="mt-3 neo-inset h-1.5 rounded-full overflow-hidden">
            <div className="h-full rounded-full" style={{
              width: `${atRisk}%`,
              background: atRisk <= 25
                ? "linear-gradient(90deg,#22c55e,#86efac)"
                : atRisk <= 45
                ? "linear-gradient(90deg,#f59e0b,#facc15)"
                : "linear-gradient(90deg,#ef4444,#f87171)",
            }} />
          </div>
        </NC>

        <NC depth="raised">
          <Label>Dominant Segment</Label>
          <p className="text-2xl font-black leading-none text-primary">{topRfm.name}</p>
          <p className="text-3xl font-black text-foreground mt-1">{topRfm.value.toFixed(0)}%</p>
          <p className="text-xs text-muted-foreground mt-1">of predicted buyers</p>
        </NC>

        {/* Diverging chart — full width */}
        <NC depth="raised" className="col-span-3">
          <Label>Full RFM Breakdown</Label>
          <LoyaltyDiverging rfm={cp.rfm} />
        </NC>

      </div>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════
   5. COMPETITOR GAP ANALYSIS
══════════════════════════════════════════════════════════════ */
export function CompetitorSection({
  result, competitors,
}: {
  result: AnalysisResult
  competitors: CompetitorResult | null
}) {
  const { market_position: mp } = result
  const priceDelta = mp.price_vs_category_pct
  const priceSign  = priceDelta >= 0 ? "+" : ""
  const priceColor = priceDelta > 15 ? "text-red-400" : priceDelta < -5 ? "text-emerald-400" : "text-amber-400"

  return (
    <div>
      <SectionHeader emoji="⚔️" title="Competitor Gap Analysis"
        sub="Radar comparison · spec similarity · KNN-matched competitors" />

      <div className="grid grid-cols-2 gap-4">
        {/* Price delta */}
        <NC depth="raised">
          <Label>Price vs Category Avg</Label>
          <p className={cn("text-3xl font-black leading-none", priceColor)}>
            {priceSign}{priceDelta.toFixed(1)}%
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            {priceDelta > 15 ? "Above peers — justify with specs"
              : priceDelta < -5 ? "Value leader — strong positioning"
              : "In-line with market average"}
          </p>
        </NC>

        {/* Segment */}
        <NC depth="raised">
          <Label>Market Segment</Label>
          <p className={cn("text-2xl font-black leading-none", SEG_COLORS[mp.predicted_segment]?.split(" ")[0] ?? "text-primary")}>
            {mp.predicted_segment.toUpperCase()}
          </p>
          <div className="mt-2 neo-inset h-1.5 rounded-full overflow-hidden">
            <div className="h-full rounded-full"
              style={{ width: `${mp.confidence_pct}%`, background: "linear-gradient(90deg,#6366f1,#a855f7)" }} />
          </div>
          <p className="text-[10px] text-muted-foreground mt-1">{mp.confidence_pct}% classifier confidence</p>
        </NC>

        {/* Radar — full width */}
        <NC depth="raised" className="col-span-2">
          {competitors ? (
            <CompetitorRadar data={competitors} />
          ) : (
            <div className="flex h-48 items-center justify-center flex-col gap-3">
              <div className="h-7 w-7 rounded-full border-2 border-primary border-t-transparent animate-spin" />
              <p className="text-sm text-muted-foreground">Loading competitor radar…</p>
            </div>
          )}
        </NC>
      </div>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════
   6. OPTIMAL PRICE RECOMMENDER
══════════════════════════════════════════════════════════════ */
export function PriceSection({
  priceData, result,
}: {
  priceData: PriceOptimizeResult | null
  result: AnalysisResult
}) {
  const { input, market_position: mp } = result
  const priceDelta = mp.price_vs_category_pct
  const priceSign  = priceDelta >= 0 ? "+" : ""
  const priceColor = priceDelta > 15 ? "text-red-400" : priceDelta < -5 ? "text-emerald-400" : "text-amber-400"

  return (
    <div>
      <SectionHeader emoji="💰" title="Optimal Price Recommender"
        sub="14-point demand/revenue sweep · spec-ratio adjustment · margin modelling" />

      <div className="grid grid-cols-3 gap-4">
        <NC depth="raised">
          <Label>Your Price</Label>
          <p className="text-3xl font-black leading-none text-foreground">${input.price_usd.toLocaleString()}</p>
          <p className="text-xs text-muted-foreground mt-1">{input.model} · {input.region}</p>
        </NC>

        <NC depth={priceData ? "raised" : "flat"}>
          <Label>Recommended Price</Label>
          {priceData ? (
            <>
              <p className="text-3xl font-black leading-none text-primary">
                ${priceData.recommended_price.toLocaleString()}
              </p>
              <p className="text-xs text-muted-foreground mt-1">max revenue point on demand curve</p>
            </>
          ) : (
            <div className="flex items-center gap-2 h-10">
              <div className="h-4 w-4 rounded-full border-2 border-primary border-t-transparent animate-spin" />
              <p className="text-xs text-muted-foreground">Computing…</p>
            </div>
          )}
        </NC>

        <NC depth="raised">
          <Label>vs Category Avg</Label>
          <p className={cn("text-3xl font-black leading-none", priceColor)}>
            {priceSign}{priceDelta.toFixed(1)}%
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            {priceDelta > 15 ? "Premium — back it up with specs"
              : priceDelta < -5 ? "Value leader — strong position"
              : "In-line with peers"}
          </p>
        </NC>

        {/* Price optimizer chart — full width */}
        <NC depth="raised" className="col-span-3">
          {priceData ? (
            <PriceOptimizer data={priceData} />
          ) : (
            <div className="flex h-48 items-center justify-center flex-col gap-3">
              <div className="h-8 w-8 rounded-full border-2 border-primary border-t-transparent animate-spin" />
              <p className="text-sm text-muted-foreground">Computing 14-point price sweep…</p>
            </div>
          )}
        </NC>

        {/* Assumptions */}
        <NC depth="inset" className="text-center">
          <Label>Gross Margin</Label>
          <p className="text-2xl font-black text-foreground">40%</p>
          <p className="text-[10px] text-muted-foreground mt-0.5">assumed hardware margin</p>
        </NC>
        <NC depth="inset" className="text-center">
          <Label>Demand Elasticity β</Label>
          <p className="text-2xl font-black text-foreground">1.30</p>
          <p className="text-[10px] text-muted-foreground mt-0.5">price sensitivity coefficient</p>
        </NC>
        <NC depth="inset" className="text-center">
          <Label>Spec Ratio Cap</Label>
          <p className="text-2xl font-black text-foreground">0.8–1.25×</p>
          <p className="text-[10px] text-muted-foreground mt-0.5">vs competitor spec average</p>
        </NC>
      </div>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════
   7. PRODUCT MARKET FIT
══════════════════════════════════════════════════════════════ */
export function PMFSection({ result }: { result: AnalysisResult }) {
  const { success: suc, demand_forecast: df, clv_per_buyer_usd: clv, market_position: mp } = result

  const isGo      = suc.recommendation.includes("GO FOR")
  const isCaution = suc.recommendation.includes("CAUTION")
  const icon      = isGo ? "✅" : isCaution ? "⚠️" : "🚫"
  const recClass  = isGo
    ? "bg-emerald-950/60 border-emerald-600/40 text-emerald-300"
    : isCaution
    ? "bg-amber-950/60 border-amber-600/40 text-amber-300"
    : "bg-red-950/60 border-red-600/40 text-red-300"

  const pmfScore  = Math.round(suc.pmf_score * 100)
  const pmfColor  = suc.pmf_score >= 0.65 ? "text-emerald-400" : suc.pmf_score >= 0.4 ? "text-amber-400" : "text-red-400"
  const riskColor = suc.failure_risk_pct <= 20 ? "text-emerald-400" : suc.failure_risk_pct <= 45 ? "text-amber-400" : "text-red-400"

  const kpis = [
    { label: "PMF Score",    value: `${pmfScore}/100`,                       color: pmfColor },
    { label: "Failure Risk", value: `${suc.failure_risk_pct}%`,              color: riskColor },
    { label: "6-Mo Demand",  value: df.total_horizon_units.toLocaleString(), color: "text-sky-400" },
    { label: "CLV / Buyer",  value: `$${clv.toLocaleString()}`,              color: "text-emerald-400" },
    { label: "Confidence",   value: `${mp.confidence_pct}%`,                 color: "text-indigo-400" },
    { label: "Segment",      value: mp.predicted_segment.toUpperCase(),      color: "text-purple-400" },
  ]

  return (
    <div>
      <SectionHeader emoji="🎯" title="Product-Market Fit"
        sub="Random Forest failure predictor · PMF composite score · go/no-go verdict" />

      <div className="grid grid-cols-2 gap-4">
        {/* Verdict */}
        <div className={cn("col-span-2 neo-raised rounded-2xl border p-5", recClass)}>
          <div className="flex items-center gap-3">
            <span className="text-3xl">{icon}</span>
            <div>
              <p className="text-2xl font-black tracking-tight leading-none">{suc.recommendation}</p>
              <p className="text-sm opacity-80 mt-1">
                PMF score: {pmfScore}/100 &nbsp;·&nbsp; Failure risk: {suc.failure_risk_pct}%
              </p>
            </div>
          </div>
        </div>

        {/* PMF Gauge — spans 2 rows */}
        <NC depth="raised" className="row-span-2 flex flex-col">
          <Label>Product-Market Fit Gauge</Label>
          <div className="flex-1">
            <PMFGauge score={suc.pmf_score} failureRisk={suc.failure_risk_pct} />
          </div>
          <p className="text-[10px] text-muted-foreground mt-2 leading-relaxed">
            PMF = 0.25×NPS + 0.20×repeat + 0.20×recommend + 0.15×(1−return) + 0.10×(1−warranty) + 0.10×growth_slope.
            Thresholds: &lt;0.15 → GO · &lt;0.35 → CAUTION · else RECONSIDER
          </p>
        </NC>

        {/* PMF score */}
        <NC depth="raised">
          <Label>PMF Score</Label>
          <p className={cn("text-5xl font-black leading-none", pmfColor)}>{pmfScore}</p>
          <p className="text-xs text-muted-foreground mt-1">out of 100</p>
          <div className="mt-3 neo-inset h-2 rounded-full overflow-hidden">
            <div className="h-full rounded-full" style={{
              width: `${pmfScore}%`,
              background: suc.pmf_score >= 0.65 ? "linear-gradient(90deg,#22c55e,#86efac)"
                : suc.pmf_score >= 0.4 ? "linear-gradient(90deg,#f59e0b,#facc15)"
                : "linear-gradient(90deg,#ef4444,#f87171)",
            }} />
          </div>
        </NC>

        {/* Failure risk */}
        <NC depth="raised">
          <Label>Failure Risk</Label>
          <p className={cn("text-5xl font-black leading-none", riskColor)}>{suc.failure_risk_pct}%</p>
          <p className="text-xs text-muted-foreground mt-1">Random Forest failure probability</p>
          <div className="mt-3 neo-inset h-2 rounded-full overflow-hidden">
            <div className="h-full rounded-full" style={{
              width: `${suc.failure_risk_pct}%`,
              background: suc.failure_risk_pct <= 20 ? "linear-gradient(90deg,#22c55e,#86efac)"
                : suc.failure_risk_pct <= 45 ? "linear-gradient(90deg,#f59e0b,#facc15)"
                : "linear-gradient(90deg,#ef4444,#f87171)",
            }} />
          </div>
        </NC>

        {/* KPI tiles */}
        <div className="col-span-2 grid grid-cols-6 gap-2">
          {kpis.map(k => (
            <NC key={k.label} depth="inset" className="p-3 text-center">
              <p className="text-[9px] font-bold uppercase tracking-widest text-muted-foreground">{k.label}</p>
              <p className={cn("text-sm font-black mt-0.5", k.color)}>{k.value}</p>
            </NC>
          ))}
        </div>
      </div>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════
   8. MARKET INTELLIGENCE — Live News
══════════════════════════════════════════════════════════════ */
export function NewsSection({
  news, newsLoading, newsError,
}: {
  news: NewsResult | null
  newsLoading: boolean
  newsError: string | null
}) {
  return (
    <div>
      <SectionHeader emoji="🌐" title="Market Intelligence"
        sub="Live news & articles about your phone, competitors, and market trends" />

      {newsError && (
        <NC depth="flat" className="mb-4">
          <p className="text-sm text-red-400"><strong>News error:</strong> {newsError}</p>
        </NC>
      )}

      <NC depth="raised" className="p-0 overflow-hidden">
        <NewsPanel news={news} loading={newsLoading} error={newsError} />
      </NC>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════
   Legacy exports for backward-compat
══════════════════════════════════════════════════════════════ */
export { PriceSection as MarketSection }

export function Dashboard({
  result, news, newsLoading, newsError, competitors, priceData,
}: {
  result: AnalysisResult
  news: NewsResult | null
  newsLoading: boolean
  newsError: string | null
  competitors: CompetitorResult | null
  priceData: PriceOptimizeResult | null
}) {
  return (
    <main className="p-5 space-y-10">
      <MarketSummary result={result} />
      <DemandSection result={result} />
      <CustomerSection result={result} />
      <AffordabilitySection result={result} />
      <LoyaltySection result={result} />
      <CompetitorSection result={result} competitors={competitors} />
      <PriceSection priceData={priceData} result={result} />
      <PMFSection result={result} />
      <NewsSection news={news} newsLoading={newsLoading} newsError={newsError} />
    </main>
  )
}
