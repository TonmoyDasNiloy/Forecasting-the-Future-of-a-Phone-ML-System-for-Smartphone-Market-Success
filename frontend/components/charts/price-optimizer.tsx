"use client"

import { PriceOptimizeResult } from "@/lib/types"
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip,
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface Props { data: PriceOptimizeResult }

export function PriceOptimizer({ data }: Props) {
  const recLabel = `$${data.recommended_price.toLocaleString()}`
  const curLabel = `$${data.current_price.toLocaleString()}`

  const rec = data.curve.find(p => p.price === data.recommended_price)
  const cur = data.curve.find(p => p.price === data.current_price) ?? data.curve[0]

  const diff = data.recommended_price - data.current_price
  const dir  = diff > 0 ? "raise" : diff < 0 ? "lower" : "keep"
  const nearOptimal = Math.abs(diff) / Math.max(1, data.recommended_price) < 0.03

  const profitUplift = rec && cur && cur.profit > 0
    ? Math.round(((rec.profit - cur.profit) / cur.profit) * 100)
    : 0

  const vsMarket = Math.round(((data.recommended_price - data.market_price) / Math.max(1, data.market_price)) * 100)

  const specStrength =
    data.spec_ratio > 1.03 ? `${Math.round((data.spec_ratio - 1) * 100)}% stronger`
    : data.spec_ratio < 0.97 ? `${Math.round((1 - data.spec_ratio) * 100)}% weaker`
    : "on par"

  // ── Radar: Your Price vs Recommended Price across normalised metrics ──
  const maxDemand  = Math.max(...data.curve.map(p => p.demand), 1)
  const maxRevenue = Math.max(...data.curve.map(p => p.revenue), 1)
  const maxProfit  = Math.max(...data.curve.map(p => p.profit), 1)

  const norm = (p: typeof rec) => ({
    Demand:    p ? Math.round((p.demand / maxDemand) * 100) : 0,
    Revenue:   p ? Math.round((p.revenue / maxRevenue) * 100) : 0,
    Profit:    p ? Math.round((Math.max(0, p.profit) / maxProfit) * 100) : 0,
    "PMF Fit": p ? Math.round(p.pmf_score * 100) : 0,
    "Low Risk": p ? Math.round(100 - p.failure_risk) : 0,
  })

  const curN = norm(cur)
  const recN = norm(rec)
  const AXES = ["Demand", "Revenue", "Profit", "PMF Fit", "Low Risk"] as const

  const radarData = AXES.map(axis => ({
    axis,
    current:     curN[axis],
    recommended: recN[axis],
  }))

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const renderTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null
    return (
      <div className="rounded-xl border border-border bg-popover px-3 py-2 shadow-lg text-xs">
        <p className="font-bold text-foreground mb-1.5">{label}</p>
        {payload.map((p: { dataKey: string; value: number; color: string }, i: number) => (
          <div key={i} className="flex items-center justify-between gap-4 mb-0.5">
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
              <span className="text-muted-foreground">{p.dataKey === "current" ? `Your ${curLabel}` : `Optimal ${recLabel}`}</span>
            </span>
            <span className="font-bold text-foreground">{p.value}/100</span>
          </div>
        ))}
      </div>
    )
  }

  const reasons: { icon: string; title: string; body: string }[] = [
    {
      icon: "🏷️",
      title: "Matches the competitive market",
      body: `The ${data.competitors.length} most spec-similar phones sell for $${data.competitor_min.toLocaleString()}–$${data.competitor_max.toLocaleString()} (median $${data.market_price.toLocaleString()}). ${recLabel} sits ${vsMarket >= 0 ? `${vsMarket}% above` : `${Math.abs(vsMarket)}% below`} that median — a position buyers will accept.`,
    },
    {
      icon: "⚙️",
      title: "Reflects your spec strength",
      body: `Your specs are ${specStrength} than these rivals, justifying a fair price of $${data.fair_price.toLocaleString()}. Pricing here captures that advantage without overreaching.`,
    },
    {
      icon: "💰",
      title: "Maximises profit, not just sales",
      body: `Profit = (price − unit cost $${data.assumptions.unit_cost.toLocaleString()}) × demand. ${recLabel} is the exact point where this trade-off yields the highest profit${rec ? ` (≈ $${rec.profit.toLocaleString()})` : ""}${profitUplift > 0 ? `, +${profitUplift}% vs your current price` : ""}.`,
    },
    {
      icon: "🎯",
      title: "Keeps product-market fit healthy",
      body: `At ${recLabel} the PMF score stays strong (${rec ? Math.round(rec.pmf_score * 100) : "—"}%). The recommendation is profit weighted by PMF, so it never chases profit at the cost of poor fit.`,
    },
    {
      icon: "📈",
      title: "Profit-optimal, not revenue-optimal",
      body: `Peak revenue would be at $${data.peak_revenue_price.toLocaleString()}, but that sells more units at thinner margins. ${recLabel} earns more actual profit — the metric that matters for a launch.`,
    },
  ]

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">💰 Optimal Price Recommender</CardTitle>
        <p className="text-xs text-muted-foreground mt-0.5">
          Anchored to <span className="font-semibold text-foreground">competitor pricing</span> (not your input),
          so the recommendation is stable. Assumes {data.assumptions.gross_margin_pct}% gross margin
          and a price-elasticity demand curve.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">

        {/* ── Summary cards ── */}
        <div className="grid grid-cols-3 gap-3">
          <div className="rounded-xl bg-secondary border border-border px-4 py-3 text-center">
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Your Price</p>
            <p className="text-xl font-black text-foreground">{curLabel}</p>
            <p className="text-[11px] text-muted-foreground mt-0.5">Profit ≈ ${(cur?.profit ?? 0).toLocaleString()}</p>
          </div>
          <div className="rounded-xl bg-primary/10 border border-primary/30 px-4 py-3 text-center">
            <p className="text-[10px] text-primary uppercase tracking-wider mb-1 font-semibold">Recommended</p>
            <p className="text-xl font-black text-primary">{recLabel}</p>
            <p className="text-[11px] text-primary/70 mt-0.5">Profit ≈ ${(rec?.profit ?? 0).toLocaleString()}</p>
          </div>
          <div className={`rounded-xl border px-4 py-3 text-center
            ${nearOptimal ? "bg-secondary border-border"
              : dir === "raise" ? "bg-emerald-500/10 border-emerald-500/30"
              : "bg-amber-500/10 border-amber-500/30"}`}>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Suggestion</p>
            <p className={`text-sm font-black ${nearOptimal ? "text-muted-foreground" : dir === "raise" ? "text-emerald-600 dark:text-emerald-500" : "text-amber-600 dark:text-amber-500"}`}>
              {nearOptimal ? "✓ Near optimal"
                : dir === "raise" ? `↑ Raise $${Math.abs(diff).toLocaleString()}`
                : `↓ Lower $${Math.abs(diff).toLocaleString()}`}
            </p>
            <p className="text-[11px] text-muted-foreground mt-0.5">
              {nearOptimal ? "Well priced" : profitUplift > 0 ? `+${profitUplift}% profit` : "Best profit + fit"}
            </p>
          </div>
        </div>

        {/* ── Radar: Your price vs Recommended price ── */}
        <div className="rounded-xl border border-border bg-secondary/40 p-3">
          <div className="flex items-center justify-between mb-1">
            <p className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground">
              Your Price vs Optimal — outcome comparison
            </p>
            <div className="flex items-center gap-3 text-[11px]">
              <span className="flex items-center gap-1 text-red-600 dark:text-red-400 font-semibold">
                <span className="h-2.5 w-2.5 rounded-full bg-red-500" /> Your {curLabel}
              </span>
              <span className="flex items-center gap-1 text-primary font-semibold">
                <span className="h-2.5 w-2.5 rounded-full bg-primary" /> Optimal {recLabel}
              </span>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData} outerRadius="70%" margin={{ top: 16, right: 30, bottom: 16, left: 30 }}>
              <PolarGrid gridType="polygon" stroke="rgba(148,163,184,0.55)" strokeWidth={1.2} />
              <PolarAngleAxis dataKey="axis" tick={{ fill: "var(--foreground)", fontSize: 12, fontWeight: 700 }} />
              <PolarRadiusAxis angle={90} domain={[0, 100]} tickCount={5}
                tick={{ fill: "var(--muted-foreground)", fontSize: 9 }} axisLine={false}
                stroke="rgba(148,163,184,0.4)" />
              <Radar name="current" dataKey="current" stroke="#ef4444" strokeWidth={2.5}
                fill="#ef4444" fillOpacity={0.14}
                dot={{ r: 3, fill: "#ef4444", strokeWidth: 0 }}
                isAnimationActive animationDuration={550} animationEasing="ease-out" />
              <Radar name="recommended" dataKey="recommended" stroke="#6366f1" strokeWidth={3}
                fill="#6366f1" fillOpacity={0.22}
                dot={{ r: 3.5, fill: "#6366f1", strokeWidth: 0 }}
                isAnimationActive animationDuration={650} animationEasing="ease-out" />
              <Tooltip content={renderTooltip} />
            </RadarChart>
          </ResponsiveContainer>
          <p className="text-[11px] text-muted-foreground text-center -mt-1">
            All five outcomes scored 0–100. A larger indigo web means the optimal price wins on more fronts.
          </p>
        </div>

        {/* ── Why this price is best — reasoning ── */}
        <div>
          <p className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground mb-2">
            Why {recLabel} is the best price
          </p>
          <div className="space-y-2">
            {reasons.map((r, i) => (
              <div key={i} className="flex items-start gap-3 rounded-xl border border-border bg-secondary/50 px-4 py-3">
                <span className="text-base leading-none mt-0.5 shrink-0">{r.icon}</span>
                <div>
                  <p className="text-xs font-bold text-foreground">{r.title}</p>
                  <p className="text-xs text-muted-foreground leading-relaxed mt-0.5">{r.body}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

      </CardContent>
    </Card>
  )
}
