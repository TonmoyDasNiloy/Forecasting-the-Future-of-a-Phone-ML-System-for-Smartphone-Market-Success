"use client"

interface LoyaltyDivergingProps {
  rfm: {
    rfm_recent_pct: number
    rfm_champion_pct: number
    rfm_loyal_pct: number
    rfm_at_risk_pct: number
    rfm_hibernating_pct: number
  }
}

export function LoyaltyDiverging({ rfm }: LoyaltyDivergingProps) {
  const healthy = rfm.rfm_champion_pct + rfm.rfm_loyal_pct + rfm.rfm_recent_pct
  const risk    = rfm.rfm_at_risk_pct + rfm.rfm_hibernating_pct
  const healthyPct = Math.round((healthy / (healthy + risk || 1)) * 100)
  const riskPct    = 100 - healthyPct

  return (
    <div className="space-y-2">
      {/* Engaged vs at-risk split bar */}
      <div className="flex h-7 rounded-lg overflow-hidden border border-border">
        <div className="h-full bg-emerald-500 flex items-center justify-center" style={{ width: `${healthyPct}%` }}>
          {healthyPct >= 16 && <span className="text-[11px] font-bold text-white">{healthyPct}%</span>}
        </div>
        <div className="h-full bg-red-500 flex items-center justify-center" style={{ width: `${riskPct}%` }}>
          {riskPct >= 16 && <span className="text-[11px] font-bold text-white">{riskPct}%</span>}
        </div>
      </div>
      <div className="flex justify-between text-xs">
        <span className="font-semibold text-emerald-600 dark:text-emerald-400">{healthyPct}% engaged</span>
        <span className="font-semibold text-red-600 dark:text-red-400">{riskPct}% at-risk</span>
      </div>
    </div>
  )
}
