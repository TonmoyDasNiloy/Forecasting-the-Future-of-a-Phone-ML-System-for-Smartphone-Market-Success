"use client"

interface PMFGaugeProps {
  score: number
  failureRisk: number
}

export function PMFGauge({ score, failureRisk }: PMFGaugeProps) {
  const pct     = Math.round(score * 100)
  const color   = pct >= 65 ? "#22c55e" : pct >= 40 ? "#f59e0b" : "#ef4444"
  const bgTrack = pct >= 65 ? "#14532d33" : pct >= 40 ? "#78350f33" : "#7f1d1d33"

  // SVG semicircle gauge (180° arc, left→right)
  const W = 280
  const H = 160      // half of W so the arc sits in the top half
  const cx = W / 2
  const cy = H        // centre of the full circle sits at the bottom edge
  const R  = 110      // radius

  // Arc path helper: draws an arc of `angleDeg` degrees (0 = left, 180 = right)
  const arcPath = (angleDeg: number) => {
    const rad     = (angleDeg / 180) * Math.PI
    const x       = cx - R * Math.cos(rad)
    const y       = cy - R * Math.sin(rad)
    const large   = angleDeg > 180 ? 1 : 0
    return `M ${cx - R} ${cy} A ${R} ${R} 0 ${large} 1 ${x} ${y}`
  }

  const filled = (pct / 100) * 180

  return (
    <div className="flex flex-col items-center gap-4 py-2">

      {/* Gauge SVG */}
      <div className="relative w-full max-w-xs">
        <svg viewBox={`0 0 ${W} ${H + 10}`} className="w-full overflow-visible">
          {/* Background track */}
          <path
            d={arcPath(180)}
            fill="none"
            stroke={bgTrack}
            strokeWidth={18}
            strokeLinecap="round"
          />
          {/* Filled arc */}
          {pct > 0 && (
            <path
              d={arcPath(filled)}
              fill="none"
              stroke={color}
              strokeWidth={18}
              strokeLinecap="round"
            />
          )}
        </svg>

        {/* Score — sits inside the arc opening, no overlap */}
        <div className="absolute inset-x-0 bottom-0 flex flex-col items-center pb-1">
          <span className="text-4xl font-black leading-none" style={{ color }}>{pct}%</span>
          <span className="text-xs text-muted-foreground mt-1 font-medium tracking-wide">PMF Score</span>
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 gap-3 w-full">
        <div className="rounded-xl bg-secondary border border-border px-3 py-3 text-center">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Failure Risk</p>
          <p className="text-xl font-black text-destructive">{failureRisk}%</p>
        </div>
        <div className="rounded-xl bg-secondary border border-border px-3 py-3 text-center">
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">PMF Score</p>
          <p className="text-xl font-black" style={{ color }}>{pct}%</p>
        </div>
      </div>
    </div>
  )
}
