"use client"

import { useState } from "react"
import { CompetitorResult, CompetitorPhone, SpecMeta } from "@/lib/types"
import { Treemap, ResponsiveContainer, Tooltip } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"

const PALETTE = [
  "#6366f1",  // indigo  — your phone
  "#f59e0b",  // amber   — comp 1
  "#22c55e",  // emerald — comp 2
  "#ef4444",  // red     — comp 3
]

const SPEC_CONFIG = [
  { key: "ram_capacity",        label: "RAM",     unit: "GB",   scale: 16   },
  { key: "battery_capacity",    label: "Battery", unit: "mAh",  scale: 7000 },
  { key: "primary_camera_rear", label: "Camera",  unit: "MP",   scale: 250  },
  { key: "refresh_rate",        label: "Refresh", unit: "Hz",   scale: 144  },
  { key: "screen_size",         label: "Screen",  unit: "in",   scale: 7.5  },
  { key: "processor_speed",     label: "CPU",     unit: "GHz",  scale: 4    },
]

function fmtVal(v: number, unit: string) {
  if (unit === "in")  return `${v.toFixed(1)}"`
  if (unit === "GHz") return `${v.toFixed(1)}GHz`
  return `${Number.isInteger(v) ? v.toLocaleString() : v.toFixed(1)}${unit}`
}

/* ── Spec modal ─────────────────────────────────────────────── */
function SpecsModal({ comp, yours, meta, onClose }: {
  comp: CompetitorPhone
  yours: Record<string, number>
  meta: SpecMeta[]
  onClose: () => void
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" />
      <div
        className="relative z-10 w-full max-w-2xl max-h-[80vh] overflow-y-auto rounded-2xl border border-border bg-card shadow-2xl"
        onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between border-b border-border px-5 py-4">
          <div>
            <h3 className="font-bold text-base text-foreground">{comp.name}</h3>
            <p className="text-xs text-muted-foreground mt-0.5">Full spec comparison</p>
          </div>
          <button onClick={onClose}
            className="rounded-lg p-2 text-muted-foreground hover:bg-secondary transition-colors">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div className="p-5">
          <div className="grid grid-cols-3 text-[11px] font-bold uppercase tracking-wider text-muted-foreground mb-3 px-2">
            <span>Spec</span>
            <span className="text-center" style={{ color: PALETTE[0] }}>Your Phone</span>
            <span className="text-center" style={{ color: PALETTE[1] }}>{comp.name.split(" ").slice(0, 2).join(" ")}</span>
          </div>
          <div className="space-y-1">
            {meta.map(({ key, label, unit }) => {
              const y = yours[key] ?? 0
              const c = comp.specs[key] ?? 0
              if (y === 0 && c === 0) return null
              const fmt = (v: number) => {
                if (key === "5g_or_not") return v ? "Yes" : "No"
                if (key === "price_usd") return `$${v.toLocaleString()}`
                if (unit === "in") return `${v.toFixed(1)}"`
                return `${Number.isInteger(v) ? v : v.toFixed(1)}${unit ? " " + unit : ""}`
              }
              return (
                <div key={key} className="grid grid-cols-3 items-center rounded-lg px-2 py-2.5 hover:bg-secondary transition-colors">
                  <span className="text-xs font-medium text-foreground">{label}</span>
                  <span className={cn("text-center text-sm font-bold",
                    y > c + 0.01 ? "text-emerald-700 dark:text-emerald-400" : "text-foreground")}>{fmt(y)}</span>
                  <span className={cn("text-center text-sm font-bold",
                    c > y + 0.01 ? "text-emerald-700 dark:text-emerald-400" : "text-foreground")}>{fmt(c)}</span>
                </div>
              )
            })}
          </div>
          <div className="mt-4 rounded-xl bg-secondary border border-border p-3">
            {(() => {
              let wins = 0, losses = 0
              meta.forEach(({ key }) => {
                if (key === "5g_or_not" || key === "price_usd") return
                const yv = yours[key] ?? 0, cv = comp.specs[key] ?? 0
                if (yv > cv + 0.01) wins++
                else if (cv > yv + 0.01) losses++
              })
              return (
                <div className="flex gap-4">
                  <span className="text-xs font-semibold text-emerald-700 dark:text-emerald-400">✅ Lead on {wins} specs</span>
                  <span className="text-xs font-semibold text-red-700 dark:text-red-400">⚠️ Behind on {losses} specs</span>
                </div>
              )
            })()}
          </div>
        </div>
      </div>
    </div>
  )
}

/* ── Main component ─────────────────────────────────────────── */
interface Props { data: CompetitorResult }

export function CompetitorRadar({ data }: Props) {
  const [activeModal, setActiveModal] = useState<CompetitorPhone | null>(null)

  const allPhones = [
    { name: "Your Phone", specs: data.your_phone.specs, index: 0 },
    ...data.competitors.map((c, i) => ({ name: c.name, specs: c.specs, index: i + 1 })),
  ]

  // Flat treemap: one tile per phone×spec, sized by normalised value with a
  // high floor so even small specs keep a readable tile.
  const treeData = allPhones.flatMap(phone =>
    SPEC_CONFIG.map(sc => {
      const raw  = phone.specs[sc.key] ?? 0
      const norm = Math.max(0.35, Math.min(1, raw / sc.scale))
      return {
        name:        `${phone.name} · ${sc.label}`,
        phoneName:   phone.name,
        phoneIndex:  phone.index,
        specLabel:   sc.label,
        actualValue: Number.isInteger(raw) ? raw : parseFloat(raw.toFixed(1)),
        unit:        sc.unit,
        size:        Math.round(norm * 100),
      }
    })
  )

  // Defensive cell renderer — skips root/invalid nodes (never crashes)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const renderCell = (props: any) => {
    const { x, y, width, height, phoneIndex, specLabel, actualValue, unit } = props
    if (
      specLabel == null ||
      typeof x !== "number" || typeof y !== "number" ||
      typeof width !== "number" || typeof height !== "number" ||
      width < 2 || height < 2
    ) {
      return <g />
    }
    const color    = PALETTE[phoneIndex ?? 0] ?? PALETTE[0]
    const isYours  = phoneIndex === 0
    const canLabel = width > 36 && height > 24
    const canValue = width > 46 && height > 38
    const fs       = Math.max(8, Math.min(13, Math.floor(width / 5)))

    return (
      <g>
        <rect x={x + 1.5} y={y + 1.5} width={Math.max(0, width - 3)} height={Math.max(0, height - 3)}
          fill={color} fillOpacity={isYours ? 0.9 : 0.6}
          stroke="white" strokeWidth={1.5} rx={4} />
        {canLabel && (
          <text x={x + width / 2} y={y + height / 2 + (canValue ? -6 : 4)}
            textAnchor="middle" dominantBaseline="middle"
            fill="white" fontSize={fs} fontWeight={700}>
            {specLabel}
          </text>
        )}
        {canValue && (
          <text x={x + width / 2} y={y + height / 2 + 10}
            textAnchor="middle" dominantBaseline="middle"
            fill="rgba(255,255,255,0.92)" fontSize={Math.max(7, fs - 2)} fontWeight={600}>
            {fmtVal(actualValue, unit)}
          </text>
        )}
      </g>
    )
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const renderTooltip = ({ active, payload }: any) => {
    if (!active || !payload?.length) return null
    const d = payload[0]?.payload
    if (!d?.specLabel) return null
    return (
      <div className="rounded-xl border border-border bg-popover px-3 py-2 shadow-lg text-xs">
        <div className="flex items-center gap-2 mb-1">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: PALETTE[d.phoneIndex] }} />
          <span className="font-bold text-foreground">{d.phoneName}</span>
        </div>
        <span className="text-muted-foreground">{d.specLabel}: </span>
        <span className="font-bold text-foreground">{fmtVal(d.actualValue, d.unit)}</span>
      </div>
    )
  }

  return (
    <>
      {activeModal && (
        <SpecsModal
          comp={activeModal}
          yours={data.your_phone.specs}
          meta={data.spec_meta}
          onClose={() => setActiveModal(null)}
        />
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">⚔️ Competitor Gap Analysis</CardTitle>
          <p className="text-xs text-muted-foreground mt-0.5">
            Treemap — each section is one phone, larger tiles = stronger specs within it. Hover any tile for the exact value.
          </p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">

            {/* ── Treemap ── */}
            <div className="lg:col-span-2">
              {/* Legend */}
              <div className="flex flex-wrap gap-2 mb-3">
                {allPhones.map(p => (
                  <span key={p.name}
                    className="flex items-center gap-1.5 rounded-full border border-border bg-secondary px-3 py-1 text-[11px] font-semibold text-foreground">
                    <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ background: PALETTE[p.index] }} />
                    {p.name === "Your Phone" ? "Your Phone" : p.name.split(" ").slice(0, 2).join(" ")}
                  </span>
                ))}
              </div>

              <ResponsiveContainer width="100%" height={360}>
                <Treemap
                  data={treeData}
                  dataKey="size"
                  aspectRatio={4 / 3}
                  stroke="transparent"
                  content={renderCell}
                  isAnimationActive
                  animationDuration={500}
                  animationEasing="ease-out"
                >
                  <Tooltip content={renderTooltip} />
                </Treemap>
              </ResponsiveContainer>

              {/* Spec legend */}
              <div className="flex flex-wrap justify-center gap-2 mt-3">
                {SPEC_CONFIG.map(sc => (
                  <span key={sc.key}
                    className="rounded-full bg-secondary border border-border px-2.5 py-0.5 text-[10px] font-medium text-muted-foreground">
                    {sc.label} ({sc.unit})
                  </span>
                ))}
              </div>
            </div>

            {/* ── Right panel ── */}
            <div className="flex flex-col gap-4">
              <div>
                <p className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground mb-2">
                  Spec Differences
                </p>
                {data.insights.length === 0 ? (
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    Specs are closely matched — no significant differences.
                  </p>
                ) : (
                  <div className="flex flex-col gap-2">
                    {data.insights.map((ins, i) => (
                      <div key={i} className={cn(
                        "flex items-start gap-2 rounded-lg border px-3 py-2 text-xs font-medium",
                        ins.type === "win"
                          ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                          : "border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-400"
                      )}>
                        <span className="mt-px shrink-0">{ins.type === "win" ? "✅" : "⚠️"}</span>
                        <span className="leading-snug">{ins.text}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div>
                <p className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground mb-2">
                  Competitors — click for full specs
                </p>
                <div className="flex flex-col gap-2">
                  {data.competitors.map((c, i) => (
                    <button key={i} onClick={() => setActiveModal(c)}
                      className="flex items-center gap-3 rounded-xl border border-border bg-secondary px-4 py-3 text-left transition-all hover:border-primary/40 hover:bg-accent hover:shadow-sm group"
                      style={{ borderLeftWidth: 3, borderLeftColor: PALETTE[i + 1] }}>
                      <span className="h-3 w-3 rounded-full shrink-0" style={{ background: PALETTE[i + 1] }} />
                      <span className="flex-1 text-xs font-semibold text-foreground leading-snug line-clamp-2">{c.name}</span>
                      <div className="flex flex-col items-end gap-0.5 shrink-0">
                        <span className="text-sm font-black" style={{ color: PALETTE[i + 1] }}>
                          ${c.price.toLocaleString()}
                        </span>
                        <span className="text-[10px] text-muted-foreground group-hover:text-primary transition-colors">
                          View specs →
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>

          </div>
        </CardContent>
      </Card>
    </>
  )
}
