"use client"

import { DemandForecast } from "@/lib/types"
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Line, ComposedChart, ResponsiveContainer, LabelList,
} from "recharts"

export function DemandChart({ forecast }: { forecast: DemandForecast }) {
  const data = [1, 2, 3, 4, 5, 6].map((m) => ({
    month: `M${m}`,
    units: forecast[`month_${m}` as keyof DemandForecast] as number,
  }))

  return (
    <ResponsiveContainer width="100%" height={220}>
      <ComposedChart data={data} margin={{ top: 24, right: 12, left: -8, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
        <XAxis dataKey="month" tick={{ fill: "#94a3b8", fontSize: 12 }} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={false} tickLine={false} />
        <Tooltip
          cursor={{ fill: "rgba(255,255,255,0.04)" }}
          contentStyle={{ background: "#1e2337", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, fontSize: 12 }}
          labelStyle={{ color: "#94a3b8" }}
          itemStyle={{ color: "#e2e8f0" }}
          formatter={(v) => [`${v} units`, "Forecast"]}
        />
        <Bar dataKey="units" fill="#4f6ef7" radius={[6, 6, 0, 0]} maxBarSize={48}
          isAnimationActive animationDuration={500} animationEasing="ease-out">
          <LabelList dataKey="units" position="top" style={{ fill: "#94a3b8", fontSize: 11 }} />
        </Bar>
        <Line type="monotone" dataKey="units" stroke="#818cf8" strokeWidth={2} strokeDasharray="4 2"
          dot={{ fill: "#818cf8", r: 4 }} activeDot={{ r: 6 }}
          isAnimationActive animationDuration={500} animationEasing="ease-out" />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
