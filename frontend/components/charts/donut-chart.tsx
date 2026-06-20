"use client"

import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from "recharts"

interface DonutChartProps {
  data: { name: string; value: number; color: string }[]
  centerLabel?: string
  centerValue?: string
}

export function DonutChart({ data, centerLabel, centerValue }: DonutChartProps) {
  return (
    <div className="relative">
      <ResponsiveContainer width="100%" height={200}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={52}
            outerRadius={78}
            paddingAngle={2}
            dataKey="value"
            isAnimationActive
            animationDuration={500}
            animationEasing="ease-out"
          >
            {data.map((entry, i) => (
              <Cell key={i} fill={entry.color} stroke="transparent" />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
            itemStyle={{ color: "var(--foreground)" }}
            formatter={(v) => [typeof v === "number" ? `${v.toFixed(1)}%` : `${v}%`]}
          />
          <Legend
            iconType="circle"
            iconSize={8}
            wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
            formatter={(value) => <span style={{ color: "var(--muted-foreground)" }}>{value}</span>}
          />
        </PieChart>
      </ResponsiveContainer>

      {/* Center label — dominant group, sits in the donut hole */}
      {centerLabel && (
        <div className="pointer-events-none absolute inset-x-0 top-0 flex flex-col items-center justify-center"
          style={{ height: 200 }}>
          <span className="text-lg font-black text-foreground leading-none">{centerValue}</span>
          <span className="text-[10px] text-muted-foreground mt-0.5 max-w-20 text-center leading-tight">{centerLabel}</span>
        </div>
      )}
    </div>
  )
}
