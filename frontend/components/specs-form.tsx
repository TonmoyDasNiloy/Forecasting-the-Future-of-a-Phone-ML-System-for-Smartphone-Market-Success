"use client"

import { useState } from "react"
import { BIAgentLogo } from "@/components/logo"
import { PhoneSpec } from "@/lib/types"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"

const PROC_BRANDS = ["snapdragon","mediatek","apple","exynos","unisoc","kirin","dimensity","helio","other"]
const REGIONS = ["Asia-Pacific","North America","Europe","South Asia","Middle East & Africa","Latin America"]

function Select({ label, value, options, onChange }: {
  label: string; value: string; options: string[]; onChange: (v: string) => void
}) {
  return (
    <div className="space-y-1.5">
      <Label className="text-xs text-muted-foreground uppercase tracking-wider">{label}</Label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-border bg-secondary px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
      >
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  )
}

function NumberInput({ label, value, min, max, step = 1, onChange, testid }: {
  label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void; testid?: string
}) {
  return (
    <div className="space-y-1.5">
      <Label className="text-xs text-muted-foreground uppercase tracking-wider">{label}</Label>
      <input
        type="number"
        data-testid={testid}
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full rounded-lg border border-border bg-secondary px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
      />
    </div>
  )
}

function SliderInput({ label, value, min, max, step = 1, unit, onChange }: {
  label: string; value: number; min: number; max: number; step?: number; unit?: string; onChange: (v: number) => void
}) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <Label className="text-xs text-muted-foreground uppercase tracking-wider">{label}</Label>
        <span className="text-xs font-semibold text-primary">{value}{unit}</span>
      </div>
      <Slider min={min} max={max} step={step} value={[value]} onValueChange={(v) => onChange(Array.isArray(v) ? v[0] : (v as number))} className="w-full" />
    </div>
  )
}

function Toggle({ label, checked, onChange }: {
  label: string; checked: boolean; onChange: (v: boolean) => void
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className={cn(
        "flex w-full items-center justify-between rounded-lg border px-3 py-2 text-sm transition-colors",
        checked
          ? "border-primary bg-primary/10 text-primary"
          : "border-border bg-secondary text-muted-foreground"
      )}
    >
      <span>{label}</span>
      <div className={cn(
        "h-5 w-9 rounded-full transition-colors relative",
        checked ? "bg-primary" : "bg-muted"
      )}>
        <div className={cn(
          "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform",
          checked ? "translate-x-4" : "translate-x-0.5"
        )} />
      </div>
    </button>
  )
}

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <div className="pt-1">
      <p className="text-[10px] font-semibold uppercase tracking-widest text-primary pb-1.5">{children}</p>
      <Separator className="mb-3 opacity-30" />
    </div>
  )
}

const DEFAULTS: PhoneSpec = {
  brand_name: "samsung", model: "", price_usd: 320, region: "Asia-Pacific",
  "5g_or_not": 1, processor_brand: "snapdragon", num_cores: 8, processor_speed: 2.4,
  battery_capacity: 5000, fast_charging_available: 1, fast_charging: 25,
  ram_capacity: 8, internal_memory: 128, screen_size: 6.5, refresh_rate: 120,
  num_rear_cameras: 3, os: "android", primary_camera_rear: 50, primary_camera_front: 16,
  extended_memory_available: 1, resolution_height: 2400, resolution_width: 1080, avg_rating: 7.5,
}

export function SpecsForm({ onSubmit, loading }: {
  onSubmit: (spec: PhoneSpec) => void
  loading: boolean
}) {
  const [spec, setSpec] = useState<PhoneSpec>(DEFAULTS)
  const set = <K extends keyof PhoneSpec>(k: K, v: PhoneSpec[K]) => setSpec((s) => ({ ...s, [k]: v }))

  return (
    <aside className="w-80 shrink-0 overflow-y-auto border-r border-border bg-sidebar flex flex-col">
      {/* Header */}
      <div className="sticky top-0 z-10 border-b border-border bg-sidebar px-4 py-4">
        <div className="flex items-center gap-3">
          {/* Logo */}
          <BIAgentLogo size={52} />
          <div className="flex flex-col leading-tight">
            <h2 className="font-black text-sm tracking-tight text-foreground">
              Launch Profile
            </h2>
            <p className="text-[11px] text-muted-foreground mt-0.5">
              Configure specs · Run AI analysis
            </p>
          </div>
        </div>
      </div>

      <div className="flex-1 px-4 py-3 space-y-4 text-sm">

        {/* Identity */}
        <SectionHeader>Identity</SectionHeader>
        <div className="space-y-1.5">
          <Label className="text-xs text-muted-foreground uppercase tracking-wider">Model Name</Label>
          <input
            value={spec.model}
            onChange={(e) => set("model", e.target.value)}
            placeholder="Enter model name e.g. Galaxy A55"
            className="w-full rounded-lg border border-border bg-secondary px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
          />
        </div>
        <Select label="Target Region" value={spec.region} options={REGIONS} onChange={(v) => set("region", v)} />

        {/* Pricing */}
        <SectionHeader>Pricing</SectionHeader>
        <div className="space-y-1.5">
          <Label className="text-xs text-muted-foreground uppercase tracking-wider">Price (USD)</Label>
          <div className="relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-sm font-bold text-muted-foreground">$</span>
            <input
              type="number"
              min={20}
              max={5000}
              step={1}
              value={spec.price_usd}
              onChange={(e) => {
                const v = parseFloat(e.target.value)
                if (!isNaN(v) && v >= 0) set("price_usd", v)
              }}
              className="w-full rounded-lg border border-border bg-secondary pl-7 pr-3 py-2 text-sm font-semibold text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              placeholder="Enter price e.g. 320"
            />
          </div>
          <p className="text-[10px] text-muted-foreground">
            Enter your intended price — the system analyses it against the market.
          </p>
        </div>

        {/* Connectivity */}
        <SectionHeader>Connectivity & OS</SectionHeader>
        <Toggle label="5G Enabled" checked={!!spec["5g_or_not"]} onChange={(v) => set("5g_or_not", v ? 1 : 0)} />
        <Select
          label="Operating System"
          value={spec.os}
          options={["android", "ios"]}
          onChange={(v) => {
            set("os", v)
            // iOS must use Apple silicon — lock processor automatically
            if (v === "ios") set("processor_brand", "apple")
          }}
        />

        {/* Processor */}
        <SectionHeader>Processor</SectionHeader>
        <Select
          label="Processor"
          value={PROC_BRANDS.filter(p => p !== "apple").includes(spec.processor_brand) || spec.processor_brand === "apple"
            ? spec.processor_brand
            : "other"}
          options={spec.os === "ios" ? ["apple"] : PROC_BRANDS.filter(p => p !== "apple")}
          onChange={(v) => {
            if (v === "other") {
              set("processor_brand", "other")
            } else {
              set("processor_brand", v)
            }
          }}
        />
        {/* Custom processor name input — shown when "other" is selected */}
        {spec.processor_brand === "other" || (!PROC_BRANDS.includes(spec.processor_brand) && spec.processor_brand !== "") ? (
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground uppercase tracking-wider">
              Processor Name
            </Label>
            <input
              type="text"
              placeholder="e.g. Dimensity 9300, Exynos 2400…"
              value={spec.processor_brand === "other" ? "" : spec.processor_brand}
              onChange={(e) => {
                const val = e.target.value.trim()
                set("processor_brand", val === "" ? "other" : val)
              }}
              className="w-full rounded-lg border border-primary/40 bg-secondary px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              autoFocus
            />
            <p className="text-[10px] text-muted-foreground">
              Type the exact processor name for accurate results.
            </p>
          </div>
        ) : null}
        {/* Cores — preset chips + unlimited stepper */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <Label className="text-xs text-muted-foreground uppercase tracking-wider">Cores</Label>
            <span className="text-xs font-bold text-primary">{spec.num_cores} cores</span>
          </div>
          <div className="flex flex-wrap gap-1 items-center">
            {[2,4,6,8,10,12].map(v => (
              <button key={v} data-testid={`cores-${v}`} onClick={() => set("num_cores", v)}
                className={cn("px-2 py-1 rounded text-xs font-medium transition-colors",
                  spec.num_cores === v ? "bg-primary text-primary-foreground" : "bg-secondary text-muted-foreground hover:text-foreground"
                )}>{v}</button>
            ))}
            <div className="flex items-center gap-1 ml-1">
              <button
                onClick={() => set("num_cores", Math.max(1, spec.num_cores - 1))}
                disabled={spec.num_cores <= 1}
                className="h-5 w-5 rounded text-xs font-bold bg-secondary text-muted-foreground hover:text-foreground hover:bg-accent transition-colors disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center"
              >−</button>
              <button
                onClick={() => set("num_cores", spec.num_cores < 12 ? 12 : spec.num_cores + 2)}
                className="h-5 w-5 rounded text-xs font-bold bg-primary/20 text-primary hover:bg-primary/30 transition-colors flex items-center justify-center"
              >+</button>
            </div>
          </div>
        </div>

        {/* Memory */}
        <SectionHeader>Memory</SectionHeader>
        <div className="grid grid-cols-2 gap-3">
          {/* RAM — preset chips + unlimited + button after 16 */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <Label className="text-xs text-muted-foreground uppercase tracking-wider">RAM (GB)</Label>
              <span className="text-xs font-bold text-primary">{spec.ram_capacity} GB</span>
            </div>
            <div className="flex flex-wrap gap-1 items-center">
              {[2,3,4,6,8,12,16].map(v => (
                <button key={v} data-testid={`ram-${v}`} onClick={() => set("ram_capacity", v)}
                  className={cn("px-2 py-1 rounded text-xs font-medium transition-colors",
                    spec.ram_capacity === v ? "bg-primary text-primary-foreground" : "bg-secondary text-muted-foreground hover:text-foreground"
                  )}>{v}</button>
              ))}
              {/* + / - controls beyond 16 GB */}
              <div className="flex items-center gap-1 ml-1">
                <button
                  onClick={() => set("ram_capacity", Math.max(2, spec.ram_capacity - (spec.ram_capacity > 16 ? 4 : 1)))}
                  disabled={spec.ram_capacity <= 16}
                  className="h-5 w-5 rounded text-xs font-bold bg-secondary text-muted-foreground hover:text-foreground hover:bg-accent transition-colors disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center"
                >−</button>
                <button
                  onClick={() => set("ram_capacity", spec.ram_capacity < 16 ? 16 : spec.ram_capacity + 4)}
                  className="h-5 w-5 rounded text-xs font-bold bg-primary/20 text-primary hover:bg-primary/30 transition-colors flex items-center justify-center"
                >+</button>
              </div>
            </div>
          </div>

          {/* Storage — presets up to 2 TB + unlimited + / − beyond */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <Label className="text-xs text-muted-foreground uppercase tracking-wider">Storage</Label>
              <span className="text-xs font-bold text-primary">
                {spec.internal_memory >= 1024
                  ? `${(spec.internal_memory / 1024) % 1 === 0
                      ? spec.internal_memory / 1024
                      : (spec.internal_memory / 1024).toFixed(1)} TB`
                  : `${spec.internal_memory} GB`}
              </span>
            </div>
            <div className="flex flex-wrap gap-1 items-center">
              {[32,64,128,256,512,1024,2048].map(v => (
                <button key={v} data-testid={`storage-${v}`} onClick={() => set("internal_memory", v)}
                  className={cn("px-2 py-1 rounded text-xs font-medium transition-colors",
                    spec.internal_memory === v ? "bg-primary text-primary-foreground" : "bg-secondary text-muted-foreground hover:text-foreground"
                  )}>
                  {v >= 1024 ? `${v / 1024}TB` : v}
                </button>
              ))}
              {/* + / − beyond 2 TB in 512 GB steps */}
              <div className="flex items-center gap-1 ml-1">
                <button
                  onClick={() => set("internal_memory", Math.max(32, spec.internal_memory - (spec.internal_memory > 2048 ? 512 : 32)))}
                  disabled={spec.internal_memory <= 2048}
                  className="h-5 w-5 rounded text-xs font-bold bg-secondary text-muted-foreground hover:text-foreground hover:bg-accent transition-colors disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center"
                >−</button>
                <button
                  onClick={() => set("internal_memory", spec.internal_memory < 2048 ? 2048 : spec.internal_memory + 512)}
                  className="h-5 w-5 rounded text-xs font-bold bg-primary/20 text-primary hover:bg-primary/30 transition-colors flex items-center justify-center"
                >+</button>
              </div>
            </div>
          </div>
        </div>
        <Toggle label="Expandable Storage" checked={!!spec.extended_memory_available} onChange={(v) => set("extended_memory_available", v ? 1 : 0)} />

        {/* Battery */}
        <SectionHeader>Battery</SectionHeader>
        <NumberInput
          label="Capacity (mAh)"
          testid="battery"
          value={spec.battery_capacity}
          min={1000}
          max={30000}
          step={100}
          onChange={(v) => set("battery_capacity", v)}
        />
        <Toggle label="Fast Charging" checked={!!spec.fast_charging_available} onChange={(v) => set("fast_charging_available", v ? 1 : 0)} />
        {!!spec.fast_charging_available && (
          <SliderInput label="Fast Charging (W)" value={spec.fast_charging} min={5} max={120} step={5} unit="W" onChange={(v) => set("fast_charging", v)} />
        )}

        {/* Display */}
        <SectionHeader>Display</SectionHeader>
        <SliderInput label="Screen Size (in)" value={spec.screen_size} min={4.5} max={7.5} step={0.1} unit="″" onChange={(v) => set("screen_size", Math.round(v * 10) / 10)} />
        {/* Refresh rate — presets + stepper beyond 144 */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <Label className="text-xs text-muted-foreground uppercase tracking-wider">Refresh Rate</Label>
            <span className="text-xs font-bold text-primary">{spec.refresh_rate} Hz</span>
          </div>
          <div className="flex flex-wrap gap-1 items-center">
            {[60,90,120,144].map(v => (
              <button key={v} data-testid={`refresh-${v}`} onClick={() => set("refresh_rate", v)}
                className={cn("px-2.5 py-1 rounded text-xs font-semibold transition-colors",
                  spec.refresh_rate === v ? "bg-primary text-primary-foreground" : "bg-secondary text-muted-foreground hover:text-foreground"
                )}>{v}Hz</button>
            ))}
            <div className="flex items-center gap-1 ml-1">
              <button
                onClick={() => set("refresh_rate", Math.max(60, spec.refresh_rate - (spec.refresh_rate > 144 ? 15 : 30)))}
                disabled={spec.refresh_rate <= 144}
                className="h-5 w-5 rounded text-xs font-bold bg-secondary text-muted-foreground hover:text-foreground hover:bg-accent transition-colors disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center"
              >−</button>
              <button
                onClick={() => set("refresh_rate", spec.refresh_rate < 144 ? 144 : spec.refresh_rate + 15)}
                className="h-5 w-5 rounded text-xs font-bold bg-primary/20 text-primary hover:bg-primary/30 transition-colors flex items-center justify-center"
              >+</button>
            </div>
          </div>
        </div>

        {/* Camera */}
        <SectionHeader>Camera</SectionHeader>
        <div className="space-y-1.5">
          <Label className="text-xs text-muted-foreground uppercase tracking-wider">Rear Cameras</Label>
          <div className="flex gap-2">
            {[1,2,3,4,5].map(v => (
              <button key={v} onClick={() => set("num_rear_cameras", v)}
                className={cn("flex-1 py-1.5 rounded text-xs font-semibold transition-colors",
                  spec.num_rear_cameras === v ? "bg-primary text-primary-foreground" : "bg-secondary text-muted-foreground hover:text-foreground"
                )}>{v}</button>
            ))}
          </div>
        </div>
        {/* Main camera — slider + stepper (can exceed slider max) */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <Label className="text-xs text-muted-foreground uppercase tracking-wider">Main Camera</Label>
            <span className="text-xs font-bold text-primary">{spec.primary_camera_rear} MP</span>
          </div>
          <div className="flex items-center gap-2">
            <Slider min={5} max={250} step={5} value={[Math.min(spec.primary_camera_rear, 250)]}
              onValueChange={(v) => set("primary_camera_rear", Array.isArray(v) ? v[0] : (v as number))}
              className="flex-1" />
            <button onClick={() => set("primary_camera_rear", Math.max(5, spec.primary_camera_rear - 10))}
              className="h-5 w-5 shrink-0 rounded text-xs font-bold bg-secondary text-muted-foreground hover:text-foreground hover:bg-accent transition-colors flex items-center justify-center">−</button>
            <button onClick={() => set("primary_camera_rear", spec.primary_camera_rear + 10)}
              className="h-5 w-5 shrink-0 rounded text-xs font-bold bg-primary/20 text-primary hover:bg-primary/30 transition-colors flex items-center justify-center">+</button>
          </div>
        </div>

        {/* Front camera — slider + stepper */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <Label className="text-xs text-muted-foreground uppercase tracking-wider">Front Camera</Label>
            <span className="text-xs font-bold text-primary">{spec.primary_camera_front} MP</span>
          </div>
          <div className="flex items-center gap-2">
            <Slider min={2} max={64} step={2} value={[Math.min(spec.primary_camera_front, 64)]}
              onValueChange={(v) => set("primary_camera_front", Array.isArray(v) ? v[0] : (v as number))}
              className="flex-1" />
            <button onClick={() => set("primary_camera_front", Math.max(2, spec.primary_camera_front - 4))}
              className="h-5 w-5 shrink-0 rounded text-xs font-bold bg-secondary text-muted-foreground hover:text-foreground hover:bg-accent transition-colors flex items-center justify-center">−</button>
            <button onClick={() => set("primary_camera_front", spec.primary_camera_front + 4)}
              className="h-5 w-5 shrink-0 rounded text-xs font-bold bg-primary/20 text-primary hover:bg-primary/30 transition-colors flex items-center justify-center">+</button>
          </div>
        </div>

        <div className="pb-2" />
      </div>

      {/* Sticky submit */}
      <div className="sticky bottom-0 border-t border-border bg-sidebar p-4">
        <button
          type="button"
          disabled={loading || spec.model.trim() === ""}
          onClick={() => onSubmit(spec)}
          className="w-full rounded-xl bg-linear-to-r from-primary to-violet-600 py-3 text-sm font-bold text-white shadow-lg transition-all hover:opacity-90 active:scale-95 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              Analyzing…
            </span>
          ) : (
    <span className="flex items-center justify-center gap-2">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth="2.5"
        strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8" />
        <path d="m21 21-4.35-4.35" />
      </svg>
      Analyze This Phone
    </span>
  )}
        </button>
      </div>
    </aside>
  )
}
