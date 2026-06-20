export interface PhoneSpec {
  brand_name: string
  model: string
  price_usd: number
  region: string
  "5g_or_not": number
  processor_brand: string
  num_cores: number
  processor_speed: number
  battery_capacity: number
  fast_charging_available: number
  fast_charging: number
  ram_capacity: number
  internal_memory: number
  screen_size: number
  refresh_rate: number
  num_rear_cameras: number
  os: string
  primary_camera_rear: number
  primary_camera_front: number
  extended_memory_available: number
  resolution_height: number
  resolution_width: number
  avg_rating: number
}

export interface MarketPosition {
  predicted_segment: "budget" | "mid-range" | "premium" | "flagship"
  confidence_pct: number
  closest_competitors: string[]
  price_vs_category_pct: number
}

export interface AgeProfile {
  age_18_24_pct: number
  age_25_34_pct: number
  age_35_44_pct: number
  age_45_54_pct: number
  age_55_plus_pct: number
}

export interface IncomeProfile {
  income_low_pct: number
  income_lower_mid_pct: number
  income_mid_pct: number
  income_upper_mid_pct: number
  income_high_pct: number
}

export interface RfmProfile {
  rfm_recent_pct: number
  rfm_champion_pct: number
  rfm_loyal_pct: number
  rfm_at_risk_pct: number
  rfm_hibernating_pct: number
}

export interface GenderProfile {
  gender_male_pct: number
  gender_female_pct: number
  gender_nonbinary_pct: number
}

export interface CustomerProfile {
  age: AgeProfile
  income: IncomeProfile
  rfm: RfmProfile
  gender: GenderProfile
}

export interface DemandForecast {
  month_1: number
  month_2: number
  month_3: number
  month_4: number
  month_5: number
  month_6: number
  total_horizon_units: number
}

export interface SuccessMetrics {
  pmf_score: number
  failure_risk_pct: number
  recommendation: string
}

// ── Competitor Gap ────────────────────────────────────────────
export interface SpecMeta { key: string; label: string; unit: string }

export interface CompetitorPhone {
  name: string
  scores: number[]
  specs: Record<string, number>
  price: number
  distance: number
}

export interface CompetitorResult {
  axes: string[]
  your_phone: { name: string; scores: number[]; specs: Record<string, number> }
  competitors: CompetitorPhone[]
  insights: { type: "win" | "loss"; text: string }[]
  spec_meta: SpecMeta[]
}

// ── Price Optimizer ───────────────────────────────────────────
export interface PricePoint {
  price: number
  demand: number
  revenue: number
  profit: number
  pmf_score: number
  failure_risk: number
  recommendation: string
}

export interface PriceAssumptions {
  gross_margin_pct: number
  unit_cost: number
  demand_sensitivity: number
}

export interface PriceCompetitor {
  name: string
  price: number
}

export interface PriceOptimizeResult {
  curve: PricePoint[]
  recommended_price: number
  current_price: number
  peak_revenue_price: number
  market_price: number
  competitor_min: number
  competitor_max: number
  fair_price: number
  spec_ratio: number
  competitors: PriceCompetitor[]
  assumptions: PriceAssumptions
}

export interface NewsArticle {
  title: string
  url: string
  snippet: string
  source: string
  date: string | null
  tag: "Your Phone" | "Competitor" | "Market"
}

export interface NewsResult {
  articles: NewsArticle[]
  queries_used: string[]
  distribution?: Record<string, number>
}

export interface AnalysisResult {
  input: {
    brand: string
    model: string
    price_usd: number
    region: string
  }
  market_position: MarketPosition
  customer_profile: CustomerProfile
  demand_forecast: DemandForecast
  success: SuccessMetrics
  clv_per_buyer_usd: number
}
