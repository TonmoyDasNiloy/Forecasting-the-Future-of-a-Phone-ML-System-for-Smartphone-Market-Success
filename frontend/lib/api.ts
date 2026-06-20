import { PhoneSpec, AnalysisResult, NewsResult, CompetitorResult, PriceOptimizeResult } from "./types"

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

export async function analyzePhone(spec: PhoneSpec): Promise<AnalysisResult> {
  const payload = {
    ...spec,
    "5g_or_not": spec["5g_or_not"],
  }

  const res = await fetch(`${BASE_URL}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? "Analysis failed")
  }

  return res.json()
}

export async function fetchCompetitors(spec: PhoneSpec): Promise<CompetitorResult> {
  const res = await fetch(`${BASE_URL}/api/competitors`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(spec),
  })
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? "Competitor fetch failed")
  return res.json()
}

export async function fetchPriceOptimize(spec: PhoneSpec): Promise<PriceOptimizeResult> {
  const res = await fetch(`${BASE_URL}/api/price-optimize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(spec),
  })
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? "Price optimize failed")
  return res.json()
}

export async function fetchMarketNews(spec: PhoneSpec): Promise<NewsResult> {
  const res = await fetch(`${BASE_URL}/api/news`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(spec),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? "News fetch failed")
  }
  return res.json()
}
