"use client"

import { useState } from "react"
import { NewsArticle, NewsResult } from "@/lib/types"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"

const TAG_STYLES: Record<string, string> = {
  "Your Phone": "bg-indigo-500/15 text-indigo-400 border-indigo-500/30",
  Competitor:   "bg-amber-500/15  text-amber-400  border-amber-500/30",
}

const TAG_ICONS: Record<string, string> = {
  "Your Phone": "📱",
  Competitor:   "⚔️",
}

function ArticleCard({ article }: { article: NewsArticle }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <a
      href={article.url}
      target="_blank"
      rel="noopener noreferrer"
      className="group block rounded-xl border border-border bg-secondary/50 p-4 transition-all hover:border-primary/40 hover:bg-secondary hover:shadow-lg hover:shadow-primary/5"
    >
      {/* top row */}
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={cn(
            "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider",
            TAG_STYLES[article.tag] ?? "bg-muted text-muted-foreground border-border"
          )}>
            {TAG_ICONS[article.tag]} {article.tag}
          </span>
          <span className="text-[11px] text-muted-foreground">{article.source}</span>
          {article.date && (
            <span className="text-[11px] text-muted-foreground opacity-70">
              · {new Date(article.date).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
            </span>
          )}
        </div>
        {/* external link icon */}
        <svg className="h-3.5 w-3.5 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
        </svg>
      </div>

      {/* title */}
      <h4 className="text-sm font-semibold text-foreground leading-snug group-hover:text-primary transition-colors">
        {article.title}
      </h4>

      {/* snippet */}
      {article.snippet && (
        <p className={cn(
          "mt-1.5 text-xs text-muted-foreground leading-relaxed overflow-hidden transition-all",
          expanded ? "max-h-40" : "max-h-10 line-clamp-2"
        )}>
          {article.snippet}
        </p>
      )}
    </a>
  )
}

function FilterButton({ label, active, count, onClick }: {
  label: string; active: boolean; count: number; onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium transition-colors",
        active
          ? "border-primary bg-primary/10 text-primary"
          : "border-border bg-secondary text-muted-foreground hover:text-foreground"
      )}
    >
      {label}
      <span className={cn(
        "rounded-full px-1.5 py-px text-[10px] font-bold",
        active ? "bg-primary/20 text-primary" : "bg-muted text-muted-foreground"
      )}>{count}</span>
    </button>
  )
}

export function NewsPanel({ news, loading, error }: {
  news: NewsResult | null
  loading: boolean
  error: string | null
}) {
  const [filter, setFilter] = useState<string>("All")

  const allTags = ["All", "Your Phone", "Competitor"]
  const articles = news?.articles ?? []

  const counts: Record<string, number> = { All: articles.length }
  for (const t of ["Your Phone", "Competitor"]) {
    counts[t] = articles.filter(a => a.tag === t).length
  }

  const filtered = filter === "All" ? articles : articles.filter(a => a.tag === filter)

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between flex-wrap gap-3">
          <CardTitle className="text-sm flex items-center gap-2">
            🌐 Market Intelligence — Live Web Search
            {loading && (
              <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground font-normal">
                <svg className="h-3 w-3 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                Searching…
              </span>
            )}
            {!loading && news && (
              <span className="text-[11px] text-emerald-400 font-normal">
                {articles.length} articles found
              </span>
            )}
          </CardTitle>

          {/* filter chips */}
          {!loading && news && articles.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {allTags.map(t => (
                <FilterButton
                  key={t}
                  label={t}
                  active={filter === t}
                  count={counts[t] ?? 0}
                  onClick={() => setFilter(t)}
                />
              ))}
            </div>
          )}
        </div>

        {/* legend */}
        {!loading && news && (
          <div className="flex flex-wrap gap-3 mt-1">
            <span className="text-[11px] text-muted-foreground flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-indigo-400 inline-block" /> Your Phone — your model &amp; spec-similar handsets
            </span>
            <span className="text-[11px] text-muted-foreground flex items-center gap-1">
              <span className="h-2 w-2 rounded-full bg-amber-400 inline-block" /> Competitor — the rivals the system predicts
            </span>
          </div>
        )}
      </CardHeader>

      <CardContent>
        {/* loading skeleton */}
        {loading && (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="rounded-xl border border-border bg-secondary/40 p-4 animate-pulse space-y-2">
                <div className="flex gap-2">
                  <div className="h-4 w-16 rounded-full bg-muted" />
                  <div className="h-4 w-24 rounded bg-muted" />
                </div>
                <div className="h-4 w-full rounded bg-muted" />
                <div className="h-4 w-3/4 rounded bg-muted" />
                <div className="h-3 w-full rounded bg-muted opacity-60" />
              </div>
            ))}
          </div>
        )}

        {/* error state */}
        {!loading && error && (
          <div className="rounded-xl border border-red-500/30 bg-red-950/40 px-4 py-3 text-sm text-red-300">
            <strong>News search failed:</strong> {error}
            <p className="mt-1 text-xs opacity-70">
              Make sure the backend is running on port 8000: <code>python -m uvicorn backend.main:app --reload --port 8000</code>
            </p>
          </div>
        )}

        {/* empty state */}
        {!loading && !news && !error && (
          <div className="py-10 text-center text-sm text-muted-foreground">
            Run an analysis first to fetch live market intelligence.
          </div>
        )}

        {/* no results for filter */}
        {!loading && news && filtered.length === 0 && (
          <div className="py-6 text-center text-sm text-muted-foreground">
            No articles found for <strong>{filter}</strong>.
          </div>
        )}

        {/* article grid */}
        {!loading && filtered.length > 0 && (
          <>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {filtered.map((a, i) => (
                <ArticleCard key={i} article={a} />
              ))}
            </div>

            {/* Distribution + search queries */}
            {news?.queries_used && (
              <>
                <Separator className="mt-5 mb-3 opacity-20" />
                <div className="flex flex-wrap items-center justify-between gap-3">
                  {/* Balanced distribution badge */}
                  {news.distribution && (
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Article balance:</span>
                      {Object.entries(news.distribution).map(([tag, count]) => (
                        count > 0 && (
                          <span key={tag} className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold
                            ${tag === "Your Phone" ? "border-indigo-500/30 text-indigo-600 dark:text-indigo-400 bg-indigo-500/10"
                            : tag === "Competitor" ? "border-amber-500/30 text-amber-600 dark:text-amber-400 bg-amber-500/10"
                            : "border-emerald-500/30 text-emerald-600 dark:text-emerald-400 bg-emerald-500/10"}`}>
                            {tag}: {count}
                          </span>
                        )
                      ))}
                    </div>
                  )}
                  {/* Spec-based queries used */}
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Searched by specs:</span>
                    {news.queries_used.slice(0, 3).map((q, i) => (
                      <span key={i} className="rounded border border-border bg-secondary px-2 py-0.5 text-[10px] text-muted-foreground max-w-50 truncate">
                        {q}
                      </span>
                    ))}
                  </div>
                </div>
              </>
            )}
          </>
        )}
      </CardContent>
    </Card>
  )
}
