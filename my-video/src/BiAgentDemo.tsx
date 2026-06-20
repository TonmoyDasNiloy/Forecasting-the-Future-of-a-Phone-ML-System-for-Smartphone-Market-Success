import React from "react";
import {
  AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig,
} from "remotion";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";
import { wipe } from "@remotion/transitions/wipe";

/* ─── tokens ─────────────────────────────────────────────────── */
const BG = "#0a0c14";
const CARD = "#131620";
const BORDER = "#ffffff14";
const PRIMARY = "#6366f1";
const VIOLET = "#a855f7";
const EMERALD = "#22c55e";
const AMBER = "#f59e0b";
const SKY = "#38bdf8";
const RED = "#ef4444";
const PINK = "#ec4899";
const MUTED = "#8b8fa8";
const TEXT = "#e8eaf6";

/* ─── helpers ────────────────────────────────────────────────── */
const useFade = (s: number, e: number) => {
  const f = useCurrentFrame();
  return interpolate(f, [s, e], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
};
const useSlide = (s: number, e: number, d = 40) => {
  const f = useCurrentFrame();
  const p = interpolate(f, [s, e], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return { opacity: p, transform: `translateY(${interpolate(p, [0, 1], [d, 0])}px)` };
};
const useSpringAt = (s: number) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  return spring({ frame: f - s, fps, config: { damping: 14, stiffness: 110 } });
};

/* ─── particles bg ───────────────────────────────────────────── */
const DOTS = Array.from({ length: 50 }, (_, i) => ({
  x: (i * 137.5) % 100, y: (i * 71.3) % 100, r: 1 + (i % 3),
  speed: 0.006 + (i % 5) * 0.002, opacity: 0.05 + (i % 4) * 0.03,
}));
const Particles: React.FC = () => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{ overflow: "hidden" }}>
      {DOTS.map((d, i) => (
        <div key={i} style={{
          position: "absolute", left: `${d.x}%`,
          top: `${((d.y + f * d.speed * 10) % 110) - 5}%`,
          width: d.r * 2, height: d.r * 2, borderRadius: "50%",
          background: PRIMARY, opacity: d.opacity,
        }} />
      ))}
      {Array.from({ length: 9 }, (_, i) => (
        <div key={`h${i}`} style={{ position: "absolute", left: 0, right: 0, top: `${i * 12.5}%`, height: 1, background: `${PRIMARY}07` }} />
      ))}
      {Array.from({ length: 14 }, (_, i) => (
        <div key={`v${i}`} style={{ position: "absolute", top: 0, bottom: 0, left: `${i * 7.2}%`, width: 1, background: `${PRIMARY}07` }} />
      ))}
    </AbsoluteFill>
  );
};

/* ─── Siri orb ───────────────────────────────────────────────── */
const SiriOrb: React.FC<{ size?: number }> = ({ size = 220 }) => {
  const f = useCurrentFrame();
  const rot = (f * 4) % 360;
  const rotRev = (-f * 2.6) % 360;
  const pulse = 1 + Math.sin(f / 14) * 0.04;
  const ripple = (f % 96) / 96;
  return (
    <div style={{ position: "relative", width: size, height: size, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div style={{
        position: "absolute", width: size * 0.8 * (0.85 + ripple), height: size * 0.8 * (0.85 + ripple),
        borderRadius: "50%", border: `3px solid ${PRIMARY}`, opacity: 0.5 * (1 - ripple),
      }} />
      <div style={{ position: "relative", width: size * 0.72, height: size * 0.72, transform: `scale(${pulse})` }}>
        <div style={{ position: "absolute", inset: 0, borderRadius: "50%", overflow: "hidden", boxShadow: `0 0 70px ${PRIMARY}aa, 0 0 30px ${VIOLET}88` }}>
          <div style={{ position: "absolute", inset: -45, background: `conic-gradient(from ${rot}deg, ${PRIMARY}, ${SKY}, ${VIOLET}, ${PINK}, ${PRIMARY})`, filter: "blur(16px)" }} />
          <div style={{ position: "absolute", inset: -35, background: `conic-gradient(from ${rotRev}deg, #22d3ee, #818cf8, #c084fc, transparent, #22d3ee)`, filter: "blur(18px)", mixBlendMode: "screen" }} />
          <div style={{ position: "absolute", inset: 0, borderRadius: "50%", background: "radial-gradient(circle at 35% 28%, rgba(255,255,255,0.42), transparent 55%)" }} />
        </div>
      </div>
    </div>
  );
};

/* ─── shared scene chrome ────────────────────────────────────── */
const StepBadge: React.FC<{ n: string; title: string; sub: string; style?: React.CSSProperties }> = ({ n, title, sub, style }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 16, ...style }}>
    <div style={{ width: 52, height: 52, borderRadius: 14, background: `${PRIMARY}22`, border: `1px solid ${PRIMARY}55`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 24, fontWeight: 900, color: PRIMARY }}>{n}</div>
    <div>
      <div style={{ fontSize: 34, fontWeight: 800, color: TEXT, letterSpacing: "-0.01em" }}>{title}</div>
      <div style={{ fontSize: 17, color: MUTED }}>{sub}</div>
    </div>
  </div>
);

const Card: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <div style={{ background: CARD, border: `1px solid ${BORDER}`, borderRadius: 16, padding: 20, ...style }}>{children}</div>
);

/* ════════════════════════════════════════════════════════════════
   SCENE 1 — Welcome
════════════════════════════════════════════════════════════════ */
const SWelcome: React.FC = () => {
  const titleS = useSlide(20, 60, 40);
  const subS = useSlide(45, 85, 24);
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 10 }}>
      <Particles />
      <SiriOrb size={260} />
      <div style={{ ...titleS, textAlign: "center", marginTop: 20 }}>
        <div style={{ fontSize: 76, fontWeight: 900, background: `linear-gradient(135deg, ${TEXT}, ${PRIMARY} 60%, ${VIOLET})`, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", letterSpacing: "-0.02em" }}>
          Smartphone BI Agent
        </div>
      </div>
      <div style={{ ...subS, fontSize: 24, color: MUTED }}>
        Predict a phone&apos;s market success — before it launches
      </div>
    </AbsoluteFill>
  );
};

/* ════════════════════════════════════════════════════════════════
   SCENE 2 — Step 1: Enter specifications
════════════════════════════════════════════════════════════════ */
const SPECS = [
  ["Model", "Galaxy A55"], ["Price", "$320"], ["5G", "On"], ["Processor", "Snapdragon"],
  ["RAM", "8 GB"], ["Storage", "128 GB"], ["Battery", "5000 mAh"], ["Screen", "6.5\""],
  ["Refresh", "120 Hz"], ["Camera", "50 MP"],
];
const SSpecs: React.FC = () => {
  const head = useSlide(5, 35, 30);
  return (
    <AbsoluteFill style={{ padding: 80, justifyContent: "center" }}>
      <Particles />
      <StepBadge n="1" title="Enter the Specs" sub="Type any new phone's specifications in the sidebar" style={head} />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginTop: 40, maxWidth: 1000 }}>
        {SPECS.map(([k, v], i) => {
          const s = useSpringAt(20 + i * 6);
          return (
            <div key={k} style={{
              opacity: s, transform: `translateX(${interpolate(s, [0, 1], [-30, 0])}px)`,
              display: "flex", justifyContent: "space-between", alignItems: "center",
              background: CARD, border: `1px solid ${BORDER}`, borderRadius: 12, padding: "16px 22px",
            }}>
              <span style={{ fontSize: 20, color: MUTED }}>{k}</span>
              <span style={{ fontSize: 22, fontWeight: 800, color: TEXT }}>{v}</span>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

/* ════════════════════════════════════════════════════════════════
   SCENE 3 — Step 2: Analyze (loading)
════════════════════════════════════════════════════════════════ */
const STAGES = ["Researching market…", "Forecasting demand…", "Profiling customers…", "Scoring product-fit…", "Fetching live news…"];
const SLoading: React.FC = () => {
  const f = useCurrentFrame();
  const idx = Math.min(STAGES.length - 1, Math.floor(f / 16));
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 30 }}>
      <Particles />
      <SiriOrb size={200} />
      <div style={{ fontSize: 26, fontWeight: 700, color: TEXT }}>{STAGES[idx]}</div>
      <div style={{ width: 320, height: 6, borderRadius: 99, background: "#ffffff14", overflow: "hidden" }}>
        <div style={{ width: `${((idx + 1) / STAGES.length) * 100}%`, height: "100%", borderRadius: 99, background: `linear-gradient(90deg, ${PRIMARY}, ${VIOLET})` }} />
      </div>
    </AbsoluteFill>
  );
};

/* ════════════════════════════════════════════════════════════════
   SCENE 4 — Verdict + KPI cards
════════════════════════════════════════════════════════════════ */
const KPIS = [
  ["MARKET SEGMENT", "MID-RANGE", PRIMARY],
  ["PMF SCORE", "95/100", EMERALD],
  ["6-MONTH DEMAND", "703", SKY],
  ["CLV / BUYER", "$160", EMERALD],
  ["PRICE VS MARKET", "+2.4%", AMBER],
];
const SVerdict: React.FC = () => {
  const banner = useSpringAt(5);
  return (
    <AbsoluteFill style={{ padding: 80, justifyContent: "center" }}>
      <Particles />
      <StepBadge n="2" title="Instant BI Dashboard" sub="One input → a full launch-intelligence report" style={useSlide(0, 25, 24)} />
      {/* verdict banner */}
      <div style={{ transform: `scale(${banner})`, opacity: banner, marginTop: 36, background: "#0d3b2e", border: `1px solid ${EMERALD}`, borderRadius: 16, padding: "22px 30px", display: "flex", alignItems: "center", gap: 16 }}>
        <span style={{ fontSize: 38 }}>✅</span>
        <div>
          <div style={{ fontSize: 34, fontWeight: 900, color: "#86efac" }}>GO FOR LAUNCH</div>
          <div style={{ fontSize: 18, color: MUTED }}>Galaxy A55 · $320 · Asia-Pacific · Failure risk 5.2%</div>
        </div>
      </div>
      {/* KPI cards */}
      <div style={{ display: "flex", gap: 16, marginTop: 28 }}>
        {KPIS.map(([label, val, color], i) => {
          const s = useSpringAt(25 + i * 8);
          return (
            <div key={label as string} style={{ flex: 1, opacity: s, transform: `translateY(${interpolate(s, [0, 1], [30, 0])}px)`, background: CARD, border: `1px solid ${BORDER}`, borderRadius: 14, padding: "20px 16px", textAlign: "center" }}>
              <div style={{ fontSize: 12, color: MUTED, letterSpacing: "0.08em", marginBottom: 8 }}>{label}</div>
              <div style={{ fontSize: 30, fontWeight: 900, color: color as string }}>{val}</div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

/* ════════════════════════════════════════════════════════════════
   SCENE 5 — Demand forecast
════════════════════════════════════════════════════════════════ */
const DEMAND = [106, 110, 108, 130, 125, 124];
const SDemand: React.FC = () => {
  const f = useCurrentFrame();
  const maxD = Math.max(...DEMAND);
  return (
    <AbsoluteFill style={{ padding: 80, justifyContent: "center" }}>
      <Particles />
      <StepBadge n="📈" title="6-Month Demand Forecast" sub="Cold-start unit projection from specs alone" style={useSlide(0, 25, 24)} />
      <Card style={{ marginTop: 36, height: 460, display: "flex", alignItems: "flex-end", justifyContent: "space-around", padding: 50 }}>
        {DEMAND.map((d, i) => {
          const grow = interpolate(f, [20 + i * 6, 50 + i * 6], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
          const h = (d / maxD) * 320 * grow;
          return (
            <div key={i} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12 }}>
              <span style={{ fontSize: 22, fontWeight: 800, color: TEXT, opacity: grow }}>{Math.round(d * grow)}</span>
              <div style={{ width: 90, height: h, borderRadius: "10px 10px 0 0", background: `linear-gradient(180deg, ${SKY}, ${PRIMARY})` }} />
              <span style={{ fontSize: 18, color: MUTED }}>M{i + 1}</span>
            </div>
          );
        })}
      </Card>
    </AbsoluteFill>
  );
};

/* ════════════════════════════════════════════════════════════════
   SCENE 6 — Customer profile
════════════════════════════════════════════════════════════════ */
const Donut: React.FC<{ segs: { v: number; c: string }[]; label: string; top: string }> = ({ segs, label, top }) => {
  const f = useCurrentFrame();
  const reveal = interpolate(f, [25, 60], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  let acc = 0;
  const grad = segs.map((s) => {
    const start = acc; acc += s.v * reveal;
    return `${s.c} ${start}% ${acc}%`;
  }).join(", ");
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12 }}>
      <div style={{ width: 180, height: 180, borderRadius: "50%", background: `conic-gradient(${grad}, #1e2233 0%)`, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div style={{ width: 110, height: 110, borderRadius: "50%", background: CARD, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
          <span style={{ fontSize: 26, fontWeight: 900, color: TEXT }}>{top}</span>
        </div>
      </div>
      <span style={{ fontSize: 19, fontWeight: 700, color: TEXT }}>{label}</span>
    </div>
  );
};
const SProfile: React.FC = () => {
  const persona = useSlide(15, 45, 24);
  return (
    <AbsoluteFill style={{ padding: 80, justifyContent: "center" }}>
      <Particles />
      <StepBadge n="👥" title="Customer Profile" sub="Who actually buys this phone — age, income, gender, loyalty" style={useSlide(0, 25, 24)} />
      <div style={{ ...persona, marginTop: 28, background: CARD, border: `1px solid ${BORDER}`, borderRadius: 14, padding: "18px 24px", fontSize: 22, color: TEXT }}>
        🧑‍💼 <b>Most likely buyer:</b> a 25–34 year-old, mid-income shopper, typically loyal.
      </div>
      <div style={{ display: "flex", justifyContent: "space-around", marginTop: 36 }}>
        <Donut label="Age" top="25–34" segs={[{ v: 20, c: PRIMARY }, { v: 25, c: "#818cf8" }, { v: 24, c: VIOLET }, { v: 17, c: PINK }, { v: 14, c: SKY }]} />
        <Donut label="Income" top="Mid" segs={[{ v: 13, c: RED }, { v: 20, c: AMBER }, { v: 31, c: "#facc15" }, { v: 22, c: "#86efac" }, { v: 14, c: EMERALD }]} />
        <Donut label="Gender" top="Male" segs={[{ v: 50, c: PRIMARY }, { v: 48, c: PINK }, { v: 2, c: VIOLET }]} />
      </div>
    </AbsoluteFill>
  );
};

/* ════════════════════════════════════════════════════════════════
   SCENE 7 — Competitor gap (treemap)
════════════════════════════════════════════════════════════════ */
const SCompetitor: React.FC = () => {
  const tiles = [
    { x: 0, y: 0, w: 50, h: 55, c: PRIMARY, l: "RAM 8GB" },
    { x: 50, y: 0, w: 50, h: 35, c: PRIMARY, l: "Battery" },
    { x: 50, y: 35, w: 25, h: 20, c: PRIMARY, l: "Cam" },
    { x: 75, y: 35, w: 25, h: 20, c: PRIMARY, l: "120Hz" },
    { x: 0, y: 55, w: 33, h: 45, c: AMBER, l: "Nothing 1" },
    { x: 33, y: 55, w: 34, h: 45, c: EMERALD, l: "Redmi N12" },
    { x: 67, y: 55, w: 33, h: 45, c: RED, l: "Galaxy A23" },
  ];
  return (
    <AbsoluteFill style={{ padding: 80, justifyContent: "center" }}>
      <Particles />
      <StepBadge n="⚔️" title="Competitor Gap Analysis" sub="Your phone vs the rivals the system predicts" style={useSlide(0, 25, 24)} />
      <div style={{ position: "relative", marginTop: 36, height: 460, width: "100%", background: CARD, border: `1px solid ${BORDER}`, borderRadius: 16, overflow: "hidden" }}>
        {tiles.map((t, i) => {
          const s = useSpringAt(20 + i * 7);
          return (
            <div key={i} style={{
              position: "absolute", left: `${t.x}%`, top: `${t.y}%`, width: `${t.w}%`, height: `${t.h}%`,
              padding: 6, opacity: s, transform: `scale(${s})`,
            }}>
              <div style={{ width: "100%", height: "100%", background: t.c, opacity: 0.82, borderRadius: 10, border: "2px solid #0a0c14", display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontWeight: 800, fontSize: 22 }}>
                {t.l}
              </div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

/* ════════════════════════════════════════════════════════════════
   SCENE 8 — Price recommender (radar)
════════════════════════════════════════════════════════════════ */
const SPrice: React.FC = () => {
  const cards = [
    ["Your Price", "$320", MUTED], ["Recommended", "$383", PRIMARY], ["Action", "↑ Raise $63", EMERALD],
  ];
  return (
    <AbsoluteFill style={{ padding: 80, justifyContent: "center" }}>
      <Particles />
      <StepBadge n="💰" title="Optimal Price Recommender" sub="Profit-maximising price, anchored to competitor pricing" style={useSlide(0, 25, 24)} />
      <div style={{ display: "flex", gap: 18, marginTop: 36 }}>
        {cards.map(([l, v, c], i) => {
          const s = useSpringAt(20 + i * 10);
          return (
            <div key={l as string} style={{ flex: 1, opacity: s, transform: `scale(${s})`, background: i === 1 ? `${PRIMARY}1a` : CARD, border: `1px solid ${i === 1 ? PRIMARY + "55" : BORDER}`, borderRadius: 16, padding: "26px 20px", textAlign: "center" }}>
              <div style={{ fontSize: 14, color: MUTED, letterSpacing: "0.08em", marginBottom: 10 }}>{(l as string).toUpperCase()}</div>
              <div style={{ fontSize: 40, fontWeight: 900, color: c as string }}>{v}</div>
            </div>
          );
        })}
      </div>
      <Card style={{ marginTop: 24, ...useSlide(50, 80, 20) }}>
        <div style={{ fontSize: 19, color: TEXT, lineHeight: 1.6 }}>
          <b style={{ color: PRIMARY }}>Why $383?</b> &nbsp;Similar phones sell for $301–$361 (median $325). Your specs justify a small premium, and $383 is where <b>profit × product-market-fit</b> peaks.
        </div>
      </Card>
    </AbsoluteFill>
  );
};

/* ════════════════════════════════════════════════════════════════
   SCENE 9 — PMF gauge + Live news
════════════════════════════════════════════════════════════════ */
const SPmfNews: React.FC = () => {
  const f = useCurrentFrame();
  const fill = interpolate(f, [20, 55], [0, 0.95], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const articles = [
    ["Your Phone", "Galaxy A55 Review — AndroidAuthority", PRIMARY],
    ["Competitor", "Nothing Phone 1 Full Specs & Price", AMBER],
    ["Your Phone", "Best mid-range 5G phones tested", PRIMARY],
    ["Competitor", "Redmi Note 12 Pro vs Galaxy A", AMBER],
  ];
  return (
    <AbsoluteFill style={{ padding: 80, justifyContent: "center" }}>
      <Particles />
      <StepBadge n="🎯" title="Product-Market Fit + Live News" sub="Success score plus real-time web articles" style={useSlide(0, 25, 24)} />
      <div style={{ display: "flex", gap: 24, marginTop: 36 }}>
        {/* gauge */}
        <Card style={{ width: 420, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
          <svg viewBox="0 0 200 120" style={{ width: 320 }}>
            <path d="M 20 110 A 80 80 0 0 1 180 110" fill="none" stroke="#1e2233" strokeWidth={18} strokeLinecap="round" />
            <path d="M 20 110 A 80 80 0 0 1 180 110" fill="none" stroke={EMERALD} strokeWidth={18} strokeLinecap="round"
              strokeDasharray={`${fill * 251} 251`} />
          </svg>
          <div style={{ fontSize: 52, fontWeight: 900, color: EMERALD, marginTop: -20 }}>{Math.round(fill * 100)}%</div>
          <div style={{ fontSize: 18, color: MUTED }}>Product-Market Fit</div>
        </Card>
        {/* news */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 12 }}>
          {articles.map(([tag, title, c], i) => {
            const s = useSpringAt(30 + i * 8);
            return (
              <div key={i} style={{ opacity: s, transform: `translateX(${interpolate(s, [0, 1], [40, 0])}px)`, display: "flex", alignItems: "center", gap: 14, background: CARD, border: `1px solid ${BORDER}`, borderRadius: 12, padding: "16px 20px" }}>
                <span style={{ background: `${c}22`, color: c as string, border: `1px solid ${c}55`, borderRadius: 99, padding: "4px 12px", fontSize: 13, fontWeight: 700, whiteSpace: "nowrap" }}>{tag}</span>
                <span style={{ fontSize: 19, color: TEXT, flex: 1 }}>{title}</span>
                <span style={{ color: PRIMARY, fontSize: 20 }}>↗</span>
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};

/* ════════════════════════════════════════════════════════════════
   SCENE 10 — Segment showcase
════════════════════════════════════════════════════════════════ */
const SEGMENTS = [
  { name: "BUDGET", price: "$95", verdict: "RECONSIDER", color: SKY, vc: RED, icon: "🟦" },
  { name: "MID-RANGE", price: "$320", verdict: "GO FOR LAUNCH", color: PRIMARY, vc: EMERALD, icon: "🟪" },
  { name: "PREMIUM", price: "$720", verdict: "PROCEED", color: VIOLET, vc: AMBER, icon: "🟣" },
  { name: "FLAGSHIP", price: "$1,200", verdict: "GO FOR LAUNCH", color: AMBER, vc: EMERALD, icon: "🟡" },
];
const SSegments: React.FC = () => {
  return (
    <AbsoluteFill style={{ padding: 80, justifyContent: "center" }}>
      <Particles />
      <StepBadge n="🎚️" title="Works Across Every Segment" sub="From $95 budget phones to $1,200 flagships" style={useSlide(0, 25, 24)} />
      <div style={{ display: "flex", gap: 18, marginTop: 40 }}>
        {SEGMENTS.map((s, i) => {
          const sp = useSpringAt(20 + i * 12);
          return (
            <div key={s.name} style={{ flex: 1, opacity: sp, transform: `translateY(${interpolate(sp, [0, 1], [40, 0])}px) scale(${sp})`, background: CARD, border: `1px solid ${s.color}55`, borderRadius: 18, padding: 24, textAlign: "center" }}>
              <div style={{ fontSize: 34 }}>{s.icon}</div>
              <div style={{ fontSize: 22, fontWeight: 900, color: s.color, marginTop: 10, letterSpacing: "0.05em" }}>{s.name}</div>
              <div style={{ fontSize: 34, fontWeight: 900, color: TEXT, margin: "12px 0" }}>{s.price}</div>
              <div style={{ display: "inline-block", background: `${s.vc}1f`, border: `1px solid ${s.vc}66`, color: s.vc, borderRadius: 99, padding: "6px 14px", fontSize: 14, fontWeight: 800 }}>{s.verdict}</div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

/* ════════════════════════════════════════════════════════════════
   SCENE 11 — Outro
════════════════════════════════════════════════════════════════ */
const SOutro: React.FC = () => {
  const t = useSlide(8, 35, 24);
  const chips = useSlide(20, 50, 20);
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 18 }}>
      <Particles />
      <div style={{ ...useSlide(0, 20, 0) }}><SiriOrb size={150} /></div>
      <div style={{ ...t, fontSize: 64, fontWeight: 900, background: `linear-gradient(135deg, ${TEXT}, ${PRIMARY} 60%, ${VIOLET})`, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", marginTop: 10 }}>
        Smartphone BI Agent
      </div>
      <div style={{ ...chips, display: "flex", gap: 12, flexWrap: "wrap", justifyContent: "center", maxWidth: 900 }}>
        {["Market Segment", "Demand Forecast", "Customer Profile", "Competitor Gap", "Price Optimizer", "PMF Score", "Live News"].map((c) => (
          <span key={c} style={{ background: `${PRIMARY}18`, border: `1px solid ${PRIMARY}33`, borderRadius: 99, padding: "8px 18px", fontSize: 16, color: PRIMARY, fontWeight: 600 }}>{c}</span>
        ))}
      </div>
    </AbsoluteFill>
  );
};

/* ════════════════════════════════════════════════════════════════
   ROOT — sequence all scenes (30fps)
════════════════════════════════════════════════════════════════ */
const T = 18; // transition length in frames (~0.6s)

export const BiAgentDemo: React.FC = () => {
  return (
    <AbsoluteFill style={{ background: BG, fontFamily: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", color: TEXT }}>
      <TransitionSeries>
        <TransitionSeries.Sequence durationInFrames={120}><SWelcome /></TransitionSeries.Sequence>
        <TransitionSeries.Transition presentation={fade()} timing={linearTiming({ durationInFrames: T })} />

        <TransitionSeries.Sequence durationInFrames={170}><SSpecs /></TransitionSeries.Sequence>
        <TransitionSeries.Transition presentation={slide({ direction: "from-right" })} timing={linearTiming({ durationInFrames: T })} />

        <TransitionSeries.Sequence durationInFrames={90}><SLoading /></TransitionSeries.Sequence>
        <TransitionSeries.Transition presentation={fade()} timing={linearTiming({ durationInFrames: T })} />

        <TransitionSeries.Sequence durationInFrames={160}><SVerdict /></TransitionSeries.Sequence>
        <TransitionSeries.Transition presentation={slide({ direction: "from-right" })} timing={linearTiming({ durationInFrames: T })} />

        <TransitionSeries.Sequence durationInFrames={140}><SDemand /></TransitionSeries.Sequence>
        <TransitionSeries.Transition presentation={wipe({ direction: "from-left" })} timing={linearTiming({ durationInFrames: T })} />

        <TransitionSeries.Sequence durationInFrames={160}><SProfile /></TransitionSeries.Sequence>
        <TransitionSeries.Transition presentation={slide({ direction: "from-right" })} timing={linearTiming({ durationInFrames: T })} />

        <TransitionSeries.Sequence durationInFrames={150}><SCompetitor /></TransitionSeries.Sequence>
        <TransitionSeries.Transition presentation={wipe({ direction: "from-bottom" })} timing={linearTiming({ durationInFrames: T })} />

        <TransitionSeries.Sequence durationInFrames={160}><SPrice /></TransitionSeries.Sequence>
        <TransitionSeries.Transition presentation={slide({ direction: "from-right" })} timing={linearTiming({ durationInFrames: T })} />

        <TransitionSeries.Sequence durationInFrames={160}><SPmfNews /></TransitionSeries.Sequence>
        <TransitionSeries.Transition presentation={fade()} timing={linearTiming({ durationInFrames: T })} />

        <TransitionSeries.Sequence durationInFrames={170}><SSegments /></TransitionSeries.Sequence>
        <TransitionSeries.Transition presentation={fade()} timing={linearTiming({ durationInFrames: T })} />

        <TransitionSeries.Sequence durationInFrames={130}><SOutro /></TransitionSeries.Sequence>
      </TransitionSeries>
    </AbsoluteFill>
  );
};
