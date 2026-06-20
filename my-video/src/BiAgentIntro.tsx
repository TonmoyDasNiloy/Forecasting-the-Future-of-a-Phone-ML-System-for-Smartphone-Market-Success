import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  Sequence,
} from "remotion";

/* ─── shared tokens ─────────────────────────────────────────── */
const BG       = "#0a0c14";
const CARD     = "#13162080";
const BORDER   = "#ffffff18";
const PRIMARY  = "#6366f1";
const VIOLET   = "#7c3aed";
const EMERALD  = "#22c55e";
const AMBER    = "#f59e0b";
const RED      = "#ef4444";
const SKY      = "#38bdf8";
const MUTED    = "#64748b";
const TEXT     = "#e2e8f0";

/* ─── helpers ───────────────────────────────────────────────── */
function useFade(startFrame: number, endFrame: number) {
  const frame = useCurrentFrame();
  return interpolate(frame, [startFrame, endFrame], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
}

function useSlideUp(startFrame: number, endFrame: number, distance = 40) {
  const frame = useCurrentFrame();
  const progress = interpolate(frame, [startFrame, endFrame], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const y = interpolate(progress, [0, 1], [distance, 0]);
  return { opacity: progress, transform: `translateY(${y}px)` };
}

function useSpring(startFrame: number) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return spring({ frame: frame - startFrame, fps, config: { damping: 14, stiffness: 100 } });
}

/* ─── particles background ──────────────────────────────────── */
const DOTS = Array.from({ length: 60 }, (_, i) => ({
  x: (i * 137.5) % 100,
  y: (i * 73.1)  % 100,
  r: 1 + (i % 3),
  speed: 0.008 + (i % 5) * 0.003,
  opacity: 0.06 + (i % 4) * 0.04,
}));

const Particles: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{ overflow: "hidden" }}>
      {DOTS.map((d, i) => (
        <div
          key={i}
          style={{
            position: "absolute",
            left: `${d.x}%`,
            top: `${((d.y + frame * d.speed * 10) % 110) - 5}%`,
            width: d.r * 2,
            height: d.r * 2,
            borderRadius: "50%",
            background: PRIMARY,
            opacity: d.opacity,
          }}
        />
      ))}
      {/* grid lines */}
      {Array.from({ length: 8 }, (_, i) => (
        <div key={`h${i}`} style={{
          position: "absolute", left: 0, right: 0,
          top: `${i * 14.3}%`, height: 1,
          background: `${PRIMARY}08`,
        }} />
      ))}
      {Array.from({ length: 12 }, (_, i) => (
        <div key={`v${i}`} style={{
          position: "absolute", top: 0, bottom: 0,
          left: `${i * 8.5}%`, width: 1,
          background: `${PRIMARY}08`,
        }} />
      ))}
    </AbsoluteFill>
  );
};

/* ─── SCENE 1: Title (frames 0–120, 4s) ─────────────────────── */
const SceneTitle: React.FC = () => {
  const fade      = useFade(0, 40);
  const titleSlide = useSlideUp(10, 60, 60);
  const subSlide   = useSlideUp(40, 90, 30);
  const tagSlide   = useSlideUp(70, 110, 20);

  return (
    <AbsoluteFill style={{ display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center", gap: 24 }}>
      <Particles />

      {/* glow blob */}
      <div style={{
        position: "absolute", width: 600, height: 600,
        borderRadius: "50%",
        background: `radial-gradient(circle, ${PRIMARY}20 0%, transparent 70%)`,
        opacity: fade,
      }} />

      {/* badge */}
      <div style={{ ...tagSlide, background: `${PRIMARY}22`,
        border: `1px solid ${PRIMARY}44`, borderRadius: 100,
        padding: "6px 20px", fontSize: 14, color: PRIMARY,
        letterSpacing: "0.15em", textTransform: "uppercase", fontWeight: 700 }}>
        Thesis Research Project
      </div>

      {/* main title */}
      <div style={{ ...titleSlide, textAlign: "center" }}>
        <div style={{ fontSize: 88, fontWeight: 900, lineHeight: 1,
          background: `linear-gradient(135deg, ${TEXT} 0%, ${PRIMARY} 60%, ${VIOLET} 100%)`,
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          letterSpacing: "-0.02em" }}>
          Smartphone
        </div>
        <div style={{ fontSize: 88, fontWeight: 900, lineHeight: 1,
          background: `linear-gradient(135deg, ${PRIMARY} 0%, ${VIOLET} 100%)`,
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          letterSpacing: "-0.02em" }}>
          BI Agent
        </div>
      </div>

      {/* subtitle */}
      <div style={{ ...subSlide, fontSize: 26, color: MUTED,
        letterSpacing: "0.05em", textAlign: "center", fontWeight: 400 }}>
        AI-Powered Market Intelligence for New Product Launches
      </div>
    </AbsoluteFill>
  );
};

/* ─── SCENE 2: Dataset stats (frames 120–270, 5s) ───────────── */
const STATS = [
  { value: "116,481", label: "Real Phone Records", color: PRIMARY },
  { value: "968",     label: "Unique Phones",       color: SKY },
  { value: "6",       label: "Global Regions",      color: EMERALD },
  { value: "46",      label: "Brands",              color: AMBER },
];

const SceneDataset: React.FC = () => {
  const titleSlide = useSlideUp(0, 40, 30);

  return (
    <AbsoluteFill style={{ display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center", gap: 48 }}>
      <Particles />

      <div style={{ ...titleSlide, textAlign: "center" }}>
        <div style={{ fontSize: 18, color: PRIMARY, letterSpacing: "0.2em",
          textTransform: "uppercase", fontWeight: 700, marginBottom: 12 }}>
          Training Data
        </div>
        <div style={{ fontSize: 48, fontWeight: 800, color: TEXT }}>
          Trained on Real-World Smartphone Data
        </div>
      </div>

      <div style={{ display: "flex", gap: 28 }}>
        {STATS.map((s, i) => {
          const slide = useSlideUp(20 + i * 15, 60 + i * 15, 30);
          return (
            <div key={i} style={{ ...slide,
              background: CARD, border: `1px solid ${s.color}33`,
              borderRadius: 20, padding: "32px 40px", textAlign: "center",
              minWidth: 180 }}>
              <div style={{ fontSize: 52, fontWeight: 900, color: s.color,
                lineHeight: 1, marginBottom: 8 }}>{s.value}</div>
              <div style={{ fontSize: 15, color: MUTED, letterSpacing: "0.05em" }}>
                {s.label}
              </div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

/* ─── SCENE 3: Four Pillars (frames 270–480, 7s) ────────────── */
const PILLARS = [
  { num: "01", title: "Market Research",   desc: "Segment classification + nearest-competitor retrieval",  color: PRIMARY,  icon: "🔍", metric: "99.4% Accuracy" },
  { num: "02", title: "Demand Forecast",   desc: "6-month cold-start demand curve from specs only",         color: SKY,      icon: "📈", metric: "R² = 0.854" },
  { num: "03", title: "Customer Profile",  desc: "Age, income & loyalty via KNN similarity retrieval",      color: EMERALD,  icon: "👥", metric: "R² = 0.771" },
  { num: "04", title: "PMF & Failure",     desc: "Product-Market Fit score with GO / CAUTION / REWORK",     color: AMBER,    icon: "🎯", metric: "AUC = 0.88" },
];

const ScenePillars: React.FC = () => {
  const titleSlide = useSlideUp(0, 35, 30);

  return (
    <AbsoluteFill style={{ display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center", gap: 40 }}>
      <Particles />

      <div style={{ ...titleSlide, textAlign: "center" }}>
        <div style={{ fontSize: 18, color: PRIMARY, letterSpacing: "0.2em",
          textTransform: "uppercase", fontWeight: 700, marginBottom: 8 }}>
          Four ML Pillars
        </div>
        <div style={{ fontSize: 48, fontWeight: 800, color: TEXT }}>
          End-to-End Intelligence Pipeline
        </div>
      </div>

      <div style={{ display: "flex", gap: 20, width: 1600 }}>
        {PILLARS.map((p, i) => {
          const s = useSpring(i * 12);
          return (
            <div key={i} style={{
              flex: 1,
              transform: `scale(${s}) translateY(${interpolate(s,[0,1],[40,0])}px)`,
              opacity: s,
              background: CARD,
              border: `1px solid ${p.color}40`,
              borderRadius: 20,
              padding: "28px 24px",
              display: "flex", flexDirection: "column", gap: 14,
            }}>
              {/* header */}
              <div style={{ display: "flex", alignItems: "center",
                justifyContent: "space-between" }}>
                <span style={{ fontSize: 28 }}>{p.icon}</span>
                <span style={{ fontSize: 11, color: p.color, fontWeight: 700,
                  letterSpacing: "0.15em", background: `${p.color}18`,
                  padding: "4px 10px", borderRadius: 100 }}>
                  PILLAR {p.num}
                </span>
              </div>

              {/* title */}
              <div style={{ fontSize: 20, fontWeight: 800, color: TEXT,
                lineHeight: 1.2 }}>{p.title}</div>

              {/* desc */}
              <div style={{ fontSize: 13, color: MUTED, lineHeight: 1.5,
                flex: 1 }}>{p.desc}</div>

              {/* metric badge */}
              <div style={{ background: `${p.color}20`, border: `1px solid ${p.color}40`,
                borderRadius: 10, padding: "8px 14px", fontSize: 15,
                fontWeight: 800, color: p.color, textAlign: "center" }}>
                {p.metric}
              </div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

/* ─── SCENE 4: Metric Reveal (frames 480–630, 5s) ───────────── */
const METRICS = [
  { label: "Segment Accuracy",     value: "99.4%",  sub: "XGBoost Classifier",    color: PRIMARY },
  { label: "Demand R²",            value: "0.854",  sub: "Cold-start Regression", color: SKY },
  { label: "Income Profile R²",    value: "0.771",  sub: "KNN Retrieval",         color: EMERALD },
  { label: "Failure AUC-ROC",      value: "0.880",  sub: "Random Forest",         color: AMBER },
  { label: "Composite Score",      value: "85.7%",  sub: "System Grade: A+",      color: VIOLET },
];

const SceneMetrics: React.FC = () => {
  const titleSlide = useSlideUp(0, 35, 30);
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{ display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center", gap: 44 }}>
      <Particles />

      <div style={{ ...titleSlide, textAlign: "center" }}>
        <div style={{ fontSize: 18, color: PRIMARY, letterSpacing: "0.2em",
          textTransform: "uppercase", fontWeight: 700, marginBottom: 8 }}>
          Accuracy Report
        </div>
        <div style={{ fontSize: 48, fontWeight: 800, color: TEXT }}>
          Honest, Leakage-Free Evaluation
        </div>
      </div>

      <div style={{ display: "flex", gap: 20 }}>
        {METRICS.map((m, i) => {
          const start = i * 14;
          const opacity = interpolate(frame, [start, start + 25], [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
          const y = interpolate(frame, [start, start + 25], [30, 0],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

          return (
            <div key={i} style={{
              opacity, transform: `translateY(${y}px)`,
              background: CARD, border: `1px solid ${m.color}44`,
              borderRadius: 20, padding: "28px 24px", textAlign: "center",
              minWidth: 220,
            }}>
              <div style={{ fontSize: 13, color: MUTED, textTransform: "uppercase",
                letterSpacing: "0.12em", fontWeight: 600, marginBottom: 10 }}>
                {m.label}
              </div>
              <div style={{ fontSize: 52, fontWeight: 900, color: m.color,
                lineHeight: 1, marginBottom: 10 }}>{m.value}</div>
              <div style={{ fontSize: 13, color: MUTED }}>{m.sub}</div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

/* ─── SCENE 5: Live Web Search (frames 630–750, 4s) ─────────── */
const ARTICLES = [
  { tag: "Your Phone", title: "Samsung Galaxy A23 5G Review — AndroidAuthority", source: "androidauthority.com", color: PRIMARY },
  { tag: "Competitor", title: "Nothing Phone 1 Full Specs & Price 2025",          source: "gsmarena.com",        color: AMBER },
  { tag: "Market",     title: "Mid-Range Smartphone Trends Asia-Pacific 2025",     source: "techcrunch.com",      color: EMERALD },
  { tag: "Your Phone", title: "Best Budget 5G Phones — Expert Roundup",            source: "wired.com",           color: PRIMARY },
  { tag: "Competitor", title: "Xiaomi Redmi Note 12 Pro vs Galaxy A Series",       source: "phonearena.com",      color: AMBER },
];

const SceneNews: React.FC = () => {
  const titleSlide = useSlideUp(0, 35, 30);
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{ display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center", gap: 36 }}>
      <Particles />

      <div style={{ ...titleSlide, textAlign: "center" }}>
        <div style={{ fontSize: 18, color: PRIMARY, letterSpacing: "0.2em",
          textTransform: "uppercase", fontWeight: 700, marginBottom: 8 }}>
          Live Web Search
        </div>
        <div style={{ fontSize: 48, fontWeight: 800, color: TEXT }}>
          Market Intelligence — Real-Time News
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 12, width: 900 }}>
        {ARTICLES.map((a, i) => {
          const start = i * 10;
          const opacity = interpolate(frame, [start, start + 20], [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
          const x = interpolate(frame, [start, start + 20], [-40, 0],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

          return (
            <div key={i} style={{
              opacity, transform: `translateX(${x}px)`,
              background: CARD, border: `1px solid ${BORDER}`,
              borderRadius: 14, padding: "14px 20px",
              display: "flex", alignItems: "center", gap: 16,
            }}>
              <span style={{
                background: `${a.color}22`, border: `1px solid ${a.color}44`,
                color: a.color, borderRadius: 100, padding: "3px 12px",
                fontSize: 11, fontWeight: 700, letterSpacing: "0.1em",
                whiteSpace: "nowrap",
              }}>{a.tag}</span>
              <span style={{ color: TEXT, fontSize: 15, fontWeight: 600, flex: 1 }}>
                {a.title}
              </span>
              <span style={{ color: MUTED, fontSize: 13 }}>{a.source}</span>
              <span style={{ color: PRIMARY, fontSize: 16 }}>↗</span>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

/* ─── SCENE 6: GO FOR LAUNCH (frames 750–870, 4s) ───────────── */
const SceneLaunch: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 12, stiffness: 80 } });

  const ring1 = interpolate(frame, [0, 60], [0, 1.4],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const ring2 = interpolate(frame, [10, 70], [0, 1.4],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const ring1op = interpolate(frame, [30, 70], [0.3, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const ring2op = interpolate(frame, [40, 80], [0.2, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  return (
    <AbsoluteFill style={{ display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center", gap: 28 }}>
      <Particles />

      {/* ripple rings */}
      <div style={{ position: "absolute",
        width: 500 * ring1, height: 500 * ring1,
        borderRadius: "50%", border: `2px solid ${EMERALD}`,
        opacity: ring1op,
      }} />
      <div style={{ position: "absolute",
        width: 500 * ring2, height: 500 * ring2,
        borderRadius: "50%", border: `2px solid ${EMERALD}`,
        opacity: ring2op,
      }} />

      {/* green glow */}
      <div style={{ position: "absolute",
        width: 400, height: 400, borderRadius: "50%",
        background: `radial-gradient(circle, ${EMERALD}25 0%, transparent 70%)`,
        opacity: s,
      }} />

      <div style={{ opacity: s, transform: `scale(${s})`, textAlign: "center" }}>
        <div style={{ fontSize: 36, marginBottom: 16 }}>✅</div>
        <div style={{ fontSize: 80, fontWeight: 900,
          background: `linear-gradient(135deg, ${EMERALD}, #86efac)`,
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          letterSpacing: "-0.02em", lineHeight: 1 }}>
          GO FOR LAUNCH
        </div>
        <div style={{ fontSize: 22, color: MUTED, marginTop: 16 }}>
          PMF Score: 95% · Failure Risk: 5.2% · CLV: $160/buyer
        </div>
        <div style={{ display: "flex", gap: 16, justifyContent: "center",
          marginTop: 28 }}>
          {["Samsung Galaxy Test A", "$320", "Asia-Pacific", "Mid-Range"].map((t, i) => (
            <div key={i} style={{ background: `${EMERALD}18`,
              border: `1px solid ${EMERALD}40`, borderRadius: 100,
              padding: "8px 20px", fontSize: 15, color: EMERALD, fontWeight: 600 }}>
              {t}
            </div>
          ))}
        </div>
      </div>
    </AbsoluteFill>
  );
};

/* ─── SCENE 7: Outro (frames 870–900, 1s) ───────────────────── */
const SceneOutro: React.FC = () => {
  const fade      = useFade(0, 20);
  const titleSlide = useSlideUp(5, 30, 20);
  const subSlide   = useSlideUp(15, 40, 20);

  return (
    <AbsoluteFill style={{ display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center", gap: 20 }}>
      <Particles />

      <div style={{ position: "absolute",
        width: 700, height: 700, borderRadius: "50%",
        background: `radial-gradient(circle, ${PRIMARY}18 0%, transparent 70%)`,
        opacity: fade,
      }} />

      <div style={{ ...titleSlide, textAlign: "center" }}>
        <div style={{ fontSize: 72, fontWeight: 900,
          background: `linear-gradient(135deg, ${TEXT} 0%, ${PRIMARY} 60%, ${VIOLET} 100%)`,
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          letterSpacing: "-0.02em" }}>
          Smartphone BI Agent
        </div>
      </div>

      <div style={{ ...subSlide, textAlign: "center", display: "flex",
        flexDirection: "column", gap: 12, alignItems: "center" }}>
        <div style={{ fontSize: 20, color: MUTED }}>
          XGBoost · Random Forest · KNN Retrieval · FastAPI · Next.js
        </div>
        <div style={{ display: "flex", gap: 12, marginTop: 8 }}>
          {["4 ML Pillars", "116K Records", "6 Regions", "Real-Time News"].map((t, i) => (
            <div key={i} style={{ background: `${PRIMARY}18`,
              border: `1px solid ${PRIMARY}33`, borderRadius: 100,
              padding: "6px 16px", fontSize: 13, color: PRIMARY, fontWeight: 600 }}>
              {t}
            </div>
          ))}
        </div>
      </div>
    </AbsoluteFill>
  );
};

/* ─── ROOT COMPOSITION ──────────────────────────────────────── */
export const BiAgentIntro: React.FC = () => {
  return (
    <AbsoluteFill style={{ background: BG, fontFamily:
      "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", color: TEXT }}>

      {/* Scene 1: Title           0  – 120  (4s) */}
      <Sequence from={0}   durationInFrames={120}><SceneTitle   /></Sequence>

      {/* Scene 2: Dataset        120 – 270  (5s) */}
      <Sequence from={120} durationInFrames={150}><SceneDataset /></Sequence>

      {/* Scene 3: Four Pillars   270 – 480  (7s) */}
      <Sequence from={270} durationInFrames={210}><ScenePillars /></Sequence>

      {/* Scene 4: Metrics        480 – 630  (5s) */}
      <Sequence from={480} durationInFrames={150}><SceneMetrics /></Sequence>

      {/* Scene 5: Live News      630 – 750  (4s) */}
      <Sequence from={630} durationInFrames={120}><SceneNews    /></Sequence>

      {/* Scene 6: GO FOR LAUNCH  750 – 870  (4s) */}
      <Sequence from={750} durationInFrames={120}><SceneLaunch  /></Sequence>

      {/* Scene 7: Outro          870 – 900  (1s) */}
      <Sequence from={870} durationInFrames={30}><SceneOutro   /></Sequence>
    </AbsoluteFill>
  );
};
