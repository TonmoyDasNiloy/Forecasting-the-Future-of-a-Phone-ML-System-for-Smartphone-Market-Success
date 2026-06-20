// Full "someone using the system" demo:
//  1) welcome animation
//  2) dark/light toggle shown once
//  3) all 7 segments analyzed, each scrolled through incl. Live News
// Requires the app running: frontend http://localhost:3000  +  backend :8000
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const W = 1920, H = 1080;
const URL = "http://localhost:3000";
const OUT_DIR = path.join(__dirname, "out");
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

const CURSOR_SCRIPT = `
  (() => {
    if (document.getElementById('__cursor')) return;
    const c = document.createElement('div');
    c.id = '__cursor';
    c.style.cssText = [
      'position:fixed','top:0','left:0','z-index:2147483647','pointer-events:none',
      'width:22px','height:22px','margin-left:-11px','margin-top:-11px','border-radius:50%',
      'border:2px solid rgba(99,102,241,0.95)','background:rgba(99,102,241,0.28)',
      'box-shadow:0 0 14px rgba(99,102,241,0.8)','transition:transform .08s ease','will-change:transform'
    ].join(';');
    document.documentElement.appendChild(c);
    let x = innerWidth/2, y = innerHeight/2;
    const m = (e)=>{x=e.clientX;y=e.clientY;c.style.transform='translate('+x+'px,'+y+'px)';};
    addEventListener('mousemove', m, true);
    addEventListener('mousedown', ()=>{c.style.transform='translate('+x+'px,'+y+'px) scale(0.6)';}, true);
    addEventListener('mouseup',   ()=>{c.style.transform='translate('+x+'px,'+y+'px) scale(1)';}, true);
  })();
`;

// ── the 7 segments (only the fields that differ from app defaults need setting,
//    but we set all key fields explicitly each run for correctness) ──
// app defaults after reload: price 320, region Asia-Pacific, 5g ON, os android,
//   processor snapdragon, cores 8, ram 8, storage 128, battery 5000,
//   fastCharging ON, expandable ON, refresh 120
const SEGMENTS = [
  { tag: "BUDGET", model: "Spark Go 1", region: "South Asia", price: "95",
    os: "android", proc: "mediatek", cores: 4, ram: 3, storage: 32, battery: "5000",
    refresh: 60, fiveG: false, fastCharging: false, expandable: true },
  { tag: "MID-RANGE", model: "Galaxy Test A", region: "Asia-Pacific", price: "320",
    os: "android", proc: "snapdragon", cores: 8, ram: 8, storage: 128, battery: "5000",
    refresh: 120, fiveG: true, fastCharging: true, expandable: true },
  { tag: "PREMIUM", model: "Pixel Pro 9", region: "North America", price: "720",
    os: "android", proc: "snapdragon", cores: 8, ram: 12, storage: 256, battery: "5050",
    refresh: 120, fiveG: true, fastCharging: true, expandable: false },
  { tag: "FLAGSHIP", model: "Galaxy Ultra X", region: "North America", price: "1200",
    os: "android", proc: "snapdragon", cores: 8, ram: 16, storage: 512, battery: "5000",
    refresh: 144, fiveG: true, fastCharging: true, expandable: false },
  { tag: "FLAGSHIP iOS", model: "iPhone Ultra Pro", region: "Europe", price: "1300",
    os: "ios", proc: null, cores: 6, ram: 8, storage: 256, battery: "4400",
    refresh: 120, fiveG: true, fastCharging: true, expandable: false },
  { tag: "OVERPRICED", model: "Overpriced Mid", region: "Asia-Pacific", price: "750",
    os: "android", proc: "mediatek", cores: 8, ram: 8, storage: 128, battery: "5000",
    refresh: 90, fiveG: true, fastCharging: true, expandable: true },
  { tag: "UNDERPRICED", model: "Hidden Gem", region: "South Asia", price: "450",
    os: "android", proc: "snapdragon", cores: 8, ram: 12, storage: 256, battery: "5500",
    refresh: 144, fiveG: true, fastCharging: true, expandable: true },
];

(async () => {
  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: W, height: H },
    recordVideo: { dir: OUT_DIR, size: { width: W, height: H } },
  });
  await context.addInitScript(CURSOR_SCRIPT);
  const page = await context.newPage();

  async function moveTo(locator, steps = 24) {
    try {
      const b = await locator.boundingBox();
      if (!b) return;
      await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps });
      await sleep(180);
    } catch {}
  }
  async function clickEl(locator, steps = 22) {
    await moveTo(locator, steps);
    await locator.click({ timeout: 5000 }).catch(() => {});
    await sleep(200);
  }
  async function wheel(total, chunks = 12) {
    const step = total / chunks;
    for (let i = 0; i < chunks; i++) { await page.mouse.wheel(0, step); await sleep(100); }
  }

  console.log("Opening app…");
  await page.goto(URL).catch(() => {});
  await page.waitForLoadState("networkidle").catch(() => {});
  await sleep(3000);                       // ① welcome Siri orb

  // ② show dark / light toggle once
  console.log("Toggling theme…");
  const themeBtn = page.getByTestId("theme-toggle");
  await clickEl(themeBtn); await sleep(1400);   // -> light
  await clickEl(themeBtn); await sleep(1200);   // -> back to dark

  // ── run each segment ──
  for (let s = 0; s < SEGMENTS.length; s++) {
    const seg = SEGMENTS[s];
    console.log(`Segment ${s + 1}/7: ${seg.tag}`);
    if (s > 0) {
      await page.reload().catch(() => {});
      await page.waitForLoadState("networkidle").catch(() => {});
      await sleep(1500);
    }

    // Model (cursor + typing)
    const model = page.getByPlaceholder(/Enter model name/i);
    await moveTo(model); await model.click(); await page.keyboard.press("Control+A").catch(()=>{});
    await model.type(seg.model, { delay: 70 }); await sleep(300);

    // Region / OS / Processor (native selects)
    await page.locator('select:has(option[value="South Asia"])').selectOption(seg.region).catch(()=>{});
    await page.locator('select:has(option[value="ios"])').selectOption(seg.os).catch(()=>{});
    if (seg.proc) {
      await page.locator('select:has(option[value="mediatek"])').selectOption(seg.proc).catch(()=>{});
    }

    // Price (cursor + typing)
    const price = page.getByPlaceholder(/Enter price/i);
    await moveTo(price); await price.click(); await page.keyboard.press("Control+A").catch(()=>{});
    await price.type(seg.price, { delay: 70 }); await sleep(250);

    // Chips
    await clickEl(page.getByTestId(`cores-${seg.cores}`)).catch(()=>{});
    await clickEl(page.getByTestId(`ram-${seg.ram}`)).catch(()=>{});
    await clickEl(page.getByTestId(`storage-${seg.storage}`)).catch(()=>{});
    await clickEl(page.getByTestId(`refresh-${seg.refresh}`)).catch(()=>{});

    // Battery
    const bat = page.getByTestId("battery");
    await bat.click().catch(()=>{}); await page.keyboard.press("Control+A").catch(()=>{});
    await bat.type(seg.battery, { delay: 40 }).catch(()=>{}); await sleep(200);

    // Toggles (defaults: 5g ON, fast ON, expandable ON → click only to change)
    if (!seg.fiveG)        await clickEl(page.getByRole("button", { name: /5G Enabled/i }));
    if (!seg.fastCharging) await clickEl(page.getByRole("button", { name: /Fast Charging/i }));
    if (!seg.expandable)   await clickEl(page.getByRole("button", { name: /Expandable Storage/i }));

    // Analyze
    const analyze = page.getByRole("button", { name: /Analyze This Phone/i });
    await clickEl(analyze);
    await page.getByText(/6-Month Demand Forecast/i).waitFor({ timeout: 60000 }).catch(() => {});
    await sleep(1500);

    // Tour the dashboard, pausing on Live News
    await page.mouse.move(W / 2, H / 2, { steps: 8 });
    await wheel(750); await sleep(700);    // KPIs + demand
    await wheel(750); await sleep(700);    // customer profile
    await wheel(750); await sleep(700);    // competitor treemap
    await wheel(750); await sleep(700);    // price recommender
    await wheel(750); await sleep(700);    // PMF
    // ensure Live News is in view and wait for it to populate
    await page.getByText(/Market Intelligence/i).scrollIntoViewIfNeeded().catch(()=>{});
    await sleep(2600);                      // ③ show live news working
    await wheel(500); await sleep(1500);
    // back to top
    await wheel(-5200, 22); await sleep(900);
  }

  console.log("Finishing…");
  await context.close();
  await browser.close();

  const vids = fs.readdirSync(OUT_DIR).filter(f => f.endsWith(".webm"));
  if (vids.length) {
    const newest = vids.map(f => ({ f, t: fs.statSync(path.join(OUT_DIR, f)).mtimeMs }))
      .sort((a, b) => b.t - a.t)[0].f;
    fs.renameSync(path.join(OUT_DIR, newest), path.join(OUT_DIR, "segments-demo.webm"));
    console.log("Saved:", path.join(OUT_DIR, "segments-demo.webm"));
  }
})();
