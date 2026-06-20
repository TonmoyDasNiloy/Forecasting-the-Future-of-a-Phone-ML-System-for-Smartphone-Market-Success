// Converts docs/technical_documentation.html → a styled PDF using headless Chromium.
const { chromium } = require("playwright");
const path = require("path");

(async () => {
  const htmlPath = "file:///" + path.resolve(__dirname, "../docs/technical_documentation.html").replace(/\\/g, "/");
  const outPath  = path.resolve(__dirname, "../docs/BI_Agent_Technical_Documentation.pdf");

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto(htmlPath, { waitUntil: "networkidle" });
  await page.pdf({
    path: outPath,
    format: "A4",
    printBackground: true,
    margin: { top: "16mm", bottom: "16mm", left: "14mm", right: "14mm" },
    displayHeaderFooter: true,
    headerTemplate: "<div></div>",
    footerTemplate:
      '<div style="width:100%;font-size:8px;color:#9ca3af;padding:0 14mm;display:flex;justify-content:space-between;">' +
      '<span>Smartphone BI Agent — Technical Documentation</span>' +
      '<span>Page <span class="pageNumber"></span> / <span class="totalPages"></span></span></div>',
  });
  await browser.close();
  console.log("PDF saved:", outPath);
})();
