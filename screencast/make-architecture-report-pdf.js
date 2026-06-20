const { chromium } = require("playwright");
const path = require("path");

(async () => {
  const htmlPath =
    "file:///" +
    path.resolve(__dirname, "../docs/system_architecture_report.html").replace(/\\/g, "/");
  const outPath = path.resolve(__dirname, "../docs/System_Architecture_Report.pdf");

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  await page.goto(htmlPath, { waitUntil: "networkidle" });

  await page.pdf({
    path: outPath,
    format: "A4",
    printBackground: true,
    margin: { top: "0mm", bottom: "0mm", left: "0mm", right: "0mm" },
    displayHeaderFooter: false,
  });

  await browser.close();
  console.log("PDF saved:", outPath);
})();
