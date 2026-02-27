#!/usr/bin/env node

import { execSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = resolve(fileURLToPath(import.meta.url), "..");
const ROOT = resolve(__dirname, "..");

const SCREENSHOT_DIR = join(ROOT, "tests/screenshots");
const BASELINE_DIR = join(SCREENSHOT_DIR, "baseline");
const CURRENT_DIR = join(SCREENSHOT_DIR, "current");
const DIFF_DIR = join(SCREENSHOT_DIR, "diff");

const DEV_PORT = 4321;
const DEV_URL = `http://localhost:${DEV_PORT}`;
const PROD_URL = "https://jinnpan.com";

const PAGES = [
  { path: "/en/", name: "home-en" },
  { path: "/zh/", name: "home-zh" },
  { path: "/en/blog", name: "blog-en" },
  { path: "/zh/blog", name: "blog-zh" },
  { path: "/en/blog/attention-mechanisms", name: "post-en" },
  { path: "/zh/blog/attention-mechanisms", name: "post-zh" },
];

const PIXEL_DIFF_THRESHOLD = 0.005; // 0.5%

// --- Argument parsing ---

const args = process.argv.slice(2);
const isBaseline = args.includes("--baseline");
const noScreenshots = args.includes("--no-screenshots");
const messageFlag = args.find((a) => a.startsWith("--message="));
const commitMessage = messageFlag ? messageFlag.split("=").slice(1).join("=") : null;

if (!isBaseline && !noScreenshots && !commitMessage) {
  // Allow running without message for testing (won't deploy)
}

// --- Helpers ---

function log(step, msg) {
  console.log(`\n[${"=".repeat(3)} ${step} ${"=".repeat(3)}] ${msg}`);
}

function logDetail(msg) {
  console.log(`  ${msg}`);
}

function run(cmd, opts = {}) {
  return execSync(cmd, {
    cwd: ROOT,
    stdio: opts.silent ? "pipe" : "inherit",
    encoding: "utf-8",
    ...opts,
  });
}

// --- Step 1: Build with retry ---

async function buildWithRetry(maxAttempts = 3) {
  log("BUILD", `Building site (max ${maxAttempts} attempts)...`);

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      logDetail(`Attempt ${attempt}/${maxAttempts}`);
      run("npm run build");
      logDetail("Build succeeded.");
      return;
    } catch (err) {
      const stderr = err.stderr || err.message || "";
      logDetail(`Build failed (attempt ${attempt}).`);

      if (attempt === maxAttempts) {
        console.error("\nBuild failed after all attempts. Errors:");
        console.error(stderr);
        process.exit(1);
      }

      // Parse common errors and suggest (don't auto-fix non-trivial)
      if (stderr.includes("missing frontmatter")) {
        logDetail("Detected: missing frontmatter field. Retrying...");
      } else if (stderr.includes("TypeScript")) {
        logDetail("Detected: TypeScript error. Retrying...");
      } else {
        logDetail("Retrying...");
      }
    }
  }
}

// --- Step 2: Start dev server ---

function startDevServer() {
  return new Promise((resolve, reject) => {
    log("DEV SERVER", `Starting Astro dev server on port ${DEV_PORT}...`);

    const proc = spawn("npx", ["astro", "dev", "--port", String(DEV_PORT)], {
      cwd: ROOT,
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, FORCE_COLOR: "0" },
    });

    let started = false;
    const timeout = setTimeout(() => {
      if (!started) {
        proc.kill("SIGTERM");
        reject(new Error("Dev server failed to start within 30s"));
      }
    }, 30000);

    proc.stdout.on("data", (data) => {
      const text = data.toString();
      if (text.includes("localhost") && !started) {
        started = true;
        clearTimeout(timeout);
        logDetail("Dev server ready.");
        resolve(proc);
      }
    });

    proc.stderr.on("data", () => {
      // Astro sometimes logs to stderr for non-errors
    });

    proc.on("error", (err) => {
      clearTimeout(timeout);
      reject(err);
    });

    proc.on("exit", (code) => {
      if (!started) {
        clearTimeout(timeout);
        reject(new Error(`Dev server exited with code ${code}`));
      }
    });
  });
}

// --- Step 3: Take screenshots ---

async function takeScreenshots(outputDir) {
  log("SCREENSHOTS", `Capturing ${PAGES.length} pages...`);

  const { chromium } = await import("@playwright/test");
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 },
    deviceScaleFactor: 2,
  });

  mkdirSync(outputDir, { recursive: true });

  for (const page of PAGES) {
    const p = await context.newPage();
    const url = `${DEV_URL}${page.path}`;
    logDetail(`${page.name}: ${url}`);

    await p.goto(url, { waitUntil: "networkidle" });
    await p.waitForTimeout(1500); // wait for CSS animations

    await p.screenshot({
      path: join(outputDir, `${page.name}.png`),
      fullPage: true,
    });

    await p.close();
  }

  await browser.close();
  logDetail(`Screenshots saved to ${outputDir}`);
}

// --- Step 4: Visual regression ---

async function compareScreenshots() {
  log("VISUAL REGRESSION", "Comparing against baselines...");

  const { default: pixelmatch } = await import("pixelmatch");
  const { PNG } = await import("pngjs");

  let failures = [];

  for (const page of PAGES) {
    const baselinePath = join(BASELINE_DIR, `${page.name}.png`);
    const currentPath = join(CURRENT_DIR, `${page.name}.png`);

    if (!existsSync(baselinePath)) {
      logDetail(`${page.name}: SKIP (no baseline)`);
      continue;
    }
    if (!existsSync(currentPath)) {
      logDetail(`${page.name}: FAIL (no current screenshot)`);
      failures.push({ name: page.name, reason: "missing current screenshot" });
      continue;
    }

    const baseline = PNG.sync.read(readFileSync(baselinePath));
    const current = PNG.sync.read(readFileSync(currentPath));

    // Dimension mismatch = layout shift
    if (baseline.width !== current.width || baseline.height !== current.height) {
      logDetail(
        `${page.name}: FAIL (dimension mismatch: ${baseline.width}x${baseline.height} vs ${current.width}x${current.height})`
      );
      failures.push({
        name: page.name,
        reason: `dimension mismatch: ${baseline.width}x${baseline.height} -> ${current.width}x${current.height}`,
      });
      continue;
    }

    const diff = new PNG({ width: baseline.width, height: baseline.height });
    const numDiffPixels = pixelmatch(
      baseline.data,
      current.data,
      diff.data,
      baseline.width,
      baseline.height,
      { threshold: 0.1 }
    );

    const totalPixels = baseline.width * baseline.height;
    const diffPercent = numDiffPixels / totalPixels;

    if (diffPercent > PIXEL_DIFF_THRESHOLD) {
      const diffPath = join(DIFF_DIR, `${page.name}-diff.png`);
      mkdirSync(DIFF_DIR, { recursive: true });
      writeFileSync(diffPath, PNG.sync.write(diff));

      logDetail(
        `${page.name}: FAIL (${(diffPercent * 100).toFixed(2)}% diff, threshold ${PIXEL_DIFF_THRESHOLD * 100}%)`
      );
      logDetail(`  Diff image: ${diffPath}`);
      failures.push({
        name: page.name,
        reason: `${(diffPercent * 100).toFixed(2)}% pixel diff`,
        diffPath,
      });
    } else {
      logDetail(
        `${page.name}: PASS (${(diffPercent * 100).toFixed(2)}% diff)`
      );
    }
  }

  if (failures.length > 0) {
    console.error("\nVisual regressions detected:");
    for (const f of failures) {
      console.error(`  - ${f.name}: ${f.reason}`);
      if (f.diffPath) console.error(`    Diff: ${f.diffPath}`);
    }
    process.exit(1);
  }

  logDetail("All pages pass visual regression.");
}

// --- Step 5: Functional checks ---

async function functionalChecks() {
  log("FUNCTIONAL CHECKS", "Verifying language switcher and root redirect...");

  const { chromium } = await import("@playwright/test");
  const browser = await chromium.launch();
  const context = await browser.newContext();

  // Check root redirect
  const rootPage = await context.newPage();
  await rootPage.goto(`${DEV_URL}/`, { waitUntil: "networkidle" });
  const rootUrl = rootPage.url();
  if (!rootUrl.includes("/en/")) {
    await browser.close();
    console.error(`Root redirect FAILED: / landed on ${rootUrl} (expected /en/)`);
    process.exit(1);
  }
  logDetail(`Root redirect: / -> ${rootUrl} OK`);
  await rootPage.close();

  // Check language switcher on /en/
  const enPage = await context.newPage();
  await enPage.goto(`${DEV_URL}/en/`, { waitUntil: "networkidle" });
  const enToggle = await enPage.$("a.lang-toggle");
  if (enToggle) {
    const href = await enToggle.getAttribute("href");
    if (!href || !href.includes("/zh/")) {
      await browser.close();
      console.error(`Language switcher FAILED on /en/: toggle href is "${href}" (expected /zh/)`);
      process.exit(1);
    }
    logDetail(`Language switcher on /en/: href="${href}" OK`);
  } else {
    await browser.close();
    console.error("Language switcher FAILED: a.lang-toggle not found on /en/");
    process.exit(1);
  }
  await enPage.close();

  // Check language switcher on /zh/
  const zhPage = await context.newPage();
  await zhPage.goto(`${DEV_URL}/zh/`, { waitUntil: "networkidle" });
  const zhToggle = await zhPage.$("a.lang-toggle");
  if (zhToggle) {
    const href = await zhToggle.getAttribute("href");
    if (!href || !href.includes("/en/")) {
      await browser.close();
      console.error(`Language switcher FAILED on /zh/: toggle href is "${href}" (expected /en/)`);
      process.exit(1);
    }
    logDetail(`Language switcher on /zh/: href="${href}" OK`);
  } else {
    await browser.close();
    console.error("Language switcher FAILED: a.lang-toggle not found on /zh/");
    process.exit(1);
  }
  await zhPage.close();

  await browser.close();
  logDetail("All functional checks passed.");
}

// --- Step 6: Deploy ---

function deploy(message) {
  log("DEPLOY", "Committing and pushing...");

  // Stage specific files, not everything
  const status = run("git status --porcelain", { silent: true }).trim();
  if (!status) {
    logDetail("No changes to commit.");
    return;
  }

  // Stage tracked modified files and any new files the user intended
  const files = status
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => line.slice(3)) // strip status prefix
    .filter(
      (f) =>
        !f.startsWith("tests/screenshots/current/") &&
        !f.startsWith("tests/screenshots/diff/") &&
        !f.includes(".env") &&
        !f.includes("node_modules/")
    );

  if (files.length === 0) {
    logDetail("No deployable changes found.");
    return;
  }

  logDetail(`Staging ${files.length} file(s):`);
  for (const f of files) {
    logDetail(`  ${f}`);
    run(`git add "${f}"`);
  }

  run(`git commit -m "${message}"`);
  run("git push");
  logDetail("Push complete.");
}

// --- Step 7: Verify deployment ---

async function verifyDeployment() {
  log("VERIFY", `Polling ${PROD_URL} for successful deployment...`);

  const maxWait = 180000; // 3 minutes
  const interval = 15000; // 15 seconds
  const start = Date.now();

  while (Date.now() - start < maxWait) {
    try {
      const res = await fetch(PROD_URL);
      if (res.ok) {
        logDetail(`${PROD_URL} returned ${res.status}. Deployment verified.`);
        return;
      }
      logDetail(`${PROD_URL} returned ${res.status}. Retrying in ${interval / 1000}s...`);
    } catch (err) {
      logDetail(`Fetch failed: ${err.message}. Retrying in ${interval / 1000}s...`);
    }
    await new Promise((r) => setTimeout(r, interval));
  }

  console.error(`Deployment verification timed out after ${maxWait / 1000}s.`);
  process.exit(1);
}

// --- Main ---

async function main() {
  let devProc = null;

  try {
    // Step 1: Build
    await buildWithRetry();

    if (isBaseline) {
      // Baseline mode: start server, take screenshots, done
      devProc = await startDevServer();
      await takeScreenshots(BASELINE_DIR);
      logDetail("\nBaseline screenshots captured. Commit them to git.");
      return;
    }

    if (!noScreenshots) {
      // Steps 2-4: Screenshots and visual regression
      devProc = await startDevServer();
      await takeScreenshots(CURRENT_DIR);
      await compareScreenshots();
    }

    // Step 5: Functional checks (need dev server)
    if (!devProc) {
      devProc = await startDevServer();
    }
    await functionalChecks();

    // Step 6: Deploy (only if message provided)
    if (commitMessage) {
      deploy(commitMessage);
      // Step 7: Verify
      await verifyDeployment();
    } else {
      logDetail("\nNo --message= flag provided. Skipping deploy.");
      logDetail("All checks passed. Ready to deploy.");
    }
  } finally {
    if (devProc) {
      devProc.kill("SIGTERM");
      logDetail("Dev server stopped.");
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
