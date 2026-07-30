#!/usr/bin/env node
/**
 * Regenerate the data-driven plates and tables in public/sources/kimi-k3-tuning.html
 * from the archived CSVs in data/kimi-k3-bsz1-throughput/results/.
 *
 * Why this exists: the deep dive quotes ~90 numbers. Typing them by hand once is
 * fine; keeping them correct across edits is not. Everything between a
 * `<!-- GEN:name -->` / `<!-- /GEN:name -->` marker pair is owned by this script,
 * so the page can never drift from the measurements. Prose stays hand-written —
 * only the figures are mechanical.
 *
 * Output is static inline SVG with pre-computed coordinates: no runtime <script>,
 * no charting library, nothing to break when JS is off.
 *
 *   node scripts/gen-k3-plates.mjs [--check]
 *
 * --check exits non-zero if the file would change, for use before a commit.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DATA = resolve(ROOT, "data/kimi-k3-bsz1-throughput/results");
const TARGET = resolve(ROOT, "public/sources/kimi-k3-tuning.html");

/* ---------------------------------------------------------------- csv ---- */

function readCsv(file) {
  const text = readFileSync(resolve(DATA, file), "utf8").trim();
  const [head, ...lines] = text.split("\n");
  const cols = head.split(",");
  return lines.map((line) => {
    // No quoted commas in these files, so a plain split is safe.
    const cells = line.split(",");
    return Object.fromEntries(cols.map((c, i) => [c, cells[i]]));
  });
}

const num = (row, key) => {
  const v = row[key];
  return v === undefined || v === "" || v === "NA" ? NaN : Number(v);
};

const AB = readCsv("sweep-ab-isl-osl.csv");
const C = readCsv("sweep-c-concurrency.csv");

const sweepA = AB.filter((r) => r.sweep === "A").sort((a, b) => +a.isl - +b.isl);
const sweepB = AB.filter((r) => r.sweep === "B").sort((a, b) => +a.osl - +b.osl);
const sweepC = [...C].sort((a, b) => +a.conc - +b.conc);

for (const [name, rows, want] of [
  ["A", sweepA, 7],
  ["B", sweepB, 4],
  ["C", sweepC, 7],
]) {
  if (rows.length !== want) {
    throw new Error(`sweep ${name}: expected ${want} rows, found ${rows.length}`);
  }
  const bad = rows.filter((r) => r.status !== "OK");
  if (bad.length) throw new Error(`sweep ${name}: ${bad.length} non-OK rows`);
}

/* ------------------------------------------------------------ derived ---- */

// step_ms: wall time of one speculative verify step. TPOT is time per *accepted*
// token, so multiplying back by accept_len recovers the step itself — the
// quantity that isolates per-step cost from draft quality.
const stepMs = (r) => num(r, "accept_len") * num(r, "median_tpot_ms");
// prefill_tps: ISL / TTFT. At concurrency 1 nothing else is in flight, so TTFT is
// the prefill of exactly this request.
const prefillTps = (r) => +r.isl / (num(r, "median_ttft_ms") / 1000);

const fmt = (v, d = 2) =>
  Number.isFinite(v)
    ? v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d })
    : "—";
const int = (v) => (Number.isFinite(v) ? Math.round(v).toLocaleString("en-US") : "—");
const tokLabel = (n) => (n >= 1024 ? `${n / 1024}K` : String(n));

/* ---------------------------------------------------------------- svg ---- */

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/**
 * House Chinese typography (CLAUDE.md): a half-width space follows 。 ， ： when
 * the next character is content. Applied here rather than post-hoc because this
 * script owns its output regions — a formatter run over the finished file would
 * be undone the next time the generator executes.
 */
const zhSpace = (s) =>
  String(s).replace(/([。，：])(?=[^\s。，：、；！？」』）】》…])/g, "$1 ");

const bi = (zh, en) =>
  `<span lang="en">${esc(en)}</span><span lang="zh">${esc(zhSpace(zh))}</span>`;

/** Nice axis maximum: 1/2/2.5/5 x 10^k above the data. */
function niceMax(max) {
  if (max <= 0) return 1;
  const mag = 10 ** Math.floor(Math.log10(max));
  for (const m of [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10]) {
    if (max <= mag * m) return mag * m;
  }
  return mag * 10;
}

/**
 * Line chart on a categorical x-axis. Series share the y-axis unless `axis:"right"`,
 * which gets its own scale and right-hand ticks — needed when plotting a rate
 * against a count (throughput vs accept length).
 */
function lineChart({
  categories,
  series,
  width = 780,
  height = 300,
  padL = 62,
  padR = 62,
  padT = 22,
  padB = 46,
  yLabel,
  yLabelRight,
  xLabel,
  yTicks = 5,
}) {
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const x = (i) =>
    padL + (categories.length === 1 ? plotW / 2 : (i * plotW) / (categories.length - 1));

  const leftSeries = series.filter((s) => s.axis !== "right");
  const rightSeries = series.filter((s) => s.axis === "right");
  const maxOf = (list) => niceMax(Math.max(...list.flatMap((s) => s.data.filter(Number.isFinite))));
  const maxL = leftSeries.length ? maxOf(leftSeries) : 1;
  const maxR = rightSeries.length ? maxOf(rightSeries) : 1;
  const yL = (v) => padT + plotH - (v / maxL) * plotH;
  const yR = (v) => padT + plotH - (v / maxR) * plotH;

  let out = `<svg viewBox="0 0 ${width} ${height}" role="img" class="plate-svg">`;

  // Horizontal rules double as the y grid; the hairline weight is the whole
  // visual idea of this entry, so no boxed frame.
  for (let t = 0; t <= yTicks; t++) {
    const v = (maxL / yTicks) * t;
    const yy = yL(v);
    out += `<line class="ax-grid" x1="${padL}" y1="${yy.toFixed(1)}" x2="${padL + plotW}" y2="${yy.toFixed(1)}"/>`;
    out += `<text class="ax-tick" x="${padL - 8}" y="${(yy + 3.5).toFixed(1)}" text-anchor="end">${
      maxL >= 100 ? int(v) : fmt(v, maxL >= 10 ? 0 : 1)
    }</text>`;
    if (rightSeries.length) {
      const vr = (maxR / yTicks) * t;
      out += `<text class="ax-tick" x="${padL + plotW + 8}" y="${(yy + 3.5).toFixed(1)}">${fmt(vr, 1)}</text>`;
    }
  }

  out += `<line class="ax-axis" x1="${padL}" y1="${padT + plotH}" x2="${padL + plotW}" y2="${padT + plotH}"/>`;

  categories.forEach((c, i) => {
    out += `<text class="ax-tick" x="${x(i).toFixed(1)}" y="${padT + plotH + 18}" text-anchor="middle">${esc(c)}</text>`;
  });
  if (xLabel) {
    out += `<text class="ax-label" x="${(padL + plotW / 2).toFixed(1)}" y="${height - 6}" text-anchor="middle">${xLabel}</text>`;
  }
  if (yLabel) {
    out += `<text class="ax-label" transform="translate(13,${(padT + plotH / 2).toFixed(1)}) rotate(-90)" text-anchor="middle">${yLabel}</text>`;
  }
  if (yLabelRight) {
    out += `<text class="ax-label" transform="translate(${width - 8},${(padT + plotH / 2).toFixed(1)}) rotate(-90)" text-anchor="middle">${yLabelRight}</text>`;
  }

  series.forEach((s) => {
    const yf = s.axis === "right" ? yR : yL;
    const pts = s.data
      .map((v, i) => (Number.isFinite(v) ? `${x(i).toFixed(1)},${yf(v).toFixed(1)}` : null))
      .filter(Boolean)
      .join(" ");
    out += `<polyline class="ser ser-${s.key}${s.dash ? " dashed" : ""}" points="${pts}"/>`;
    s.data.forEach((v, i) => {
      if (!Number.isFinite(v)) return;
      out += `<circle class="dot dot-${s.key}" cx="${x(i).toFixed(1)}" cy="${yf(v).toFixed(1)}" r="3"/>`;
      if (s.labels !== false) {
        const above = s.labelBelow ? 16 : -9;
        out += `<text class="ser-val val-${s.key}" x="${x(i).toFixed(1)}" y="${(yf(v) + above).toFixed(1)}" text-anchor="middle">${esc(
          s.format ? s.format(v) : int(v),
        )}</text>`;
      }
    });
  });

  return out + `</svg>`;
}

function legend(items) {
  return (
    `<div class="legend">` +
    items
      .map(
        (i) =>
          `<span class="legend-item"><span class="swatch sw-${i.key}"></span>${i.label}</span>`,
      )
      .join("") +
    `</div>`
  );
}

/* ------------------------------------------------------------- plates ---- */

const blocks = {};

// Plate: bsz=1 throughput and accept length against input length. Two series on
// split axes because the point of the plate is that they fall *together*.
blocks["plate-isl"] =
  lineChart({
    categories: sweepA.map((r) => tokLabel(+r.isl)),
    series: [
      {
        key: "tps",
        data: sweepA.map((r) => num(r, "out_tps")),
        format: (v) => fmt(v, 1),
      },
      {
        key: "acc",
        axis: "right",
        dash: true,
        data: sweepA.map((r) => num(r, "accept_len")),
        format: (v) => fmt(v, 2),
        labelBelow: true,
      },
    ],
    yLabel: bi("输出吞吐 tok/s", "Output tok/s"),
    yLabelRight: bi("accept_len", "Accept length"),
    xLabel: bi("输入长度 ISL（token，OSL 固定 1024，并发 1）", "Input length ISL (tokens; OSL 1024, concurrency 1)"),
  }) +
  legend([
    { key: "tps", label: bi("输出吞吐（左轴）", "Output throughput (left)") },
    { key: "acc", label: bi("accept_len（右轴，虚线）", "Accept length (right, dashed)") },
  ]);

// Plate: the same two quantities against concurrency. Deliberately the same
// encoding as plate-isl so the contrast is readable at a glance.
blocks["plate-conc"] =
  lineChart({
    categories: sweepC.map((r) => r.conc),
    series: [
      { key: "tps", data: sweepC.map((r) => num(r, "out_tps")), format: (v) => int(v) },
      {
        key: "acc",
        axis: "right",
        dash: true,
        data: sweepC.map((r) => num(r, "accept_len")),
        format: (v) => fmt(v, 2),
        labelBelow: true,
      },
    ],
    yLabel: bi("聚合输出吞吐 tok/s", "Aggregate output tok/s"),
    yLabelRight: bi("accept_len", "Accept length"),
    xLabel: bi("客户端并发（ISL/OSL 固定 1024）", "Client concurrency (ISL/OSL 1024)"),
  }) +
  legend([
    { key: "tps", label: bi("聚合吞吐（左轴）", "Aggregate throughput (left)") },
    { key: "acc", label: bi("accept_len（右轴，虚线）", "Accept length (right, dashed)") },
  ]);

// Plate: per-request throughput and median TTFT against concurrency — the cost
// side of the scaling curve, which the aggregate plot hides.
blocks["plate-cost"] =
  lineChart({
    categories: sweepC.map((r) => r.conc),
    series: [
      {
        key: "perreq",
        data: sweepC.map((r) => num(r, "out_tps") / num(r, "conc_ach")),
        format: (v) => fmt(v, 1),
      },
      {
        key: "ttft",
        axis: "right",
        dash: true,
        data: sweepC.map((r) => num(r, "median_ttft_ms")),
        format: (v) => int(v),
        labelBelow: true,
      },
    ],
    yLabel: bi("每请求 tok/s", "tok/s per request"),
    yLabelRight: bi("TTFT 中位 ms", "Median TTFT (ms)"),
    xLabel: bi("客户端并发（ISL/OSL 固定 1024）", "Client concurrency (ISL/OSL 1024)"),
  }) +
  legend([
    { key: "perreq", label: bi("每请求吞吐（左轴）", "Per-request throughput (left)") },
    { key: "ttft", label: bi("TTFT 中位（右轴，虚线）", "Median TTFT (right, dashed)") },
  ]);

// Plate: output length. Flat is the finding, so the axis is zoomed to make the
// flatness visible rather than compressing it against a zero baseline.
blocks["plate-osl"] = (() => {
  const rows = [...sweepB];
  const anchor = sweepA.find((r) => +r.isl === 1024);
  if (anchor && !rows.some((r) => +r.osl === 1024)) {
    rows.push({ ...anchor, osl: "1024", sweep: "B" });
  }
  rows.sort((a, b) => +a.osl - +b.osl);
  const data = rows.map((r) => num(r, "out_tps"));
  return (
    lineChart({
      categories: rows.map((r) => tokLabel(+r.osl)),
      series: [{ key: "tps", data, format: (v) => fmt(v, 1) }],
      height: 240,
      yTicks: 4,
      yLabel: bi("输出吞吐 tok/s", "Output tok/s"),
      xLabel: bi("输出长度 OSL（token，ISL 固定 1024，并发 1）", "Output length OSL (tokens; ISL 1024, concurrency 1)"),
    }) +
    `<p class="caption">${bi(
      `OSL 128 到 2048 之间吞吐几乎不动（${fmt(Math.max(...data.slice(0, 4)), 1)} 到 ${fmt(Math.min(...data.slice(0, 4)), 1)} tok/s）。OSL 4096 掉到 ${fmt(data[data.length - 1], 1)}，那是上下文自身增长带来的，与 ISL 曲线同源。`,
      `Throughput barely moves between OSL 128 and 2048 (${fmt(Math.max(...data.slice(0, 4)), 1)} down to ${fmt(Math.min(...data.slice(0, 4)), 1)} tok/s). The fall to ${fmt(data[data.length - 1], 1)} at OSL 4096 comes from the context growing during generation — the same effect as the ISL curve.`,
    )}</p>`
  );
})();

/* ------------------------------------------------------------- tables ---- */

function table({ headers, rows, align = [], cls = "" }) {
  const th = headers
    .map((h, i) => `<th${align[i] ? ` class="num"` : ""}>${h}</th>`)
    .join("");
  const tb = rows
    .map(
      (r) =>
        `<tr>${r
          .map((c, i) => `<td${align[i] ? ` class="num"` : ""}>${c}</td>`)
          .join("")}</tr>`,
    )
    .join("");
  return `<div class="table-wrap"><table class="${cls}"><thead><tr>${th}</tr></thead><tbody>${tb}</tbody></table></div>`;
}

const R = 1; // right-aligned numeric column marker

blocks["table-isl"] = table({
  headers: [
    "ISL",
    bi("输出 tok/s", "Output tok/s"),
    bi("总 tok/s", "Total tok/s"),
    bi("TTFT 中位", "Median TTFT"),
    bi("prefill tok/s", "Prefill tok/s"),
    bi("TPOT 中位", "Median TPOT"),
    "accept_len",
    bi("每 step 耗时", "Per-step"),
  ],
  align: [0, R, R, R, R, R, R, R],
  rows: sweepA.map((r) => [
    `<code>${tokLabel(+r.isl)}</code>`,
    fmt(num(r, "out_tps"), 2),
    fmt(num(r, "total_tps"), 1),
    `${int(num(r, "median_ttft_ms"))} ms`,
    int(prefillTps(r)),
    `${fmt(num(r, "median_tpot_ms"), 2)} ms`,
    fmt(num(r, "accept_len"), 3),
    `${fmt(stepMs(r), 1)} ms`,
  ]),
});

blocks["table-conc"] = table({
  headers: [
    bi("并发", "Conc"),
    bi("请求数", "Reqs"),
    bi("输出 tok/s", "Output tok/s"),
    bi("实测并发", "Achieved"),
    bi("TTFT 中位", "Median TTFT"),
    bi("TPOT 中位", "Median TPOT"),
    "accept_len",
    bi("每 step 耗时", "Per-step"),
    bi("每 step token", "Tok/step"),
    bi("每请求 tok/s", "Per-req tok/s"),
  ],
  align: [0, R, R, R, R, R, R, R, R, R],
  rows: sweepC.map((r) => [
    `<code>${r.conc}</code>`,
    r.np,
    `<strong>${fmt(num(r, "out_tps"), 2)}</strong>`,
    fmt(num(r, "conc_ach"), 2),
    `${int(num(r, "median_ttft_ms"))} ms`,
    `${fmt(num(r, "median_tpot_ms"), 2)} ms`,
    fmt(num(r, "accept_len"), 3),
    `${fmt(stepMs(r), 1)} ms`,
    fmt(num(r, "accept_len") * num(r, "conc_ach"), 1),
    fmt(num(r, "out_tps") / num(r, "conc_ach"), 1),
  ]),
});

blocks["table-marginal"] = (() => {
  const rows = [];
  for (let i = 1; i < sweepC.length; i++) {
    const a = sweepC[i - 1];
    const b = sweepC[i];
    const cx = +b.conc / +a.conc;
    const tx = num(b, "out_tps") / num(a, "out_tps");
    rows.push([
      `<code>${a.conc} → ${b.conc}</code>`,
      `${fmt(cx, 2)}x`,
      `${fmt(tx, 2)}x`,
      `${fmt((tx / cx) * 100, 1)}%`,
      `${int(num(a, "median_ttft_ms"))} → ${int(num(b, "median_ttft_ms"))} ms`,
    ]);
  }
  return table({
    headers: [
      bi("并发区间", "Segment"),
      bi("并发倍数", "Conc"),
      bi("吞吐倍数", "Throughput"),
      bi("线性度", "Efficiency"),
      bi("TTFT 中位变化", "Median TTFT"),
    ],
    align: [0, R, R, R, R],
    rows,
  });
})();

/* ------------------------------------------------- inline derived facts --- */

const a1k = sweepA.find((r) => +r.isl === 1024);
const a64k = sweepA.find((r) => +r.isl === 65536);
const c1 = sweepC.find((r) => +r.conc === 1);
const c48 = sweepC.find((r) => +r.conc === 48);

const lossX = num(a1k, "out_tps") / num(a64k, "out_tps");
const stepX = stepMs(a64k) / stepMs(a1k);
const accX = num(a1k, "accept_len") / num(a64k, "accept_len");

const accsC = sweepC.map((r) => num(r, "accept_len"));
const accsA = sweepA.map((r) => num(r, "accept_len"));

const facts = {
  "f-loss": `${fmt(lossX, 2)}x`,
  "f-step": `${fmt(stepX, 2)}x`,
  "f-acc": `${fmt(accX, 2)}x`,
  "f-product": `${fmt(stepX * accX, 2)}x`,
  "f-gain": `${fmt(num(c48, "out_tps") / num(c1, "out_tps"), 2)}x`,
  "f-peak": int(num(c48, "out_tps")),
  "f-perreq1": fmt(num(c1, "out_tps") / num(c1, "conc_ach"), 1),
  "f-perreq48": fmt(num(c48, "out_tps") / num(c48, "conc_ach"), 1),
  "f-spread-conc": `${fmt((Math.max(...accsC) / Math.min(...accsC) - 1) * 100, 1)}%`,
  "f-spread-isl": `${fmt((Math.max(...accsA) / Math.min(...accsA) - 1) * 100, 1)}%`,
  "f-acc-lo": fmt(Math.min(...accsC), 3),
  "f-acc-hi": fmt(Math.max(...accsC), 3),
  "f-prefill-peak": int(Math.max(...sweepA.map(prefillTps))),
  "f-prefill-64k": int(prefillTps(a64k)),
  "f-tpot1k": fmt(num(a1k, "median_tpot_ms"), 2),
  "f-tpot64k": fmt(num(a64k, "median_tpot_ms"), 2),
  "f-points": String(sweepA.length + sweepB.length + sweepC.length),
};

/* -------------------------------------------------------------- splice --- */

let html = readFileSync(TARGET, "utf8");
const before = html;
let spliced = 0;

for (const [name, body] of Object.entries(blocks)) {
  const re = new RegExp(
    `(<!-- GEN:${name} -->)[\\s\\S]*?(<!-- /GEN:${name} -->)`,
    "g",
  );
  if (!re.test(html)) throw new Error(`marker GEN:${name} not found in ${TARGET}`);
  html = html.replace(re, `$1\n${body}\n$2`);
  spliced++;
}

for (const [name, value] of Object.entries(facts)) {
  const re = new RegExp(
    `(<span class="gen" data-gen="${name}">)[\\s\\S]*?(</span>)`,
    "g",
  );
  if (!re.test(html)) throw new Error(`fact span ${name} not found in ${TARGET}`);
  html = html.replace(re, `$1${value}$2`);
  spliced++;
}

const changed = html !== before;

if (process.argv.includes("--check")) {
  console.log(
    changed
      ? "STALE — run `node scripts/gen-k3-plates.mjs` and commit the result"
      : `up to date (${spliced} regions)`,
  );
  process.exit(changed ? 1 : 0);
}

writeFileSync(TARGET, html);
console.log(
  `${changed ? "updated" : "unchanged"}: ${TARGET}\n` +
    `  ${Object.keys(blocks).length} plate/table regions, ${Object.keys(facts).length} inline facts\n` +
    `  sweeps: A=${sweepA.length} B=${sweepB.length} C=${sweepC.length} points`,
);
