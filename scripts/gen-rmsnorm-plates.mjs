#!/usr/bin/env node
/**
 * Regenerate the data-driven plates, tables and inline figures in
 * public/sources/quack-rmsnorm-hopper.html from the archived sweep results in
 * data/quack-rmsnorm-hopper/results/.
 *
 * Why this exists: the experiment record quotes well over a hundred numbers
 * across two GPUs and three providers. Typing them by hand guarantees drift
 * the first time the sweep is re-run. Everything between a `<!-- GEN:name -->`
 * / `<!-- /GEN:name -->` marker pair, and every `<span class="gen"
 * data-gen="name">`, is owned by this script. Prose stays hand-written.
 *
 * Output is static inline SVG with pre-computed coordinates — no runtime
 * script, no chart library, so the page renders identically without JS.
 *
 *   node scripts/gen-rmsnorm-plates.mjs [--check]
 *
 * --check exits non-zero if the file would change, for use before a commit.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const DATA = resolve(ROOT, "data/quack-rmsnorm-hopper/results");
const TARGET = resolve(ROOT, "public/sources/quack-rmsnorm-hopper.html");

const HOSTS = [
  { key: "h200", label: "H200" },
  { key: "h100", label: "H100" },
];
/** Shape order is the harness's COMPACT_SHAPES, smallest working set first. */
const SHAPES = [
  [1, 4096],
  [256, 4096],
  [512, 4096],
  [4096, 3000],
  [4096, 4096],
  [32768, 1024],
  [32768, 2048],
  [32768, 4096],
  [32768, 8192],
];
const DTYPE_MODES = [
  ["float16", "float16"],
  ["float16", "float32"],
  ["bfloat16", "bfloat16"],
  ["bfloat16", "float32"],
  ["float32", "float32"],
];
/** The reference dtype for every per-shape figure: 16-bit activations and weight. */
const REF = ["bfloat16", "bfloat16"];
const BIG = [32768, 8192];

/**
 * Split one CSV record. `provider_detail` carries a comma — "(CuTe, autotuned)"
 * — so the writer quotes it and a plain `split(",")` silently shifts every
 * later column. Handles quoted fields and doubled quotes; that is all this
 * schema needs.
 */
function splitRow(line) {
  const cells = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i += 1;
        } else quoted = false;
      } else cur += ch;
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      cells.push(cur);
      cur = "";
    } else cur += ch;
  }
  cells.push(cur);
  return cells;
}

function readCsv(path) {
  const text = readFileSync(path, "utf8").trim();
  const [head, ...lines] = text.split("\n");
  const cols = splitRow(head);
  return lines.map((line) => {
    const cells = splitRow(line);
    return Object.fromEntries(cols.map((c, i) => [c, cells[i]]));
  });
}

function loadHost(key) {
  const rows = readCsv(resolve(DATA, key, "results.csv"));
  const env = JSON.parse(readFileSync(resolve(DATA, key, "environment.json"), "utf8"));
  const index = new Map();
  for (const r of rows) {
    index.set(
      [r.provider, r.operation, r.m, r.n, r.activation_dtype, r.weight_dtype].join("|"),
      r,
    );
  }
  const get = (provider, op, [m, n], [act, w] = REF) => {
    const row = index.get([provider, op, m, n, act, w].join("|"));
    if (!row) throw new Error(`${key}: missing ${provider} ${op} ${m}x${n} ${act}/${w}`);
    return row;
  };
  return { key, rows, env, get };
}

const hosts = Object.fromEntries(HOSTS.map((h) => [h.key, loadHost(h.key)]));

// ---------------------------------------------------------------- formatting
const f1 = (v) => v.toFixed(1);
const f2 = (v) => v.toFixed(2);
/** Sub-1% cells are real latency-bound measurements, not missing data. */
const pct = (v) => (v < 1 ? `${v.toFixed(2)}%` : `${v.toFixed(1)}%`);
const us = (v) => (v < 10 ? v.toFixed(2) : v.toFixed(1));
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const shapeLabel = ([m, n]) => `${m}x${n}`;

// -------------------------------------------------------------------- charts
/**
 * Grouped column chart. One column per shape, `series.length` bars per group.
 * Coordinates are absolute so the SVG needs no layout at render time.
 */
function columnChart({
  series,
  categories,
  width = 760,
  height = 300,
  yMax,
  yLabel,
  valueFmt,
  refLines = [],
  tickFmt = (v) => f1(v).replace(/\.0$/, ""),
}) {
  const padL = 54;
  const padR = 14;
  const padT = 18;
  const padB = 54;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const bandW = plotW / categories.length;
  const groupW = bandW * 0.62;
  const barW = groupW / series.length;
  const y = (v) => padT + plotH - (v / yMax) * plotH;

  const parts = [];
  // horizontal grid + y ticks
  const ticks = 5;
  for (let i = 0; i <= ticks; i += 1) {
    const v = (yMax / ticks) * i;
    const yy = y(v).toFixed(1);
    parts.push(`<line class="ax-grid" x1="${padL}" y1="${yy}" x2="${width - padR}" y2="${yy}"/>`);
    parts.push(`<text class="ax-tick" x="${padL - 8}" y="${(Number(yy) + 3.5).toFixed(1)}" text-anchor="end">${tickFmt(v)}</text>`);
  }
  parts.push(`<line class="ax-axis" x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}"/>`);

  // Measured ceilings, drawn behind the bars so a bar can visibly touch one.
  for (const r of refLines) {
    const yy = y(r.value).toFixed(1);
    parts.push(
      `<line class="ax-roof roof-${r.cls}" x1="${padL}" y1="${yy}" x2="${width - padR}" y2="${yy}"/>`,
    );
    parts.push(
      `<text class="ax-roof-label roof-${r.cls}" x="${width - padR}" y="${(Number(yy) - 4).toFixed(1)}" text-anchor="end">${esc(r.label)}</text>`,
    );
  }

  categories.forEach((cat, ci) => {
    const gx = padL + bandW * ci + (bandW - groupW) / 2;
    series.forEach((s, si) => {
      const v = s.data[ci];
      const bx = gx + barW * si;
      const by = y(Math.max(v, 0));
      const bh = Math.max(padT + plotH - by, 0.8);
      parts.push(
        `<rect class="bar bar-${s.cls}" x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${(barW - 1.6).toFixed(1)}" height="${bh.toFixed(1)}"><title>${esc(cat)} ${esc(s.name)}: ${esc(valueFmt(v))}</title></rect>`,
      );
    });
    parts.push(
      `<text class="ax-tick" x="${(padL + bandW * ci + bandW / 2).toFixed(1)}" y="${padT + plotH + 16}" text-anchor="middle">${esc(cat)}</text>`,
    );
  });

  // legend
  const legendY = padT + plotH + 40;
  let lx = padL;
  series.forEach((s) => {
    parts.push(`<rect class="bar bar-${s.cls}" x="${lx}" y="${legendY - 8}" width="10" height="10"/>`);
    parts.push(`<text class="ax-label" x="${lx + 15}" y="${legendY + 1}">${esc(s.name)}</text>`);
    lx += 22 + s.name.length * 6.4;
  });
  parts.push(
    `<text class="ax-label" x="10" y="${padT + plotH / 2}" transform="rotate(-90 10 ${(padT + plotH / 2).toFixed(1)})" text-anchor="middle">${esc(yLabel)}</text>`,
  );

  return `<svg viewBox="0 0 ${width} ${height}" role="img" class="plate-svg">${parts.join("")}</svg>`;
}

/** Diverging column chart around a zero baseline, for signed gains. */
function gainChart({ series, categories, width = 760, height = 280, yMin, yMax }) {
  const padL = 54;
  const padR = 14;
  const padT = 18;
  const padB = 54;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const bandW = plotW / categories.length;
  const groupW = bandW * 0.6;
  const barW = groupW / series.length;
  const y = (v) => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH;
  const zero = y(0);

  const parts = [];
  const step = 5;
  for (let v = Math.ceil(yMin / step) * step; v <= yMax; v += step) {
    const yy = y(v).toFixed(1);
    parts.push(`<line class="ax-grid" x1="${padL}" y1="${yy}" x2="${width - padR}" y2="${yy}"/>`);
    parts.push(`<text class="ax-tick" x="${padL - 8}" y="${(Number(yy) + 3.5).toFixed(1)}" text-anchor="end">${v > 0 ? "+" : ""}${v}</text>`);
  }
  parts.push(`<line class="ax-axis" x1="${padL}" y1="${zero.toFixed(1)}" x2="${width - padR}" y2="${zero.toFixed(1)}"/>`);

  categories.forEach((cat, ci) => {
    const gx = padL + bandW * ci + (bandW - groupW) / 2;
    series.forEach((s, si) => {
      const v = s.data[ci];
      const bx = gx + barW * si;
      const top = v >= 0 ? y(v) : zero;
      const bh = Math.max(Math.abs(y(v) - zero), 0.8);
      parts.push(
        `<rect class="bar bar-${s.cls}" x="${bx.toFixed(1)}" y="${top.toFixed(1)}" width="${(barW - 1.6).toFixed(1)}" height="${bh.toFixed(1)}"><title>${esc(cat)} ${esc(s.name)}: ${v >= 0 ? "+" : ""}${f1(v)}%</title></rect>`,
      );
    });
    parts.push(
      `<text class="ax-tick" x="${(padL + bandW * ci + bandW / 2).toFixed(1)}" y="${padT + plotH + 16}" text-anchor="middle">${esc(cat)}</text>`,
    );
  });

  const legendY = padT + plotH + 40;
  let lx = padL;
  series.forEach((s) => {
    parts.push(`<rect class="bar bar-${s.cls}" x="${lx}" y="${legendY - 8}" width="10" height="10"/>`);
    parts.push(`<text class="ax-label" x="${lx + 15}" y="${legendY + 1}">${esc(s.name)}</text>`);
    lx += 22 + s.name.length * 6.4;
  });
  parts.push(
    `<text class="ax-label" x="10" y="${padT + plotH / 2}" transform="rotate(-90 10 ${(padT + plotH / 2).toFixed(1)})" text-anchor="middle">autotune gain (%)</text>`,
  );
  return `<svg viewBox="0 0 ${width} ${height}" role="img" class="plate-svg">${parts.join("")}</svg>`;
}

// --------------------------------------------------------------- data slices
const peakPct = (host, provider, op, shape, mode = REF) =>
  Number(hosts[host].get(provider, op, shape, mode).peak_bw_pct);
const medianUs = (host, provider, op, shape, mode = REF) =>
  Number(hosts[host].get(provider, op, shape, mode).median_us);
const gbps = (host, provider, op, shape, mode = REF) =>
  Number(hosts[host].get(provider, op, shape, mode).logical_gbps);
const spreadPct = (host, provider, op, shape, mode = REF) => {
  const r = hosts[host].get(provider, op, shape, mode);
  return ((Number(r.p90_us) - Number(r.p10_us)) / Number(r.median_us)) * 100;
};
/** The config path a caller should actually deploy for this cell. */
const best = (host, op, shape, mode = REF) => {
  const a = hosts[host].get("quack", op, shape, mode);
  const t = hosts[host].get("quack_tuned", op, shape, mode);
  return Number(t.median_us) < Number(a.median_us)
    ? { row: t, path: "tuned" }
    : { row: a, path: "analytical" };
};
/** Positive = the autotuned config is faster than the analytical one. */
const gain = (host, op, shape, mode = REF) =>
  (1 - medianUs(host, "quack_tuned", op, shape, mode) / medianUs(host, "quack", op, shape, mode)) * 100;

const cats = SHAPES.map(shapeLabel);

const blocks = {};

/**
 * Achieved bandwidth in GB/s, both config paths and both hosts, with each
 * host's measured roofline drawn as a ceiling. Absolute units so the numbers
 * transfer to another machine; the ceilings carry the efficiency reading that
 * a percentage would have given.
 */
const roofH200 = hosts.h200.env.achievable_bandwidth.median_gbps;
const roofH100 = hosts.h100.env.achievable_bandwidth.median_gbps;
const gbpsPlate = (op) =>
  columnChart({
    categories: cats,
    yMax: Math.ceil((Math.max(roofH200, roofH100) * 1.08) / 500) * 500,
    yLabel: "achieved bandwidth (GB/s)",
    valueFmt: (v) => `${f1(v)} GB/s`,
    tickFmt: (v) => String(Math.round(v)),
    refLines: [
      { value: roofH200, label: `H200 roofline ${f1(roofH200)}`, cls: "a" },
      { value: roofH100, label: `H100 roofline ${f1(roofH100)}`, cls: "b" },
    ],
    series: [
      { name: "H200 analytical", cls: "c", data: SHAPES.map((s) => gbps("h200", "quack", op, s)) },
      { name: "H200 tuned", cls: "a", data: SHAPES.map((s) => gbps("h200", "quack_tuned", op, s)) },
      { name: "H100 analytical", cls: "d", data: SHAPES.map((s) => gbps("h100", "quack", op, s)) },
      { name: "H100 tuned", cls: "b", data: SHAPES.map((s) => gbps("h100", "quack_tuned", op, s)) },
    ],
  });

blocks["plate-fwd"] = gbpsPlate("fwd");
blocks["plate-bwd"] = gbpsPlate("bwd");

{
  const series = [];
  for (const h of HOSTS) {
    series.push({
      name: `${h.label} fwd`,
      cls: h.key === "h200" ? "a" : "b",
      data: SHAPES.map((s) => gain(h.key, "fwd", s)),
    });
  }
  for (const h of HOSTS) {
    series.push({
      name: `${h.label} bwd`,
      cls: h.key === "h200" ? "c" : "d",
      data: SHAPES.map((s) => gain(h.key, "bwd", s)),
    });
  }
  const all = series.flatMap((s) => s.data);
  const lo = Math.min(-5, Math.floor(Math.min(...all) / 5) * 5);
  const hi = Math.max(5, Math.ceil(Math.max(...all) / 5) * 5);
  blocks["plate-gain"] = gainChart({ categories: cats, series, yMin: lo, yMax: hi });
}

/**
 * The absolute reference table, one per operation. Latency and achieved
 * bandwidth are the primary measurements; percent-of-peak is derived and shown
 * beside them so the ratio can be checked rather than taken on trust.
 */
const absTable = (op) => {
  const head =
    `<thead><tr><th rowspan="2">shape</th><th class="num" rowspan="2">moved</th>` +
    HOSTS.map((h) => `<th class="num" colspan="4">${h.label}</th>`).join("") +
    `</tr><tr>${HOSTS.map(
      () =>
        `<th class="num">anal. us</th><th class="num">tuned us</th><th class="num">best GB/s</th><th class="num">% peak</th>`,
    ).join("")}</tr></thead>`;
  const body = SHAPES.map((s) => {
    const bytes = Number(hosts.h200.get("quack", op, s).logical_bytes);
    const moved = bytes >= 1024 ** 2 ? `${f1(bytes / 1024 ** 2)} MiB` : `${f1(bytes / 1024)} KiB`;
    const cells = HOSTS.map((h) => {
      const b = best(h.key, op, s);
      return (
        `<td class="num">${us(medianUs(h.key, "quack", op, s))}</td>` +
        `<td class="num">${us(medianUs(h.key, "quack_tuned", op, s))}</td>` +
        `<td class="num">${f1(Number(b.row.logical_gbps))}</td>` +
        `<td class="num">${pct(Number(b.row.peak_bw_pct))}</td>`
      );
    }).join("");
    return `<tr><td><code>${shapeLabel(s)}</code></td><td class="num">${moved}</td>${cells}</tr>`;
  }).join("");
  return `<table>${head}<tbody>${body}</tbody></table>`;
};

blocks["table-fwd"] = absTable("fwd");
blocks["table-bwd"] = absTable("bwd");

blocks["table-dtype"] = (() => {
  const head =
    `<thead><tr><th rowspan="2">activation / weight</th>` +
    HOSTS.map(
      (h) => `<th class="num" colspan="2">${h.label} fwd</th><th class="num" colspan="2">${h.label} bwd</th>`,
    ).join("") +
    `</tr><tr>${HOSTS.map(
      () => `<th class="num">us</th><th class="num">GB/s</th><th class="num">us</th><th class="num">GB/s</th>`,
    ).join("")}</tr></thead>`;
  const body = DTYPE_MODES.map((mode) => {
    const cells = HOSTS.map((h) =>
      ["fwd", "bwd"]
        .map((op) => {
          const b = best(h.key, op, BIG, mode);
          return `<td class="num">${us(Number(b.row.median_us))}</td><td class="num">${f1(Number(b.row.logical_gbps))}</td>`;
        })
        .join(""),
    ).join("");
    return `<tr><td><code>${mode[0]} / ${mode[1]}</code></td>${cells}</tr>`;
  }).join("");
  return `<table>${head}<tbody>${body}</tbody></table>`;
})();

blocks["table-env"] = (() => {
  const rows = [
    ["GPU", (h) => h.env.gpu.name],
    ["visible index", (h) => h.env.visibility.CUDA_VISIBLE_DEVICES],
    ["SMs", (h) => String(h.env.gpu.compute_units)],
    ["reported L2", (h) => `${(h.env.gpu.l2_cache_bytes_reported / 1024 / 1024).toFixed(0)} MiB`],
    ["PyTorch", (h) => h.env.versions.torch],
    ["CUDA", (h) => h.env.versions.cuda],
    ["commit", (h) => (h.env.git_commit || "").slice(0, 7)],
    [
      "achievable bandwidth",
      (h) =>
        `${f1(h.env.achievable_bandwidth.median_gbps)} GB/s (${h.env.achievable_bandwidth.best_probe.replace(/_/g, " ")})`,
    ],
    [
      "contention canary",
      (h) => `${h.env.contention_canary.closing_over_opening.toFixed(3)} (${h.env.contention_canary.quiet ? "quiet" : "NOISY"})`,
    ],
    ["result rows", (h) => String(h.env.result_rows)],
  ];
  const head = `<thead><tr><th>property</th>${HOSTS.map((h) => `<th>${h.label}</th>`).join("")}</tr></thead>`;
  const body = rows
    .map(
      ([label, fn]) =>
        `<tr><td>${esc(label)}</td>${HOSTS.map((h) => `<td>${esc(fn(hosts[h.key]))}</td>`).join("")}</tr>`,
    )
    .join("");
  return `<table>${head}<tbody>${body}</tbody></table>`;
})();

// -------------------------------------------------------------- inline spans
const inline = {};
for (const h of HOSTS) {
  const k = h.key;
  for (const op of ["fwd", "bwd"]) {
    // "best" = the better of the two config paths, which is the number a
    // future implementation should actually be measured against.
    inline[`${k}-${op}-best`] = pct(
      Math.max(...SHAPES.flatMap((s) => ["quack", "quack_tuned"].map((p) => peakPct(k, p, op, s)))),
    );
    inline[`${k}-${op}-big`] = pct(peakPct(k, "quack_tuned", op, BIG));
    inline[`${k}-${op}-big-analytical`] = pct(peakPct(k, "quack", op, BIG));
    inline[`${k}-${op}-big-best`] = pct(
      Math.max(peakPct(k, "quack", op, BIG), peakPct(k, "quack_tuned", op, BIG)),
    );
    const g = gain(k, op, BIG);
    inline[`${k}-${op}-big-gain`] = `${g >= 0 ? "+" : ""}${f1(g)}%`;
    // Absolute figures at the widest row — the numbers another machine's run
    // can actually be placed next to.
    const b = best(k, op, BIG);
    inline[`${k}-${op}-big-us`] = `${us(Number(b.row.median_us))} us`;
    inline[`${k}-${op}-big-gbps`] = `${f1(Number(b.row.logical_gbps))} GB/s`;
    inline[`${k}-${op}-big-analytical-us`] = `${us(medianUs(k, "quack", op, BIG))} us`;
    inline[`${k}-${op}-big-tuned-us`] = `${us(medianUs(k, "quack_tuned", op, BIG))} us`;
    inline[`${k}-${op}-big-spread`] = `${f1(spreadPct(k, "quack_tuned", op, BIG))}%`;
  }
  inline[`${k}-big-bytes`] = `${f1(Number(hosts[k].get("quack", "bwd", BIG).logical_bytes) / 1024 ** 2)} MiB`;
  inline[`${k}-bw`] = `${f1(hosts[k].env.achievable_bandwidth.median_gbps)} GB/s`;
  inline[`${k}-canary`] = hosts[k].env.contention_canary.closing_over_opening.toFixed(3);
  inline[`${k}-torch`] = hosts[k].env.versions.torch;
}
inline["rows-per-gpu"] = String(hosts.h200.env.result_rows);
inline["cells-per-gpu"] = String(SHAPES.length * DTYPE_MODES.length * 2);

// ------------------------------------------------------------------- splice
let html = readFileSync(TARGET, "utf8");
const before = html;

for (const [name, payload] of Object.entries(blocks)) {
  const re = new RegExp(`(<!-- GEN:${name} -->)([\\s\\S]*?)(<!-- /GEN:${name} -->)`);
  if (!re.test(html)) throw new Error(`marker GEN:${name} not found in ${TARGET}`);
  html = html.replace(re, `$1\n${payload}\n$3`);
}

for (const [name, value] of Object.entries(inline)) {
  const re = new RegExp(`(<span class="gen" data-gen="${name}">)([\\s\\S]*?)(</span>)`, "g");
  html = html.replace(re, `$1${value}$3`);
}

const unresolved = [...html.matchAll(/data-gen="([a-z0-9-]+)"/g)]
  .map((m) => m[1])
  .filter((n) => !(n in inline));
if (unresolved.length) {
  throw new Error(`unknown data-gen names in page: ${[...new Set(unresolved)].join(", ")}`);
}

if (process.argv.includes("--check")) {
  if (html !== before) {
    console.error("quack-rmsnorm-hopper.html is stale — run: node scripts/gen-rmsnorm-plates.mjs");
    process.exit(1);
  }
  console.log("quack-rmsnorm-hopper.html is up to date");
} else {
  writeFileSync(TARGET, html);
  console.log(
    `regenerated ${Object.keys(blocks).length} blocks and ${Object.keys(inline).length} inline figures`,
  );
}
