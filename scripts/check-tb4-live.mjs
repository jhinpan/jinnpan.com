#!/usr/bin/env node

import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const PAGE = resolve(ROOT, "public/sources/glm53-mxfp4-tb4-live.html");
const HUB = resolve(ROOT, "public/sources/index.html");
const EXPERIMENT = resolve(ROOT, "public/sources/glm53-mxfp4-mi355x.html");
const FEED =
  "https://gist.githubusercontent.com/jhinpan/e73f4d28e91332c8524f03a682923a1d/raw/tb4-status.json";

const requiredIds = [
  "status-dot",
  "status-title",
  "progress-ring",
  "progress-bar",
  "scored",
  "complete-tasks",
  "issues",
  "initial-data",
];
const forbidden = [
  [/\b(?:10|127|169\.254|172\.(?:1[6-9]|2\d|3[01])|192\.168)\./, "private IP"],
  [/\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/, "tailnet IP"],
  [/\b(?:smci|mia1)[-.\w]*\b/i, "internal hostname"],
  [/\/(?:root|home|tmp)\//, "local path"],
  [/\b(?:api[_-]?key|auth[_-]?token|password|secret)\b/i, "credential field"],
];

function embeddedJson(html, id) {
  const opening = `<script id="${id}" type="application/json">`;
  const start = html.indexOf(opening);
  if (start === -1) throw new Error(`missing #${id}`);
  const contentStart = start + opening.length;
  const end = html.indexOf("</script>", contentStart);
  if (end === -1) throw new Error(`unclosed #${id}`);
  return JSON.parse(html.slice(contentStart, end));
}

function validate(data, label) {
  if (data.schemaVersion !== 1) throw new Error(`${label}: schemaVersion`);
  if (data.targetScoredAttempts !== 315) throw new Error(`${label}: target`);
  if (data.scoredAttempts + data.remainingAttempts !== 315) {
    throw new Error(`${label}: scored + remaining`);
  }
  if (data.totalTasks !== 63 || data.attemptsPerTask !== 5) {
    throw new Error(`${label}: task contract`);
  }
  if (data.configuredConcurrency !== 24 || data.pools?.length !== 2) {
    throw new Error(`${label}: pool contract`);
  }
}

const [page, hub, experiment] = await Promise.all([
  readFile(PAGE, "utf8"),
  readFile(HUB, "utf8"),
  readFile(EXPERIMENT, "utf8"),
]);

for (const id of requiredIds) {
  if (!page.includes(`id="${id}"`)) throw new Error(`missing DOM id #${id}`);
}
if (!page.includes(FEED)) throw new Error("page does not use the canonical feed");
if (!hub.includes("./glm53-mxfp4-tb4-live.html")) {
  throw new Error("library hub does not link the live board");
}
if (!experiment.includes("./glm53-mxfp4-tb4-live.html")) {
  throw new Error("Experiment 004 does not link the live board");
}

const en = [...page.matchAll(/(?<!data-)lang="en"/g)].length;
const zh = [...page.matchAll(/(?<!data-)lang="zh"/g)].length;
if (en !== zh) throw new Error(`language pairs differ: en=${en}, zh=${zh}`);

for (const [pattern, label] of forbidden) {
  if (pattern.test(page)) throw new Error(`page contains ${label}`);
}

const initial = embeddedJson(page, "initial-data");
validate(initial, "embedded snapshot");

if (process.argv.includes("--live")) {
  const response = await fetch(`${FEED}?check=${Date.now()}`, {
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`live feed HTTP ${response.status}`);
  validate(await response.json(), "live feed");
}

console.log(
  `OK: language pairs ${en}/${zh}; embedded scored=${initial.scoredAttempts}; live=${process.argv.includes("--live")}`,
);
