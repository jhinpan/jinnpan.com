#!/usr/bin/env node

import { readdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(SCRIPT_DIR, "..");
const SERIES_DIR = resolve(ROOT, "public/sources/daily-sync");
const INDEX_FILE = resolve(SERIES_DIR, "index.html");
const API_VERSION = "2022-11-28";
const token = process.env.GITHUB_TOKEN || process.env.GH_TOKEN || "";

function jsonScriptBounds(html, id) {
  const opening = `<script id="${id}" type="application/json">`;
  const start = html.indexOf(opening);
  if (start === -1) throw new Error(`Missing JSON script #${id}`);
  const contentStart = start + opening.length;
  const end = html.indexOf("</script>", contentStart);
  if (end === -1) throw new Error(`Unclosed JSON script #${id}`);
  return { contentStart, end };
}

function readJsonScript(html, id) {
  const { contentStart, end } = jsonScriptBounds(html, id);
  return JSON.parse(html.slice(contentStart, end).trim());
}

function replaceJsonScript(html, id, value) {
  const { contentStart, end } = jsonScriptBounds(html, id);
  // PR titles are untrusted. Escaping "<" prevents a literal </script> sequence
  // from terminating an embedded JSON script block.
  const serialized = JSON.stringify(value, null, 2).replaceAll("<", "\\u003c");
  return `${html.slice(0, contentStart)}\n${serialized}\n  ${html.slice(end)}`;
}

async function githubApi(path) {
  const headers = {
    Accept: "application/vnd.github+json",
    "User-Agent": "jinnpan-daily-sync",
    "X-GitHub-Api-Version": API_VERSION,
  };
  if (token) headers.Authorization = `Bearer ${token}`;

  let lastError;
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      const response = await fetch(`https://api.github.com${path}`, { headers });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(
          `GitHub API ${response.status} for ${path}: ${detail.slice(0, 300)}`,
        );
      }
      return response.json();
    } catch (error) {
      lastError = error;
      if (attempt < 3) {
        await new Promise((delay) => setTimeout(delay, attempt * 1_000));
      }
    }
  }
  throw lastError;
}

async function mapWithConcurrency(items, concurrency, mapper) {
  const output = new Array(items.length);
  let cursor = 0;

  async function worker() {
    while (cursor < items.length) {
      const index = cursor;
      cursor += 1;
      output[index] = await mapper(items[index], index);
    }
  }

  const workerCount = Math.min(concurrency, items.length);
  await Promise.all(Array.from({ length: workerCount }, () => worker()));
  return output;
}

// The window is stated in the author's own days, so the search bounds carry that
// zone's offset. A bare date would make GitHub cut the window on UTC midnight and
// silently move merges into the neighbouring entry.
function zoneOffset(timeZone, onDate) {
  const formatter = new Intl.DateTimeFormat("en-US", {
    timeZone,
    timeZoneName: "longOffset",
  });
  const part = formatter
    .formatToParts(new Date(`${onDate}T12:00:00Z`))
    .find((p) => p.type === "timeZoneName");
  const match = /GMT([+-]\d{2}:\d{2})/.exec(part?.value || "");
  return match ? match[1] : "+00:00";
}

async function searchPullRequests(query) {
  const collected = [];
  for (let page = 1; page <= 5; page += 1) {
    const path = `/search/issues?q=${encodeURIComponent(query)}&per_page=100&page=${page}`;
    const result = await githubApi(path);
    collected.push(...result.items);
    if (collected.length >= result.total_count || result.items.length === 0) break;
  }
  return collected;
}

function repoFromApiUrl(repositoryUrl) {
  return repositoryUrl.replace(/^https:\/\/api\.github\.com\/repos\//, "");
}

async function pullRequestDetail(item, role, previousByNumber) {
  const repository = repoFromApiUrl(item.repository_url);
  const live = await githubApi(`/repos/${repository}/pulls/${item.number}`);
  const previous = previousByNumber.get(`${repository}#${item.number}`) || {};
  return {
    number: live.number,
    repository,
    url: live.html_url,
    title: live.title,
    // `short` and `note` are editorial. The sync refreshes measurements only, so a
    // hand-written line survives every later run.
    short: previous.short || live.title,
    role,
    mergedAt: live.merged_at,
    additions: live.additions,
    deletions: live.deletions,
    changedFiles: live.changed_files,
    commits: live.commits,
    note: previous.note || "",
  };
}

async function latestEntryFile() {
  const explicit = process.argv.find((arg) => arg.startsWith("--entry="));
  if (explicit) {
    const slug = explicit.slice("--entry=".length);
    return { slug, file: resolve(SERIES_DIR, `${slug}.html`) };
  }
  const slugs = (await readdir(SERIES_DIR))
    .filter((name) => /^\d{3}\.html$/.test(name))
    .map((name) => name.replace(/\.html$/, ""))
    .sort();
  if (!slugs.length) throw new Error(`No numbered entries in ${SERIES_DIR}`);
  const slug = slugs[slugs.length - 1];
  return { slug, file: resolve(SERIES_DIR, `${slug}.html`) };
}

const { slug, file } = await latestEntryFile();
const originalHtml = await readFile(file, "utf8");
const meta = readJsonScript(originalHtml, "sync-meta");
const previous = readJsonScript(originalHtml, "pr-data");

const previousByNumber = new Map(
  previous.map((pr) => [`${pr.repository}#${pr.number}`, pr]),
);

const offset = zoneOffset(meta.timeZone, meta.windowStart);
const range = `${meta.windowStart}T00:00:00${offset}..${meta.windowEnd}T23:59:59${offset}`;
const repoFilter = (meta.repositories || [])
  .map((repository) => `repo:${repository}`)
  .join(" ");
const base = `type:pr is:merged merged:${range} ${repoFilter}`.trim();

console.log(
  `Syncing entry ${slug} for ${meta.githubLogin} over ${range}${token ? " with authentication" : " without authentication"}...`,
);

// Write nothing unless every search and every pull request resolved.
const [authoredItems, reviewedItems] = await Promise.all([
  searchPullRequests(`author:${meta.githubLogin} ${base}`),
  searchPullRequests(`reviewed-by:${meta.githubLogin} -author:${meta.githubLogin} ${base}`),
]);

const [authored, reviewed] = await Promise.all([
  mapWithConcurrency(authoredItems, 4, (item) =>
    pullRequestDetail(item, "authored", previousByNumber),
  ),
  mapWithConcurrency(reviewedItems, 4, (item) =>
    pullRequestDetail(item, "reviewed", previousByNumber),
  ),
]);

const pullRequests = [...authored, ...reviewed].sort((a, b) => a.number - b.number);
const sum = (list, key) => list.reduce((acc, pr) => acc + (pr[key] || 0), 0);

const syncedAt = new Date().toISOString();
const syncedMeta = {
  ...meta,
  lastSuccessfulSync: syncedAt,
  source: "GitHub REST API (search/issues + pulls)",
  generatedBy: "scripts/sync-daily-sync.mjs",
};

let nextHtml = replaceJsonScript(originalHtml, "sync-meta", syncedMeta);
nextHtml = replaceJsonScript(nextHtml, "pr-data", pullRequests);
await writeFile(file, nextHtml, "utf8");

// The series index repeats the headline figures, so a stale index would contradict
// the entry it links to.
const indexHtml = await readFile(INDEX_FILE, "utf8");
const entries = readJsonScript(indexHtml, "entry-data");
const record = entries.find((entry) => entry.slug === slug);
if (record) {
  record.authored = authored.length;
  record.reviewed = reviewed.length;
  record.additions = sum(authored, "additions");
  record.deletions = sum(authored, "deletions");
  await writeFile(
    INDEX_FILE,
    replaceJsonScript(indexHtml, "entry-data", entries),
    "utf8",
  );
} else {
  console.warn(`Index has no record for entry ${slug}; left untouched.`);
}

console.log(
  `Sync completed at ${syncedAt}: ${authored.length} authored, ${reviewed.length} reviewed, ` +
    `+${sum(authored, "additions")} −${sum(authored, "deletions")} across ` +
    `${sum(authored, "changedFiles")} files.`,
);
