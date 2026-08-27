#!/usr/bin/env node

// Refresh the MEASURED fields of the latest Daily Sync-Up entry from GitHub.
//
// The entry's #pr-data block mixes two kinds of fields:
//   measured  — state, dates, line counts, review decision, CI — owned by this script
//   editorial — short, thread, status, what, why, waiting, next, points … — hand-written
// Every run re-derives the measured fields and carries the editorial ones over
// untouched, so a nightly cron can keep the numbers honest without ever
// overwriting a sentence a person wrote.
//
// Discovery is bounded by the entry's window (PRs the author opened, reviewed or
// commented on that were updated inside it) plus an explicit `pinned` list for
// older work still worth tracking. Refresh is unbounded: once a PR is in the
// entry its live state is always re-read, so an "in flight" bar turns into a
// "landed" one on its own.
//
// Usage:
//   GITHUB_TOKEN=… node scripts/sync-daily-sync.mjs [--entry=001] [--dry-run]

import { readdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(SCRIPT_DIR, "..");
const SERIES_DIR = resolve(ROOT, "public/sources/daily-sync");
const INDEX_FILE = resolve(SERIES_DIR, "index.html");
const token = process.env.GITHUB_TOKEN || process.env.GH_TOKEN || "";
const dryRun = process.argv.includes("--dry-run");

if (!token) {
  console.error("GITHUB_TOKEN (or GH_TOKEN) is required: the GraphQL API does not serve anonymous requests.");
  process.exit(2);
}

/* ---------- JSON-in-HTML plumbing ---------- */

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

/* ---------- GitHub GraphQL ---------- */

async function graphql(query, variables) {
  let lastError;
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      const response = await fetch("https://api.github.com/graphql", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
          "User-Agent": "jinnpan-daily-sync",
        },
        body: JSON.stringify({ query, variables }),
      });
      if (!response.ok) {
        throw new Error(`GraphQL HTTP ${response.status}: ${(await response.text()).slice(0, 300)}`);
      }
      const payload = await response.json();
      if (payload.errors?.length) {
        // A search can succeed as a whole while individual nodes are withheld
        // (SAML-enforced orgs, private repos). Those come back null and are
        // dropped by the caller; only a response with no data at all is fatal.
        const message = payload.errors.map((e) => e.message).join("; ").slice(0, 400);
        if (!payload.data) throw new Error(`GraphQL: ${message}`);
        const withheld = payload.errors.filter((e) => e.extensions?.saml_failure || e.type === "FORBIDDEN").length;
        console.warn(`GraphQL returned partial data (${withheld || payload.errors.length} withheld node(s)): ${message.slice(0, 120)}…`);
      }
      return payload.data;
    } catch (error) {
      lastError = error;
      if (attempt < 3) await new Promise((r) => setTimeout(r, attempt * 1500));
    }
  }
  throw lastError;
}

const PR_FRAGMENT = `
  fragment PR on PullRequest {
    number title url state isDraft createdAt updatedAt mergedAt closedAt
    additions deletions changedFiles
    baseRefName headRefName
    reviewDecision
    author { login }
    mergedBy { login }
    repository { nameWithOwner isPrivate }
    labels(first: 12) { nodes { name } }
    commits(last: 1) {
      totalCount
      nodes { commit { statusCheckRollup { state } } }
    }
    reviews(last: 40) { nodes { author { login } state submittedAt } }
  }
`;

async function searchPullRequests(query) {
  const found = [];
  let after = null;
  for (let page = 0; page < 5; page += 1) {
    const data = await graphql(
      `${PR_FRAGMENT}
       query ($q: String!, $after: String) {
         search(type: ISSUE, query: $q, first: 100, after: $after) {
           pageInfo { hasNextPage endCursor }
           nodes { ...PR }
         }
       }`,
      { q: query, after },
    );
    found.push(...data.search.nodes.filter((n) => n && n.number));
    if (!data.search.pageInfo.hasNextPage) break;
    after = data.search.pageInfo.endCursor;
  }
  return found;
}

// One unreadable ref (a private org the token cannot see, SAML enforcement, a
// deleted fork) must not abort the whole sync: the caller keeps that PR's
// previous record and the rest of the entry still refreshes.
async function fetchPinned(refs) {
  const out = [];
  for (const ref of refs) {
    const match = /^([^/]+)\/([^#]+)#(\d+)$/.exec(ref);
    if (!match) {
      console.warn(`Ignoring malformed pinned ref "${ref}" (expected owner/repo#N).`);
      continue;
    }
    const [, owner, name, number] = match;
    try {
      const data = await graphql(
        `${PR_FRAGMENT}
         query ($owner: String!, $name: String!, $number: Int!) {
           repository(owner: $owner, name: $name) { pullRequest(number: $number) { ...PR } }
         }`,
        { owner, name, number: Number(number) },
      );
      if (data.repository?.pullRequest) out.push(data.repository.pullRequest);
      else console.warn(`Pinned ${ref} not found or not visible with this token; keeping its previous record.`);
    } catch (error) {
      console.warn(`Could not refresh ${ref}: ${String(error.message).slice(0, 160)}; keeping its previous record.`);
    }
  }
  return out;
}

/* ---------- shaping ---------- */

// The window is stated in the author's own days, so the search bounds carry that
// zone's offset. A bare date would make GitHub cut the window on UTC midnight and
// silently move activity into the neighbouring entry.
function zoneOffset(timeZone, onDate) {
  const part = new Intl.DateTimeFormat("en-US", { timeZone, timeZoneName: "longOffset" })
    .formatToParts(new Date(`${onDate}T12:00:00Z`))
    .find((p) => p.type === "timeZoneName");
  const match = /GMT([+-]\d{2}:\d{2})/.exec(part?.value || "");
  return match ? match[1] : "+00:00";
}

const CI_STATES = { SUCCESS: "success", FAILURE: "failure", ERROR: "failure", PENDING: "pending", EXPECTED: "pending" };

// Latest review state per reviewer; a COMMENTED review after an APPROVED one does
// not withdraw the approval, so only decisive states overwrite.
function summarizeReviews(nodes, login) {
  const latest = new Map();
  for (const review of nodes || []) {
    const who = review.author?.login;
    if (!who) continue;
    const prev = latest.get(who);
    if (review.state === "COMMENTED" && prev && prev !== "COMMENTED") continue;
    latest.set(who, review.state);
  }
  const approvedBy = [];
  const changesRequestedBy = [];
  for (const [who, state] of latest) {
    if (who === login) continue;
    if (state === "APPROVED") approvedBy.push(who);
    if (state === "CHANGES_REQUESTED") changesRequestedBy.push(who);
  }
  return { approvedBy: approvedBy.sort(), changesRequestedBy: changesRequestedBy.sort(), myReview: latest.get(login) || null };
}

// Hand-written from the PR thread, never derivable from the API. The sync
// refreshes measurements only, so these survive every later run.
const EDITORIAL_KEYS = [
  "short", "thread", "status", "what", "why", "waiting", "next", "points",
  "byline", "related", "decision", "hidden", "note",
];

function toRecord(node, login, previous, showPrivate) {
  const repository = node.repository.nameWithOwner;
  const reviews = summarizeReviews(node.reviews?.nodes, login);
  const rollup = node.commits?.nodes?.[0]?.commit?.statusCheckRollup?.state;
  // The page is public. A private repository's PR keeps its number, state and
  // timing on the board — the existence of work is not a secret — but its title
  // and any prose stay out of the published HTML unless the entry opts in.
  const scrub = Boolean(node.repository.isPrivate) && !showPrivate;
  const record = {
    number: node.number,
    repository,
    private: Boolean(node.repository.isPrivate),
    url: node.url,
    title: scrub ? "(private repository)" : node.title,
    author: node.author?.login || "ghost",
    role: node.author?.login === login ? "authored" : "reviewed",
    state: node.mergedAt ? "merged" : node.state === "CLOSED" ? "closed" : "open",
    isDraft: Boolean(node.isDraft),
    createdAt: node.createdAt,
    updatedAt: node.updatedAt,
    mergedAt: node.mergedAt,
    closedAt: node.closedAt,
    mergedBy: node.mergedBy?.login || null,
    additions: node.additions,
    deletions: node.deletions,
    changedFiles: node.changedFiles,
    commits: node.commits?.totalCount ?? 0,
    reviewDecision: node.reviewDecision || null,
    ciState: rollup ? CI_STATES[rollup] || "pending" : "none",
    approvedBy: reviews.approvedBy,
    changesRequestedBy: reviews.changesRequestedBy,
    myReview: reviews.myReview,
    base: node.baseRefName,
    head: node.headRefName,
    labels: scrub ? [] : (node.labels?.nodes || []).map((l) => l.name),
    short: scrub ? "(private repository)" : previous?.short || node.title,
  };
  for (const key of EDITORIAL_KEYS) {
    if (key === "short" || !previous || previous[key] === undefined) continue;
    if (scrub && !["thread", "status", "hidden"].includes(key)) continue;
    record[key] = previous[key];
  }
  return record;
}

/* ---------- main ---------- */

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
const login = meta.githubLogin;
const key = (repository, number) => `${repository}#${number}`;
const previousByKey = new Map(previous.map((pr) => [key(pr.repository, pr.number), pr]));

const offset = zoneOffset(meta.timeZone, meta.windowStart);
const range = `${meta.windowStart}T00:00:00${offset}..${meta.windowEnd}T23:59:59${offset}`;
const repoFilter = (meta.repositories || []).map((r) => `repo:${r}`).join(" ");
const excludeFilter = (meta.exclude || []).map((r) => `-repo:${r}`).join(" ");
const base = `is:pr updated:${range} ${repoFilter} ${excludeFilter}`.replace(/\s+/g, " ").trim();

console.log(`Syncing entry ${slug} for ${login} over ${range}${dryRun ? " (dry run)" : ""}…`);

// Write nothing unless every search resolved.
const [authoredNodes, reviewedNodes, commentedNodes, pinnedNodes] = await Promise.all([
  searchPullRequests(`author:${login} ${base}`),
  searchPullRequests(`reviewed-by:${login} -author:${login} ${base}`),
  searchPullRequests(`commenter:${login} -author:${login} ${base}`),
  fetchPinned(meta.pinned || []),
]);

const excluded = new Set(meta.exclude || []);
const byKey = new Map();
for (const node of [...authoredNodes, ...reviewedNodes, ...commentedNodes, ...pinnedNodes]) {
  const repository = node.repository.nameWithOwner;
  if (excluded.has(repository)) continue;
  byKey.set(key(repository, node.number), node);
}
// A PR already in the entry stays in it even if the search no longer returns it:
// re-read it when the token can, and otherwise carry the previous record forward
// untouched rather than silently dropping it from the page.
const stale = [];
for (const [k, prev] of previousByKey) {
  if (!byKey.has(k)) stale.push(k);
}
const carried = [];
if (stale.length) {
  const refreshed = await fetchPinned(stale);
  for (const node of refreshed) byKey.set(key(node.repository.nameWithOwner, node.number), node);
  for (const k of stale) if (!byKey.has(k)) carried.push(previousByKey.get(k));
}

const pullRequests = [
  ...[...byKey.values()].map((node) =>
    toRecord(node, login, previousByKey.get(key(node.repository.nameWithOwner, node.number)), Boolean(meta.showPrivate)),
  ),
  ...carried,
].sort((a, b) => a.repository.localeCompare(b.repository) || a.number - b.number);
if (carried.length) console.warn(`${carried.length} record(s) carried forward unrefreshed: ${carried.map((pr) => key(pr.repository, pr.number)).join(", ")}`);

const added = pullRequests.filter((pr) => !previousByKey.has(key(pr.repository, pr.number)));
const visible = pullRequests.filter((pr) => !pr.hidden);
const authored = visible.filter((pr) => pr.role === "authored");
const reviewed = visible.filter((pr) => pr.role === "reviewed");
const count = (list, predicate) => list.filter(predicate).length;
const sum = (list, k) => list.reduce((acc, pr) => acc + (pr[k] || 0), 0);
const landedAuthored = authored.filter((pr) => pr.state === "merged");

// "decide" is an editorial status: open, but parked pending a call. It is
// counted apart from in-flight so the headline number stays honest.
const summary = {
  landed: landedAuthored.length,
  inFlight: count(authored, (pr) => pr.state === "open" && pr.status !== "decide"),
  undecided: count(visible, (pr) => pr.state === "open" && pr.status === "decide"),
  reviewedMerged: count(reviewed, (pr) => pr.state === "merged"),
  reviewing: count(reviewed, (pr) => pr.state === "open" && pr.status !== "decide"),
  closed: count(authored, (pr) => pr.state === "closed"),
  additions: sum(landedAuthored, "additions"),
  deletions: sum(landedAuthored, "deletions"),
  repositories: new Set(visible.map((pr) => pr.repository)).size,
};

console.log(
  `${pullRequests.length} pull requests (${added.length} new): ` +
    `${summary.landed} landed, ${summary.inFlight} in flight, ${summary.undecided} to decide, ${summary.closed} closed; ` +
    `reviewed ${summary.reviewedMerged} merged + ${summary.reviewing} open; ` +
    `+${summary.additions} −${summary.deletions} across ${summary.repositories} repositories.`,
);
for (const pr of added) console.log(`  + ${pr.repository}#${pr.number} [${pr.role}/${pr.state}] ${pr.title}`);

if (dryRun) process.exit(0);

const syncedAt = new Date().toISOString();
const syncedMeta = {
  ...meta,
  lastSuccessfulSync: syncedAt,
  source: "GitHub GraphQL API (search + pullRequest)",
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
  Object.assign(record, {
    landed: summary.landed,
    inFlight: summary.inFlight,
    undecided: summary.undecided,
    reviewed: summary.reviewedMerged + summary.reviewing,
    closed: summary.closed,
    additions: summary.additions,
    deletions: summary.deletions,
    repositories: summary.repositories,
  });
  await writeFile(INDEX_FILE, replaceJsonScript(indexHtml, "entry-data", entries), "utf8");
} else {
  console.warn(`Index has no record for entry ${slug}; left untouched.`);
}

console.log(`Sync completed at ${syncedAt}.`);
