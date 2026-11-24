#!/usr/bin/env node

// Build the data behind the PR board (src/pages/[lang]/prs.astro) from GitHub.
//
// The board answers three questions about open pull requests:
//   follow-up    mine, and the next move is mine
//   review       someone else's, and it waits on my review
//   in-progress  mine, and it waits on someone else (or is still a draft)
// Every lane is derived from GitHub state; nothing on the board is typed by hand.
// The only editorial input is src/data/pr-board.config.json: the login, the
// importance tier of each repository, per-PR pins and the stale thresholds.
//
// The site is public, so the board covers public repositories only. A private
// repository never reaches the output, whatever the token can see.
//
// Usage:
//   node scripts/sync-pr-board.mjs [--dry-run]
// CI passes GITHUB_TOKEN. Without a token the request goes through the gh CLI,
// which signs it with its own stored login, so no token is exported to a shell.

import { execFile } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const CONFIG_FILE = resolve(ROOT, "src/data/pr-board.config.json");
const OUTPUT_FILE = resolve(ROOT, "src/data/pr-board.json");
const token = process.env.GITHUB_TOKEN || process.env.GH_TOKEN || "";
const dryRun = process.argv.includes("--dry-run");
const DAY_MS = 24 * 60 * 60 * 1000;

/* ---------- GitHub GraphQL ---------- */

async function postWithToken(body) {
  const response = await fetch("https://api.github.com/graphql", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      "User-Agent": "jinnpan-pr-board",
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`GraphQL HTTP ${response.status}: ${(await response.text()).slice(0, 300)}`);
  }
  return response.json();
}

function postWithGh(body) {
  return new Promise((done, fail) => {
    const child = execFile(
      "gh",
      ["api", "graphql", "--input", "-"],
      { maxBuffer: 64 * 1024 * 1024 },
      (error, stdout, stderr) => {
        // gh exits non-zero when the response carries errors but still prints the payload.
        try {
          done(JSON.parse(stdout));
        } catch {
          fail(new Error(`gh api graphql failed: ${String(stderr || error?.message).slice(0, 300)}`));
        }
      },
    );
    child.stdin.end(JSON.stringify(body));
  });
}

async function graphql(query, variables) {
  let lastError;
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      const payload = await (token ? postWithToken : postWithGh)({ query, variables });
      if (payload.errors?.length) {
        // A search can succeed while single nodes are withheld (SAML-enforced
        // orgs). Those come back null and are dropped by the caller.
        const message = payload.errors.map((e) => e.message).join("; ").slice(0, 400);
        if (!payload.data) throw new Error(`GraphQL: ${message}`);
        console.warn(`GraphQL returned partial data: ${message.slice(0, 160)}`);
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
    number title url isDraft createdAt updatedAt
    additions deletions
    reviewDecision mergeable
    author { login }
    repository { nameWithOwner isPrivate }
    commits(last: 1) { nodes { commit { committedDate statusCheckRollup { state } } } }
    reviews(last: 50) { nodes { author { login __typename } state submittedAt } }
    comments(last: 30) { nodes { author { login __typename } createdAt } }
    reviewRequests(first: 10) {
      nodes { requestedReviewer { __typename ... on User { login } ... on Team { name } } }
    }
  }
`;

async function searchPullRequests(query) {
  const found = [];
  let after = null;
  // GitHub search returns at most 1,000 results: 20 pages of 50.
  for (let page = 0; page < 20; page += 1) {
    const data = await graphql(
      `${PR_FRAGMENT}
       query ($q: String!, $after: String) {
         search(type: ISSUE, query: $q, first: 50, after: $after) {
           pageInfo { hasNextPage endCursor }
           nodes { ...PR }
         }
       }`,
      { q: query, after },
    );
    found.push(...data.search.nodes.filter((node) => node && node.number));
    if (!data.search.pageInfo.hasNextPage) break;
    after = data.search.pageInfo.endCursor;
  }
  return found;
}

/* ---------- classification ---------- */

const CI_STATES = { SUCCESS: "success", FAILURE: "failure", ERROR: "failure", PENDING: "pending", EXPECTED: "pending" };

const latest = (dates) => dates.filter(Boolean).sort().at(-1) || null;
const daysBetween = (from, to) => Math.floor((new Date(to) - new Date(from)) / DAY_MS);

// Latest decisive review per reviewer. A later COMMENTED review does not
// withdraw an approval or a change request.
function reviewStates(reviews) {
  const states = new Map();
  for (const review of reviews) {
    const who = review.author?.login;
    if (!who) continue;
    const previous = states.get(who);
    if (review.state === "COMMENTED" && previous && previous !== "COMMENTED") continue;
    states.set(who, review.state);
  }
  return states;
}

function importanceOf(config, repo, number) {
  const pinned = config.pins?.[`${repo}#${number}`];
  if (pinned) return pinned;
  const matches = (patterns = []) =>
    patterns.some((pattern) => (pattern.endsWith("/*") ? repo.startsWith(pattern.slice(0, -1)) : pattern === repo));
  if (matches(config.importance?.high)) return "high";
  if (matches(config.importance?.low)) return "low";
  return "medium";
}

function baseRecord(config, node, role) {
  const repo = node.repository.nameWithOwner;
  const rollup = node.commits?.nodes?.[0]?.commit?.statusCheckRollup?.state;
  return {
    repo,
    number: node.number,
    title: node.title,
    url: node.url,
    author: node.author?.login || "ghost",
    role,
    importance: importanceOf(config, repo, node.number),
    draft: Boolean(node.isDraft),
    ci: rollup ? CI_STATES[rollup] || "pending" : "none",
    createdAt: node.createdAt,
    updatedAt: node.updatedAt,
    additions: node.additions,
    deletions: node.deletions,
  };
}

// A pull request I opened. It needs follow-up when the next move is mine.
function classifyAuthored(config, node, now) {
  const login = config.login;
  const reviews = node.reviews?.nodes || [];
  const comments = node.comments?.nodes || [];
  const isMe = (author) => author?.login === login;
  const isOtherHuman = (author) => author && author.__typename !== "Bot" && author.login !== login;

  const states = reviewStates(reviews);
  const changesRequestedBy = [...states].filter(([who, state]) => who !== login && state === "CHANGES_REQUESTED").map(([who]) => who).sort();
  const approvedBy = [...states].filter(([who, state]) => who !== login && state === "APPROVED").map(([who]) => who).sort();

  // My last move: a push, a comment, or a reply in a review thread.
  const myLast = latest([
    node.createdAt,
    node.commits?.nodes?.[0]?.commit?.committedDate,
    ...comments.filter((c) => isMe(c.author)).map((c) => c.createdAt),
    ...reviews.filter((r) => isMe(r.author)).map((r) => r.submittedAt),
  ]);
  // Their last word. An approval is not a question, so it does not count here.
  const theirs = [
    ...comments.filter((c) => isOtherHuman(c.author)).map((c) => ({ at: c.createdAt, by: c.author.login })),
    ...reviews
      .filter((r) => isOtherHuman(r.author) && r.state !== "APPROVED" && r.state !== "DISMISSED")
      .map((r) => ({ at: r.submittedAt, by: r.author.login })),
  ].sort((a, b) => (a.at < b.at ? -1 : 1));
  const theirLast = theirs.at(-1);

  const record = baseRecord(config, node, "author");
  const idleDays = daysBetween(node.updatedAt, now);
  const staleAfter = node.isDraft ? config.staleDays.draft : config.staleDays.ready;

  const reasons = [];
  if (changesRequestedBy.length || node.reviewDecision === "CHANGES_REQUESTED") {
    reasons.push({ code: "changes-requested", by: changesRequestedBy });
  }
  if (record.ci === "failure") reasons.push({ code: "ci-failing" });
  if (node.mergeable === "CONFLICTING") reasons.push({ code: "conflict" });
  if (node.reviewDecision === "APPROVED" && !node.isDraft) reasons.push({ code: "approved", by: approvedBy });
  if (theirLast && theirLast.at > myLast) {
    reasons.push({ code: "reply-needed", by: [theirLast.by], days: daysBetween(theirLast.at, now) });
  }
  if (idleDays >= staleAfter) reasons.push({ code: "stale", days: idleDays });

  if (reasons.length) return { ...record, lane: "follow-up", reasons };

  const waitingOn = (node.reviewRequests?.nodes || [])
    .map((request) => request.requestedReviewer?.login || request.requestedReviewer?.name)
    .filter(Boolean)
    .sort();
  if (node.isDraft) return { ...record, lane: "in-progress", reasons: [{ code: "draft" }] };
  if (record.ci === "pending") return { ...record, lane: "in-progress", reasons: [{ code: "ci-running" }] };
  return { ...record, lane: "in-progress", reasons: [{ code: "waiting-review", by: waitingOn }] };
}

// Someone else's pull request that I reviewed before. It waits on me again when
// my last review was not an approval and commits arrived after it.
function needsReReview(config, node) {
  const mine = (node.reviews?.nodes || []).filter((r) => r.author?.login === config.login);
  const last = mine.sort((a, b) => (a.submittedAt < b.submittedAt ? -1 : 1)).at(-1);
  if (!last || last.state === "APPROVED") return false;
  const pushedAt = node.commits?.nodes?.[0]?.commit?.committedDate;
  return Boolean(pushedAt && pushedAt > last.submittedAt);
}

/* ---------- main ---------- */

const config = JSON.parse(await readFile(CONFIG_FILE, "utf8"));
const login = config.login;
const now = new Date().toISOString();
const scope = `is:pr is:open is:public archived:false ${(config.exclude || []).map((r) => `-repo:${r}`).join(" ")}`.trim();

console.log(`Building the PR board for ${login}${dryRun ? " (dry run)" : ""} via ${token ? "GITHUB_TOKEN" : "gh"}…`);

// Write nothing unless every search resolved.
const [authored, requested, reviewed] = await Promise.all([
  searchPullRequests(`${scope} author:${login}`),
  searchPullRequests(`${scope} review-requested:${login}`),
  searchPullRequests(`${scope} reviewed-by:${login} -author:${login}`),
]);

const key = (node) => `${node.repository.nameWithOwner}#${node.number}`;
const isPublic = (node) => node.repository.isPrivate === false;
const records = new Map();

for (const node of authored.filter(isPublic)) {
  records.set(key(node), classifyAuthored(config, node, now));
}
for (const node of requested.filter(isPublic)) {
  if (node.author?.login === login) continue;
  records.set(key(node), { ...baseRecord(config, node, "reviewer"), lane: "review", reasons: [{ code: "review-requested" }] });
}
for (const node of reviewed.filter(isPublic)) {
  if (records.has(key(node)) || !needsReReview(config, node)) continue;
  records.set(key(node), { ...baseRecord(config, node, "reviewer"), lane: "review", reasons: [{ code: "re-review" }] });
}

const prs = [...records.values()].sort((a, b) => a.repo.localeCompare(b.repo) || a.number - b.number);
const count = (lane) => prs.filter((pr) => pr.lane === lane).length;
console.log(
  `${prs.length} open pull requests in ${new Set(prs.map((pr) => pr.repo)).size} public repositories: ` +
    `${count("follow-up")} follow-up, ${count("review")} to review, ${count("in-progress")} in progress.`,
);

if (dryRun) process.exit(0);

const board = { generatedAt: now, login, staleDays: config.staleDays, importance: config.importance, prs };
await writeFile(OUTPUT_FILE, `${JSON.stringify(board, null, 2)}\n`);
console.log(`Wrote ${OUTPUT_FILE}`);
