#!/usr/bin/env node

import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(SCRIPT_DIR, "..");
const RADAR_FILE = resolve(
  ROOT,
  "public/sources/kernel-inference-optimization-radar.html",
);
const PACIFIC_TIME_ZONE = "America/Los_Angeles";
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
  // GitHub titles and labels are untrusted. Escaping "<" prevents a literal
  // </script> sequence from terminating an embedded JSON script block.
  const serialized = JSON.stringify(value, null, 2).replaceAll("<", "\\u003c");
  return `${html.slice(0, contentStart)}\n${serialized}\n  ${html.slice(end)}`;
}

function githubTarget(item, expectedKind) {
  const url = new URL(item.url);
  const match = url.pathname.match(
    /^\/([^/]+)\/([^/]+)\/(issues|pull)\/(\d+)\/?$/,
  );
  if (!match) throw new Error(`Unsupported GitHub URL: ${item.url}`);
  const [, owner, repo, kind, number] = match;
  if (expectedKind === "issue" && kind !== "issues") {
    throw new Error(`Expected issue URL, received: ${item.url}`);
  }
  if (expectedKind === "pull" && kind !== "pull") {
    throw new Error(`Expected pull request URL, received: ${item.url}`);
  }
  if (Number(number) !== item.number) {
    throw new Error(`Number mismatch in ${item.url}`);
  }
  return { owner, repo, number: Number(number) };
}

async function githubApi(path) {
  const headers = {
    Accept: "application/vnd.github+json",
    "User-Agent": "jinnpan-kernel-inference-radar",
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
        await new Promise((resolveDelay) =>
          setTimeout(resolveDelay, attempt * 1_000),
        );
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

function dateOnly(isoTimestamp) {
  return isoTimestamp.slice(0, 10);
}

function labelNames(labels) {
  return labels
    .map((label) => (typeof label === "string" ? label : label.name))
    .filter(Boolean);
}

async function syncIssue(item) {
  const { owner, repo, number } = githubTarget(item, "issue");
  const live = await githubApi(`/repos/${owner}/${repo}/issues/${number}`);
  if (live.pull_request) {
    throw new Error(`${item.id} resolves to a pull request, not an issue`);
  }
  const assignees = live.assignees.map((assignee) => assignee.login);
  return {
    ...item,
    title: live.title,
    owner: assignees.length ? assignees.join(", ") : "Unassigned",
    updated: dateOnly(live.updated_at),
    comments: live.comments,
    githubState: live.state,
    githubStateReason: live.state_reason || null,
    githubUpdatedAt: live.updated_at,
    githubClosedAt: live.closed_at,
    githubAssignees: assignees,
    githubLabels: labelNames(live.labels),
  };
}

async function syncPullRequest(item) {
  const { owner, repo, number } = githubTarget(item, "pull");
  const live = await githubApi(`/repos/${owner}/${repo}/pulls/${number}`);
  return {
    ...item,
    title: live.title,
    updated: dateOnly(live.updated_at),
    state: live.state,
    isDraft: live.draft,
    merged: live.merged,
    githubUpdatedAt: live.updated_at,
    closedAt: live.closed_at,
    mergedAt: live.merged_at,
    author: live.user?.login || null,
    labels: labelNames(live.labels),
  };
}

const originalHtml = await readFile(RADAR_FILE, "utf8");
const issues = readJsonScript(originalHtml, "issue-data");
const pullRequests = readJsonScript(originalHtml, "pr-data");
const metadata = readJsonScript(originalHtml, "radar-meta");

console.log(
  `Syncing ${issues.length} issues and ${pullRequests.length} pull requests${token ? " with authentication" : " without authentication"}...`,
);

// Write nothing unless every tracked GitHub record was fetched successfully.
const [syncedIssues, syncedPullRequests] = await Promise.all([
  mapWithConcurrency(issues, 5, syncIssue),
  mapWithConcurrency(pullRequests, 5, syncPullRequest),
]);

const syncedAt = new Date().toISOString();
const activePullRequestCount = syncedPullRequests.filter(
  (pullRequest) => pullRequest.state === "open",
).length;
const repositories = [
  ...new Set(
    [...syncedIssues, ...syncedPullRequests].map((item) => {
      const target = githubTarget(
        item,
        item.url.includes("/pull/") ? "pull" : "issue",
      );
      return `${target.owner}/${target.repo}`;
    }),
  ),
].sort();

const syncedMetadata = {
  ...metadata,
  lastSuccessfulSync: syncedAt,
  timeZone: PACIFIC_TIME_ZONE,
  source: "GitHub REST API",
  trackedIssueCount: syncedIssues.length,
  trackedPullRequestCount: syncedPullRequests.length,
  activePullRequestCount,
  repositories,
  generatedBy: "scripts/sync-kernel-inference-radar.mjs",
};

let nextHtml = replaceJsonScript(originalHtml, "radar-meta", syncedMetadata);
nextHtml = replaceJsonScript(nextHtml, "issue-data", syncedIssues);
nextHtml = replaceJsonScript(nextHtml, "pr-data", syncedPullRequests);
await writeFile(RADAR_FILE, nextHtml, "utf8");

console.log(
  `Sync completed at ${syncedAt}: ${syncedIssues.length} issues, ${activePullRequestCount}/${syncedPullRequests.length} active pull requests.`,
);
