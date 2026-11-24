// Typed view of pr-board.json, which scripts/sync-pr-board.mjs writes from GitHub.
import type { Lang } from "@lib/utils";
import raw from "./pr-board.json";

export type Lane = "follow-up" | "review" | "in-progress";
export type Importance = "high" | "medium" | "low";

export type ReasonCode =
  | "changes-requested"
  | "ci-failing"
  | "conflict"
  | "approved"
  | "reply-needed"
  | "stale"
  | "review-requested"
  | "re-review"
  | "waiting-review"
  | "ci-running"
  | "draft";

export type Reason = { code: ReasonCode; by?: string[]; days?: number };

export type BoardPr = {
  repo: string;
  number: number;
  title: string;
  url: string;
  author: string;
  role: "author" | "reviewer";
  importance: Importance;
  lane: Lane;
  reasons: Reason[];
  draft: boolean;
  ci: "success" | "failure" | "pending" | "none";
  createdAt: string;
  updatedAt: string;
  additions: number;
  deletions: number;
};

export type Board = {
  generatedAt: string;
  login: string;
  staleDays: { ready: number; draft: number };
  importance: { high?: string[]; low?: string[] };
  prs: BoardPr[];
};

// One row of the board: an importance tier or a repository, split into the three lanes.
export type Swimlane = {
  title: string;
  note?: string;
  tier?: Importance;
  open: boolean;
  prs: BoardPr[];
};

export const BOARD = raw as Board;

export const LANES: Lane[] = ["follow-up", "review", "in-progress"];
export const IMPORTANCE: Importance[] = ["high", "medium", "low"];

// Most pressing first: the script lists each PR's reasons in this order too.
const SEVERITY: ReasonCode[] = [
  "changes-requested",
  "ci-failing",
  "conflict",
  "approved",
  "reply-needed",
  "review-requested",
  "re-review",
  "stale",
  "waiting-review",
  "ci-running",
  "draft",
];

export function byUrgency(a: BoardPr, b: BoardPr) {
  const rank = (pr: BoardPr) => SEVERITY.indexOf(pr.reasons[0].code);
  return rank(a) - rank(b) || (a.updatedAt < b.updatedAt ? 1 : -1);
}

// Ages are measured against the refresh time, not the build or view time.
export function daysBeforeRefresh(date: string) {
  return Math.max(0, Math.floor((Date.parse(BOARD.generatedAt) - Date.parse(date)) / 86_400_000));
}

const people = (names: string[] = [], lang: Lang) => {
  if (!names.length) return "";
  const shown = names.slice(0, 2).join(lang === "zh" ? "、" : ", ");
  return names.length > 2 ? `${shown} +${names.length - 2}` : shown;
};

export function reasonText(reason: Reason, lang: Lang) {
  const who = people(reason.by, lang);
  const zh: Record<ReasonCode, string> = {
    "changes-requested": who ? `${who} 要求修改` : "要求修改",
    "ci-failing": "CI 失败",
    conflict: "有合并冲突",
    approved: "已批准， 未合并",
    "reply-needed": `待回复 ${who}`,
    stale: `${reason.days} 天没有动静`,
    "review-requested": "请求我 review",
    "re-review": "我 review 之后有新提交",
    "waiting-review": who ? `等待 review： ${who}` : "等待 review",
    "ci-running": "CI 运行中",
    draft: "草稿",
  };
  const en: Record<ReasonCode, string> = {
    "changes-requested": who ? `Changes requested by ${who}` : "Changes requested",
    "ci-failing": "CI failing",
    conflict: "Merge conflict",
    approved: "Approved, not merged",
    "reply-needed": `Reply to ${who}`,
    stale: `No activity for ${reason.days} days`,
    "review-requested": "Review requested",
    "re-review": "New commits since my review",
    "waiting-review": who ? `Waiting on ${who}` : "Waiting for review",
    "ci-running": "CI running",
    draft: "Draft",
  };
  return (lang === "zh" ? zh : en)[reason.code];
}
