# jinnpan.com — Claude Code Project Context

Astro Micro blog at jinnpan.com. Bilingual (zh/en). Deployed to Vercel via push to main.

## Chinese Typography Rules

These rules apply to all Chinese content (`src/content/blog/zh/*.md`, Chinese strings in `src/pages/[lang]/about.astro`):

1. **Space after sentence punctuation**: Add a half-width space after `。` `，` `：` when followed by content characters (Chinese chars, letters, digits, opening brackets). Do NOT add space at end of line or before closing punctuation.

2. **Space around slash separators**: Add half-width spaces around `/` when used as an "or" separator between alternatives (e.g. `MI300X / MI355X`, `SFT / RLHF / GRPO`, `TP / PP / EP`). Do NOT add spaces in:
   - Model paths (`Qwen/Qwen3-Coder-30B-A3B`)
   - Units (`tok/s`, `req/s`)
   - Single ASCII char pairs (`K/V`, `N/l`)
   - Math fractions (`1/2`, `2/3`)
   - Import/file paths

3. **Skip code blocks, math blocks, and inline code/math** when applying typography rules.

## Content Structure

```
src/content/blog/zh/   — Chinese blog posts (markdown)
src/content/blog/en/   — English blog posts (same slugs)
src/pages/[lang]/      — Bilingual page templates (.astro)
```

## Deploy

Push to `main` auto-deploys via Vercel (~60s).

## Content categories

All published writing sorts into six shelves, surfaced by the categorized hub at `public/sources/index.html` (linked from the top nav as `sources`):

- **code** 代码 — source-level readings of real codebases → self-contained HTML deep dives under `public/sources/<slug>.html` (kicker `Source Reading NNN`). **No markdown twin.**
- **paper** 论文 — close readings of research papers → HTML deep dives under `public/sources/<slug>.html` (kicker `Paper Reading NNN`). **No markdown twin.**
- **tutorial** 教程 — first-principles primers/guides → bilingual markdown under `src/content/blog/{en,zh}/`, plus an HTML primer when the topic deserves hand-coded plates.
- **blog** 博客 — original writing (benchmarks, comparisons, project notes) → bilingual markdown only.
- **experiments** 实验 — reproducible kernel and system measurements → HTML deep dives under `public/sources/<slug>.html` (kicker `Experiment NNN`), with raw CSVs, run logs and the exact commands archived under `data/<slug>/`. **No markdown twin.** These exist to be cited later, so the raw data is committed alongside the write-up rather than living in a canvas that dies with the container.
- **sync** 日常同步 — numbered status entries → `public/sources/daily-sync/NNN.html`, indexed by `public/sources/daily-sync/index.html` (page title "Open Source PR Tracker", kicker `Open Source PR Tracker · Entry NNN`; the directory name `daily-sync/` is historical). Light theme only, by request — no dark mode block. **No markdown twin.** Each entry is a board of every pull request in its window — authored or reviewed; landed, in flight, needs-me, blocked, decide, draft, closed — grouped by the **thread** (workstream) it advances, followed by per-PR cards (what / why / waiting on / next), then blockers, questions and ideas. Four JSON blocks drive the page: `#sync-meta` (window, `pinned` refs, `exclude`, `showPrivate`), `#thread-data` (slug, name, goal, next, `serves`), `#pr-data`, `#notes-data`. Measurements are never typed by hand: `scripts/sync-daily-sync.mjs` (GraphQL) refreshes every measured field of `#pr-data` — state, dates, line counts, reviewDecision, CI, approvers — on a nightly cron and discovers new PRs updated inside the window, while the editorial keys (`short`, `thread`, `status`, `what`, `why`, `waiting`, `next`, `points`, `byline`, `related`, `decision`, `note`, `hidden`) survive every run. Editorial strings are `{en, zh}` pairs; a missing `zh` falls back to `en`. PRs in private repositories keep number, state and timing on the board but have their title and prose scrubbed unless `showPrivate` is true. A PR the sync cannot re-read (SAML-enforced org, private repo) is carried forward unchanged, never dropped. To open a new entry, copy the newest `NNN.html`, bump `#sync-meta.issue` and the window, reset `#pr-data` to `[]`, add a record to the index's `#entry-data`, run `npm run sync:daily`, then write the editorial fields.

As of 2026-06-08 the per-entry "appetizer" markdown for code/paper readings was pruned — the HTML deep dive is the canonical artifact, discovered via the `/sources/` hub. Do not recreate appetizer markdown for code/paper entries.

## Skills

This repo has a local, **category-aware** skill at `.claude/skills/source2blog/SKILL.md` for turning a source (code repo, paper, concept, or original work) into a polished portfolio artifact routed onto one of the four shelves above. Use it whenever the user wants a new code/paper deep dive, a tutorial primer, a blog post, or anything matching "把 X 做成 portfolio html / blog" / "deep dive on X" / "source-reading NNN" / "write a primer on X". The skill always registers the new entry as a card in the `/sources/` hub. Read the SKILL.md for the category routing table, aesthetic differentiation rules, SVG hand-coding conventions, the EN/ZH toggle pattern, hub-registration step, Vercel deploy verification, and Chinese typography checks.

**Writing bar (SKILL.md Step 1.5):** these artifacts are for Jhin to read and think with — the standard is technical depth *and* interpretability. Explain the **Why / How / Why-Not**, not just the *What*: state the problem before the solution, give the concrete mechanism (not a label), and say why the naive alternative fails. Two thinking tools, applied in order: **first-principles** (reason up from constraints true of any implementation to find the real Why — not "because this library does it this way"), then **Occam's razor** (cut everything not load-bearing; litmus: delete the proper nouns — does a transferable idea survive?). Flat "here are the four layers / features" lists without causality are the main failure mode.
