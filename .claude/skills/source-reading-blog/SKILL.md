---
name: source-reading-blog
description: Author a bilingual (zh + en) "Source Reading" deep-dive blog post for jinnpan.com. Each entry consists of (a) a richly-designed self-contained HTML deep-dive with hand-coded SVG diagrams placed under public/sources/, and (b) two short markdown summaries under src/content/blog/{zh,en}/ that link to the full HTML. Use when the user says things like "write a source reading post on X", "do a deep dive on repo Y", "source-reading 004", or "add a new entry to the source reading series".
---

This skill produces one numbered entry in Jhin's "Source Reading" series on jinnpan.com. Each entry has three artifacts created together:

1. **`public/sources/<slug>.html`** — a richly designed, self-contained HTML file with hand-coded SVG diagrams. No Mermaid, no runtime JS dependencies (except Google Fonts CDN).
2. **`src/content/blog/en/source-reading-NNN-<slug>.md`** — English short-form blog (~600-1000 words) introducing and linking to the HTML.
3. **`src/content/blog/zh/source-reading-NNN-<slug>.md`** — Chinese mirror of the same blog (same slug, same date, same images), following Chinese typography rules from this repo's CLAUDE.md.

## Step 0 · Pick the number and slug

List existing entries first:

```bash
ls public/sources/
ls src/content/blog/en/source-reading-*.md
```

The next entry's number is `max(existing) + 1` (zero-padded to 3 digits: `001`, `002`, ...). Slug is the lowercase repo name or short identifier (e.g. `skypilot`, `sglang`, `vllm`, `triton`, `pytorch-distributed`).

## Step 1 · Read the source code

You must actually read the target repository, not just summarize from external knowledge. Clone shallowly:

```bash
mkdir -p ~/Documents/GitHub
cd ~/Documents/GitHub
git clone --depth 1 https://github.com/<org>/<repo>.git
cd <repo>
wc -l $(find . -name "*.py" -not -path "./.*") | tail -1
```

Map the repo: list top-level dirs, find the entry-point files, get the biggest files (these are usually load-bearing). Read the README, any `CLAUDE.md` or `AGENTS.md`, and `design_docs/` if present.

Aim for **6 hours of equivalent reading** condensed to the post. The post is the appetizer; the HTML deep dive is the entrée. Both must be backed by real code references with `file_path:line_number` precision — invent nothing.

## Step 2 · Pick a distinctive aesthetic for the HTML

**Critical rule**: never reuse the aesthetic of a previous entry. Read the existing files first to see what's taken:

```bash
grep -l "font-family:" public/sources/*.html
```

Each entry commits to one bold aesthetic direction. Past examples:
- 001 SkyPilot — dark "engineering atelier" (Fraunces + Geist + JetBrains Mono · navy/bone/rust/brass)
- 002 SGLang — light "lab notebook" (DM Serif Display + DM Sans + JetBrains Mono · cream/cobalt/crimson)
- 003 vLLM — dark "navigational chart" (Cormorant Garamond + Spectral + IBM Plex Mono · abyss/cream/gold/teal)

For the new entry, pick a fresh direction with intent. Some untried angles: "brutalist Swiss poster" (light, Helvetica-adjacent sans + condensed display + grid breaking), "1990s scientific viz" (dark teal/lime/magenta on near-black, monospace-heavy), "Japanese minimalism" (warm off-white + ink red + single hairline grid), "art-deco geometric" (gold/black/cream + Marcellus or Cinzel display).

**Forbidden defaults**: Inter, Roboto, Arial, system-ui as body fonts; Space Grotesk (called out by the frontend-design skill); generic purple-gradient-on-white; "rounded blue card" UI patterns. The frontend-design skill at `~/.claude/plugins/marketplaces/claude-plugins-official/plugins/frontend-design/skills/frontend-design/SKILL.md` is the authoritative guide.

## Step 3 · Hand-code the SVG diagrams

Aim for 4-7 SVG "plates" per entry. Every diagram must be **inline `<svg viewBox="...">` with hand-calculated coordinates** — no Mermaid, no GraphViz, no `<script>` rendering. This makes the HTML bulletproof (zero parse errors possible) and gives full design control.

Typical plate inventory:
- Plate I — high-level architecture (zones, processes, big components)
- Plate II — a sequence / timeline (how one user action flows through)
- Plate III — a key data structure (cache tree, block table, state machine)
- Plate IV — a comparison or branching decision (DP vs ILP, two engines side-by-side)
- Plate V-VII — domain-specific: kernel paths, file change maps, integration points

Use a small shared set of SVG CSS classes (defined once in `<style>`): `.diag-node`, `.diag-text`, `.diag-edge`, `.zone-box`, `.zone-label`, plus color variants matching the post's palette. Reuse markers (`<marker id="arrowhead">`) defined once in `<defs>`.

Each plate has a `.plate-meta` header (number, italic name, scale) and a `.caption` (one sentence, italic). Wide plates use `.plate.wide` to break out of the text column.

## Step 4 · Write the full HTML

Structure (copy from any of the existing three for the skeleton, change palette + fonts + content):

```
<head>
  Google Fonts <link>
  CSS variables (--bg, --fg, accents, fonts, spacing)
  Body, masthead, layout, rail, typography, components, plate styles
</head>
<body>
  <nav class="rail"> sticky TOC with section numbers </nav>
  <header class="masthead"> kicker · h1 · subtitle · spec sheet </header>
  <main class="article">
    Prologue
    Plate I — architecture
    Module sections (M0-M9 or fewer for smaller repos)
      Each module: hook prose + code excerpt + insight callout + table or list
    Plates interleaved between modules
    Traps (or "Reefs" or "Corrections" — language matches the aesthetic)
    Red-line questions
    AMD-specific takeaways (when relevant to Jhin's work)
    Epilogue
  </main>
  <footer class="colophon"> source · typography · palette · compiled-for </footer>
</body>
```

Target file size: **70-120 KB · 1700-2300 lines**. Keep the article column to `max-width: 720px` for readability; widen only the plates.

## Step 5 · Write the bilingual markdown blogs

Each markdown blog is a **~600-1000 word distillation**. Structure:

```markdown
---
title: "Source Reading NNN — Repo, Tagline"
description: "One sentence on what this entry covers."
date: YYYY-MM-DD
tags: ["source-reading", "MLSys", ...]
category: "Technical"
lang: "en" or "zh"
---

Hook paragraph — why this repo, why now.

## Why this matters

A few sentences on the target audience and why the repo is worth 6 hours.

## Five findings worth carrying

1. **Bold-titled finding.** Two to three sentences explaining with concrete file_path:line_number references.
2. ...

## ★ The one insight that reframed my mental model

> Block-quote insight. The most non-obvious takeaway, in one paragraph.

## What's in the full reading

List of plates + brief preview.

**→ Full deep dive at [/sources/<slug>.html](/sources/<slug>.html)** — describes the aesthetic.

---

*Previous: link · Next: link. Series context.*
```

### Chinese typography rules (mandatory)

Apply these to all `src/content/blog/zh/` content per the repo CLAUDE.md:

- **Half-width space after `。` `，` `：`** when followed by content characters (Chinese chars, letters, digits, opening brackets). Not at end of line, not before closing punctuation.
- **Half-width spaces around `/`** as separator for alternatives: `MI300X / MI355X`, `SFT / RLHF / GRPO`, `TP / PP / EP`. Do NOT add spaces in: model paths (`Qwen/Qwen3-Coder-30B-A3B`), units (`tok/s`, `req/s`), single ASCII char pairs (`K/V`, `N/l`), math fractions (`1/2`), import paths.
- **Skip code blocks, math blocks, inline code/math** — typography rules do not apply inside.

The zh blog should be a natural translation, not literal word-for-word. Preserve technical English terms when they're proper nouns or have no clean Chinese equivalent (e.g., "scheduler", "attention backend" can stay English; "thread", "process" should be "线程", "进程").

## Step 6 · Verify locally

```bash
# from ~/jinnpan.com
npm run dev
# → opens http://localhost:4321
# Check that:
#   - new blog appears in /blog list (both /en/blog and /zh/blog)
#   - clicking it renders correctly with frontmatter
#   - the /sources/<slug>.html link works (browser tab opens the deep dive)
#   - on mobile width the rail collapses gracefully
```

## Step 7 · Commit and push

```bash
git -C ~/jinnpan.com add public/sources/<slug>.html src/content/blog/en/source-reading-NNN-<slug>.md src/content/blog/zh/source-reading-NNN-<slug>.md
git -C ~/jinnpan.com commit -m "blog: source reading NNN — <repo>"
git -C ~/jinnpan.com push origin main
# Vercel auto-deploys in ~60 seconds.
```

## Quality checklist before declaring done

- [ ] HTML has 4+ inline SVG plates, all hand-coded coordinates
- [ ] Aesthetic distinctly different from all previous entries (fonts AND colors)
- [ ] Every numeric claim (line count, file count) cross-checked against actual files via `wc -l`
- [ ] Every code reference includes a `file_path:line_number`
- [ ] zh blog passes typography rules (run a mental scan or grep for `[。，：][^ \n]` outside code blocks)
- [ ] Both blogs have the same date, same slug, same tags
- [ ] At least one section ties to Jhin's AMD / kernel-optimization work where applicable
- [ ] No emojis in the HTML or blog body (Jhin's style — only use if he explicitly asks)
- [ ] No generic AI-writing patterns (run `/humanizer` mentally — avoid "delve into", "comprehensive", "leverage", em-dash overuse, rule-of-three lists when not natural)

## Skipping rules

If the user asks for a source reading on a repo that:
- has fewer than 20K lines of code, propose a single-blog post instead (no HTML deep dive needed)
- is a fork or a documentation-only repo, ask whether they want the parent repo instead
- is already covered in a previous entry, propose updating the existing entry rather than a new one
