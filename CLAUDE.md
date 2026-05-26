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

## Skills

This repo has a local skill at `.claude/skills/source2blog/SKILL.md` for turning code / docs / papers into a polished, shareable, podcast-style HTML deep dive — a self-contained richly-designed HTML with hand-coded SVG plates and an EN/ZH language toggle (under `public/sources/`) plus a bilingual markdown blog pair (under `src/content/blog/{zh,en}/`). Use it whenever the user wants a new "Source Reading" entry, a deep dive on a specific repo, a podcast-style HTML on a topic, or anything matching "把 X 做成 portfolio html / blog". Read the SKILL.md for the full workflow including aesthetic differentiation rules, SVG hand-coding conventions, bilingual toggle pattern, Vercel deploy verification, and Chinese typography checks.
