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

This repo has a local skill at `.claude/skills/source-reading-blog/SKILL.md` for authoring numbered "Source Reading" entries — each one is a self-contained richly-designed HTML deep dive (under `public/sources/`) plus a bilingual markdown blog pair (under `src/content/blog/{zh,en}/`). Use it whenever the user asks for a new source-reading entry, a deep dive on a specific repo, or wants to extend the series. Read the SKILL.md for the full workflow including aesthetic differentiation rules, SVG hand-coding conventions, and the Chinese typography checks that apply to all `zh/` content.
