---
name: source-reading-blog
description: Turn a pile of code + docs into a shareable, narrative-driven, "podcast-style" deep-dive on jinnpan.com. Produces (a) a richly-designed self-contained HTML with hand-coded SVG diagrams + bilingual EN/ZH toggle under public/sources/, and (b) two short markdown blog summaries under src/content/blog/{zh,en}/. Use whenever the user wants to take a body of knowledge — a repo, a paper, a kernel, a system internals walkthrough — and make it easy to read, easy to learn, easy to share. Triggers include "write a source reading post on X", "do a deep dive on repo Y", "source-reading NNN", "把 X 做成 portfolio html", "把这堆代码 / 文档变成一个 blog", "deep dive on X", "write a podcast-style HTML about X", "make X shareable / learnable".
---

This skill is Jhin's workflow for **converting raw knowledge (code, docs, papers) into a polished, podcast-style HTML deep dive** on jinnpan.com. Each invocation produces three artifacts together:

1. **`public/sources/<slug>.html`** — a richly designed, self-contained HTML file with hand-coded SVG diagrams and a built-in EN/ZH language toggle. No Mermaid, no runtime JS dependencies (except Google Fonts CDN).
2. **`src/content/blog/en/source-reading-NNN-<slug>.md`** — English short-form blog (~600-1000 words) introducing and linking to the HTML.
3. **`src/content/blog/zh/source-reading-NNN-<slug>.md`** — Chinese mirror of the same blog (same slug, same date, same images), following Chinese typography rules from this repo's CLAUDE.md.

### When the entry is *not* a numbered "Source Reading"

The default series is numbered. For one-off deep dives that don't belong to the series (a talk transcript, a project retrospective, a paper-reading write-up), skip Step 0's numbering and pick a freeform slug; place the HTML under `public/sources/<slug>.html` without the NNN prefix, and drop the `Source Reading NNN —` prefix from the markdown titles. Every other step in this skill still applies.

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

Target file size: **70-120 KB · 1700-2300 lines** (English-only). With the bilingual toggle from Step 4.5 this grows to **140-180 KB · 2200-2800 lines**. Keep the article column to `max-width: 720px` for readability; widen only the plates.

## Step 4.5 · Add the EN/ZH language toggle (default-on)

The HTML deep dive ships bilingual: every translatable paragraph, heading, callout, plate caption, table cell, and colophon row exists in both English and Chinese, with a floating button in the top-right that switches the visible language without a page reload. Default to having the toggle unless the user explicitly says "English only" or "single language."

**Mechanics — the four moving parts:**

1. **CSS visibility rule** in the `<style>` block, near the bottom:

   ```css
   body[data-lang="en"] [lang="zh"]:not(html) { display: none !important; }
   body[data-lang="zh"] [lang="en"]:not(html) { display: none !important; }
   ```

2. **The toggle button**, placed right after `<body data-lang="en">`:

   ```html
   <div class="lang-toggle" role="group" aria-label="Language">
     <button type="button" data-set="en" aria-label="English">EN</button>
     <button type="button" data-set="zh" aria-label="中文">中文</button>
   </div>
   ```

   Style it as a fixed-position pill in the top-right corner (sample CSS in `flydsl.html`). Match the aesthetic of the entry — borrow palette and typography from the existing `:root` variables.

3. **The JS** just before `</body>` — first-visit picks browser language (`navigator.language.startsWith('zh')` → `zh`, else `en`), thereafter persists choice in `localStorage` under a per-page key like `<slug>-source-lang`:

   ```html
   <script>
   (function() {
     var KEY = '<slug>-source-lang';
     var body = document.body;
     var stored = null;
     try { stored = localStorage.getItem(KEY); } catch (e) {}
     if (stored === 'en' || stored === 'zh') {
       body.setAttribute('data-lang', stored);
     } else {
       var nav = (navigator.language || 'en').toLowerCase();
       body.setAttribute('data-lang', nav.indexOf('zh') === 0 ? 'zh' : 'en');
     }
     document.querySelectorAll('.lang-toggle button[data-set]').forEach(function(btn) {
       btn.addEventListener('click', function() {
         var v = btn.getAttribute('data-set');
         body.setAttribute('data-lang', v);
         try { localStorage.setItem(KEY, v); } catch (e) {}
       });
     });
   })();
   </script>
   ```

4. **The `[lang]` content pairs**. For every translatable block, add a sibling with the alternate language. Inline spans for short phrases inside a single sentence, separate elements for paragraphs and headings:

   ```html
   <h3 lang="en">The atom · Shape, Stride, Layout</h3>
   <h3 lang="zh">原子 · Shape、 Stride、 Layout</h3>

   <p lang="en">A <code>!fly.layout</code> is a pair of integer tuples...</p>
   <p lang="zh">一个 <code>!fly.layout</code> 是两个整数 tuple 的对...</p>

   <table class="tbl">
     <thead>
       <tr lang="en"><th>Call</th><th>Returns</th></tr>
       <tr lang="zh"><th>调用</th><th>返回</th></tr>
     </thead>
     <tbody>
       <tr>
         <td><code>partition_S(bA)</code></td>
         <td lang="en">per-thread view of source</td>
         <td lang="zh">source 的 per-thread 视图</td>
       </tr>
     </tbody>
   </table>
   ```

**What to translate vs leave language-neutral:**

| Translate | Leave as-is |
|---|---|
| `<p>` body prose | Code blocks (`.code` elements) |
| `<h2>`, `<h3>`, `<h4>` headings | Short SVG labels (`STAGE 0`, `MFMA`, `ds_read`) |
| `<th>` / `<td>` prose | Inline `<code>` API names |
| Callout `.ctag` and prose | URLs, file paths, line numbers |
| Plate captions | The masthead `<h1>` brand name |
| Colophon `.lbl` and `.val` prose | Numerical values in meta blocks |

**Chinese typography in HTML content:** the same rules as `src/content/blog/zh/` markdown apply — half-width space after `。` `，` `：` followed by content characters, half-width spaces around `/` as alternative separator. Mentally scan `[。，：][^ \n*]` patterns before declaring done. Skip code blocks and inline code.

**Pairing check** before commit:

```bash
# Excluding data-lang attribute and CSS [lang=...] selectors:
python3 -c "
import re
html = open('public/sources/<slug>.html').read()
en = len(re.findall(r'(?<!data-)lang=\"en\"', html))
zh = len(re.findall(r'(?<!data-)lang=\"zh\"', html))
print(f'en={en} zh={zh} {\"BALANCED\" if en == zh else \"MISMATCH\"}')"
```

The two counts must be equal — every English block has a Chinese counterpart, and vice versa. A 1-block discrepancy means one language has an orphaned block that will only render in one mode.

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

## Step 7.5 · Poll Vercel + spot-check the live page

After pushing, don't tell the user "it's live" until it actually is. Poll for the new content and verify it renders:

```bash
# Background poll — finishes when the new HTML is reachable AND contains
# a string unique to this entry (a translated phrase, the title, or a slug).
URL=https://jinnpan.com/sources/<slug>.html
until curl -sf -o /tmp/check.html "$URL" 2>/dev/null && \
      grep -q '<unique-string-from-this-entry>' /tmp/check.html; do
  sleep 8
done
echo "Vercel deploy live ✓ ($(wc -c < /tmp/check.html) bytes)"
```

Run this in the background with `run_in_background: true` so you keep working while it polls. Vercel typically finishes in 30–90 seconds; if it's been > 3 minutes, the deploy probably failed — check `gh run list` or the Vercel dashboard.

Once the poll exits, do a quick WebFetch spot-check on **three URLs** to verify the live pages render as expected:

- `https://jinnpan.com/sources/<slug>.html` — masthead title, rail anchors, plate count, language toggle markup
- `https://jinnpan.com/en/blog/source-reading-NNN-<slug>/` — frontmatter title, date, tags, link to /sources/<slug>.html
- `https://jinnpan.com/zh/blog/source-reading-NNN-<slug>/` — same checks in Chinese

Note the URL prefix is `/en/blog/...` (not `/blog/...`) for English; the unprefixed `/blog/` path is the legacy alias and may 404.

Only report "deployed and verified" once all three return 200 and contain the expected content. If the user asks to see the result, surface the URLs and a one-line summary of what each contains — don't claim it works without checking.

## Quality checklist before declaring done

- [ ] HTML has 4+ inline SVG plates, all hand-coded coordinates
- [ ] Aesthetic distinctly different from all previous entries (fonts AND colors)
- [ ] **EN/ZH toggle present and content blocks paired** (see Step 4.5; `lang="en"` count == `lang="zh"` count, excluding `data-lang` and CSS selectors)
- [ ] Every numeric claim (line count, file count) cross-checked against actual files via `wc -l`
- [ ] Every code reference includes a `file_path:line_number`
- [ ] zh blog AND zh HTML content pass typography rules (half-width space after `。` `，` `：` + half-width spaces around `/` separators; skip code blocks)
- [ ] Both blogs have the same date, same slug, same tags
- [ ] At least one section ties to Jhin's AMD / kernel-optimization work where applicable
- [ ] No emojis in the HTML or blog body (Jhin's style — only use if he explicitly asks)
- [ ] No generic AI-writing patterns (run `/humanizer` mentally — avoid "delve into", "comprehensive", "leverage", em-dash overuse, rule-of-three lists when not natural)
- [ ] **Anchor cross-check passes** (see below)
- [ ] **Vercel deploy verified live** (Step 7.5: HTML URL + en blog URL + zh blog URL all return 200 with expected content)

### Anchor cross-check — MANDATORY before commit

Before declaring the HTML done, run this cross-grep to verify that every rail/TOC link has a matching `<section id="...">` and vice versa. This catches "rail drift" — where modules get merged during writing but the rail wasn't updated.

```bash
F=public/sources/<slug>.html
rail=$(grep -oE 'href="#[a-z0-9-]+"' "$F" | sort -u | sed 's/href="#//;s/"//')
body=$(grep -oE 'section[^>]*id="[a-z0-9-]+"' "$F" | sed 's/.*id="//;s/"//' | sort -u)
echo "Rail-only (broken links): $(comm -23 <(echo "$rail") <(echo "$body") | tr '\n' ' ')"
echo "Body-only (orphan sections): $(comm -13 <(echo "$rail") <(echo "$body") | tr '\n' ' ')"
```

Both lines must print empty. If "Rail-only" is non-empty, your rail points to nothing — clicking falls back to the page top silently. If "Body-only" is non-empty, you have a real section without a nav entry — the most common case is forgetting an Epilogue link.

The first three entries in this series (skypilot, sglang, vllm) all shipped with broken rails because writing-time module merges weren't reflected back into the rail. Don't repeat that — run the check.

## Skipping rules

If the user asks for a source reading on a repo that:
- has fewer than 20K lines of code, propose a single-blog post instead (no HTML deep dive needed)
- is a fork or a documentation-only repo, ask whether they want the parent repo instead
- is already covered in a previous entry, propose updating the existing entry rather than a new one
