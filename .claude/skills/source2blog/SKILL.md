---
name: source2blog
description: >-
  Turn a source (a code repo, a paper, or a body of knowledge) into a polished, shareable artifact on jinnpan.com — routed into one of four categories. CODE and PAPER readings become richly-designed self-contained HTML deep dives under public/sources/ (hand-coded SVG plates + EN/ZH toggle), with NO markdown twin. TUTORIAL is a first-principles primer (bilingual markdown blog, plus an HTML primer when it deserves visual plates). BLOG is original writing (benchmark, comparison, project note — bilingual markdown only). Every entry is registered as a card in the /sources/ library hub. Use whenever the user wants to take a body of knowledge — a repo, a paper, a kernel, a concept, a benchmark — and make it easy to read, learn, and share. Triggers: "source2blog X", "do a deep dive on repo Y", "source-reading NNN", "paper-reading NNN", "write a primer on X", "把 X 做成 portfolio html / blog", "把这堆代码 / 文档变成一个 deep dive", "deep dive on X", "make X shareable / learnable".
---

`source2blog` is Jhin's workflow for **converting a source into a polished portfolio artifact** on jinnpan.com. It is **category-aware**: the first thing you do is decide which of four shelves the entry lives on, because the category determines the output shape.

## The four categories — and what each produces

| Category | What it is | Output | Markdown? |
|---|---|---|---|
| **code** 代码 | A source-level reading of a real **code repository / codebase** | `public/sources/<slug>.html` — HTML deep dive, kicker `Source Reading NNN` | **No** |
| **paper** 论文 | A close reading of a research **paper** | `public/sources/<slug>.html` — HTML deep dive, kicker `Paper Reading NNN` | **No** |
| **tutorial** 教程 | A first-principles **primer / explainer / guide** on a concept (not one repo or paper) | bilingual `src/content/blog/{en,zh}/<slug>.md`; **optionally also** an HTML primer at `public/sources/<slug>.html` when it deserves hand-coded plates | **Yes** |
| **blog** 博客 | **Original writing** — a benchmark, a framework comparison, an opinion, a project note | bilingual `src/content/blog/{en,zh}/<slug>.md` | **Yes** |

**In every case, you also register the entry as a card in the `/sources/` library hub (`public/sources/index.html`).** That hub is the single, categorized map of everything — it is linked from the site nav and replaces the old per-entry "appetizer" markdown.

### The big architectural rule (changed 2026-06-08)

**Code and paper readings no longer get a markdown twin.** Previously every deep dive shipped with a short `source-reading-NNN-*.md` / `paper-reading-NNN-*.md` "appetizer" whose only job was to link to the HTML. Those have all been pruned. The HTML deep dive is now the canonical, self-contained artifact, discoverable through the `/sources/` hub and the top-nav `sources` link. **Do not recreate appetizer markdown for code/paper entries.** Markdown is reserved for `tutorial` and `blog`, where the markdown *is* the content (or the primer), not an ad for an HTML page.

### Deciding the category when it's ambiguous

- Am I **reading someone else's artifact**? → repo = `code`, paper = `paper`.
- Am I **teaching a concept from first principles** (no single repo/paper at the center)? → `tutorial`.
- Am I **sharing original work / measurements / an opinion** (a benchmark I ran, a comparison I reasoned through, my own project)? → `blog`.
- For a `tutorial`, does the topic deserve hand-coded SVG plates and a long visual walkthrough (e.g. `from-python-to-silicon`)? → also build the HTML primer. Otherwise markdown alone is enough.

When in doubt, ask the user which shelf they want.

---

## Step 0 · Pick the category, slug, and number

Decide the category (above). Then:

```bash
ls public/sources/                       # existing HTML deep dives
```

- **Slug** = lowercase repo name / paper short-name / concept identifier (e.g. `skypilot`, `polar`, `from-python-to-silicon`).
- **Number** (code/paper only) lives **inside the HTML masthead**, not in any filename. The next number = highest existing on that shelf + 1. Find it from the hub:

  ```bash
  # highest Source Reading number on the code shelf:
  grep -oE 'No\. 0[0-9]+' public/sources/index.html | sort -u | tail -1
  ```

  Numbering is a soft convention — a one-off reading (like `codex-goal`) can be unnumbered and labelled `Source-level` instead.

## Step 1 · Read the source (code / paper)

For `code` and `paper`, you must actually read the target, not summarize from external knowledge. Clone shallowly:

```bash
mkdir -p ~/Documents/GitHub && cd ~/Documents/GitHub
git clone --depth 1 https://github.com/<org>/<repo>.git && cd <repo>
wc -l $(find . -name "*.py" -not -path "./.*") | tail -1
```

Map the repo: list top-level dirs, find entry points, read the biggest files (usually load-bearing), the README, any `CLAUDE.md`/`AGENTS.md`, and `design_docs/`. For a paper, read the PDF end-to-end and pull the real equations/algorithms.

Aim for **6 hours of equivalent reading** condensed into the artifact. Everything must be backed by real references — `file_path:line_number` for code, section/figure numbers for papers. Invent nothing.

When naming a concrete file in prose, tables, callouts, captions, or summaries, make the filename itself clickable whenever possible. For external code readings, use commit-pinned source links with line fragments, e.g. `<a href="https://github.com/org/repo/blob/<sha>/path/file.py#L12-L34"><code>path/file.py</code></a>`. For site-local files in final handoff notes, use clickable markdown file links. Jhin often clicks these names to inspect the source directly, so avoid leaving important filenames as inert plain text.

## Step 1.5 · Explain the Why / How / Why-Not, not just the What

**These artifacts are for Jhin to read and think with, not marketing copy.** The bar is technical depth *and* interpretability. The most common failure — and one Jhin will call out — is a paragraph that lists *what* something is (a feature, a set of layers, an API) without explaining *why it exists, how it works, or why the alternative was rejected*. A reader finishes such a paragraph knowing the vocabulary but not understanding anything.

For every non-trivial claim, mechanism, or design choice, make sure the prose answers as many of these as apply:

- **Why** does this exist? What problem or failure mode forced it? State the problem *before* the solution, so the solution has something to attach to.
- **How** does it actually work? The concrete mechanism — the key, the check, the data flow — not a label for it.
- **Why not** the obvious alternative? What breaks if you do the simpler/naive thing? This is often the most illuminating sentence in a section.
- **What would go wrong without it?** The concrete failure the mechanism prevents.

### Two thinking tools — apply both

These are complementary, not the same move:

- **First-principles thinking (decompose, then rebuild).** Don't explain a mechanism by analogy to the framework's own vocabulary or by restating its docs. Break it down to the bedrock facts that must be true regardless of this particular library — the physical or logical constraints (compilation costs seconds; GPU occupancy depends on register/shared-memory budget; a cache is only sound if its key covers everything that changes the output) — and rebuild the design up from there. The test: *could the reader re-derive this design themselves from the constraints, without having seen this codebase?* When you find yourself writing "X does A, B, C," stop and ask "what forces A, B, C to exist at all?" — that root cause is the thing worth writing. First-principles is generative: it produces the Why.

- **Occam's razor (cut to the essential).** Once the idea is built up, strip everything that isn't load-bearing. No sentence that only re-labels the previous one; no adjective that adds no information; no third example when two carry the point. The litmus below is the razor in practice. Occam is subtractive: it produces clarity.

Order of operations: reason up from first principles to find the real Why, then take Occam's razor to the prose. Depth first, then economy.

Litmus test (Occam) before you ship a paragraph: *if I deleted every proper noun and API name, would a reader still learn a transferable idea?* If the paragraph collapses into name-dropping, it fails.

First-principles check before you ship a section: *does it bottom out in a constraint that would hold in any implementation, or does it stop at "because this library does it this way"?* If it stops at the library, dig one level deeper.

Anti-pattern (what NOT to do) — a flat list of layers with no causality:

> "quack's cache has four layers: an in-memory dict, an on-disk `.o` keyed by `(qualname, *args)`, a source fingerprint, and an optional result cache."

Fixed — problem first, then each layer catches a case the one above can't:

> "Compiling is slow but the same kernel is called thousands of times, so you compile once and reuse — but reuse is only safe if nothing affecting the compiled code changed. **Layer 1** (in-memory dict) is instant but dies with the process. **Layer 2** (on-disk `.o`) survives runs — but here's the trap: if you *edit the kernel source*, the `(name, args)` key is unchanged, so it hands back a stale object. **Layer 3** folds a source fingerprint into the key so an edit invalidates it automatically. That's what makes the disk cache safe to keep."

Same facts, but the second teaches. Apply this to prose, callouts, and plate captions alike. Diagrams should show a mechanism or a causal flow, not just box-and-label taxonomy. Depth beats coverage: better to explain four things with their Why/How/Why-Not than to name twelve.

## Step 2 · Pick a distinctive aesthetic (HTML entries)

**Critical rule**: never reuse the aesthetic of a previous entry. See what's taken:

```bash
grep -l "font-family:" public/sources/*.html
```

Each HTML entry commits to one bold direction. Past examples:
- 001 SkyPilot — dark "engineering atelier" (Fraunces + Geist + JetBrains Mono · navy/bone/rust/brass)
- 002 SGLang — light "lab notebook" (DM Serif Display + DM Sans + JetBrains Mono · cream/cobalt/crimson)
- 003 vLLM — dark "navigational chart" (Cormorant Garamond + Spectral + IBM Plex Mono · abyss/cream/gold/teal)

For a new entry, pick a fresh direction with intent. Untried angles: "brutalist Swiss poster", "1990s scientific viz" (dark teal/lime/magenta on near-black), "Japanese minimalism" (warm off-white + ink red + hairline grid), "art-deco geometric" (gold/black/cream + Marcellus/Cinzel).

**Forbidden defaults**: Inter, Roboto, Arial, system-ui as body fonts; Space Grotesk; generic purple-gradient-on-white; "rounded blue card" UI. The frontend-design skill at `~/.claude/plugins/marketplaces/claude-plugins-official/plugins/frontend-design/skills/frontend-design/SKILL.md` is the authoritative guide.

## Step 3 · Hand-code the SVG diagrams

4–7 SVG "plates" per HTML entry. Every diagram is **inline `<svg viewBox="...">` with hand-calculated coordinates** — no Mermaid, no GraphViz, no `<script>` rendering. Bulletproof and fully controllable.

Typical inventory: architecture (zones/processes), a sequence/timeline, a key data structure, a comparison/branch, plus domain-specific plates (kernel paths, file maps, integration points). Use a small shared set of SVG CSS classes (`.diag-node`, `.diag-text`, `.diag-edge`, `.zone-box`, `.zone-label`) defined once, and one `<marker id="arrowhead">` in `<defs>`. Each plate gets a `.plate-meta` header and a one-sentence italic `.caption`; wide plates use `.plate.wide`.

## Step 4 · Write the full HTML

Copy the skeleton from any existing entry; change palette + fonts + content:

```
<head> Google Fonts · CSS variables (--bg/--fg/accents/fonts) · layout/typography/plate styles </head>
<body>
  <nav class="rail"> "← The Library" back-link to /sources/ (top of rail) · sticky TOC with section numbers </nav>
  <header class="masthead"> kicker (e.g. "Source Reading 009") · h1 · subtitle · spec sheet </header>
  <main class="article">
    Prologue · Plate I (architecture) · module sections (each: hook + code/figure excerpt + insight callout + table)
    · plates interleaved · Traps/Reefs · red-line questions · AMD takeaways (when relevant) · Epilogue
  </main>
  <footer class="colophon"> source · typography · palette · compiled-for </footer>
</body>
```

Target file size: **70–120 KB · 1700–2300 lines** (English-only); **140–180 KB · 2200–2800 lines** with the bilingual toggle. Article column `max-width: 720px`; widen only plates.

## Step 4.5 · Add the EN/ZH language toggle (default-on)

Every HTML deep dive ships bilingual: every translatable paragraph, heading, callout, plate caption, table cell, and colophon row exists in both languages, with a floating top-right toggle that switches visible language without reload. Default on unless the user says "English only."

Four moving parts:

1. **CSS visibility rule** near the bottom of `<style>`:
   ```css
   body[data-lang="en"] [lang="zh"]:not(html) { display: none !important; }
   body[data-lang="zh"] [lang="en"]:not(html) { display: none !important; }
   ```
2. **The toggle button**, right after `<body data-lang="en">`:
   ```html
   <div class="lang-toggle" role="group" aria-label="Language">
     <button type="button" data-set="en" aria-label="English">EN</button>
     <button type="button" data-set="zh" aria-label="中文">中文</button>
   </div>
   ```
   Style as a fixed top-right pill; borrow palette/typography from the entry's `:root` (sample in `flydsl.html`).
3. **The JS** before `</body>` — first visit picks browser language, then persists in `localStorage` under `<slug>-source-lang`:
   ```html
   <script>
   (function() {
     var KEY = '<slug>-source-lang', body = document.body, stored = null;
     try { stored = localStorage.getItem(KEY); } catch (e) {}
     if (stored === 'en' || stored === 'zh') { body.setAttribute('data-lang', stored); }
     else { var nav = (navigator.language || 'en').toLowerCase(); body.setAttribute('data-lang', nav.indexOf('zh') === 0 ? 'zh' : 'en'); }
     document.querySelectorAll('.lang-toggle button[data-set]').forEach(function(btn) {
       btn.addEventListener('click', function() {
         var v = btn.getAttribute('data-set'); body.setAttribute('data-lang', v);
         try { localStorage.setItem(KEY, v); } catch (e) {}
       });
     });
   })();
   </script>
   ```
4. **The `[lang]` content pairs** — for every translatable block add a sibling in the other language. Inline `<span>` for short phrases; separate elements for paragraphs/headings/cells.

**Translate vs leave language-neutral:** translate `<p>` prose, `<h2/3/4>`, `<th>/<td>` prose, callouts, plate captions, colophon prose. Leave code blocks, short SVG labels (`STAGE 0`, `MFMA`), inline `<code>` API names, URLs, file paths, line numbers, the masthead brand `<h1>`, and numeric meta values.

**Chinese typography in HTML content:** same rules as `src/content/blog/zh/` (CLAUDE.md) — half-width space after `。` `，` `：` before content characters, half-width spaces around `/` as an alternative separator. Mentally scan `[。，：][^ \n*]` before declaring done. Skip code blocks and inline code.

**Pairing check** before commit — the two counts must be equal:
```bash
python3 -c "
import re
html = open('public/sources/<slug>.html').read()
en = len(re.findall(r'(?<!data-)lang=\"en\"', html))
zh = len(re.findall(r'(?<!data-)lang=\"zh\"', html))
print(f'en={en} zh={zh} {\"BALANCED\" if en == zh else \"MISMATCH\"}')"
```

## Step 5 · Write the bilingual markdown (tutorial / blog only)

**Skip this step for `code` and `paper`** — they have no markdown.

For `tutorial` and `blog`, write a bilingual pair under `src/content/blog/{en,zh}/<slug>.md` (same slug, same date, same tags). This markdown is the real content — a standalone, substantive post — **not** an appetizer pointing at an HTML page.

```markdown
---
title: "<Plain title — no 'Source Reading NNN' prefix>"
description: "One sentence on what this covers."
date: YYYY-MM-DD
tags: ["...", "..."]
category: "Technical"
lang: "en"   # or "zh"
---

Hook paragraph.

## ...substantive sections, math, runnable code, tables, original analysis...
```

- A `tutorial` teaches from first principles and is self-contained; if you also built an HTML primer (Step 4), link to it once (`Full read: [/sources/<slug>.html](/sources/<slug>.html)`) — but the markdown must still stand on its own.
- A `blog` post is original writing and never has an HTML twin.

### Chinese typography (mandatory)

Apply to all `src/content/blog/zh/` content per CLAUDE.md: half-width space after `。` `，` `：` before content characters; half-width spaces around `/` for alternatives (`MI300X / MI355X`, `SFT / RLHF`) but NOT in model paths (`Qwen/Qwen3-...`), units (`tok/s`), single ASCII pairs (`K/V`), fractions (`1/2`), import paths; skip code/math blocks. The zh post is a natural translation, not word-for-word; keep proper-noun technical terms in English.

## Step 5.5 · Register the entry in the library hub (ALL categories)

Add a card to `public/sources/index.html` in the correct shelf, and bump the counts. This is mandatory — the hub is how every entry is discovered.

1. **Add a card** inside the right `<section class="shelf" data-section="...">`'s `.grid`:

   ```html
   <!-- code / paper: HTML deep dive, single primary link -->
   <article class="entry" data-kind="code" data-search="lowercase keywords topic tags for search">
     <div class="entry-top"><span class="entry-kind">Source Reading</span><span>No. 009</span></div>
     <div class="entry-body">
       <h3>Title</h3>
       <p>One- to two-sentence description.</p>
       <div class="tags"><span class="tag">tag</span><span class="tag">tag</span><span class="tag">tag</span></div>
     </div>
     <div class="links"><a class="link primary" href="./<slug>.html">Deep dive →</a></div>
   </article>

   <!-- tutorial / blog: bilingual blog page, EN + 中文 links (add a "Deep dive →" too if an HTML primer exists) -->
   <article class="entry" data-kind="blog" data-search="lowercase keywords">
     <div class="entry-top"><span class="entry-kind">Benchmark</span><span>short meta</span></div>
     <div class="entry-body">
       <h3>Title</h3><p>Description.</p>
       <div class="tags"><span class="tag">tag</span></div>
     </div>
     <div class="links"><a class="link primary" href="/en/blog/<slug>/">EN</a><a class="link" href="/zh/blog/<slug>/">中文</a></div>
   </article>
   ```

   `data-kind` ∈ `code|paper|tutorial|blog` (drives the accent color + filter). `entry-kind` label: `Source Reading` / `Paper Reading` for code/paper; for tutorial/blog use a fitting label (`Primer`, `Guide`, `Benchmark`, `Comparison`, `Project`, `Note`).

2. **Bump the counts**: the matching `.stat[data-c="..."] .num`, the shelf's `.shelf-count` text, and the SVG legend count in the hero plate (and add a small `<rect>` book on that shelf's row if you want it visually exact).

## Step 6 · Verify locally

```bash
npm run dev   # http://localhost:4321
```
- `/sources/` hub: new card appears in the right shelf, search + filter still work, counts updated.
- code/paper: `/sources/<slug>.html` renders, rail/plates/toggle all work, mobile rail collapses.
- tutorial/blog: post appears in `/en/blog` and `/zh/blog`, frontmatter renders, any `/sources/` link works.

## Step 7 · Commit and push

```bash
# code / paper:
git -C ~/jinnpan.com add public/sources/<slug>.html public/sources/index.html
# tutorial / blog:
git -C ~/jinnpan.com add src/content/blog/en/<slug>.md src/content/blog/zh/<slug>.md public/sources/index.html
git -C ~/jinnpan.com commit -m "<category>: <slug>"
git -C ~/jinnpan.com push origin main      # Vercel auto-deploys in ~60s
```

## Step 7.5 · Poll Vercel + spot-check live

Don't say "it's live" until it is. Poll in the background (`run_in_background: true`) for a string unique to this entry:

```bash
URL=https://jinnpan.com/sources/<slug>.html   # or https://jinnpan.com/en/blog/<slug>/
until curl -sf -o /tmp/check.html "$URL" 2>/dev/null && grep -q '<unique-string>' /tmp/check.html; do sleep 8; done
echo "live ✓ ($(wc -c < /tmp/check.html) bytes)"
```

Vercel usually finishes in 30–90s; > 3 min means a likely failure — check `gh run list` or the Vercel dashboard. Then WebFetch spot-check the live URLs for this entry:
- code/paper: `https://jinnpan.com/sources/<slug>.html` (masthead, rail anchors, plate count, toggle) **and** `https://jinnpan.com/sources/` (the new card shows in the right shelf).
- tutorial/blog: `https://jinnpan.com/en/blog/<slug>/` and `https://jinnpan.com/zh/blog/<slug>/` (frontmatter, content) **and** the hub card.

Only report "deployed and verified" once all return 200 with the expected content. The English blog path is `/en/blog/...` (the unprefixed `/blog/` is a legacy alias that may 404).

## Quality checklist before declaring done

- [ ] **Category decided correctly**, and the output shape matches the table (code/paper = HTML only, no markdown twin; tutorial/blog = markdown, HTML primer only if warranted)
- [ ] **Why / How / Why-Not, not just What** (Step 1.5) — every non-trivial mechanism explains the problem it solves before naming the solution; no flat feature/layer lists. Both thinking tools applied: **first-principles** (each section bottoms out in a constraint true of any implementation, not "because this library does it this way") and **Occam** (passes the "delete the proper nouns, does an idea survive?" litmus)
- [ ] **Entry registered in `/sources/index.html`** — card in the right shelf, `data-kind` + `data-search` set, counts bumped
- [ ] (HTML) 4+ inline SVG plates, all hand-coded coordinates
- [ ] (HTML) Aesthetic distinctly different from all previous entries (fonts AND colors)
- [ ] (HTML) **EN/ZH toggle present and content blocks paired** (`lang="en"` count == `lang="zh"` count, excluding `data-lang` and CSS selectors)
- [ ] (HTML) **Back-link to the `/sources/` hub present** — `← The Library` at the top of the rail (or in the masthead if no rail); bilingual paired spans when the toggle is on, so it stays language-balanced
- [ ] Every numeric claim cross-checked against actual files (`wc -l`) / the paper
- [ ] Every code reference includes `file_path:line_number`
- [ ] Concrete filenames mentioned in prose, tables, callouts, captions, or final handoff notes are clickable links where possible; prefer commit-pinned source links with line fragments for external repos
- [ ] (zh) markdown AND zh HTML pass typography rules (half-width space after `。` `，` `：`; spaces around `/` separators; skip code)
- [ ] (tutorial/blog) both language files share the same date, slug, tags
- [ ] At least one section ties to Jhin's AMD / kernel-optimization work where applicable
- [ ] No emojis in HTML or blog body (Jhin's style — only if he asks)
- [ ] No generic AI-writing patterns (avoid "delve into", "comprehensive", "leverage", em-dash overuse, forced rule-of-three)
- [ ] **Anchor cross-check passes** (HTML — see below)
- [ ] **Vercel deploy verified live** (Step 7.5)

### Anchor cross-check — MANDATORY before commit (HTML entries)

Verify every rail/TOC link has a matching `<section id="...">` and vice versa — catches "rail drift" when modules get merged during writing:

```bash
F=public/sources/<slug>.html
rail=$(grep -oE 'href="#[a-z0-9-]+"' "$F" | sort -u | sed 's/href="#//;s/"//')
body=$(grep -oE 'section[^>]*id="[a-z0-9-]+"' "$F" | sed 's/.*id="//;s/"//' | sort -u)
echo "Rail-only (broken links): $(comm -23 <(echo "$rail") <(echo "$body") | tr '\n' ' ')"
echo "Body-only (orphan sections): $(comm -13 <(echo "$rail") <(echo "$body") | tr '\n' ' ')"
```

Both lines must print empty. Rail-only = links pointing nowhere; Body-only = sections with no nav entry (commonly a forgotten Epilogue).

## Skipping rules

- A `code` repo with fewer than ~20K lines, or a docs-only / fork repo → consider a `tutorial` or `blog` post instead of a full HTML deep dive; or ask whether the parent repo is meant.
- A topic already covered by an existing entry → update that entry rather than adding a duplicate.
- If the category is genuinely unclear (e.g. "is this a tutorial or a blog?"), ask the user which shelf they want before building.
