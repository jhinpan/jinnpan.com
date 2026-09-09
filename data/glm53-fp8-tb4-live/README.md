# GLM-5.3 FP8 Terminal-Bench 4.0 live board

Public data contract for the live board at `/sources/glm53-fp8-tb4-live.html`. It is
the sibling of [`../glm53-mxfp4-tb4-live/`](../glm53-mxfp4-tb4-live/README.md): same
instrument, same denominator, different weights.

## What is under test

The full `zai-org/GLM-5.3` at revision `935644c05e76fc198714f4cca449fd8b970ff6d7` —
743B total, 39B active, block-FP8 weights, `fp8_e4m3` KV cache. This is the 743B
`glm_moe_dsa` model, not the 320B GLM-5.3-Flash architecture, and not the
third-party MXFP4 requantization the previous board measured.

Served on one 8x MI355X (gfx950) node at TP=8 under the cookbook
[`gfx950:high-throughput`](https://jhinpan.github.io/sglang-amd-cookbook/#m=glm-5.3&c=gfx950:high-throughput)
cell, image `rocm/sgl-dev:v0.5.18-rocm724-mi35x-20260827`.

**One deviation from the published cell, and it is a fix rather than a preference.**
That image ships SGLang `20a491d1d3`, which predates
[#36960](https://github.com/sgl-project/sglang/pull/36960) (merged 2026-09-01). Without
it, `--chunked-prefill-size 32768` on AITER `c16d44b9` plus Triton 3.7 aborts with an
LLVM `iota_range` assertion on cold prompts past 23,170 tokens, because the DSA
indexer's fp32 logits cross the 2 GiB ceiling that AITER's `fp8_mqa_logits` compiles
under. Terminal-Bench agent turns cross 23k tokens routinely, so the unpatched image
would have looked healthy at launch and died hours into the run. The patch is applied
on top of the stock image and verified by the PR's own contract test.

## Data path

The evaluation host derives one sanitized snapshot per minute from the Harbor job
directory, the endpoint's `/get_server_info`, and the valid-attempt ledger:

- Gist: <https://gist.github.com/jhinpan/162d506a3e4c788987b60e5bfc49ff1e>
- Raw feed:
  <https://gist.githubusercontent.com/jhinpan/162d506a3e4c788987b60e5bfc49ff1e/raw/tb4-status.json>

The board fetches that feed every 30 s with cache busting, marks data older than
180 s as stale, and keeps the last good snapshot visible when a fetch fails rather
than pretending it is current. Its `localStorage` key is namespaced away from the
MXFP4 board's: the two pages share an origin, and each one's `validate()` rejects the
other's pool count, so an unnamespaced cache would make whichever page loaded second
fall back to its initial paint.

## Counting contract

Fixed target: **63 CPU tasks x 5 attempts = 315 scored attempts**.

Terminal-Bench 4.0 ships 66 tasks. Three of them — `fp8-rmsnorm-gemm`,
`jax-speedrun-gpu`, `math-eval-grader` — declare `gpus = 1` in `task.toml`, and this
node cannot honour that while all eight GPUs serve the model under test. Excluding
exactly those three reproduces the official comparator's 136 / 315 independently,
which is the check that the subset is the same one.

An attempt advances the counter only when a verifier returns a reward. A zero reward
is still a scored model outcome and counts. An infrastructure or verifier fault with
no reward does not count, stays in the issue ledger, and is replayed from the
narrowest durable boundary.

## Official comparator

[`official-glm53-tb4.json`](./official-glm53-tb4.json) is the same frozen file the
MXFP4 board used, copied verbatim so the two runs are scored against an identical
reference:

- official all-task result: 138 / 330, or 41.82% (95% CI +/- 3.23 pp);
- official result on the 63 CPU tasks: 136 / 315, or 43.17%;
- Claude Code 2.1.207 at max reasoning effort.

The official row came from an opaque hosted API. This run serves published block-FP8
weights on our own SGLang stack. That makes the line an end-to-end reference, not a
same-runtime A/B — the same caveat the MXFP4 board carries, and the reason the
FP8-vs-MXFP4 comparison between the *two boards* is the sharper one: those two share
this harness, this node and this agent, and differ only in the weights.
