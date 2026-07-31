#!/usr/bin/env python3
"""Why chunk 2's attention costs 12.8x chunk 1's.

The grid recorded in the trace settles it. extend_attention_fwd launches
`(batch, heads, cdiv(max_extend_len, BLOCK_M))`, and on gfx950
_get_block_sizes_for_extend_attention picks BLOCK_M=128 only when
128 < Lq <= 256, else 64. Both chunks extend the same 16384 tokens, so:

    chunk 1  grid z = 128  ->  BLOCK_M = 128  ->  Lq = 192  (decompressed MHA form)
    chunk 2  grid z = 256  ->  BLOCK_M =  64  ->  Lq = 576  (absorbed latent form)

Once a prefix exists SGLang switches MLA to the absorbed form, whose head
dimension is 3.4x larger. So chunk 2 is not running the same maths less
efficiently -- it is running different, much heavier maths.
"""
from __future__ import annotations

HEADS_PER_RANK = 96 // 8
CHUNK = 16384

# decompressed MHA form: q/k = qk_nope + qk_rope = 128 + 64, v = 128
MHA_QK, MHA_V = 192, 128
# absorbed latent form: q/k = kv_lora_rank + qk_rope = 512 + 64, v = kv_lora_rank
ABS_QK, ABS_V = 576, 512

MEASURED = {"chunk1_ms": 66.27, "chunk2_ms": 849.87, "layers": 24}


def pairs(queries: int, prefix: int) -> float:
    """Query-key pairs the kernel evaluates: the whole prefix, plus the causal
    half of the chunk against itself."""
    return queries * prefix + queries * queries / 2


def flops(n_pairs: float, qk: int, v: int) -> float:
    return n_pairs * (qk + v) * 2 * HEADS_PER_RANK


def main() -> int:
    c1_pairs = pairs(CHUNK, 0)
    c2_pairs = pairs(CHUNK, CHUNK)
    c1_flops = flops(c1_pairs, MHA_QK, MHA_V)
    c2_flops = flops(c2_pairs, ABS_QK, ABS_V)

    c1_ms = MEASURED["chunk1_ms"] / MEASURED["layers"]
    c2_ms = MEASURED["chunk2_ms"] / MEASURED["layers"]

    print(f"{'':22} {'chunk 1':>16} {'chunk 2':>16} {'ratio':>8}")
    print("-" * 66)
    print(f"{'MLA form':22} {'decompressed':>16} {'absorbed':>16}")
    print(f"{'Lq / Lv':22} {f'{MHA_QK} / {MHA_V}':>16} {f'{ABS_QK} / {ABS_V}':>16} "
          f"{(ABS_QK+ABS_V)/(MHA_QK+MHA_V):8.2f}")
    print(f"{'BLOCK_M / grid z':22} {'128 / 128':>16} {'64 / 256':>16}")
    print(f"{'query-key pairs':22} {c1_pairs:16.3e} {c2_pairs:16.3e} "
          f"{c2_pairs/c1_pairs:8.2f}")
    print(f"{'FLOP per layer':22} {c1_flops:16.3e} {c2_flops:16.3e} "
          f"{c2_flops/c1_flops:8.2f}")
    print(f"{'ms per layer':22} {c1_ms:16.2f} {c2_ms:16.2f} {c2_ms/c1_ms:8.2f}")
    print(f"{'achieved TFLOP/s':22} {c1_flops/(c1_ms/1e3)/1e12:16.0f} "
          f"{c2_flops/(c2_ms/1e3)/1e12:16.0f} "
          f"{(c2_flops/c2_ms)/(c1_flops/c1_ms):8.2f}")
    print()
    print(f"So the 12.8x time is {c2_flops/c1_flops:.1f}x more arithmetic at "
          f"{(c1_flops/c1_ms)/(c2_flops/c2_ms):.2f}x lower efficiency --")
    print("not a 4.3x efficiency collapse. The earlier reading assumed both chunks")
    print("ran the same form and so attributed the whole gap to the kernel.")
    print()
    print("The absorbed form is the right choice for decode (one query, many keys:")
    print("reading the latent once beats decompressing it). For a 16384-query")
    print("prefill chunk it is the wrong side of the trade: 3.4x the FLOPs per pair")
    print("to avoid a decompression that would have been amortized over 16384")
    print("queries. Chunked prefix cache exists to run the prefix part in the")
    print("decompressed form instead -- and it is unavailable on the triton")
    print("backend (CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS), which is")
    print("why this configuration pays the penalty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
