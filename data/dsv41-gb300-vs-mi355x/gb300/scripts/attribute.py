#!/usr/bin/env python3
"""Per-cycle attribution of an nsys capture (TP0) into the categories shared with the
MI355X trace analysis: MoE, dense GEMM, attention/indexer/compressor, mHC, all-reduce,
Engram, draft, other, and host gap (GPU idle).

A cycle is one launch of the main graph (target verify with DSpark, decode without).
Kernels on different streams overlap, so two views are reported per category:
  kernel_us  - summed kernel durations (can exceed the cycle)
  wall_us    - share of the cycle's wall time; overlapping intervals are split evenly
               among the kernels running in them, so the categories plus host gap sum to
               the cycle length.
"""
import argparse
import collections
import json
import re
import sqlite3
import statistics as st

# First match wins; order resolves fused kernels (e.g. MoE finalize + all-reduce is
# counted as all-reduce) and keeps MoE GEMMs out of dense GEMM.
RULES = [
    ('all_reduce', 'moe_finalize_allreduce', r'moe_finalize_all_reduce'),
    ('all_reduce', 'allreduce', r'all_reduce|allreduce|nvlink_push|nccl|_all_gather_kernel'),
    ('moe', 'moe_gemm', r'^bmm_.*E2m1|MxE2m1'),
    ('moe', 'routing', r'moe::dev::routing|_router_triton|GEMMTraitN<\(unsigned int\)384,'),
    ('moe', 'activation', r'silu_mul_clamp'),
    ('mhc', 'mhc', r'_hc_|_mhc_'),
    ('engram', 'engram', r'engram'),
    ('attention', 'attention', r'partial_gluon|^_combine$|flashmla|_q_rope_store|deepseek_rope'),
    ('attention', 'indexer', r'paged_mqa|index_q_kernel|index_k_kernel|topk_small_batch'),
    ('attention', 'compressor', r'flash_c[0-9]_decode|Compress'),
    ('attention', 'kv_metadata', r'page_table|page_indices|swa_locations|_low_ratio_metadata|'
                                 r'_window_gather|_block_seq_lens|assign_extend_cache_locs|'
                                 r'alloc_extend|assign_req_to_token|_expand_prefill_causally'),
    ('dense_gemm', 'mxfp8_gemm', r'BlockScaledPersistentDenseGemm'),
    ('dense_gemm', 'small_gemm', r'wo_a_fused|_wo_a_partial|_wo_a_reduce|n32k5120|n128k512|tiny_n_gemm'),
    ('dense_gemm', 'cublas_gemm', r'nvjet|cublasLt|gemm|Gemm'),
    ('dense_gemm', 'act_quant', r'mxfp8_quantize|MXFP8Quantize'),
    ('other', 'norm', r'rmsnorm|RMSNorm|_rmsnorm_mxfp8|MeanOps'),
    ('other', 'sampling_accept', r'argmax|greedy|accept|correct_drafts|_build_out_tokens|embedding'),
    ('other', 'elementwise_copy', r'.'),
]
RULES = [(c, s, re.compile(rx)) for c, s, rx in RULES]
CATEGORIES = ['moe', 'dense_gemm', 'attention', 'mhc', 'all_reduce', 'engram', 'draft', 'other']


def classify(name):
    for cat, sub, rx in RULES:
        if rx.search(name):
            return cat, sub
    return 'other', 'unmatched'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('sqlite')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--out')
    a = p.parse_args()
    c = sqlite3.connect(a.sqlite)
    names = dict(c.execute('select id, value from StringIds'))
    rows = c.execute('select start, end, graphId, correlationId, demangledName, shortName, graphNodeId '
                     'from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? order by start', (a.device,)).fetchall()
    copies = c.execute('select start, end from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=? '
                       'order by start', (a.device,)).fetchall()

    # Graph launches do not always carry distinct correlation ids (some are all one id),
    # so a launch is counted by its first graph node, which runs exactly once per launch.
    # Main graph: most kernel time. Another graph launched about as often is the DSpark draft.
    graph_time = collections.Counter()
    first_node = {}
    for s, e, g, corr, dn, sn, node in rows:
        if g is not None:
            graph_time[g] += e - s
            first_node.setdefault(g, node)
    node_starts = collections.defaultdict(list)
    for s, e, g, corr, dn, sn, node in rows:
        if g is not None and node == first_node[g]:
            node_starts[g].append(s)
    main_graph = max(graph_time, key=graph_time.get)
    n_main = len(node_starts[main_graph])
    draft_graphs = {g for g in graph_time if g != main_graph
                    and abs(len(node_starts[g]) - n_main) <= 0.1 * n_main}
    bounds = sorted(node_starts[main_graph])

    events = [(s, e, ('memcpy', 'memcpy')) for s, e in copies]
    for s, e, g, corr, dn, sn, node in rows:
        name = names.get(dn) or names.get(sn) or ''
        cat, sub = classify(name)
        if g in draft_graphs:
            cat, sub = 'draft', f'draft:{cat}'
        events.append((s, e, (cat, sub)))
    events.sort()

    per_cycle = []
    j = 0
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        while j < len(events) and events[j][0] < lo:
            j += 1
        cyc = []
        k = j
        while k < len(events) and events[k][0] < hi:
            s, e, lab = events[k]
            cyc.append((s, min(e, hi), lab))
            k += 1
        kern = collections.Counter()
        count = collections.Counter()
        for s, e, lab in cyc:
            kern[lab] += e - s
            count[lab] += 1
        # Sweep line: split each interval's duration evenly among the active events.
        marks = sorted([(s, 1, lab) for s, e, lab in cyc] + [(e, -1, lab) for s, e, lab in cyc],
                       key=lambda m: (m[0], m[1]))
        active = collections.Counter()
        wall = collections.Counter()
        busy = 0
        prev = lo
        for x, delta, lab in marks:
            total = sum(active.values())
            if total and x > prev:
                busy += x - prev
                for act, cnt in active.items():
                    wall[act] += (x - prev) * cnt / total
            prev = max(prev, x)
            active[lab] += delta
            if active[lab] == 0:
                del active[lab]
        per_cycle.append({'len': hi - lo, 'busy': busy, 'kern': kern, 'count': count, 'wall': wall})

    n = len(per_cycle)
    labels = sorted({lab for pc in per_cycle for lab in pc['kern']})
    subs = []
    for lab in labels:
        subs.append({'category': lab[0], 'sub': lab[1],
                     'kernel_us': sum(pc['kern'][lab] for pc in per_cycle) / n / 1e3,
                     'wall_us': sum(pc['wall'][lab] for pc in per_cycle) / n / 1e3,
                     'kernels': sum(pc['count'][lab] for pc in per_cycle) / n})
    cats = []
    for cat in CATEGORIES + ['memcpy']:
        sel = [x for x in subs if x['category'] == cat]
        if sel:
            cats.append({'category': cat, 'kernel_us': sum(x['kernel_us'] for x in sel),
                         'wall_us': sum(x['wall_us'] for x in sel), 'kernels': sum(x['kernels'] for x in sel)})
    cycle_us = [pc['len'] / 1e3 for pc in per_cycle]
    idle_us = sum(pc['len'] - pc['busy'] for pc in per_cycle) / n / 1e3
    result = {'sqlite': a.sqlite, 'device': a.device, 'main_graph': main_graph,
              'draft_graphs': sorted(draft_graphs), 'cycles': n,
              'cycle_us_median': st.median(cycle_us), 'cycle_us_mean': st.mean(cycle_us),
              'host_gap_us': idle_us, 'categories': cats, 'subcategories': subs}
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(result, f, indent=2)
    print(f"cycles={n} cycle median={result['cycle_us_median']:.1f}us mean={result['cycle_us_mean']:.1f}us "
          f"host gap (GPU idle)={idle_us:.1f}us main_graph={main_graph} draft_graphs={sorted(draft_graphs)}")
    print(f"{'category':12s} {'wall_us':>8s} {'kernel_us':>9s} {'kernels':>8s}")
    for x in cats:
        print(f"{x['category']:12s} {x['wall_us']:8.1f} {x['kernel_us']:9.1f} {x['kernels']:8.1f}")
    print(f"{'host_gap':12s} {idle_us:8.1f}")
    print('--- subcategories')
    for x in sorted(subs, key=lambda x: -x['wall_us']):
        print(f"  {x['category']:11s} {x['sub']:24s} {x['wall_us']:8.1f} {x['kernel_us']:9.1f} {x['kernels']:8.1f}")


if __name__ == '__main__':
    main()
