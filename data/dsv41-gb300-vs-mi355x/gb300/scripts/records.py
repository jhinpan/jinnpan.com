#!/usr/bin/env python3
"""Per-arm launch record: resolved backends, KV cache, memory lines, NUMA binding, driver,
power limit and clocks. Writes records/arms.json and prints one line per arm."""
import json
import re
import sys
from pathlib import Path

WORK = Path('$GB300_WORK')
INFO_KEYS = ['version', 'attention_backend', 'moe_runner_backend', 'moe_a2a_backend',
             'fp8_gemm_runner_backend', 'kv_cache_dtype', 'page_size', 'mem_fraction_static',
             'max_total_num_tokens', 'max_running_requests', 'chunked_prefill_size',
             'cuda_graph_max_bs_decode', 'disable_radix_cache', 'tp_size', 'ep_size',
             'speculative_algorithm', 'speculative_dspark_block_size', 'speculative_num_draft_tokens']
TP0_LINES = {
    'load_weight_end': r'TP0.*Load weight end\. (.*)',
    'memory_pool_end': r'TP0.*Memory pool end\. (.*)',
    'graph_capture_end': r'TP0.*(Capture .* CUDA graph end\. .*)',
    'all_reduce_config': r'TP0.*All Reduce config: (.*)',
    'max_total_num_tokens_line': r'TP0.*(max_total_num_tokens=.*)',
}


def smi_field(text, label):
    m = re.search(rf'{label}\s*:\s*(.+)', text)
    return m.group(1).strip() if m else None


def record(arm_dir):
    rec = {'arm': arm_dir.name}
    info_path = arm_dir / 'get_server_info.json'
    if info_path.exists():
        info = json.loads(info_path.read_text())
        rec.update({k: info.get(k) for k in INFO_KEYS})
    log = (arm_dir / 'server.log').read_text(errors='replace') if (arm_dir / 'server.log').exists() else ''
    for key, rx in TP0_LINES.items():
        rec[key] = [m.group(1)[:200] for m in re.finditer(rx, log)]
    rec['numa_binding_skipped'] = 'lacks permission to set NUMA affinity' in log
    rec['fp8_dense_log'] = re.findall(r'Use (\S+) for DeepSeek-V4.1 MXFP8 dense GEMMs', log)[:1]
    pre = (arm_dir / 'manifest.json')
    if pre.exists():
        smi = json.loads(pre.read_text()).get('nvidia_smi_pre', '')
        rec['driver'] = smi_field(smi, 'Driver Version')
        rec['power_limit'] = smi_field(smi, 'Current Power Limit')
        rec['max_sm_clock'] = smi_field(smi, 'SM')
        rec['docker_args'] = json.loads(pre.read_text()).get('docker_args', [])
    return rec


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else '.'
    recs = [record(d) for d in sorted((WORK / 'results').iterdir())
            if d.is_dir() and not d.name.startswith('_') and re.search(pattern, d.name)]
    (WORK / 'records' / 'arms.json').write_text(json.dumps(recs, indent=2) + '\n')
    for r in recs:
        mem = (r['load_weight_end'][:1] or [''])[0]
        pool = (r['memory_pool_end'][:1] or [''])[0]
        print(f"{r['arm']:14s} {r.get('version')} attn={r.get('attention_backend')} "
              f"moe={r.get('moe_runner_backend')} fp8={r.get('fp8_gemm_runner_backend')} "
              f"kv={r.get('kv_cache_dtype')}/p{r.get('page_size')} tp{r.get('tp_size')}ep{r.get('ep_size')} "
              f"numa_skipped={r['numa_binding_skipped']} driver={r.get('driver')} "
              f"| {mem[-60:]} | pool {pool}")


if __name__ == '__main__':
    main()
