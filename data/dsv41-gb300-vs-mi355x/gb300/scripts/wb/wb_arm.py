#!/usr/bin/env python3
"""One measurement arm inside the dsv41-workbench container: fresh server process ->
benchmarks -> teardown.

The workbench has no Docker CLI, so each launch is a fresh server process of this container
rather than a fresh container (arm.py). Cells, client, JIT caches and records match arm.py,
and the container holds CAP_SYS_NICE, so SGLang's NUMA binding applies to every arm here.
With --profile-steps the server runs under nsys and TP0 captures that many scheduler steps,
as profile_arm.py does; the report is exported to SQLite and attributed with attribute.py.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

WORK = Path('/work')
PORT = 30000
BASE = f'http://127.0.0.1:{PORT}'
SGLANG_TREE = Path('/sgl-workspace/sglang')
RECORD_PATTERNS = [
    r'backend', r'Load weight', r'Memory pool end', r'KV Cache is allocated', r'kv_cache_dtype',
    r'Capture cuda graph', r'cuda graph', r'max_total_num_tokens', r'avail mem', r'mem usage',
    r'DSPARK', r'speculative', r'padding', r'engram', r'Engram', r'MXFP', r'mxfp', r'cutedsl',
    r'NUMA', r'numa', r'serial-moe', r'WARNING', r'Error', r'error', r'Traceback',
]


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sh(cmd, check=True, timeout=None):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), text=True, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check and r.returncode != 0:
        raise RuntimeError(f'command failed rc={r.returncode}: {cmd}\n{r.stdout}')
    return r.stdout


def gpu_state():
    rows = sh('nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw,clocks.sm,'
              'clocks.mem,temperature.gpu,power.limit --format=csv,noheader,nounits').strip().splitlines()
    apps = sh('nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader').strip()
    return {'gpus': [dict(zip(['index', 'mem_used_mib', 'util', 'power_w', 'sm_mhz', 'mem_mhz',
                               'temp_c', 'power_limit_w'], [x.strip() for x in r.split(',')]))
                     for r in rows],
            'compute_apps': apps.splitlines() if apps else []}


def other_servers():
    # List form: pgrep excludes itself, and no shell whose argv carries the pattern exists.
    out = subprocess.run(['pgrep', '-af', r'sglang::|sglang serve|sglang.launch_server'],
                         text=True, stdout=subprocess.PIPE).stdout
    return [line for line in out.splitlines() if line.strip()]


def idle_check(timeout_s=300):
    end = time.monotonic() + timeout_s
    while True:
        st = gpu_state()
        busy = [g for g in st['gpus'] if float(g['mem_used_mib']) > 1024 or float(g['util']) > 0]
        others = other_servers()
        if not (busy or st['compute_apps'] or others):
            return st
        if time.monotonic() > end:
            raise RuntimeError(f'machine not idle: busy={busy} apps={st["compute_apps"]} procs={others}')
        time.sleep(5)


def wait_ready(proc, timeout_s):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise RuntimeError(f'server exited before ready: rc={proc.returncode}')
        try:
            if requests.get(BASE + '/health', timeout=5).status_code == 200 and \
                    requests.get(BASE + '/v1/models', timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError(f'server not ready within {timeout_s}s')


def stop(proc, grace_s=90):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def capture(out, steps, trigger_after, proc):
    """Warm once, then start a CUDA_PROFILER window of `steps` scheduler steps mid-stream."""
    prompt = json.loads((WORK / 'bbuf/prompt.json').read_text())
    body = {'input_ids': prompt['input_ids'],
            'sampling_params': {'temperature': 0, 'max_new_tokens': 1024, 'ignore_eos': True,
                                'stream_interval': 1}}
    requests.post(BASE + '/generate', json=body, timeout=600).raise_for_status()
    started, reply, streamed = threading.Event(), {}, 0

    def start_profile():
        r = requests.post(BASE + '/start_profile', timeout=600,
                          json={'activities': ['CUDA_PROFILER'], 'num_steps': steps})
        reply.update(code=r.status_code, text=r.text)

    status = {}
    try:
        with requests.post(BASE + '/generate', json={**body, 'stream': True}, stream=True,
                           timeout=(30, 600)) as r:
            for line in r.iter_lines():
                if not line.startswith(b'data: ') or line[6:] == b'[DONE]':
                    continue
                streamed = json.loads(line[6:]).get('meta_info', {}).get('completion_tokens', 0)
                if streamed >= trigger_after and not started.is_set():
                    started.set()
                    threading.Thread(target=start_profile, daemon=True).start()
    except requests.RequestException as e:
        # Expected: nsys shuts the server down after the captured steps.
        status['stream_end'] = repr(e)[:200]
    status['streamed_tokens'] = streamed
    proc.wait(timeout=1800)
    status['server_exit'] = proc.returncode
    status['profile_reply'] = reply
    return status


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True, help='arm id; results go to /work/results/<name>')
    p.add_argument('--server-args', required=True, help='arguments after `sglang serve`')
    p.add_argument('--env', action='append', default=[], help='KEY=VALUE for the server')
    p.add_argument('--pythonpath', help='prepended to PYTHONPATH (experiment-only patches)')
    p.add_argument('--bench', action='append', default=[],
                   help='shell command run from /work once ready; {out} expands to the arm dir')
    p.add_argument('--record-placement', action='store_true')
    p.add_argument('--profile-steps', type=int, help='run under nsys; TP0 captures N steps')
    p.add_argument('--trigger-after', type=int, default=256)
    p.add_argument('--ready-timeout', type=int, default=1800)
    a = p.parse_args()

    out = WORK / 'results' / a.name
    if out.exists():
        sys.exit(f'{out} exists; refusing to overwrite an arm')
    out.mkdir(parents=True)
    env = os.environ.copy()
    env.update(TRITON_CACHE_DIR='/root/.cache/triton', DG_JIT_CACHE_DIR='/root/.cache/deep_gemm',
               TILELANG_CACHE_DIR='/root/.cache/tilelang', CUPY_CACHE_DIR='/root/.cache/cupy')
    for kv in a.env:
        k, v = kv.split('=', 1)
        env[k] = v
    if a.pythonpath:
        env['PYTHONPATH'] = a.pythonpath + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    server = 'sglang serve ' + a.server_args
    rep = out / 'nsys' / 'tp0'
    if a.profile_steps:
        rep.parent.mkdir()
        server = ('nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi '
                  '--capture-range-end=stop-shutdown --sample=none --cpuctxsw=none --force-overwrite=true '
                  f'--output={rep} ' + server)
    manifest = {'name': a.name, 'server_cmd': server, 'env': a.env, 'pythonpath': a.pythonpath,
                'bench': a.bench, 'profile_steps': a.profile_steps, 'trigger_after': a.trigger_after,
                'host': os.uname().nodename, 'container': 'dsv41-workbench', 'started_utc': utc(),
                'sglang_rev': sh(['git', '-C', str(SGLANG_TREE), 'rev-parse', 'HEAD'], check=False).strip(),
                'sglang_dirty': sh(['git', '-C', str(SGLANG_TREE), 'status', '--porcelain',
                                    '--untracked-files=no'], check=False).strip().splitlines()[:20]}
    manifest['idle_check'] = idle_check()
    manifest['nvidia_smi_pre'] = sh('nvidia-smi -q -d POWER,CLOCK,PERFORMANCE')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    t0 = time.monotonic()
    log = open(out / 'server.log', 'w')
    proc = subprocess.Popen(['bash', '-c', 'exec ' + server], cwd=WORK, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    status = {'ok': False}
    try:
        wait_ready(proc, a.ready_timeout)
        status['ready_s'] = round(time.monotonic() - t0, 1)
        for route in ('/get_server_info', '/v1/models'):
            r = requests.get(BASE + route, timeout=30)
            (out / (route.strip('/').replace('/', '_') + '.json')).write_text(r.text)
        if a.record_placement:
            (out / 'placement_ready.txt').write_text(sh(['bash', '/work/scripts/placement.sh'], check=False))
        if a.profile_steps:
            status.update(capture(out, a.profile_steps, a.trigger_after, proc))
            status['report_exists'] = rep.with_suffix('.nsys-rep').exists()
            status['ok'] = status['report_exists']
        else:
            status['bench_rc'] = []
            for i, cmd in enumerate(a.bench):
                cmd = cmd.format(out=out)
                with open(out / f'bench{i}.log', 'w') as f:
                    f.write(f'$ {cmd}\n')
                    f.flush()
                    rc = subprocess.run(cmd, shell=True, cwd=WORK, stdout=f, stderr=subprocess.STDOUT).returncode
                status['bench_rc'].append(rc)
                if rc != 0:
                    raise RuntimeError(f'bench {i} failed rc={rc}')
            if a.record_placement:
                (out / 'placement_post.txt').write_text(sh(['bash', '/work/scripts/placement.sh'], check=False))
            (out / 'nvidia_smi_post.txt').write_text(sh('nvidia-smi -q -d POWER,CLOCK,PERFORMANCE'))
            status['ok'] = True
    except Exception as e:  # recorded, then re-raised after teardown
        status['error'] = repr(e)[:500]
        raise
    finally:
        stop(proc)
        log.close()
        try:
            idle_check(300)
            status['released'] = True
        except RuntimeError as e:
            status['released'] = False
            status['release_error'] = str(e)[:500]
        text = (out / 'server.log').read_text(errors='replace').splitlines()
        rx = re.compile('|'.join(RECORD_PATTERNS))
        (out / 'server_record_lines.txt').write_text('\n'.join(l for l in text if rx.search(l)) + '\n')
        status['finished_utc'] = utc()
        (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        print('ARM', a.name, json.dumps(status), flush=True)
    if a.profile_steps and status['ok']:
        sh(['nsys', 'export', '--type', 'sqlite', '--force-overwrite', 'true', '--output',
            str(rep) + '.sqlite', str(rep) + '.nsys-rep'], timeout=3600)
        print(sh([sys.executable, '/work/scripts/attribute.py', str(rep) + '.sqlite',
                  '--out', str(out / 'attribution.json')]), flush=True)
    if not (status['ok'] and status.get('released')):
        sys.exit(1)


if __name__ == '__main__':
    main()
