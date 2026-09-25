#!/usr/bin/env python3
"""Run one measurement arm on this GB300 tray: fresh server container -> benchmarks -> teardown.

Each arm gets its own container so every measurement starts from a fresh server process.
Containers run as root (the host account has no sudo); host-side directories are created
before launch so the account can still read and delete everything the container writes.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import requests

WORK = Path('$GB300_WORK')
MODEL_HOST = Path('$GB300_MODEL_DIR')
MODEL_CONTAINER = '/models/DeepSeek-V4.1-Flash'

# Log lines that show which kernels/backends were chosen and where memory went.
RECORD_PATTERNS = [
    r'backend', r'Load weight', r'Memory pool end', r'KV Cache is allocated', r'kv_cache_dtype',
    r'Capture cuda graph', r'cuda graph', r'max_total_num_tokens', r'avail mem', r'mem usage',
    r'DSPARK', r'speculative', r'padding', r'engram', r'Engram', r'MXFP', r'mxfp', r'cutedsl',
    r'WARNING', r'Error', r'error', r'Traceback',
]


def sh(cmd, check=True, capture=True, timeout=None):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), text=True, timeout=timeout,
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.STDOUT if capture else None)
    if check and r.returncode != 0:
        raise RuntimeError(f'command failed rc={r.returncode}: {cmd}\n{r.stdout}')
    return r.stdout if capture else ''


def gpu_state():
    rows = sh('nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw,clocks.sm,'
              'clocks.mem,temperature.gpu,power.limit --format=csv,noheader,nounits').strip().splitlines()
    apps = sh('nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader').strip()
    return {'gpus': [dict(zip(['index', 'mem_used_mib', 'util', 'power_w', 'sm_mhz', 'mem_mhz',
                               'temp_c', 'power_limit_w'], [x.strip() for x in r.split(',')]))
                     for r in rows],
            'compute_apps': apps.splitlines() if apps else []}


def idle_check(timeout_s=120):
    # GPU memory of a just-stopped container is released asynchronously. The workbench
    # container may stay up; the GPU checks still reject a server running inside it.
    end = time.monotonic() + timeout_s
    while True:
        st = gpu_state()
        busy = [g for g in st['gpus'] if float(g['mem_used_mib']) > 1024 or float(g['util']) > 0]
        running = '\n'.join(
            line for line in sh("docker ps --format '{{.Names}} {{.Label \"dsv41.role\"}}'").splitlines()
            if line.strip() and not line.endswith(' workbench'))
        if not (busy or st['compute_apps'] or running):
            return st
        if time.monotonic() > end:
            raise RuntimeError(f'machine not idle: busy={busy} apps={st["compute_apps"]} '
                               f'containers={running!r}')
        time.sleep(5)


def wait_ready(port, container, timeout_s):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        state = sh(f"docker inspect -f '{{{{.State.Status}}}} {{{{.State.ExitCode}}}}' {container}",
                   check=False).strip()
        if not state.startswith('running'):
            raise RuntimeError(f'server container stopped before ready: {state}')
        try:
            if requests.get(f'http://127.0.0.1:{port}/health', timeout=5).status_code == 200:
                if requests.get(f'http://127.0.0.1:{port}/v1/models', timeout=5).status_code == 200:
                    return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError(f'server not ready within {timeout_s}s')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True, help='arm id; results go to results/<name>')
    p.add_argument('--image', required=True)
    p.add_argument('--cache-key', required=True, help='separate JIT cache per code version')
    p.add_argument('--workdir', default='/work')
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--env', action='append', default=[], help='KEY=VALUE for the server')
    p.add_argument('--server-cmd', required=True, help='shell command run inside the container')
    p.add_argument('--bench', action='append', default=[],
                   help='host shell command run from WORK after the server is ready; '
                        '{out} expands to the arm result dir')
    p.add_argument('--docker-arg', action='append', default=[],
                   help='extra `docker run` argument, e.g. --docker-arg=--cap-add=SYS_NICE')
    p.add_argument('--record-placement', action='store_true',
                   help='record scheduler CPU/NUMA placement after ready and after benchmarks')
    p.add_argument('--sample-gpu', action='store_true',
                   help='sample SM/memory clocks, power and clock-event reasons every 200 ms '
                        'while benchmarks run (diagnostic runs only)')
    p.add_argument('--ready-timeout', type=int, default=5400)
    p.add_argument('--keep', action='store_true', help='leave the server running after benchmarks')
    a = p.parse_args()

    out = WORK / 'results' / a.name
    if out.exists():
        sys.exit(f'{out} exists; refusing to overwrite an arm')
    out.mkdir(parents=True)
    cache = WORK / 'cache' / a.cache_key
    cache.mkdir(parents=True, exist_ok=True)
    container = f'dsv41-{a.name}'

    manifest = {'name': a.name, 'image': a.image, 'server_cmd': a.server_cmd, 'env': a.env,
                'docker_args': a.docker_arg, 'sample_gpu': a.sample_gpu,
                'workdir': a.workdir, 'port': a.port, 'bench': a.bench, 'host': os.uname().nodename,
                'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    manifest['idle_check'] = idle_check()
    manifest['nvidia_smi_pre'] = sh('nvidia-smi -q -d POWER,CLOCK,PERFORMANCE')
    image_id = sh(f"docker image inspect -f '{{{{.Id}}}} {{{{index .RepoDigests 0}}}}' {a.image}").strip()
    manifest['image_id'] = image_id
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    envs = ['TRITON_CACHE_DIR=/root/.cache/triton', 'DG_JIT_CACHE_DIR=/root/.cache/deep_gemm',
            'TILELANG_CACHE_DIR=/root/.cache/tilelang', 'CUPY_CACHE_DIR=/root/.cache/cupy'] + a.env
    docker = ['docker', 'run', '-d', '--init', '--name', container, '--gpus', 'all', '--ipc=host',
              '--shm-size', '32g', '--network', 'host',
              '-v', f'{MODEL_HOST}:{MODEL_CONTAINER}:ro', '-v', f'{WORK}:/work',
              '-v', f'{cache}:/root/.cache', '-w', a.workdir] + a.docker_arg
    for e in envs:
        docker += ['-e', e]
    docker += [a.image, 'bash', '-c', a.server_cmd]
    (out / 'docker_run.txt').write_text(shlex.join(docker) + '\n')

    t0 = time.monotonic()
    sh(docker)
    status = {'ok': False}
    try:
        wait_ready(a.port, container, a.ready_timeout)
        status['ready_s'] = round(time.monotonic() - t0, 1)
        base = f'http://127.0.0.1:{a.port}'
        for route in ('/get_server_info', '/v1/models'):
            r = requests.get(base + route, timeout=30)
            (out / (route.strip('/').replace('/', '_') + '.json')).write_text(r.text)
        if a.record_placement:
            (out / 'placement_ready.txt').write_text(
                sh(['docker', 'exec', container, 'bash', '/work/scripts/placement.sh'], check=False))
        sampler = None
        if a.sample_gpu:
            sampler = subprocess.Popen(
                ['nvidia-smi', '--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,'
                 'temperature.gpu,utilization.gpu,clocks_event_reasons.active',
                 '--format=csv,noheader', '-lms', '200'],
                stdout=open(out / 'gpu_samples.csv', 'w'), stderr=subprocess.STDOUT)
        status['bench_rc'] = []
        try:
            for i, cmd in enumerate(a.bench):
                cmd = cmd.format(out=out)
                with open(out / f'bench{i}.log', 'w') as f:
                    f.write(f'$ {cmd}\n')
                    f.flush()
                    rc = subprocess.run(cmd, shell=True, cwd=WORK, stdout=f,
                                        stderr=subprocess.STDOUT).returncode
                status['bench_rc'].append(rc)
                if rc != 0:
                    raise RuntimeError(f'bench {i} failed rc={rc}')
        finally:
            if sampler is not None:
                sampler.terminate()
                sampler.wait()
        if a.record_placement:
            (out / 'placement_post.txt').write_text(
                sh(['docker', 'exec', container, 'bash', '/work/scripts/placement.sh'], check=False))
        (out / 'nvidia_smi_post.txt').write_text(sh('nvidia-smi -q -d POWER,CLOCK,PERFORMANCE'))
        status['ok'] = True
    finally:
        with open(out / 'server.log', 'w') as f:
            subprocess.run(['docker', 'logs', container], stdout=f, stderr=subprocess.STDOUT)
        if not a.keep or not status['ok']:
            sh(['docker', 'stop', '-t', '60', container], check=False)
            sh(['docker', 'rm', container], check=False)
        log = (out / 'server.log').read_text(errors='replace').splitlines()
        rx = re.compile('|'.join(RECORD_PATTERNS))
        (out / 'server_record_lines.txt').write_text('\n'.join(l for l in log if rx.search(l)) + '\n')
        status['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        print('ARM', a.name, json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
