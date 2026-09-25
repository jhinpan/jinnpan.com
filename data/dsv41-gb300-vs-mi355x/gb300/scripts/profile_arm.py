#!/usr/bin/env python3
"""Step 2 attribution run: server under nsys, capture exactly N scheduler steps of
steady-state BS1 decode on TP0, then export the report to SQLite.

Only TP0 calls cudaProfilerStart/Stop for the CUDA_PROFILER activity, and
--capture-range-end=stop-shutdown makes nsys stop the server once the N steps are done,
so the report is complete when the container exits. Timing runs never use this script.
"""
import argparse
import json
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm import MODEL_CONTAINER, MODEL_HOST, WORK, idle_check, sh, wait_ready  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--cache-key', required=True)
    p.add_argument('--port', type=int, default=30000)
    p.add_argument('--env', action='append', default=[])
    p.add_argument('--docker-arg', action='append', default=[])
    p.add_argument('--server-cmd', required=True)
    p.add_argument('--prompt', default='bbuf/prompt.json')
    p.add_argument('--max-tokens', type=int, default=1024)
    p.add_argument('--trigger-after', type=int, default=256,
                   help='start the capture once this many tokens have streamed')
    p.add_argument('--num-steps', type=int, default=100)
    a = p.parse_args()

    out = WORK / 'results' / a.name
    if out.exists():
        sys.exit(f'{out} exists; refusing to overwrite')
    (out / 'nsys').mkdir(parents=True)
    cache = WORK / 'cache' / a.cache_key
    container = f'dsv41-{a.name}'
    rep = f'/work/results/{a.name}/nsys/tp0'
    nsys = (f'exec nsys profile --trace=cuda,nvtx --cuda-graph-trace=node '
            f'--capture-range=cudaProfilerApi --capture-range-end=stop-shutdown '
            f'--sample=none --cpuctxsw=none --force-overwrite=true --output={rep} ')
    envs = ['TRITON_CACHE_DIR=/root/.cache/triton', 'DG_JIT_CACHE_DIR=/root/.cache/deep_gemm',
            'TILELANG_CACHE_DIR=/root/.cache/tilelang', 'CUPY_CACHE_DIR=/root/.cache/cupy'] + a.env
    docker = ['docker', 'run', '-d', '--init', '--name', container, '--gpus', 'all', '--ipc=host',
              '--shm-size', '32g', '--network', 'host',
              '-v', f'{MODEL_HOST}:{MODEL_CONTAINER}:ro', '-v', f'{WORK}:/work',
              '-v', f'{cache}:/root/.cache', '-w', '/work'] + a.docker_arg
    for e in envs:
        docker += ['-e', e]
    docker += [a.image, 'bash', '-c', nsys + a.server_cmd]
    manifest = {'name': a.name, 'image': a.image, 'docker_run': shlex.join(docker),
                'num_steps': a.num_steps, 'trigger_after': a.trigger_after,
                'idle_check': idle_check(), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    prompt = json.loads((WORK / a.prompt).read_text())
    base = f'http://127.0.0.1:{a.port}'
    body = {'input_ids': prompt['input_ids'],
            'sampling_params': {'temperature': 0, 'max_new_tokens': a.max_tokens,
                                'ignore_eos': True, 'stream_interval': 1}}
    status = {'ok': False}
    sh(docker)
    try:
        wait_ready(a.port, container, 5400)
        requests.post(base + '/generate', json=body, timeout=600).raise_for_status()

        started = threading.Event()
        profile_reply = {}

        def start_profile():
            r = requests.post(base + '/start_profile', timeout=600,
                              json={'activities': ['CUDA_PROFILER'], 'num_steps': a.num_steps})
            profile_reply.update(code=r.status_code, text=r.text)

        streamed = 0
        try:
            with requests.post(base + '/generate', json={**body, 'stream': True}, stream=True,
                               timeout=(30, 600)) as r:
                for line in r.iter_lines():
                    if not line.startswith(b'data: ') or line[6:] == b'[DONE]':
                        continue
                    streamed = json.loads(line[6:]).get('meta_info', {}).get('completion_tokens', 0)
                    if streamed >= a.trigger_after and not started.is_set():
                        started.set()
                        threading.Thread(target=start_profile, daemon=True).start()
        except requests.RequestException as e:
            # Expected: nsys shuts the server down after the captured steps.
            status['stream_end'] = repr(e)[:200]
        status['streamed_tokens'] = streamed
        rc = sh(['docker', 'wait', container], timeout=1800).strip()
        status['container_exit'] = rc
        status['profile_reply'] = profile_reply
        status['report_exists'] = (out / 'nsys' / 'tp0.nsys-rep').exists()
        status['ok'] = status['report_exists']
    finally:
        with open(out / 'server.log', 'w') as f:
            subprocess.run(['docker', 'logs', container], stdout=f, stderr=subprocess.STDOUT)
        sh(['docker', 'rm', '-f', container], check=False)
        (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    if status['ok']:
        sh(['docker', 'run', '--rm', '-v', f'{WORK}:/work', a.image, 'nsys', 'export',
            '--type', 'sqlite', '--force-overwrite', 'true', '--output', rep + '.sqlite', rep + '.nsys-rep'],
           timeout=3600)
        sh(['docker', 'run', '--rm', '-v', f'{WORK}:/work', a.image, 'chown', '-R', '10304:10304',
            f'/work/results/{a.name}'])
    print('PROFILE', a.name, json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
