#!/usr/bin/env python3
"""Profile N decode forwards at a fixed context length, decode only.

Arms the profiler *after* prefill has already produced its first token, so the
window contains nothing but steady-state decode and there is exactly one profiler
session per context point. Both properties matter: `profile_by_stage` opens a
second session per request, and libkineto segfaults during teardown once a few
sessions have accumulated in a process (seen at the third session, taking the whole
TP group down with it).

GPU activities only. Host-side ranges are unnecessary because kernel names are
mapped to block types offline (see name_map.py), and dropping the CPU activity is
what keeps the trace small enough to be safe.

  ./profile_decode.py --isl 65536 --tag ctx64k --num-steps 24
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

PORT = int(os.environ.get("PORT", "30100"))
BASE = f"http://127.0.0.1:{PORT}"


def post(path: str, payload: dict | None = None, timeout: float = 7200.0):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload or {}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


class Generation(threading.Thread):
    """Streaming generation whose token arrival times are visible to the driver."""

    def __init__(self, input_ids: list[int], max_new: int):
        super().__init__(daemon=True)
        self.input_ids = input_ids
        self.max_new = max_new
        self.first_token_at: float | None = None
        self.stamps: list[float] = []
        self.error: str | None = None
        self.meta: dict = {}
        self.t0 = 0.0

    def run(self) -> None:
        payload = {
            "input_ids": self.input_ids,
            "sampling_params": {"max_new_tokens": self.max_new, "temperature": 0.0,
                                "ignore_eos": True},
            "stream": True,
        }
        req = urllib.request.Request(
            BASE + "/generate", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        self.t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=7200) as r:
                for raw in r:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    now = time.time()
                    if self.first_token_at is None:
                        self.first_token_at = now
                    self.stamps.append(now)
                    try:
                        self.meta = json.loads(body).get("meta_info", self.meta)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:  # noqa: BLE001 - reported to the driver, not raised
            self.error = f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--isl", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--num-steps", type=int, default=24)
    ap.add_argument("--activities", default="GPU")
    ap.add_argument("--out-root", default="/sgl-workspace/workspace/kda_prof/traces")
    ap.add_argument("--prefill-timeout", type=float, default=1800.0)
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(1234)
    input_ids = [rng.randint(1000, 100000) for _ in range(args.isl)]

    try:
        post("/flush_cache")
    except urllib.error.HTTPError as e:
        print(f"  flush_cache: {e}", flush=True)
    time.sleep(2)

    # Enough tokens to cover the profiled window plus slack for the arming latency.
    max_new = args.num_steps * 3 + 40
    gen = Generation(input_ids, max_new)
    print(f"[{args.tag}] prefill isl={args.isl} (max_new={max_new})", flush=True)
    gen.start()

    t_wait = time.time()
    while gen.first_token_at is None and gen.is_alive():
        if time.time() - t_wait > args.prefill_timeout:
            print(f"[{args.tag}] prefill timeout", flush=True)
            return 1
        time.sleep(0.25)
    if gen.error:
        print(f"[{args.tag}] generation failed: {gen.error}", flush=True)
        return 1
    ttft = gen.first_token_at - gen.t0
    print(f"[{args.tag}] ttft={ttft:.2f}s -> arming profiler for "
          f"{args.num_steps} decode steps", flush=True)

    # A few tokens of settle time so the profiled window is steady-state decode.
    time.sleep(1.0)
    n_before = len(gen.stamps)
    r = post(
        "/start_profile",
        {
            "output_dir": out_dir,
            "num_steps": args.num_steps,
            "activities": args.activities.split(","),
            "with_stack": False,
            "record_shapes": False,
            "profile_id": args.tag,
        },
    )
    print(f"  start_profile: {r}", flush=True)

    # num_steps auto-stops the profiler; wait for that many tokens to appear.
    t_p = time.time()
    while len(gen.stamps) < n_before + args.num_steps + 2 and gen.is_alive():
        if time.time() - t_p > 600:
            break
        time.sleep(0.2)
    prof_tokens = len(gen.stamps) - n_before
    prof_wall = time.time() - t_p
    time.sleep(3)

    try:
        print(f"  stop_profile: {post('/stop_profile')}", flush=True)
    except urllib.error.HTTPError as e:
        print(f"  stop_profile: already auto-stopped ({e.code})", flush=True)

    gaps = [1000.0 * (b - a) for a, b in zip(gen.stamps, gen.stamps[1:])]
    steady = sorted(gaps[2:]) if len(gaps) > 4 else sorted(gaps)
    itl_median = steady[len(steady) // 2] if steady else float("nan")

    time.sleep(8)
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".gz"))
    print(f"[{args.tag}] {len(files)} traces; itl_median={itl_median:.2f} ms "
          f"({prof_tokens} tokens in {prof_wall:.1f}s during profile)", flush=True)

    with open(os.path.join(out_dir, "run.json"), "w") as fh:
        json.dump(
            {"tag": args.tag, "isl": args.isl, "num_steps": args.num_steps,
             "ttft_s": ttft, "itl_median_ms": itl_median,
             "itl_all_ms": gaps, "profiled_tokens": prof_tokens,
             "meta_info": gen.meta},
            fh, indent=1,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
