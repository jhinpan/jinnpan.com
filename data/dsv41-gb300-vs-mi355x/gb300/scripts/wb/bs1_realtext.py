#!/usr/bin/env python3
"""The MI355X real-text BS=1 contract, replayed against a running server.

Same four chat-encoded 4,096-token prompts (token ids only), same request body (temperature 0,
ignore_eos, 1,024 streamed tokens), same order (two unscored warm-ups, then 24 samples per
prompt, alternating direction) and the same per-request metrics as the frozen client
pr39857-bs1-decode-20260921/bench/run_model.py through its model_pa_throughput.Client.
Output text is not kept: each request records the sha256 of its output ids.
"""
import argparse
import asyncio
import hashlib
import json
import math
import statistics as st
import time
from pathlib import Path

import aiohttp

ROW = dict(name="primary4k_o1024_c1", prompt=4096, cached=0, output=1024, concurrency=1)


def sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def body(ids, output):
    request = dict(input_ids=ids, sampling_params=dict(temperature=0, max_new_tokens=output,
                   ignore_eos=True), stream=True)
    return json.dumps(request, separators=(",", ":")).encode()


async def request(session, base, data, expected):
    start = time.perf_counter_ns()
    arrivals, output_ids, previous, final, done = [], [], 0, None, False
    async with session.post(base + "/generate", data=data,
                            headers={"Content-Type": "application/json"}) as response:
        response.raise_for_status()
        async for line in response.content:
            line = line.strip()
            if not line:
                continue
            assert line.startswith(b"data: "), line[:1000]
            if line == b"data: [DONE]":
                done = True
                continue
            result = json.loads(line[6:])
            assert "error" not in result, result
            meta = result["meta_info"]
            count = meta["completion_tokens"]
            assert count >= previous, (count, previous)
            ids = result.get("output_ids", [])
            if count > previous:
                arrivals.append(dict(ms=(time.perf_counter_ns() - start) / 1e6,
                                     total_tokens=count, delta_tokens=count - previous))
                if len(ids) == count:
                    output_ids = ids
                else:
                    assert len(ids) == count - previous, (len(ids), count, previous)
                    output_ids.extend(ids)
                previous = count
            final = result
    elapsed = (time.perf_counter_ns() - start) / 1e6
    assert done and final is not None and arrivals
    meta = final["meta_info"]
    assert meta["prompt_tokens"] == expected["prompt"], (expected, meta)
    assert meta["cached_tokens"] == expected["cached"], (expected, meta)
    assert len(output_ids) == meta["completion_tokens"] == expected["output"], meta
    assert meta["finish_reason"]["type"] == "length", meta
    server_ms = meta["e2e_latency"] * 1000
    ttft_ms = meta["first_token_latency"] * 1000
    assert all(math.isfinite(x) and x > 0 for x in (server_ms, ttft_ms, elapsed))
    assert ttft_ms <= server_ms
    return dict(client_ms=elapsed, client_ttft_ms=arrivals[0]["ms"], server_ms=server_ms,
                server_ttft_ms=ttft_ms, server_tpot_ms=(server_ms - ttft_ms) / max(1, len(output_ids) - 1),
                output_ids=output_ids, meta_info=meta, arrivals=arrivals)


def metrics(result):
    meta = result["meta_info"]
    # Current main names the draft counters spec_num_correct_drafts / spec_num_proposed_drafts.
    accepted = meta.get("spec_accepted_drafts", meta.get("spec_num_correct_drafts"))
    proposed = meta.get("spec_proposed_drafts", meta.get("spec_num_proposed_drafts"))
    arrivals = result["arrivals"]
    count = meta["completion_tokens"]
    span_ms = arrivals[-1]["ms"] - arrivals[0]["ms"]
    delivered_after_first = arrivals[-1]["total_tokens"] - arrivals[0]["total_tokens"]
    assert span_ms > 0 and delivered_after_first > 0
    return dict(
        request_output_tps=count / (result["client_ms"] / 1000),
        server_request_output_tps=count / (result["server_ms"] / 1000),
        stream_decode_tps=delivered_after_first / (span_ms / 1000),
        pr_style_stream_tps=count / (span_ms / 1000),
        stream_decode_ms=span_ms,
        stream_first_count=arrivals[0]["total_tokens"],
        server_decode_tps=(count - 1) / ((result["server_ms"] - result["server_ttft_ms"]) / 1000),
        server_post_ttft_ms=result["server_ms"] - result["server_ttft_ms"],
        wall_per_verify_proxy_ms=(result["server_ms"] - result["server_ttft_ms"]) / meta["spec_verify_ct"],
        acceptance_rate=accepted / proposed if accepted is not None and proposed else None,
        accepted_length=meta["spec_accept_length"],
        verify_count=meta["spec_verify_ct"],
    )


async def run(args):
    prompts = json.loads(Path(args.inputs).read_text())["prompts"]
    for p in prompts:
        assert sha(p["input_ids"]) == p["sha256"], p["index"]
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    timeout = aiohttp.ClientTimeout(total=300, sock_read=180)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        async def generate(i, repetition, scored):
            ids = prompts[i]["input_ids"]
            result = await request(session, args.url, body(ids, ROW["output"]), ROW)
            d = metrics(result)
            row = dict(index=i, repetition=repetition, scored=scored, input_sha256=prompts[i]["sha256"],
                       output_sha256=sha(result["output_ids"]), client_ms=result["client_ms"],
                       client_ttft_ms=result["client_ttft_ms"], server_ms=result["server_ms"],
                       server_ttft_ms=result["server_ttft_ms"], **d)
            rows.append(row)
            with (args.out / "requests.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(dict(index=i, repetition=repetition, scored=scored,
                                  accept=round(d["accepted_length"], 3),
                                  cycle_ms=round(d["wall_per_verify_proxy_ms"], 3),
                                  decode_tps=round(d["stream_decode_tps"], 1))), flush=True)

        for i in range(2):
            await generate(i, -1, False)
        for repeat in range(args.samples_per_prompt):
            for i in (range(4) if repeat % 2 == 0 else reversed(range(4))):
                await generate(i, repeat, True)
    scored = [r for r in rows if r["scored"]]
    med = lambda k: st.median(r[k] for r in scored)
    summary = dict(row=ROW, requests=len(scored),
                   accepted_length_median=med("accepted_length"),
                   accepted_length_by_prompt={i: st.median(r["accepted_length"] for r in scored if r["index"] == i)
                                              for i in range(4)},
                   wall_per_verify_proxy_ms_median=med("wall_per_verify_proxy_ms"),
                   stream_decode_tps_median=med("stream_decode_tps"),
                   request_output_tps_median=med("request_output_tps"),
                   server_ttft_ms_median=med("server_ttft_ms"),
                   distinct_outputs_by_prompt={i: len({r["output_sha256"] for r in scored if r["index"] == i})
                                               for i in range(4)})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:30000")
    p.add_argument("--samples-per-prompt", type=int, default=24)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
