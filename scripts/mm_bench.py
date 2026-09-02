#!/usr/bin/env python3
"""
Minimal multimodal load generator for vLLM's OpenAI-compatible server.

Replays a JSONL of {"messages": [...]} rows (see prepare_dataset.py) against
/v1/chat/completions at a fixed concurrency for a fixed duration, streaming,
and writes a JSON file shaped like GuideLLM's output so the existing
extract_metrics / summary code in this repo keeps working.

Usage:
  python3 mm_bench.py --dataset customer_attr.jsonl --port 8093 \
      --concurrency 16 --duration 90 --max-tokens 512 --output c16.json

  --max-requests N   stop after N requests (used for warmup)
  --model NAME       defaults to whatever /v1/models reports first

Requires: aiohttp  (pip install aiohttp)
"""
import argparse, asyncio, json, statistics, sys, time
import aiohttp


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(xs):
    return {
        "mean": statistics.mean(xs) if xs else 0.0,
        "percentiles": {f"p{p}": pct(xs, p) for p in (50, 90, 95, 99)},
    }


async def one_request(session, url, model, messages, max_tokens, rec):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,  # vLLM extension: forces full max_tokens so output length is fixed
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    first = None
    last = t0
    n_chunks = 0
    usage = None
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                return
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if choices and choices[0].get("delta", {}).get("content"):
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    else:
                        rec["itl"].append((now - last) * 1000)
                    last = now
                    n_chunks += 1
    except Exception as e:  # noqa
        rec["error"] = repr(e)
        return
    end = time.perf_counter()
    rec["start"], rec["end"] = t0, end
    rec["ttft_ms"] = (first - t0) * 1000 if first else None
    rec["latency_ms"] = (end - t0) * 1000
    rec["output_tokens"] = (usage or {}).get("completion_tokens", n_chunks)
    rec["prompt_tokens"] = (usage or {}).get("prompt_tokens", 0)


async def worker(q, session, url, model, max_tokens, results, deadline, stop_after):
    while True:
        if time.perf_counter() >= deadline:
            return
        if stop_after is not None and len(results) >= stop_after:
            return
        messages = await q.get()
        rec = {"itl": []}
        results.append(rec)
        await one_request(session, url, model, messages, max_tokens, rec)
        q.task_done()


async def feeder(q, rows):
    i = 0
    while True:
        await q.put(rows[i % len(rows)]["messages"])
        i += 1


async def main_async(a):
    rows = [json.loads(l) for l in open(a.dataset) if l.strip()]
    base = f"http://localhost:{a.port}/v1"
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        model = a.model
        if not model:
            async with session.get(f"{base}/models") as r:
                model = (await r.json())["data"][0]["id"]
        q = asyncio.Queue(maxsize=a.concurrency * 2)
        results = []
        start = time.perf_counter()
        deadline = start + a.duration
        feed = asyncio.create_task(feeder(q, rows))
        workers = [
            asyncio.create_task(
                worker(q, session, f"{base}/chat/completions", model, a.max_tokens, results, deadline, a.max_requests)
            )
            for _ in range(a.concurrency)
        ]
        await asyncio.gather(*workers)
        feed.cancel()
        end = time.perf_counter()

    ok = [r for r in results if "error" not in r and r.get("ttft_ms") is not None]
    bad = [r for r in results if "error" in r]
    dur = end - start
    out_tok = sum(r["output_tokens"] for r in ok)
    in_tok = sum(r["prompt_tokens"] for r in ok)
    itl_all = [x for r in ok for x in r["itl"]]
    tpot = [r["latency_ms"] / r["output_tokens"] for r in ok if r["output_tokens"]]

    benchmark = {
        "start_time": start,
        "end_time": end,
        "config": {
            "model": model, "concurrency": a.concurrency, "duration_s": a.duration,
            "max_tokens": a.max_tokens, "dataset": a.dataset, "dataset_rows": len(rows),
        },
        "requests": {
            "successful": [
                {"prompt_tokens": r["prompt_tokens"], "output_tokens": r["output_tokens"],
                 "ttft_ms": r["ttft_ms"], "latency_ms": r["latency_ms"]} for r in ok
            ],
            "errored": [{"error": r["error"]} for r in bad],
        },
        "metrics": {
            "requests_per_second": {"successful": {"mean": len(ok) / dur if dur else 0.0}},
            "output_tokens_per_second": {"successful": {"mean": out_tok / dur if dur else 0.0}},
            "prompt_tokens_per_second": {"successful": {"mean": in_tok / dur if dur else 0.0}},
            "time_to_first_token_ms": {"successful": summarize([r["ttft_ms"] for r in ok])},
            "inter_token_latency_ms": {"successful": summarize(itl_all)},
            "time_per_output_token_ms": {"successful": summarize(tpot)},
            "request_latency_ms": {"successful": summarize([r["latency_ms"] for r in ok])},
        },
    }
    if a.output:
        json.dump({"benchmarks": [benchmark]}, open(a.output, "w"))
    m = benchmark["metrics"]
    print(
        f"n={len(ok)} err={len(bad)} dur={dur:.0f}s "
        f"QPS={m['requests_per_second']['successful']['mean']:.3f} "
        f"out={m['output_tokens_per_second']['successful']['mean']:.0f}tok/s "
        f"TTFT p50={m['time_to_first_token_ms']['successful']['percentiles']['p50']:.0f}ms "
        f"p99={m['time_to_first_token_ms']['successful']['percentiles']['p99']:.0f}ms "
        f"mean_prompt_tok={in_tok / len(ok) if ok else 0:.0f}"
    )
    if bad:
        print(f"first error: {bad[0]['error']}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-requests", type=int, default=None)
    ap.add_argument("--output", default=None)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
