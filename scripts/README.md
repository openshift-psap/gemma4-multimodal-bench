# customer multimodal benchmarks (H100, TP=1)

Adapted from `scripts-synthetic-H100/` (HDCharles). Same flow — start server, warm up,
sweep concurrency, extract, summarize — but replays customer's attribute-extraction
JSONL (system prompt + 1 image + user prompt) instead of GuideLLM synthetic text.

## Run

```bash
# 0. deps (inside the vllm venv)
pip install aiohttp pillow

# 1. convert customer's file; prints system-prompt length, image sizes, etc.
mkdir -p data
python3 prepare_dataset.py /path/to/red_hat_sample.jsonl data/customer_attr.jsonl \
    --size 480 --repeat 1000 --tokenizer google/gemma-4-31B-it

# 2. run everything (2 models x 2 configs x c=1/16/64, ~1.5-2h incl. model loads)
./run-customer.sh
#   env overrides: CONCURRENCIES="1 8 16 32 64" DATASET=... ./run-customer.sh

# 3. summary lands in results/H100-customer/summary.md (also re-runnable standalone)
./generate-summary.sh
```

## What each file does
- `prepare_dataset.py` — customer JSONL → OpenAI chat JSONL with 480x480 JPEG data URIs. Prints dataset stats.
- `mm_bench.py` — fixed-concurrency streaming load generator; writes GuideLLM-shaped JSON (+ `requests_per_second`).
- `common.sh` — `start_server` (takes extra vLLM flags), `warmup`, `run_bench`, `extract_metrics`.
- `run-customer.sh` — the matrix. Edit `MODELS` / `CONFIGS` to add variants.
- `generate-summary.sh` — markdown table, QPS first, checked against the 4.4 / 2.0 targets.

## Things to check on the first real run
- `usage.prompt_tokens` in c1.json: tells you how many tokens Gemma-4 spends per 480x480 image
  and whether real requests are anywhere near the spec's 4000.
- `GPU KV cache size` line printed after server start: if `pc-fp8kv` shows ~2x the baseline, the
  fp8 KV flag took effect.
- `preemption log lines`: non-zero at c64 means KV pressure — lower `--max-num-seqs` or raise
  `--gpu-memory-utilization`.
- `ignore_eos: true` is set so every request produces exactly 512 tokens (matches Charles's
  fixed-output protocol). Drop it if you want natural stop behaviour.

## Known gaps (deliberate)
- 1 image/request — matches the sample file, not the doc's "5–9 images". Pending Rob.
- Content-moderation CSV not wired up yet.
- Cycling 100 rows to 1000 requests inflates prefix/image-cache hit rates vs production.
