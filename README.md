# Gemma-4 Multimodal Serving Benchmarks — Handoff

Benchmarking/characterization of two Gemma-4 candidates for a retail customer's
attribute-extraction and content-moderation workloads. Engagement scope is
**data collection**, not optimization — we measure, the customer validates
accuracy against their ground truth.

Status: first full matrix complete and presented to the customer. Three
follow-up runs queued (see "Next runs").

## Results so far (read these first)

- `results/H100-attr/summary.md` — main results table (unique dataset)
- `results/H100-attr-cycled-run1/` — earlier run with repeat-heavy
  traffic; only valid as an upper bound (full-prompt cache hits)
- `results/benchmarks.csv` — everything flattened for spreadsheets

Headline (1x H100 80GB SXM, TP=1, c=64, unique dataset):

| Model | Customer baseline* | Ours (default) | Ours (best) | Bound by |
|---|---|---|---|---|
| 26B-A4B MoE FP8 | 1.5 QPS | 3.14 | 3.14 | prefill/vision compute |
| 31B dense FP8 | 0.8 QPS | 0.88 | 1.06 (fp8 KV) | KV cache capacity |

*The customer revised baselines down from 2.2/1.0 mid-engagement. Confirm which
the 2x target tracks before quoting progress.

Key findings, short version:
1. Both models beat the customer's baselines with default vLLM settings.
2. 26B: no serving flag moves it (compute-bound). 31B: only KV capacity
   moves it (fp8 KV = +21%, TTFT p50 43s -> 7.5s).
3. The customer's prompt puts ~2K tokens of static rules AFTER the image, so
   they can never prefix-cache. Reordering them into the system prompt is
   the biggest untested 26B lever. Needs customer accuracy validation.
4. Repeat-heavy traffic doubles 26B throughput (6.14 vs 3.04). The customer
   says images repeat 60-70% but items usually change — realistic mix
   sits between our two datasets.
5. Per-image token cost: ~100-300 tok on this vLLM build. The customer quoted
   ~1700 (possibly Gemini tokenizer). Unresolved — affects all ISL math.

## Environment

- Cluster / namespace / pod: see INTERNAL.md (not committed) — the
  namespace name references the customer, so it stays out of the repo
- Image: `vllm/vllm-openai:gemma4-0505-cu129`
- PVCs: `model-pvc` (models at `/models`), `workspace-pvc`
  (`/workspace/datasets`, `/workspace/results`)
- Models on PVC: `/models/gemma-4-26B-A4B-it-FP8-Dynamic`,
  `/models/gemma-4-31B-it-FP8-Dynamic`
  (note: criteria doc's `google/gemma-4-31b-it-fp8-dynamic` repo name is
  wrong; correct is `RedHatAI/gemma-4-31B-it-FP8-Dynamic`)
- Dataset: customer-provided zip on the workspace PVC — NOT in this
  repo (client data, includes sensitive moderation images). New team
  members get it from the PVC or from the engagement lead.

## Scripts (`scripts/`)

| File | What it does |
|---|---|
| `prepare_dataset.py` | customer JSONL -> OpenAI-chat JSONL, 480x480 data URIs. `--repeat N` cycles; `--unique` perturbs image bytes + text per copy so only the system prompt is cache-shareable (use this for production-like runs) |
| `mm_bench.py` | Async load generator: fixed concurrency, streaming, hits /v1/chat/completions, writes GuideLLM-shaped JSON incl. QPS |
| `common.sh` | start_server / warmup / run_bench / extract_metrics. Server flags live here (max-model-len 8192, limit-mm-per-prompt 32) |
| `run-matrix.sh` | The matrix: MODELS x CONFIGS x concurrency. Edit the two assoc arrays to add variants |
| `generate-summary.sh` | Rebuilds summary.md from result JSONs |

Reproduce the whole matrix:
```bash
oc -n <namespace> rsh <pod>          # see INTERNAL.md
cd <scripts dir on pod>
pip install aiohttp pillow
python3 prepare_dataset.py <customer jsonl> data/attr_unique.jsonl \
    --size 480 --repeat 1000 --unique
nohup bash run-matrix.sh > /workspace/results/run.log 2>&1 &
```

Gotchas learned the hard way:
- Always verify the GPU is empty before starting a server
  (`nvidia-smi --query-compute-apps=pid --format=csv`); orphaned
  EngineCore processes survive soft kills. stop_server in common.sh
  force-kills and verifies.
- Longest prompts are ~5.6K tokens; max-model-len below 6144+512 causes
  400 rejections that silently inflate QPS (dropped requests are the
  expensive ones). Current setting 8192. A few residual errors per run
  were still under investigation — check the `err` column.
- Cycled datasets massively inflate prefix-cache numbers (full-prompt
  hits). Any caching experiment must use the `--unique` dataset.
- Baselines measured per the customer's method for comparison: c=16,
  100 requests, 5 warmup (their custom async script).

## Next runs (queued, in priority order)

1. **Reordered prompt** — move the static rules block before the image
   (into the system message). Predicted ~60-65% token cache-hit rate vs
   ~0% today. Measure QPS + hit rate before/after (grep "Prefix cache
   hit rate" from server.log). Largest untested 26B lever.
2. **Multi-image sweep** — 1/5/9 images per request, the customer's format:
   product text first, then images appended. Needs a prepare_dataset
   `--images N` flag packing perturbed variants.
3. **31B on W4A16** — `google/gemma-4-31B-it-qat-w4a16-ct`. Frees ~16GB
   weights -> KV. Deliver with accuracy-validation flag (community
   reports of QAT quality regressions; the customer validates anyway).


