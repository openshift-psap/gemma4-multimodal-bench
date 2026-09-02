#!/bin/bash
# Generate results/H100-customer/summary.md with QPS first and the P0 target check.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results/H100-attr}"
OUTFILE="${RESULTS_ROOT}/summary.md"

python3 - "$RESULTS_ROOT" "$OUTFILE" <<'PY'
import json, os, sys
root, outfile = sys.argv[1], sys.argv[2]

models = [
    ('26b-fp8', 'Gemma-4 26B-A4B (MoE) FP8-Dynamic', 2.2, 4.4),
    ('31b-fp8', 'Gemma-4 31B (Dense) FP8-Dynamic',   1.0, 2.0),
]
configs = [('baseline', 'baseline (no prefix cache)'), ('pc-fp8kv', 'prefix cache + fp8 KV')]
concs = [1, 16, 64]

L = ['# Attribute-Extraction Benchmark — H100, TP=1', '']
ctx = os.path.join(root, 'run-context.txt')
if os.path.exists(ctx):
    L += ['```', open(ctx).read().rstrip(), '```', '']
L += ['Customer-reported baselines: 26B = 2.2 QPS, 31B = 1.0 QPS. P0 target = 2x (4.4 / 2.0).',
      'QPS here is successful requests / wall time at fixed concurrency. customer\'s measurement method is unconfirmed; compare shapes, not just numbers, until that is resolved.', '']

for mdir, mname, wm_base, target in models:
    if not os.path.isdir(os.path.join(root, mdir)):
        continue
    L += [f'## {mname}', '',
          '| Config | C | Reqs | Err | **QPS** | vs target | Out tok/s | Prompt tok/s | TTFT p50 | TTFT p95 | TTFT p99 | ITL p50 | ITL p95 | Lat p50 (s) | Lat p99 (s) |',
          '|--------|--:|-----:|----:|--------:|----------:|----------:|-------------:|---------:|---------:|---------:|--------:|--------:|------------:|------------:|']
    for cdir, cname in configs:
        first = True
        for c in concs:
            p = os.path.join(root, mdir, cdir, f'c{c}.json')
            if not os.path.exists(p):
                continue
            b = json.load(open(p))['benchmarks'][0]
            m = b['metrics']; ok = b['requests']['successful']; err = b['requests'].get('errored', [])
            qps = m['requests_per_second']['successful']['mean']
            t = m['time_to_first_token_ms']['successful']['percentiles']
            i = m['inter_token_latency_ms']['successful']['percentiles']
            lat = m['request_latency_ms']['successful']['percentiles']
            hit = '✅' if qps >= target else f'{qps/target*100:.0f}%'
            L.append(f"| {cname if first else ''} | c{c} | {len(ok)} | {len(err)} | **{qps:.2f}** | {hit} | "
                     f"{m['output_tokens_per_second']['successful']['mean']:.0f} | {m['prompt_tokens_per_second']['successful']['mean']:.0f} | "
                     f"{t['p50']:.0f} | {t['p95']:.0f} | {t['p99']:.0f} | {i['p50']:.1f} | {i['p95']:.1f} | "
                     f"{lat['p50']/1000:.1f} | {lat['p99']/1000:.1f} |")
            first = False
    L.append('')

open(outfile, 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
print(f'\nSummary written to {outfile}')
PY
