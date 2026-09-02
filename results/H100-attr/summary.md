# customer Attribute-Extraction Benchmark — H100, TP=1

```
date: 2026-08-24T19:57:09+00:00
vllm: 0.20.2rc1.dev49+g9b4e83934
gpu: NVIDIA H100 80GB HBM3, 81559 MiB
dataset: /vllm-workspace/benchmarks/customer-eng/customer/scripts-customer/data/customer_attr_unique.jsonl (1000 rows)
output_tokens: 512  max_model_len: 8192  TP: 1
```

customer-reported baselines: 26B = 2.2 QPS, 31B = 1.0 QPS. P0 target = 2x (4.4 / 2.0).
QPS here is successful requests / wall time at fixed concurrency. customer's measurement method is unconfirmed; compare shapes, not just numbers, until that is resolved.

## Gemma-4 26B-A4B (MoE) FP8-Dynamic

| Config | C | Reqs | Err | **QPS** | vs target | Out tok/s | Prompt tok/s | TTFT p50 | TTFT p95 | TTFT p99 | ITL p50 | ITL p95 | Lat p50 (s) | Lat p99 (s) |
|--------|--:|-----:|----:|--------:|----------:|----------:|-------------:|---------:|---------:|---------:|--------:|--------:|------------:|------------:|
| baseline (no prefix cache) | c1 | 24 | 0 | **0.39** | 9% | 198 | 1207 | 206 | 232 | 261 | 4.7 | 4.9 | 2.6 | 2.7 |
|  | c16 | 192 | 4 | **2.11** | 48% | 1081 | 7009 | 1708 | 2877 | 3260 | 9.3 | 9.9 | 7.5 | 8.6 |
|  | c64 | 317 | 6 | **3.14** | 71% | 1610 | 10393 | 2541 | 8147 | 10259 | 17.4 | 18.5 | 20.2 | 28.0 |
| prefix cache + fp8 KV | c1 | 23 | 0 | **0.38** | 9% | 195 | 1180 | 197 | 224 | 249 | 4.8 | 5.0 | 2.6 | 2.7 |
|  | c16 | 208 | 4 | **2.24** | 51% | 1148 | 7455 | 1232 | 2490 | 2748 | 9.1 | 9.7 | 7.2 | 8.9 |
|  | c64 | 308 | 6 | **3.04** | 69% | 1556 | 10069 | 2259 | 7714 | 9879 | 19.4 | 20.8 | 20.8 | 28.5 |

## Gemma-4 31B (Dense) FP8-Dynamic

| Config | C | Reqs | Err | **QPS** | vs target | Out tok/s | Prompt tok/s | TTFT p50 | TTFT p95 | TTFT p99 | ITL p50 | ITL p95 | Lat p50 (s) | Lat p99 (s) |
|--------|--:|-----:|----:|--------:|----------:|----------:|-------------:|---------:|---------:|---------:|--------:|--------:|------------:|------------:|
| baseline (no prefix cache) | c1 | 7 | 0 | **0.12** | 6% | 59 | 372 | 653 | 846 | 887 | 15.6 | 16.0 | 8.6 | 9.0 |
|  | c16 | 73 | 2 | **0.68** | 34% | 350 | 2257 | 4241 | 7773 | 8715 | 23.9 | 25.0 | 21.7 | 26.1 |
|  | c64 | 127 | 2 | **0.88** | 44% | 448 | 2891 | 42668 | 49802 | 60888 | 31.6 | 32.7 | 72.5 | 95.9 |
| prefix cache + fp8 KV | c1 | 7 | 0 | **0.12** | 6% | 59 | 370 | 629 | 831 | 874 | 15.7 | 16.2 | 8.7 | 9.0 |
|  | c16 | 64 | 0 | **0.70** | 35% | 360 | 2267 | 3990 | 7349 | 8428 | 29.2 | 31.2 | 23.0 | 26.5 |
|  | c64 | 125 | 2 | **1.06** | 53% | 544 | 3500 | 7486 | 31432 | 35078 | 44.6 | 46.2 | 59.0 | 91.0 |

