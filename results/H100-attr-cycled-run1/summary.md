# customer Attribute-Extraction Benchmark — H100, TP=1

```
date: 2026-08-24T19:25:12+00:00
vllm: 0.20.2rc1.dev49+g9b4e83934
gpu: NVIDIA H100 80GB HBM3, 81559 MiB
dataset: /vllm-workspace/benchmarks/customer-eng/customer/scripts-customer/data/customer_attr.jsonl (1000 rows)
output_tokens: 512  max_model_len: 5120  TP: 1
```

customer-reported baselines: 26B = 2.2 QPS, 31B = 1.0 QPS. P0 target = 2x (4.4 / 2.0).
QPS here is successful requests / wall time at fixed concurrency. customer's measurement method is unconfirmed; compare shapes, not just numbers, until that is resolved.

## Gemma-4 26B-A4B (MoE) FP8-Dynamic

| Config | C | Reqs | Err | **QPS** | vs target | Out tok/s | Prompt tok/s | TTFT p50 | TTFT p95 | TTFT p99 | ITL p50 | ITL p95 | Lat p50 (s) | Lat p99 (s) |
|--------|--:|-----:|----:|--------:|----------:|----------:|-------------:|---------:|---------:|---------:|--------:|--------:|------------:|------------:|
| baseline (no prefix cache) | c1 | 24 | 0 | **0.39** | 9% | 198 | 1210 | 206 | 233 | 261 | 4.7 | 4.8 | 2.6 | 2.7 |
|  | c16 | 192 | 8 | **2.11** | 48% | 1078 | 6948 | 1594 | 2559 | 2943 | 9.3 | 9.9 | 7.6 | 8.6 |
|  | c64 | 313 | 10 | **3.15** | 72% | 1610 | 10332 | 1762 | 8222 | 10194 | 17.3 | 19.3 | 20.3 | 29.0 |
| prefix cache + fp8 KV | c1 | 25 | 0 | **0.40** | 9% | 207 | 1268 | 39 | 42 | 44 | 4.8 | 4.9 | 2.5 | 2.5 |
|  | c16 | 304 | 12 | **3.32** | 75% | 1696 | 10846 | 80 | 178 | 219 | 9.0 | 9.8 | 4.8 | 4.9 |
|  | c64 | 576 | 22 | **6.14** | ✅ | 3140 | 20247 | 133 | 577 | 857 | 19.3 | 26.2 | 10.4 | 11.0 |

## Gemma-4 31B (Dense) FP8-Dynamic

| Config | C | Reqs | Err | **QPS** | vs target | Out tok/s | Prompt tok/s | TTFT p50 | TTFT p95 | TTFT p99 | ITL p50 | ITL p95 | Lat p50 (s) | Lat p99 (s) |
|--------|--:|-----:|----:|--------:|----------:|----------:|-------------:|---------:|---------:|---------:|--------:|--------:|------------:|------------:|
| baseline (no prefix cache) | c1 | 7 | 0 | **0.12** | 6% | 60 | 373 | 649 | 846 | 886 | 15.6 | 16.0 | 8.6 | 9.0 |
|  | c16 | 69 | 4 | **0.68** | 34% | 348 | 2249 | 3610 | 7833 | 8715 | 23.6 | 25.0 | 21.8 | 25.8 |
|  | c64 | 128 | 2 | **0.84** | 42% | 431 | 2762 | 42255 | 51185 | 59423 | 31.5 | 32.3 | 71.9 | 93.1 |
| prefix cache + fp8 KV | c1 | 7 | 0 | **0.12** | 6% | 59 | 370 | 627 | 824 | 864 | 15.7 | 16.1 | 8.7 | 9.0 |
|  | c16 | 65 | 4 | **0.66** | 33% | 336 | 2138 | 3720 | 7123 | 7650 | 27.8 | 30.5 | 23.1 | 26.0 |
|  | c64 | 125 | 2 | **1.07** | 53% | 547 | 3497 | 7288 | 31642 | 35049 | 44.3 | 45.7 | 58.3 | 90.7 |

