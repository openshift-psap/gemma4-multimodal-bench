#!/bin/bash
# Shared functions for customer multimodal benchmarks.
# Adapted from scripts-synthetic-H100/common.sh (HDCharles). Differences:
#   - load generator is mm_bench.py (replays customer JSONL with images) instead of guidellm
#   - start_server takes an EXTRA_ARGS string so the same function serves baseline and tuned configs
#   - extract_metrics reports QPS first, since that is the success criterion

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results/H100-attr}"
MM_BENCH="${SCRIPT_DIR}/mm_bench.py"

run_bench() {
  local label="$1" streams="$2" seconds="$3" outfile="$4" port="$5" dataset="$6" output_tokens="$7"
  echo "  Running: ${label} (c=${streams}, ${seconds}s)"
  python3 "$MM_BENCH" \
    --dataset "$dataset" --port "$port" \
    --concurrency "$streams" --duration "$seconds" \
    --max-tokens "$output_tokens" --output "$outfile"
  echo ""
}

extract_metrics() {
  local jsonfile="$1"
  python3 -c "
import json
d = json.load(open('${jsonfile}'))
b = d['benchmarks'][0]
m = b['metrics']
ok = b['requests']['successful']
err = b['requests'].get('errored', [])
qps = m['requests_per_second']['successful']['mean']
otps = m['output_tokens_per_second']['successful']['mean']
ptps = m['prompt_tokens_per_second']['successful']['mean']
t = m['time_to_first_token_ms']['successful']['percentiles']
dur = b['end_time'] - b['start_time']
print(f'  n={len(ok)} err={len(err)} dur={dur:.0f}s QPS={qps:.3f} out={otps:.0f}tok/s prompt={ptps:.0f}tok/s TTFT p50={t[\"p50\"]:.0f}ms p95={t[\"p95\"]:.0f}ms p99={t[\"p99\"]:.0f}ms')
"
}

warmup() {
  local port="$1" dataset="$2" output_tokens="$3"
  echo "=== Warmup 1: serial, 5 requests (compile / CUDA graph capture) ==="
  python3 "$MM_BENCH" --dataset "$dataset" --port "$port" \
    --concurrency 1 --duration 300 --max-requests 5 --max-tokens "$output_tokens"
  echo "=== Warmup 2: concurrent c=64, 60s ==="
  python3 "$MM_BENCH" --dataset "$dataset" --port "$port" \
    --concurrency 64 --duration 60 --max-tokens "$output_tokens"
  echo "Warmup complete."
}

# start_server MODEL PORT TP LOGFILE [EXTRA_ARGS...]
# Baseline = no extra args beyond what customer's spec implies (prefix caching OFF).
# Tuned    = pass "--enable-prefix-caching --kv-cache-dtype fp8" etc.
start_server() {
  local model="$1" port="$2" tp="$3" logfile="$4"
  shift 4
  local extra=("$@")

  echo "Starting server for ${model} (TP=${tp}) extra=[${extra[*]}]..."
  # --limit-mm-per-prompt image=32 : spec says up to 30+ images; dataset currently has 1.
  # --max-model-len 8192          : 4000 in + 512 out + headroom. Do NOT over-provision; it costs KV.
  vllm serve "$model" \
    --port "$port" \
    --trust-remote-code \
    --tensor-parallel-size "$tp" \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image": 32}' \
    --no-enable-log-requests \
    "${extra[@]}" \
    > "$logfile" 2>&1 &
  VLLM_PID=$!

  ELAPSED=0
  while [[ $ELAPSED -lt 900 ]]; do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM died"; tail -30 "$logfile"; exit 1; }
    curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1 && break
    sleep 10; ELAPSED=$((ELAPSED + 10))
  done
  echo "Server ready (${ELAPSED}s)."
  # Record the KV budget the server actually got — this is the number that explains most throughput ceilings.
  grep -E "GPU KV cache size|Maximum concurrency" "$logfile" | tail -2 || true
}

stop_server() {
  pkill -P $VLLM_PID 2>/dev/null
  kill $VLLM_PID 2>/dev/null
  wait $VLLM_PID 2>/dev/null || true
  pkill -9 -f "vllm serve" 2>/dev/null
  pkill -9 -f EngineCore 2>/dev/null
  sleep 15
  local used
  used=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader | head -1)
  [[ -n "$used" ]] && { echo "WARNING: GPU still occupied after cleanup: $used"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; }
  echo "Server stopped."
}
