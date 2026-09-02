#!/bin/bash
# customer attribute-extraction workload, single H100, TP=1.
# 2 models (the FP8-Dynamic baselines named in the criteria doc)
# x 2 configs (baseline / prefix-caching + fp8 KV)
# x c = 1, 16, 64
#
# Prereqs:
#   python3 prepare_dataset.py red_hat_sample.jsonl data/attr_unique.jsonl --size 480
#   pip install aiohttp   (inside the vllm venv)
set -euo pipefail

# vllm is on PATH in the container image
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export FLASHINFER_DISABLE_VERSION_CHECK=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PORT=8093
TP=1
OUTPUT_TOKENS=512
DATASET="${DATASET:-${SCRIPT_DIR}/data/attr_unique.jsonl}"
CONCURRENCIES="${CONCURRENCIES:-1 16 64}"

[[ -f "$DATASET" ]] || { echo "dataset not found: $DATASET (run prepare_dataset.py first)"; exit 1; }

declare -A MODELS
MODELS[26b-fp8]="/models/gemma-4-26B-A4B-it-FP8-Dynamic"
MODELS[31b-fp8]="/models/gemma-4-31B-it-FP8-Dynamic"

declare -A CONFIGS
CONFIGS[baseline]="--no-enable-prefix-caching"
CONFIGS[pc-fp8kv]="--enable-prefix-caching --kv-cache-dtype fp8"
CONFIGS[pc-fp8kv-sched]="--enable-prefix-caching --kv-cache-dtype fp8 --gpu-memory-utilization 0.95 --max-num-batched-tokens 16384"

# Pin the run context into the results dir so nobody has to guess later.
mkdir -p "$RESULTS_ROOT"
{
  echo "date: $(date -Is)"
  echo "vllm: $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"
  echo "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
  echo "dataset: $DATASET ($(wc -l < "$DATASET") rows)"
  echo "output_tokens: $OUTPUT_TOKENS  max_model_len: 8192  TP: $TP"
} > "${RESULTS_ROOT}/run-context.txt"
cat "${RESULTS_ROOT}/run-context.txt"

for mlabel in 26b-fp8 31b-fp8; do
  model="${MODELS[$mlabel]}"
  for clabel in baseline pc-fp8kv pc-fp8kv-sched; do
    OUTDIR="${RESULTS_ROOT}/${mlabel}/${clabel}"
    mkdir -p "$OUTDIR"
    echo "========================================="
    echo "  ${mlabel}/${clabel}: ${model}"
    echo "========================================="

    # shellcheck disable=SC2086
    start_server "$model" "$PORT" "$TP" "${OUTDIR}/server.log" ${CONFIGS[$clabel]}
    warmup "$PORT" "$DATASET" "$OUTPUT_TOKENS"

    for c in $CONCURRENCIES; do
      dur=90; [[ $c -eq 1 ]] && dur=60
      run_bench "${mlabel}-${clabel}-c${c}" "$c" "$dur" "${OUTDIR}/c${c}.json" "$PORT" "$DATASET" "$OUTPUT_TOKENS"
      extract_metrics "${OUTDIR}/c${c}.json"
    done

    # Preemptions = KV pressure. Worth knowing per config.
    echo "  preemption log lines: $(grep -c -i preempt "${OUTDIR}/server.log" || true)"
    stop_server
    echo ""
  done
done

"${SCRIPT_DIR}/generate-summary.sh"
