#!/usr/bin/env bash
# serve_vllm_tp2.sh - launch vLLM with tensor parallelism across two GPUs, in the
# configuration the power study ran against.
#
# Recorded in the test report: TP=2, max_model_len 262144, chunked prefill with
# 4096-token chunks, automatic prefix caching on. The other flags (fp8 KV cache,
# gpu-memory-utilization 0.92, max-num-seqs 4) follow the 256K-context serving profile
# used on the same machine and are not recorded in the report itself.
# The port is published on all interfaces; put it behind a firewall or bind it to
# loopback (-p <loopback-ip>:PORT:8000) on a shared network.
#
# Usage:
#   MODEL_DIR=/path/to/models ./serve_vllm_tp2.sh
#   IMAGE=vllm/vllm-openai:<tag> PORT=8000 ./serve_vllm_tp2.sh
set -euo pipefail

IMAGE="${IMAGE:-vllm/vllm-openai:cu129-nightly}"
MODEL="${MODEL:-QuantTrio/Qwen3.6-35B-A3B-AWQ}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"          # mounted as the Hugging Face cache
PORT="${PORT:-8000}"
NAME="${NAME:-vllm-tp2}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,16p' "$0"; exit 0
fi

exec docker run --rm --name "$NAME" --gpus all --ipc=host \
  -p "${PORT}:8000" \
  -v "${MODEL_DIR}:/root/.cache/huggingface" \
  "$IMAGE" \
  --model "$MODEL" \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --enable-chunked-prefill --max-num-batched-tokens 4096 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 4
