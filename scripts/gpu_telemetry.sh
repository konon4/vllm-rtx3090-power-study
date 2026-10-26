#!/usr/bin/env bash
# gpu_telemetry.sh - log power draw, power limit, temperature, clocks and throttle
# reasons for every GPU once per second to CSV. Stop with Ctrl-C.
#
# A 1 Hz log cannot see sub-second transients. For PSU stability the real detector is
# whether the machine survives the load.
#
# Usage: ./gpu_telemetry.sh power-200w.csv [interval_ms]
set -euo pipefail
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  sed -n '2,8p' "$0"; exit 0
fi
OUT="$1"; MS="${2:-1000}"
exec nvidia-smi \
  --query-gpu=timestamp,index,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,memory.used,clocks_throttle_reasons.active \
  --format=csv -lms "$MS" > "$OUT"
