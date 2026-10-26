#!/usr/bin/env bash
# set_power_parity.sh - set the SAME power limit on every NVIDIA GPU, or refuse.
#
# Under tensor parallelism the GPUs work in lock-step: each layer waits for the slowest
# card. A lower cap on one card throttles that card first and the whole pair runs at its
# pace, so caps are always set together and always equal.
#
# If a daemon owns your power limits (LACT, a vendor tool, a systemd unit), change the
# limit through that daemon instead: a direct nvidia-smi write can be silently reverted
# or can fight the daemon. This script is for machines where nvidia-smi is the owner.
#
# Usage: sudo ./set_power_parity.sh 250
set -euo pipefail
ALLOWED="${ALLOWED:-180 200 230 250}"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -ne 1 ]]; then
  sed -n '2,12p' "$0"; echo "allowed watts: $ALLOWED (override with ALLOWED=...)"; exit 0
fi
W="$1"
[[ " $ALLOWED " == *" $W "* ]] || { echo "refusing: $W W not in {$ALLOWED}" >&2; exit 2; }
N=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
for ((i = 0; i < N; i++)); do nvidia-smi -i "$i" -pl "$W" >/dev/null; done
mapfile -t CAPS < <(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
for c in "${CAPS[@]}"; do
  [[ "$c" == "$W" ]] || { echo "FAIL: caps now ${CAPS[*]} (expected all $W)" >&2; exit 1; }
done
echo "ok: ${N} GPUs at ${W} W"
