#!/usr/bin/env bash
# verify_power_parity.sh - check that all GPUs carry an equal power limit from the
# allowed set. Exit 0 = PASS, 1 = FAIL. Suitable for a boot-time systemd timer.
#
# Usage: ./verify_power_parity.sh
set -uo pipefail
ALLOWED_RE="${ALLOWED_RE:-^(180|200|230|250)$}"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then sed -n '2,5p' "$0"; exit 0; fi
mapfile -t CAPS < <(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
[[ ${#CAPS[@]} -gt 0 ]] || { echo "FAIL: no GPUs visible"; exit 1; }
FIRST="${CAPS[0]}"
for c in "${CAPS[@]}"; do
  [[ "$c" == "$FIRST" ]] || { echo "FAIL: caps not equal: ${CAPS[*]} W"; exit 1; }
done
[[ "$FIRST" =~ $ALLOWED_RE ]] || { echo "FAIL: ${FIRST} W not in allowed set"; exit 1; }
echo "PASS: ${#CAPS[@]} GPUs at ${FIRST} W"
