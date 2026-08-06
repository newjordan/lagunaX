#!/usr/bin/env bash
# Show B70 lock holder + GPU llama processes + quick device touch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib-gpu-lock.sh"

echo "=== lx B70 GPU status ==="
lx_gpu_lock_status
echo
echo "=== drm ==="
ls -la /dev/dri/ 2>/dev/null || true
if [[ -r /sys/class/drm/card0/device/vendor ]]; then
  echo "vendor=$(cat /sys/class/drm/card0/device/vendor 2>/dev/null)"
fi
echo
echo "=== tip: never run bench + ppl/server concurrently on this card ==="
echo "  ./scripts/with-gpu-lock --status"
echo "  ./scripts/with-gpu-lock --reason myjob -- ./scripts/bench-serial.sh"
