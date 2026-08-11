#!/usr/bin/env bash
# Run xmx-dequant-gemm under the campaign B70 GPU lock. Never run the binary
# directly while campaign jobs may be using the GPU.
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
LX_ROOT=/home/frosty40/turbo/lx
set +u
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u
# shellcheck disable=SC1091
source "$LX_ROOT/scripts/lib-gpu-lock.sh"
LX_GPU_LOCK_WAIT=3600 lx_gpu_lock_enter "xmx-microbench" || exit $?
trap 'lx_gpu_lock_leave' EXIT

ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0 SYCL_CACHE_PERSISTENT=1 \
  timeout 120 "$DIR/xmx-dequant-gemm" "$@"
rc=$?
echo "rc=$rc"
exit $rc
