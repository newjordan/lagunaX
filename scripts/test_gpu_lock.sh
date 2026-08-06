#!/usr/bin/env bash
# Structural test for exclusive B70 lock (no GPU work).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
# Use a private lock file so we don't fight a real holder during CI/self-test
export LX_GPU_LOCK="$ROOT/results/.b70-gpu.lock.test.$$"
export LX_GPU_LOCK_META="${LX_GPU_LOCK}.meta"
export LX_GPU_LOCK_WAIT=0
export LX_GPU_ALLOW_BUSY=1   # test flock only; ignore unrelated system procs
export LX_GPU_LOCK_FD=8
rm -f "$LX_GPU_LOCK" "$LX_GPU_LOCK_META"

# shellcheck disable=SC1091
source "$ROOT/scripts/lib-gpu-lock.sh"

fail() { echo "FAIL: $*" >&2; exit 1; }
ok() { echo "OK: $*"; }

# 1) acquire / release
lx_gpu_lock_enter "test-a" || fail "enter failed"
[[ -f "$LX_GPU_LOCK_META" ]] || fail "meta missing after acquire"
grep -q "reason=test-a" "$LX_GPU_LOCK_META" || fail "meta reason"
lx_gpu_lock_leave
[[ ! -f "$LX_GPU_LOCK_META" ]] || fail "meta should clear on leave"
ok "acquire/release"

# 2) second shell cannot take lock while held
lx_gpu_lock_enter "test-hold" || fail "re-enter failed"
export LX_GPU_LOCK_FD=7
# subshell with different FD trying same lock file
if bash -c '
  set -euo pipefail
  source "'"$ROOT"'/scripts/lib-gpu-lock.sh"
  export LX_GPU_LOCK="'"$LX_GPU_LOCK"'"
  export LX_GPU_LOCK_META="'"$LX_GPU_LOCK_META"'"
  export LX_GPU_LOCK_WAIT=0
  export LX_GPU_ALLOW_BUSY=1
  export LX_GPU_LOCK_FD=7
  lx_gpu_lock_enter "test-b"
'; then
  fail "second holder should have been refused"
fi
ok "busy refusal"
# restore FD for leave
export LX_GPU_LOCK_FD=8
lx_gpu_lock_leave

# 3) with-gpu-lock wrapper
out="$("$ROOT/scripts/with-gpu-lock" --reason wrap-test -- bash -c 'echo WRAP_OK')"
[[ "$out" == *WRAP_OK* ]] || fail "wrapper did not run command"
ok "with-gpu-lock wrapper"

rm -f "$LX_GPU_LOCK" "$LX_GPU_LOCK_META"
echo "ALL PASS"
