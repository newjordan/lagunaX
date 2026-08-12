# shellcheck shell=bash
# B70 exclusive lock — source from harness scripts, don't execute.
#
# Concurrent Level-Zero clients on this Arc Pro B70 (xe) regularly wedge in
# xe_validation_lock / device-lost. One owner at a time. No exceptions for
# "just a quick PPL while bench runs".
#
# Usage (in a script after sourcing env.sh):
#   # shellcheck disable=SC1091
#   source "$ROOT/scripts/lib-gpu-lock.sh"
#   lx_gpu_lock_enter "bench-serial"
#   trap 'lx_gpu_lock_leave' EXIT
#
# Env:
#   LX_GPU_LOCK       lock file path (default: $LX_ROOT/results/.b70-gpu.lock)
#   LX_GPU_LOCK_WAIT  seconds to wait for lock (default 0 = fail immediately)
#   LX_GPU_LOCK_SKIP  set to 1 to disable (emergency only; log a warning)
#   LX_GPU_ALLOW_BUSY set to 1 to skip foreign-process check (still takes flock)

# Idempotent: second source is fine.
if [[ -n "${_LX_GPU_LOCK_LOADED:-}" ]]; then
  return 0 2>/dev/null || true
fi
_LX_GPU_LOCK_LOADED=1

: "${LX_ROOT:=/home/frosty40/turbo/lx}"
: "${LX_GPU_LOCK:=$LX_ROOT/results/.b70-gpu.lock}"
: "${LX_GPU_LOCK_META:=${LX_GPU_LOCK}.meta}"
: "${LX_GPU_LOCK_WAIT:=0}"
: "${LX_GPU_LOCK_FD:=9}"

_lx_gpu_log() { echo "[lx-gpu-lock] $*" >&2; }

# GPU-capable llama jobs we refuse to share the card with.
# Match by process *name* (pgrep -x) so scanners/awk patterns never false-positive.
# CPU-only embedding server (-ngl 0 / --embedding) is intentionally not matched.
lx_gpu_foreign_procs() {
  local me=$$ line pid rest
  # exact binary names only
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    pid="${line%% *}"
    rest="${line#"$pid"}"
    rest="${rest# }"
    [[ "$pid" == "$me" ]] && continue
    # drop CPU-only servers
    if [[ "$rest" == *llama-server* ]] || [[ "$(ps -p "$pid" -o comm= 2>/dev/null)" == "llama-server" ]]; then
      if [[ "$rest" == *"--embedding"* ]] || [[ "$rest" == *"-ngl 0"* ]] || [[ "$rest" == *"-ngl=0"* ]]; then
        continue
      fi
    fi
    printf '%s\n' "$line"
  done < <(
    pgrep -a -x llama-bench 2>/dev/null || true
    # Linux truncates /proc/<pid>/comm to 15 bytes, so llama-perplexity is
    # normally exposed as llama-perplexit. Recognize both representations.
    pgrep -a -x llama-perplexity 2>/dev/null || true
    pgrep -a -x llama-perplexit 2>/dev/null || true
    pgrep -a -x llama-cli 2>/dev/null || true
    pgrep -a -x llama-server 2>/dev/null || true
  )
}

lx_gpu_lock_holder() {
  if [[ -f "$LX_GPU_LOCK_META" ]]; then
    cat "$LX_GPU_LOCK_META" 2>/dev/null || true
  else
    echo "(no meta)"
  fi
}

lx_gpu_lock_status() {
  echo "lock_file=$LX_GPU_LOCK"
  if [[ -f "$LX_GPU_LOCK_META" ]]; then
    echo "meta:"
    sed 's/^/  /' "$LX_GPU_LOCK_META" 2>/dev/null || true
  else
    echo "meta: (none)"
  fi
  local foreign
  foreign="$(lx_gpu_foreign_procs || true)"
  if [[ -n "$foreign" ]]; then
    echo "foreign_gpu_procs:"
    echo "$foreign" | sed 's/^/  /'
  else
    echo "foreign_gpu_procs: (none)"
  fi
}

# Write meta while holding the lock.
_lx_gpu_write_meta() {
  local reason="${1:-unknown}"
  mkdir -p "$(dirname "$LX_GPU_LOCK")"
  cat >"$LX_GPU_LOCK_META" <<EOF
pid=$$
ppid=$PPID
user=${USER:-unknown}
host=$(hostname 2>/dev/null || echo unknown)
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
reason=$reason
cmd=$0 $*
EOF
}

_lx_gpu_clear_meta() {
  # Only clear if we own it
  if [[ -f "$LX_GPU_LOCK_META" ]]; then
    local owner
    owner="$(awk -F= '/^pid=/{print $2; exit}' "$LX_GPU_LOCK_META" 2>/dev/null || true)"
    if [[ "$owner" == "$$" ]]; then
      rm -f "$LX_GPU_LOCK_META"
    fi
  fi
}

# Acquire exclusive lock. Sets _LX_GPU_LOCK_HELD=1 on success.
lx_gpu_lock_enter() {
  local reason="${1:-lx-job}"
  shift || true

  if [[ "${LX_GPU_LOCK_SKIP:-0}" == "1" ]]; then
    _lx_gpu_log "WARNING: LX_GPU_LOCK_SKIP=1 — exclusive lock disabled (wedge risk)"
    return 0
  fi

  if [[ "${_LX_GPU_LOCK_HELD:-0}" == "1" ]]; then
    _lx_gpu_log "already held by this shell (reason was re-enter: $reason)"
    return 0
  fi

  if [[ "${LX_GPU_ALLOW_BUSY:-0}" != "1" ]]; then
    local foreign
    foreign="$(lx_gpu_foreign_procs || true)"
    if [[ -n "$foreign" ]]; then
      _lx_gpu_log "REFUSE: other GPU llama process(es) already running:"
      echo "$foreign" | sed 's/^/  /' >&2
      _lx_gpu_log "Kill them or wait. Concurrent B70 Level-Zero clients wedge the xe driver."
      _lx_gpu_log "holder meta: $(lx_gpu_lock_holder | tr '\n' ' ')"
      return 75
    fi
  fi

  mkdir -p "$(dirname "$LX_GPU_LOCK")"
  # Open FD and flock. Use eval so FD number is configurable.
  eval "exec ${LX_GPU_LOCK_FD}>\"\$LX_GPU_LOCK\""

  local wait_s="${LX_GPU_LOCK_WAIT:-0}"
  if [[ "$wait_s" =~ ^[0-9]+$ ]] && (( wait_s > 0 )); then
    _lx_gpu_log "waiting up to ${wait_s}s for exclusive B70 lock ($reason)..."
    if ! eval "flock -w \"$wait_s\" ${LX_GPU_LOCK_FD}"; then
      _lx_gpu_log "TIMEOUT waiting for lock. holder:"
      lx_gpu_lock_holder | sed 's/^/  /' >&2
      eval "exec ${LX_GPU_LOCK_FD}>&-"
      return 75
    fi
  else
    if ! eval "flock -n ${LX_GPU_LOCK_FD}"; then
      _lx_gpu_log "BUSY: another job holds the B70 exclusive lock."
      _lx_gpu_log "holder:"
      lx_gpu_lock_holder | sed 's/^/  /' >&2
      _lx_gpu_log "Set LX_GPU_LOCK_WAIT=<secs> to queue, or wait for the holder to finish."
      eval "exec ${LX_GPU_LOCK_FD}>&-"
      return 75
    fi
  fi

  # Re-check foreign procs after acquire (race with non-locked clients)
  if [[ "${LX_GPU_ALLOW_BUSY:-0}" != "1" ]]; then
    local foreign2
    foreign2="$(lx_gpu_foreign_procs || true)"
    if [[ -n "$foreign2" ]]; then
      _lx_gpu_log "REFUSE after acquire: foreign GPU procs appeared:"
      echo "$foreign2" | sed 's/^/  /' >&2
      eval "flock -u ${LX_GPU_LOCK_FD}; exec ${LX_GPU_LOCK_FD}>&-"
      return 75
    fi
  fi

  _lx_gpu_write_meta "$reason" "$@"
  _LX_GPU_LOCK_HELD=1
  _lx_gpu_log "ACQUIRED pid=$$ reason=$reason"
  return 0
}

lx_gpu_lock_leave() {
  if [[ "${_LX_GPU_LOCK_HELD:-0}" != "1" ]]; then
    return 0
  fi
  _lx_gpu_clear_meta
  eval "flock -u ${LX_GPU_LOCK_FD}" 2>/dev/null || true
  eval "exec ${LX_GPU_LOCK_FD}>&-" 2>/dev/null || true
  _LX_GPU_LOCK_HELD=0
  _lx_gpu_log "RELEASED pid=$$"
}
