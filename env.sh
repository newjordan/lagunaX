#!/usr/bin/env bash
# lx — serial Laguna B70 env. Source, don't execute.
# Serial track only. Multi-slot knobs from absolute-limit are not defaults here.

set -euo pipefail

__lx_oldopts="$(set +o)"
set +eu
# oneAPI (needed for sycl-ls / runtime)
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
eval "$__lx_oldopts"; unset __lx_oldopts
[ -n "${DNNLROOT:-}" ] && [ -d "$DNNLROOT/include" ] && export CPATH="$DNNLROOT/include:${CPATH:-}"

export LX_ROOT="${LX_ROOT:-/home/frosty40/turbo/lx}"

# Solo-fast binary — quality-safe champion (decode-only mm-add patch on tip lib)
# Unpatched tip: treebeard-base-control-latest/build-base-control/bin
export LX_BIN="${LX_BIN:-/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-mmadd-decode/bin}"
export LX_LLAMA_BENCH="${LX_LLAMA_BENCH:-$LX_BIN/llama-bench}"
export LX_LLAMA_CLI="${LX_LLAMA_CLI:-$LX_BIN/llama-cli}"
export LX_LLAMA_SERVER="${LX_LLAMA_SERVER:-$LX_BIN/llama-server}"

# Weights — same GGUF family as multi-slot campaign, but claims stay serial
export LX_MODEL="${LX_MODEL:-/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf}"

# Device — B70 only
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
# Exclusive lock (see scripts/lib-gpu-lock.sh). Concurrent Level-Zero clients
# wedge the xe driver (xe_validation_lock / device-lost). Harness scripts take
# the lock automatically. For ad-hoc llama-bench/ppl:
#   ./scripts/with-gpu-lock --reason ppl -- "$LX_BIN/llama-perplexity" ...
export LX_GPU_LOCK="${LX_GPU_LOCK:-$LX_ROOT/results/.b70-gpu.lock}"
export LX_GPU_LOCK_WAIT="${LX_GPU_LOCK_WAIT:-0}"

# Serial ship flags (single stream). Not multi-slot ship.
export NGL="${NGL:-99}"
export LX_SPLIT_MODE="${LX_SPLIT_MODE:-layer}"
export LX_MAIN_GPU="${LX_MAIN_GPU:-0}"
export LX_TENSOR_SPLIT="${LX_TENSOR_SPLIT:-0}"
export LX_DEVICE="${LX_DEVICE:-auto}"
export THREADS="${THREADS:-16}"
export LX_THREADS_BATCH="${LX_THREADS_BATCH:-$THREADS}"
# Keep KV cache device offload enabled by default; set 1 only for controlled host-KV trials.
export LX_NO_KV_OFFLOAD="${LX_NO_KV_OFFLOAD:-0}"
export LX_CPU_MOE_LAYERS="${LX_CPU_MOE_LAYERS:-0}"
export LX_NO_OP_OFFLOAD="${LX_NO_OP_OFFLOAD:-0}"
export LX_NO_HOST="${LX_NO_HOST:-0}"
# Preserve warmed measurements by default; set to 1 only for cold-start trials.
export LX_NO_WARMUP="${LX_NO_WARMUP:-0}"
export LX_CPU_MASK="${LX_CPU_MASK:-0x0}"
export LX_CPU_STRICT="${LX_CPU_STRICT:-0}"
export UBATCH="${UBATCH:-2048}"
export BBATCH="${BBATCH:-2048}"
export CTX="${CTX:-8192}"
# KV: f16 for max short-ctx decode (mlx.fast window is short)
export CTK="${CTK:-f16}"
export CTV="${CTV:-f16}"
# Attention dispatch: -1=auto, 0=disabled, 1=enabled; override for controlled A/Bs.
export FA="${FA:--1}"

# SYCL knobs — start from control-stable solo path
# DNN dispatch remains backend-default unless explicitly selected for an A/B.
# Empty preserves that default; 0/1 forces DNN enabled/disabled respectively.
export LX_SYCL_DISABLE_DNN="${LX_SYCL_DISABLE_DNN:-}"
if [[ -n "$LX_SYCL_DISABLE_DNN" ]]; then
  export GGML_SYCL_DISABLE_DNN="$LX_SYCL_DISABLE_DNN"
else
  unset GGML_SYCL_DISABLE_DNN 2>/dev/null || true
fi
unset GGML_SYCL_ENABLE_MOE_PIPELINE 2>/dev/null || true
unset GGML_SYCL_ENABLE_MOE_DOWN_GROUPED 2>/dev/null || true
# Graph capture is disabled by the compatibility default; expose it explicitly
# so interleaved trials can select the SYCL graph path without editing env.sh.
export LX_SYCL_DISABLE_GRAPH="${LX_SYCL_DISABLE_GRAPH:-1}"
export GGML_SYCL_DISABLE_GRAPH="$LX_SYCL_DISABLE_GRAPH"

# Quality-safe tip (2026-07-31 champion): dual_down + dual_multitoken stay killed
# (neg logprob / multitoken MoE). MUL_MAT_ADD reclaimed **decode-only** via
# patched tip lib (scripts/patch-mmadd-decode-only.py) — any-batch prefill
# still wrecks wikitext PPL. See notes/SHIP_20260731_mmadd_decode_only.md
export GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE="${GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE:-0}"
export GGML_SYCL_DISABLE_MOE_DUAL_DOWN="${GGML_SYCL_DISABLE_MOE_DUAL_DOWN:-1}"
export GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN="${GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN:-1}"
# Experimental QKV shared quant: never default ON (device-lost probe)
export GGML_SYCL_DISABLE_QKV_SHARED_QUANT="${GGML_SYCL_DISABLE_QKV_SHARED_QUANT:-1}"

export LD_LIBRARY_PATH="${LX_BIN}:${LD_LIBRARY_PATH:-}"

# Frozen window (mlx.fast-shaped)
export LX_PP="${LX_PP:-512}"
export LX_TG="${LX_TG:-128}"
export LX_REPS="${LX_REPS:-5}"
# Recurrent-state depth for architectures that expose llama-bench's -d control.
export LX_DEPTH="${LX_DEPTH:-0}"
# Run pp/tg in one llama-bench process when selected, avoiding a second model load.
export LX_COMBINED_BENCH="${LX_COMBINED_BENCH:-0}"
# llama-bench process/thread priority: -1 low, 0 normal, 1 medium, 2 high, 3 realtime.
export LX_PRIO="${LX_PRIO:-0}"
# Host model-loading and worker polling controls for reproducible A/B trials.
export LX_LOAD_MODE="${LX_LOAD_MODE:-mmap}"
export LX_POLL="${LX_POLL:-50}"
export LX_DELAY="${LX_DELAY:-0}"
# Optional automatic device-memory fitting. Zero keeps llama-bench fitting off.
export LX_FIT_TARGET_MIB="${LX_FIT_TARGET_MIB:-0}"
export LX_FIT_CTX="${LX_FIT_CTX:-4096}"
# Host NUMA placement: disabled (empty), distribute, isolate, or numactl.
export LX_NUMA="${LX_NUMA:-}"

export LX_BASELINE_JSON="${LX_BASELINE_JSON:-$LX_ROOT/baseline/baseline.json}"
export LX_RESULTS="${LX_RESULTS:-$LX_ROOT/results}"
export LX_GOLDEN="${LX_GOLDEN:-$LX_ROOT/correctness/golden.json}"

mkdir -p "$LX_ROOT/baseline" "$LX_RESULTS" "$LX_ROOT/correctness"

# loop-session candidate: force oneDNN off (FRONTIER_20260801_onednn_cache_hit_tax / onednn_postop_epilogue)
export GGML_SYCL_DISABLE_DNN=1
export LX_SYCL_DISABLE_DNN=1
