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

# Solo-fast binary (control wins solo on this model — see package-vs-vanilla scorecard)
export LX_BIN="${LX_BIN:-/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin}"
export LX_LLAMA_BENCH="${LX_LLAMA_BENCH:-$LX_BIN/llama-bench}"
export LX_LLAMA_CLI="${LX_LLAMA_CLI:-$LX_BIN/llama-cli}"
export LX_LLAMA_SERVER="${LX_LLAMA_SERVER:-$LX_BIN/llama-server}"

# Weights — same GGUF family as multi-slot campaign, but claims stay serial
export LX_MODEL="${LX_MODEL:-/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf}"

# Device — B70 only
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"

# Serial ship flags (single stream). Not multi-slot ship.
export NGL="${NGL:-99}"
export THREADS="${THREADS:-16}"
export UBATCH="${UBATCH:-2048}"
export BBATCH="${BBATCH:-2048}"
export CTX="${CTX:-8192}"
# KV: f16 for max short-ctx decode (mlx.fast window is short)
export CTK="${CTK:-f16}"
export CTV="${CTV:-f16}"

# SYCL knobs — start from control-stable solo path
# DNN OFF was ship for multi-slot package; control solo may differ — leave default
# unset so binary defaults apply unless a candidate overrides.
unset GGML_SYCL_DISABLE_DNN 2>/dev/null || true
unset GGML_SYCL_ENABLE_MOE_PIPELINE 2>/dev/null || true
unset GGML_SYCL_ENABLE_MOE_DOWN_GROUPED 2>/dev/null || true
export GGML_SYCL_DISABLE_GRAPH="${GGML_SYCL_DISABLE_GRAPH:-1}"

# Quality-safe tip (2026-07-30 PPL bisect): these three default-ON fuses wreck logprobs
# and/or multitoken MoE (PPL 1e5–1e7, dual_down neg-stddev, long-ctx abort).
# With them killed: PPL~1.0, golden OK, formal ~+20% (pp~1185 / tg~136).
# Do NOT unset for "speed" without re-proving PPL. See notes/SHIP_20260730_quality_safe_tip.md
export GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE="${GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE:-1}"
export GGML_SYCL_DISABLE_MOE_DUAL_DOWN="${GGML_SYCL_DISABLE_MOE_DUAL_DOWN:-1}"
export GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN="${GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN:-1}"
# Experimental QKV shared quant: never default ON (device-lost probe)
export GGML_SYCL_DISABLE_QKV_SHARED_QUANT="${GGML_SYCL_DISABLE_QKV_SHARED_QUANT:-1}"

export LD_LIBRARY_PATH="${LX_BIN}:${LD_LIBRARY_PATH:-}"

# Frozen window (mlx.fast-shaped)
export LX_PP="${LX_PP:-512}"
export LX_TG="${LX_TG:-128}"
export LX_REPS="${LX_REPS:-5}"

export LX_BASELINE_JSON="${LX_BASELINE_JSON:-$LX_ROOT/baseline/baseline.json}"
export LX_RESULTS="${LX_RESULTS:-$LX_ROOT/results}"
export LX_GOLDEN="${LX_GOLDEN:-$LX_ROOT/correctness/golden.json}"

mkdir -p "$LX_ROOT/baseline" "$LX_RESULTS" "$LX_ROOT/correctness"
