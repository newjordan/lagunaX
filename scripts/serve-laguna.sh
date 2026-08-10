#!/usr/bin/env bash
# Serve Laguna-XS-2.1 Q4_K_M on the Arc Pro B70 with the FULL 131072-token context,
# on the CHAMPION kernel build (tag lx-champion-1.3105-20260810, hub c7d3bfe6d).
#
# Overrides use SERVE_CTX / SERVE_NP / SERVE_PORT — NOT bare CTX/PORT names:
# with-gpu-lock sources env.sh, which exports CTX=8192, and bare names inherit
# it (this silently clamped serving to 8K/slot on 2026-08-10 — the same
# namespace rule as the EMB_BIN pattern in PLAN_20260807 non-goals).
#
# Context math: -c is the TOTAL KV pool, split evenly across -np slots.
#   -c 131072 -np 4  ->  32768 per request   (what bit us: prompts were already ~23.5K)
#   -c 131072 -np 1  ->  131072 per request  (full trained context, identical VRAM)
# The model's trained context is 131072 (laguna.context_length), so -np 1 is the max.
#
# ENV IS LOAD-BEARING for this binary (unlike the old treebeard server, where
# unset was correct). The champion source defaults are NOT the validated ship
# state — receipts were all taken with the block below (mirrors lx env.sh):
#   FUSE_NORM_ROPE=1            rope fuses are opt-in in source; carries ~+3-5% decode
#   DISABLE_MOE_DUAL_DOWN=1     kill-switch for a known-broken path (NaN logprobs)
#                               that is DEFAULT-ENABLED in source (ggml-sycl.cpp:6159)
#   DISABLE_MOE_DUAL_MULTITOKEN=1 / DISABLE_QKV_SHARED_QUANT=1   same class
#   DISABLE_DNN=1, DISABLE_GRAPH=1   measured choices (receipts 2026-08-09/10)
#
# -ub 2048: do NOT "fix" this to -ub 4096. A/B'd for Laguna at -c 131072 on
# 2026-08-05 (scripts/ab-ubatch-laguna.sh): 4096 is slower at every depth and
# costs +1.3 GiB VRAM. Re-check pending on the champion build (see notes).
#
# Depth receipts for THIS binary (llama-bench -p 0 -n 128, 2026-08-10):
# decode 152.5 @ d0, 129.5 @ d4096, 102.7 @ d16384; real 24K ingest+continue
# 300.5 t/s prefill / 91.2 t/s decode (llama-cli defaults).

set -euo pipefail

REPO=${REPO:-/home/frosty40/turbo/worktrees/lx-champion-tier12}
BIN="$REPO/build/bin"
MODEL=${MODEL:-/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf}
TEMPLATE=${TEMPLATE:-/home/frosty40/turbo/worktrees/treebeard-pr-private-latest/models/templates/poolside-Laguna-XS-2.1.jinja}
PORT=${SERVE_PORT:-8092}
NPARALLEL=${SERVE_NP:-1}       # 1 = full 131072 ctx per request; 2 = 65536 each, etc.
CTX=${SERVE_CTX:-131072}
LOG=${SERVE_LOG:-/home/frosty40/logs/laguna-8092.log}

# NB: do NOT `source setvars.sh` here — it exits the shell under `set -euo pipefail`.
ONEAPI=/opt/intel/oneapi
export LD_LIBRARY_PATH="$BIN:$ONEAPI/tcm/1.5/lib:$ONEAPI/umf/1.1/lib:$ONEAPI/tbb/2023.0/env/../lib/intel64/gcc4.8:$ONEAPI/mpi/2021.18/opt/mpi/libfabric/lib:$ONEAPI/mpi/2021.18/lib:$ONEAPI/mkl/2026.0/lib:$ONEAPI/ippcp/2026.0/lib/:$ONEAPI/ipp/2026.0/lib:$ONEAPI/dnnl/2026.0/lib:$ONEAPI/debugger/2026.0/opt/debugger/lib:$ONEAPI/dal/2026.0/lib:$ONEAPI/compiler/2026.0/opt/compiler/lib:$ONEAPI/compiler/2026.0/lib:$ONEAPI/ccl/2022.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Validated ship env for the champion binary — see header. Do not remove.
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
export GGML_SYCL_DISABLE_DNN=1
export GGML_SYCL_DISABLE_GRAPH=1
export GGML_SYCL_FUSE_NORM_ROPE=1
export GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=0
export GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1
export GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1
export GGML_SYCL_DISABLE_QKV_SHARED_QUANT=1

# Loader-resolution assert (FINDING_20260810 trap family): the server must run
# THIS build's kernels, not whatever else is on the path.
RESOLVED_SO="$(ldd "$BIN/llama-server" 2>/dev/null | awk '/libggml-sycl/ {print $3; exit}')"
if [[ -n "$RESOLVED_SO" && "$RESOLVED_SO" != "$BIN"/* ]]; then
  echo "FATAL: llama-server resolves libggml-sycl from $RESOLVED_SO, not $BIN" >&2
  exit 3
fi
mkdir -p "$(dirname "$LOG")"
echo "[serve-laguna] $(date -u +%FT%TZ) bin=$BIN so_sha=$(sha256sum "$(readlink -f "$BIN/libggml-sycl.so.0")" | cut -c1-8) ctx=$CTX np=$NPARALLEL port=$PORT" >>"$LOG"

exec "$BIN/llama-server" \
  -m "$MODEL" \
  --alias laguna-xs-2.1-q4-lx-champion \
  -a laguna-xs-2.1-q4-lx-champion \
  -ngl 99 -fa on -ctk f16 -ctv f16 \
  -c "$CTX" -np "$NPARALLEL" \
  -b 4096 -ub 2048 \
  -t 16 \
  --host 0.0.0.0 --port "$PORT" \
  --jinja --chat-template-file "$TEMPLATE" \
  --temp 1.0 --top-k 20 --top-p 1.0 --min-p 0.0 \
  -n -1 \
  --metrics \
  >>"$LOG" 2>&1
