#!/usr/bin/env bash
# Serve Laguna-XS-2.1 Q4_K_M on the Arc Pro B70 with the FULL 131072-token context.
#
# Context math: -c is the TOTAL KV pool, split evenly across -np slots.
#   -c 131072 -np 4  ->  32768 per request   (what bit us: prompts were already ~23.5K)
#   -c 131072 -np 1  ->  131072 per request  (full trained context, identical VRAM)
# The model's trained context is 131072 (laguna.context_length), so -np 1 is the max.
#
# GGML_SYCL_* are all deliberately LEFT UNSET = the validated B70 ship config
# (oneDNN on, top-k MoE router fusion on, RMS_NORM glue fusion on).
#
# -ub 2048: do NOT "fix" this to -ub 4096. The b70-tune ship config recommends 4096, but that
# was measured on Qwen3.6-35B-A3B at short ctx. A/B'd for Laguna at -c 131072 on 2026-08-05
# (scripts/ab-ubatch-laguna.sh, results/ab-ubatch-20260806T012435Z): 4096 is SLOWER at every
# depth -- prefill -7.1% / -6.0% / -3.6% at 6.5K / 26K / 52K tokens, decode -10%
# (84.6 -> 75.9 t/s), and costs +1.3 GiB VRAM. 2048 wins on both axes.

set -euo pipefail

REPO=/home/frosty40/turbo/worktrees/treebeard-pr-private-latest
MODEL=/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf
PORT=${PORT:-8092}
NPARALLEL=${NPARALLEL:-1}      # 1 = full 131072 ctx per request; 2 = 65536 each, etc.
CTX=${CTX:-131072}
LOG=${LOG:-/home/frosty40/logs/laguna-8092.log}

# NB: do NOT `source setvars.sh` here — it exits the shell under `set -euo pipefail`.
# This is the exact LD_LIBRARY_PATH the known-good server ran with.
ONEAPI=/opt/intel/oneapi
export LD_LIBRARY_PATH="$ONEAPI/tcm/1.5/lib:$ONEAPI/umf/1.1/lib:$ONEAPI/tbb/2023.0/env/../lib/intel64/gcc4.8:$ONEAPI/mpi/2021.18/opt/mpi/libfabric/lib:$ONEAPI/mpi/2021.18/lib:$ONEAPI/mkl/2026.0/lib:$ONEAPI/ippcp/2026.0/lib/:$ONEAPI/ipp/2026.0/lib:$ONEAPI/dnnl/2026.0/lib:$ONEAPI/debugger/2026.0/opt/debugger/lib:$ONEAPI/dal/2026.0/lib:$ONEAPI/compiler/2026.0/opt/compiler/lib:$ONEAPI/compiler/2026.0/lib:$ONEAPI/ccl/2022.0/lib/${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$(dirname "$LOG")"
cd "$REPO"

exec ./build-positive-package/bin/llama-server \
  -m "$MODEL" \
  --alias laguna-xs-2.1-q4-treebeard \
  -a laguna-xs-2.1-q4-treebeard \
  -ngl 99 -fa on -ctk f16 -ctv f16 \
  -c "$CTX" -np "$NPARALLEL" \
  -b 4096 -ub 2048 \
  -t 16 \
  --host 0.0.0.0 --port "$PORT" \
  --jinja --chat-template-file models/templates/poolside-Laguna-XS-2.1.jinja \
  --temp 1.0 --top-k 20 --top-p 1.0 --min-p 0.0 \
  -n -1 \
  --metrics \
  >>"$LOG" 2>&1
