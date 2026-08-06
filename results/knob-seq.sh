#!/usr/bin/env bash
set -uo pipefail
cd /home/frosty40/turbo/lx
export NOTE="mmid-fused-batch" KNOB_SPEC="GGML_SYCL_ENABLE_MMID_FUSED_BATCH=1"
bash scripts/bench-knob-candidate.sh; echo "MMID_FUSED_BATCH exit=$?"
export NOTE="moe-packed-reduce-off" KNOB_SPEC="GGML_SYCL_DISABLE_MOE_PACKED_REDUCE=1"
bash scripts/bench-knob-candidate.sh; echo "MOE_PACKED_REDUCE_OFF exit=$?"
