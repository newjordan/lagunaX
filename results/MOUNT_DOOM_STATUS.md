# Mount Doom status — 2026-07-30 (dense dual multi-col GEMM tip)

## LIVE NOW

**Scored tip:** MoE dual+down expert-loop (all layers) + **dense dual multi-col GEMM** + hybrid mode8 + moe-down + mmid + RMS+MUL + ADD+ADD + softplus×mul + MUL_MAT+ADD + rope+set_rows

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip dense dual multi-col GEMM** | **3396.6** | **129.1** | **+50.89%** |
| prior dual+down all-layer | 3380.5 | 128.7 | +50.40% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal: `results/20260730T101259Z/` · `notes/SHIP_20260730_dense_dual_gemm_multicol.md` · `patches/0030-*.patch`  
Kill dense dual: `GGML_SYCL_DISABLE_DENSE_DUAL_SWIGLU=1`  
Kill dense GEMM only: `GGML_SYCL_DISABLE_DENSE_DUAL_GEMM=1`

## NEXT

1. lm_head (decode-weighted).  

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
