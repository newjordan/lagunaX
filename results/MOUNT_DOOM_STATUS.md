# Mount Doom status — 2026-07-30 (dual+down all-layer expert-loop tip)

## LIVE NOW

**Scored tip:** dual decode MMVQ + prefill **expert-loop dual+down (all layers, Q4 gate + Q6 down)** + hybrid mode8 + dense dual + moe-down + mmid + RMS+MUL + ADD+ADD + softplus×mul + MUL_MAT+ADD + rope+set_rows

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip all-layer expert-loop** | **3380.5** | **128.7** | **+50.40%** |
| prior expert-loop partial | 3266.0 | 128.9 | +49.29% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal: `results/20260730T100449Z/` · `notes/SHIP_20260730_dual_down_mixed_quant.md` · `patches/0029-*.patch`  
Kill: `GGML_SYCL_DISABLE_MOE_DUAL_DOWN_EXPERT_LOOP=1` / `DISABLE_MOE_DUAL_DOWN=1` / `DISABLE_MOE_DUAL_SWIGLU=1`

## NEXT

1. lm_head (decode-weighted).  
2. Dense shexp multi-col GEMM dual.  

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
