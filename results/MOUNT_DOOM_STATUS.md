# Mount Doom status — 2026-07-30 (hybrid mode8 tip)

## LIVE NOW

**Scored tip:** dual + **hybrid mode8** + dense dual + moe-down/integrated + mmid + RMS+MUL + ADD+ADD + softplus×mul + MUL_MAT+ADD any-batch

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip mode8** | **3015.2** | **128.6** | **+46.08%** |
| prior mode7 any-batch | 3006.4 | 128.1 | +45.49% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal: `results/20260730T081402Z/` · `notes/SHIP_20260730_hybrid_mode8_default.md` · `patches/0023-*.patch`  
Fallback: `GGML_SYCL_TOPK_MOE_HYBRID_MODE=7`

## NEXT

1. Multi-token dual/MMVQ.  
2. lm_head.

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
