# Mount Doom status — 2026-07-30 (after dense dual prefill probe)

## LIVE NOW

**Scored tip:** dual + hybrid mode8 + dense dual (cap32) + moe-down/integrated + mmid + RMS+MUL + ADD+ADD + softplus×mul + MUL_MAT+ADD + rope+set_rows + **prefill two-step weighted reduce**

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip prefill moe-down weighted** | **3148.4** | **128.9** | **+47.88%** |
| dense dual prefill cap2048 (reverted) | 2627.5 | 128.4 | +40.92% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal: `results/20260730T085918Z/` · `notes/SHIP_20260730_moe_down_prefill_weighted.md` · `patches/0025-*.patch`  
Kill: `GGML_SYCL_DISABLE_MOE_DOWN_WEIGHTED=1`

## RESEARCH (not tip)

- Dense dual prefill cap 2048: golden OK, **pp −521** → reverted (`SHIP_20260730_dense_dual_prefill_cap.md`)

## NEXT

1. Multi-token dual/MMVQ via expert-batched GEMM (bitexact vs stock regroup).  
2. lm_head.  

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
