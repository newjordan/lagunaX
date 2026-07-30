# Mount Doom status — 2026-07-30 (moe-down prefill weighted tip)

## LIVE NOW

**Scored tip:** dual + hybrid mode8 + dense dual + moe-down/integrated + mmid + RMS+MUL + ADD+ADD + softplus×mul + MUL_MAT+ADD + rope+set_rows + **prefill two-step weighted reduce**

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip prefill moe-down weighted** | **3148.4** | **128.9** | **+47.88%** |
| prior rope+set_rows | 3005.5 | 128.9 | +46.18% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal: `results/20260730T085918Z/` · `notes/SHIP_20260730_moe_down_prefill_weighted.md` · `patches/0025-*.patch`  
Kill: `GGML_SYCL_DISABLE_MOE_DOWN_WEIGHTED=1` (covers prefill two-step + decode weighted)

## NEXT

1. Multi-token dual/MMVQ (bitexact vs GEMM).  
2. lm_head.  

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
