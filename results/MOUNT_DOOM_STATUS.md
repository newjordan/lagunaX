# Mount Doom status — 2026-07-30 (MUL_MAT+ADD tip)

## LIVE NOW

**Scored tip:** dual + hybrid7 + dense dual + moe-down/integrated + mmid + RMS+MUL + ADD+ADD + softplus×mul + **MUL_MAT+ADD o_proj**

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip MUL_MAT+ADD** | **1147.3** | **128.2** | **+14.46%** |
| prior softplus×mul | 1158.8 | 127.2 | +14.07% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal: `results/20260730T075349Z/` · `notes/SHIP_20260730_mul_mat_add_epilogue.md` · `patches/0021-*.patch`  
Kill: `GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1`

## NEXT

1. Multi-col MUL_MAT+ADD for prefill.  
2. Multi-token dual/MMVQ.  
3. lm_head.

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
