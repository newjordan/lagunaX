# Mount Doom status — 2026-07-30 (MUL_MAT+ADD any-batch tip)

## LIVE NOW

**Scored tip:** dual + hybrid7 + dense dual + moe-down/integrated + mmid + RMS+MUL + ADD+ADD + softplus×mul + **MUL_MAT+ADD any-batch**

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip any-batch mm+add** | **3006.4** | **128.1** | **+45.49%** |
| prior decode-only mm+add | 1147.3 | 128.2 | +14.46% |
| kill mm+add | 1155.6 | 126.9 | +13.79% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal: `results/20260730T080609Z/` · `notes/SHIP_20260730_mul_mat_add_multicol.md` · `patches/0022-*.patch`  
Kill: `GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1`

## NEXT

1. Multi-token dual/MMVQ.  
2. lm_head.

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
