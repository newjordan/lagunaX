# Mount Doom status — 2026-07-30 (mul_mat+add shexp alias tip)

## LIVE NOW

**Scored tip:** dual+down + topk + packed reduce + **mul_mat+add residual-alias (Q6 shexp)**

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip mm-add alias** | **3734.7** | **129.7** | **+55.10%** |
| prior packed reduce | 3540.3 | 130.2 | +53.41% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal: `results/20260730T124637Z/` · `notes/SHIP_20260730_mul_mat_add_shexp_alias.md` · `patches/0043-*`  
Kill mm-add: `GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1`  
Kill packed reduce: `GGML_SYCL_DISABLE_MOE_PACKED_REDUCE=1`

## NEXT

1. Optional tip rebench for noise band.
2. Attn/FA remaining from re-trace (rope fused).
3. Counts-sync / lm_head prune closed or low ROI.

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
