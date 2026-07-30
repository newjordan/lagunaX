# Mount Doom status — 2026-07-30 (down sgs=8 tip)

## LIVE NOW

**Scored tip:** dual+down + dense dual multi-col + mode8 true top-k+gather+sum + **down MMVQ sgs=8**

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip down sgs=8** | **3402.1** | **130.5** | **+52.22%** |
| prior true top-k+gather+sum | 3409.5 | 130.1 | +51.95% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal tip: `results/20260730T113629Z/` · `notes/SHIP_20260730_moe_down_sgs8.md` · `patches/0037-*.patch`  
Kill down packing: `GGML_SYCL_MOE_DOWN_SGS=1`

## NEXT

1. lm_head prune/mask only with golden oracle.  
2. Prefill multi-token dual MMVQ golden fix.  
3. Router full-norm / dual multi-sg / dense dual+down residual closed under tip.

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
