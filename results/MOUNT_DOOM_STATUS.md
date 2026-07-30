# Mount Doom status — 2026-07-30 (true top-k+gather tip)

## LIVE NOW

**Scored tip:** MoE dual+down + dense dual multi-col + hybrid mode8 + **true top-k+gather**

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip true top-k+gather** | **3412.8** | **129.9** | **+51.81%** |
| prior true top-k only | 3403.3 | 129.7 | +51.48% |
| prior dual+down decode | 3422.7 | 128.8 | +50.93% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal tip: `results/20260730T110508Z/` · `notes/SHIP_20260730_router_true_topk_gather.md` · `patches/0034-*.patch`  
Kill: `GGML_SYCL_DISABLE_ROUTER_TRUE_TOPK=1`

## NEXT

1. lm_head prune/mask only with golden oracle (packing + dual sgs closed; see `SHIP_20260730_mode9_sgs_reprobe.md`).  
2. Prefill multi-token dual MMVQ golden fix.  
3. MoE dual+down residual epilogue (optional; shexp is parallel branch).

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
