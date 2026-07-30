# Mount Doom status — 2026-07-30 (true top-k tip)

## LIVE NOW

**Scored tip:** MoE dual+down (prefill+decode) + dense dual multi-col + hybrid mode8 + **true top-k** (not full argsort 256)

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip true top-k** | **3403.3** | **129.7** | **+51.48%** |
| prior dual+down decode | 3422.7 | 128.8 | +50.93% |
| dense dual+down residual (opt-in) | 3375.8 | 128.4 | +50.06% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal tip: `results/20260730T105726Z/` · `notes/SHIP_20260730_router_true_topk.md` · `patches/0033-*.patch`  
Kill true top-k: `GGML_SYCL_DISABLE_ROUTER_TRUE_TOPK=1`

## NEXT

1. lm_head prune/mask only with golden oracle (packing A/B exhausted).  
2. Prefill multi-token dual golden fix (host-sort parity).  
3. Optional: A/B stock argsort residual noise on pp.

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
