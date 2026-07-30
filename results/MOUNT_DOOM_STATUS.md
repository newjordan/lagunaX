# Mount Doom status — 2026-07-30 (hybrid mode7 tip)

## LIVE NOW — tip + research track

**Scored tip:** MoE dual + hybrid **mode7** (stock sum/clamp + fused DIV+SCALE) + fused sigmoid+add + dense dual + moe-down + device mmid sort/prefix/event  
**Research (golden FAIL / opt-in):** full fused norm (mode6) · mode8 clamp-fuse (~tip) · integrated down · multi-token MMVQ · dual multi-token · multi-sg dual · pinned counts

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip hybrid mode7** | **1143.9** | **121.1** | **+9.57%** |
| mode8 stock-sum+fused clamp/div/scale | 1138.4 | 121.3 | +9.54% |
| prior fused sigmoid+add (mode2) | 1148.9 | 120.2 | +9.09% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal tip: `results/20260730T061617Z/` · mode8: `results/20260730T062159Z/`  
Notes: `SHIP_20260730_hybrid_mode7.md`, `SHIP_20260730_hybrid_mode8.md`  
Fallback mode2: `GGML_SYCL_TOPK_MOE_HYBRID_MODE=2` · mode8 research: `=8`  
Kill fused sig+add: `GGML_SYCL_DISABLE_ROUTER_SIGMOID_ADD=1`  
Kill mmid device sort: `GGML_SYCL_DISABLE_MMID_DEVICE_SORT=1`

## NEXT KERNEL LEVERS

1. Stay on control binary as champion.
2. Bitexact multi-token dual/MMVQ oracle.
3. Integrated weighted-MMVQ down golden fix.
4. lm_head / residual decode (high golden risk).

## Commands

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
