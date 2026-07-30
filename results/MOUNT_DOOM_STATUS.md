# Mount Doom status — 2026-07-30 (moe-down k8 unroll tip)

## LIVE NOW — tip + research track

**Scored tip:** MoE dual + hybrid **mode7** + fused sigmoid+add + dense dual + moe-down weighted (**k=8 unroll**) + device mmid sort/prefix/event  
**Research (golden FAIL / opt-in):** full fused norm (mode6) · mode8 · integrated down · multi-token MMVQ/dual · dual multi-sg (sgs=8/16)

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip + k8 unroll reduce** | **1139.5** | **121.4** | **+9.65%** |
| dual sgs=8 research | 1124.5 | 121.4 | +9.32% |
| prior hybrid mode7 | 1143.9 | 121.1 | +9.57% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal tip: `results/20260730T062910Z/` · sgs8: `results/20260730T063711Z/`  
Notes: `SHIP_20260730_moe_down_unroll.md`, `SHIP_20260730_dual_sgs8.md`  
Kill moe-down: `GGML_SYCL_DISABLE_MOE_DOWN_WEIGHTED=1`  
Dual multi-sg research: `GGML_SYCL_MOE_DUAL_SGS=8` (default **1**)

## NEXT KERNEL LEVERS

1. Stay on control binary as champion.
2. Bitexact integrated down — **still FAIL** except exact `mul_mat_id`+reduce alias (`SHIP_20260730_integrated_down_bitexact.md`).
3. Multi-token dual/MMVQ oracle.
4. lm_head / residual decode (high golden risk).

## Commands

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
