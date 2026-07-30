# Mount Doom status — 2026-07-30 (moe-down k8 unroll tip)

## LIVE NOW — tip + research track

**Scored tip:** MoE dual + hybrid **mode7** + fused sigmoid+add + dense dual + moe-down weighted (**k=8 unroll**) + device mmid sort/prefix/event  
**Research (golden FAIL / opt-in):** full fused norm (mode6) · mode8 · integrated down · multi-token MMVQ/dual · multi-sg dual

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip + k8 unroll reduce** | **1139.5** | **121.4** | **+9.65%** |
| prior hybrid mode7 | 1143.9 | 121.1 | +9.57% |
| mode8 research | 1138.4 | 121.3 | +9.54% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal tip: `results/20260730T062910Z/` · `notes/SHIP_20260730_moe_down_unroll.md` · `patches/0015-*.patch`  
Kill moe-down: `GGML_SYCL_DISABLE_MOE_DOWN_WEIGHTED=1`  
Fallback hybrid mode2: `GGML_SYCL_TOPK_MOE_HYBRID_MODE=2`

## NEXT KERNEL LEVERS

1. Stay on control binary as champion.
2. Bitexact integrated weighted-MMVQ down (still golden-fail).
3. Multi-token dual/MMVQ oracle.
4. lm_head / residual decode (high golden risk).

## Commands

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
