# Mount Doom status — 2026-07-30 (noop-reshape skip tip)

## LIVE NOW — tip + research track

**Scored tip:** MoE dual + hybrid **mode7** (noop reshape skip) + fused sigmoid+add + dense dual + moe-down k8 unroll + device mmid sort/prefix/event  
**Research (golden FAIL / opt-in):** mode6 full norm · integrated down · multi-token dual/MMVQ · dual multi-sg · pair-embd reduce

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip + noop reshape skip** | **1140.4** | **121.4** | **+9.71%** |
| prior k8 unroll reduce | 1139.5 | 121.4 | +9.65% |
| hybrid mode7 only | 1143.9 | 121.1 | +9.57% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal tip: `results/20260730T065902Z/` · `notes/SHIP_20260730_hybrid_noop_reshape.md` · `patches/0016-*.patch`  
Kill moe-down: `GGML_SYCL_DISABLE_MOE_DOWN_WEIGHTED=1`  
Hybrid fallback: `GGML_SYCL_TOPK_MOE_HYBRID_MODE=2`

## NEXT KERNEL LEVERS

1. Stay on control binary as champion.
2. Multi-token dual/MMVQ oracle (golden-fail when defaulted).
3. Integrated down mmid-buffer oracle.
4. lm_head / residual decode (high golden risk).

## Commands

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
