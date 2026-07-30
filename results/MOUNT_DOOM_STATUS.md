# Mount Doom status — 2026-07-30 (integrated decode-only tip)

## LIVE NOW — tip + research track

**Scored tip:** MoE dual + hybrid **mode7** (noop reshape skip + skip DIV store) + fused sigmoid+add + dense dual + moe-down k8 unroll + **integrated weighted-mmvq decode-only** + device mmid sort/prefix/event  
**Research (golden FAIL / opt-in):** mode6 full norm · multi-token dual/MMVQ · dual multi-sg · pair-embd reduce · integrated for ne12>1

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip integrated decode-only** | **1147.6** | **122.4** | **+10.55%** |
| prior noop reshape skip | 1140.4 | 121.4 | +9.71% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal tip: `results/20260730T072139Z/` · `notes/SHIP_20260730_moe_down_integrated_decode.md` · `patches/0017-*.patch`  
Kill integrated: `GGML_SYCL_DISABLE_MOE_DOWN_INTEGRATED=1`  
Kill moe-down all: `GGML_SYCL_DISABLE_MOE_DOWN_WEIGHTED=1`  
Hybrid fallback: `GGML_SYCL_TOPK_MOE_HYBRID_MODE=2`

## NEXT KERNEL LEVERS

1. Stay on control binary as champion.
2. Multi-token dual/MMVQ — must mirror **host-sort** multi-token mmid (not decode fused).
3. lm_head / residual decode (high golden risk).
4. Do **not** re-open integrated for ne12>1 without matching stock multi-token path.

## Commands

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
