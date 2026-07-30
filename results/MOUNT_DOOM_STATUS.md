# Mount Doom status — 2026-07-30 (RMS_NORM+MUL tip)

## LIVE NOW — tip + research track

**Scored tip:** MoE dual + hybrid mode7 + dense dual + moe-down k8 + integrated decode-only + mmid stack + **RMS_NORM+MUL fuse**  
**Research (golden FAIL / opt-in):** mode6 full norm · multi-token dual/MMVQ · dual multi-sg · integrated ne12>1

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip RMS+MUL** | **1146.4** | **123.2** | **+11.04%** |
| prior integrated decode-only | 1147.6 | 122.4 | +10.55% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal tip: `results/20260730T072939Z/` · `notes/SHIP_20260730_rms_norm_mul_fuse.md` · `patches/0018-*.patch`  
Kill RMS fuse: `GGML_SYCL_DISABLE_RMS_NORM_FUSE=1`  
Kill integrated: `GGML_SYCL_DISABLE_MOE_DOWN_INTEGRATED=1`

## NEXT KERNEL LEVERS

1. Multi-token dual/MMVQ — match **host-sort** multi-token mmid numerics.  
2. lm_head / residual decode.  
3. Do not re-open integrated for ne12>1 without matching stock multi-token.

## Commands

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
