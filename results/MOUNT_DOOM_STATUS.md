# Mount Doom status — 2026-07-30 (ADD+ADD residual tip)

## LIVE NOW — tip + research track

**Scored tip:** MoE dual + hybrid mode7 + dense dual + moe-down k8 + integrated decode-only + mmid + RMS+MUL + **ADD+ADD residual**  
**Research:** multi-token dual/MMVQ · mode6 · integrated ne12>1 · dual multi-sg

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip ADD+ADD** | **1145.0** | **125.2** | **+12.35%** |
| prior RMS+MUL | 1146.4 | 123.2 | +11.04% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal: `results/20260730T073706Z/` · `notes/SHIP_20260730_add_add_residual.md` · `patches/0019-*.patch`  
Kill ADD fuse: `GGML_SYCL_DISABLE_ADD_ADD_FUSE=1`  
Kill RMS fuse: `GGML_SYCL_DISABLE_RMS_NORM_FUSE=1`

## NEXT KERNEL LEVERS

1. Multi-token dual/MMVQ — match host-sort multi-token mmid.  
2. Softplus+mul attn gate (reshape).  
3. lm_head.

## Commands

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
