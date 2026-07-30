# Mount Doom status — 2026-07-30 (rope+set_rows tip)

## LIVE NOW

**Scored tip:** dual + hybrid mode8 + dense dual + moe-down/integrated + mmid + RMS+MUL + ADD+ADD + softplus×mul + MUL_MAT+ADD + **rope+set_rows**

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip rope+set_rows** | **3005.5** | **128.9** | **+46.18%** |
| prior mode8 | 3015.2 | 128.6 | +46.08% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal: `results/20260730T084016Z/` · `notes/SHIP_20260730_rope_set_rows_fuse.md` · `patches/0024-*.patch`  
Kill: `GGML_SYCL_DISABLE_ROPE_SET_ROWS_FUSE=1`

## NEXT

1. Multi-token dual/MMVQ.  
2. lm_head.

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
