# Mount Doom status — 2026-07-30 (dual+down decode integrated tip)

## LIVE NOW

**Scored tip:** MoE dual+down expert-loop (prefill) + **dual+down decode integrated** + dense dual multi-col GEMM + hybrid mode8 + rest of stack

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip dual+down decode** | **3422.7** | **128.8** | **+50.93%** |
| prior dense dual multi-col | 3396.6 | 129.1 | +50.89% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal: `results/20260730T101935Z/` · `notes/SHIP_20260730_dual_down_decode.md` · `patches/0031-*.patch`  
Kill dual+down: `GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1`

## NEXT

1. lm_head (decode-weighted).  

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
