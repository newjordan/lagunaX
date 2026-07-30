# Mount Doom status — 2026-07-30 (dual+down multi-token tip)

## LIVE NOW

**Scored tip:** dual (decode MMVQ + prefill expert-batched GEMM) + **dual+down multi-token** + hybrid mode8 + dense dual + moe-down/integrated + mmid + RMS+MUL + ADD+ADD + softplus×mul + MUL_MAT+ADD + rope+set_rows + prefill two-step weighted reduce

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip dual+down graph tensors** | **3161.0** | **129.0** | **+48.09%** |
| prior dual multi-token GEMM | 3167.7 | 128.7 | +47.92% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal: `results/20260730T094742Z/` · `notes/SHIP_20260730_dual_down_graph_tensors.md` · `patches/0027-*.patch`  
Kill dual+down: `GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1`  
Kill prefill dual: `GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1`  
Kill all dual: `GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU=1`

## RESEARCH (not tip)

- Dual+down with stack shells: **SEGFAULT** (fixed via graph tensors)  
- Dense dual prefill cap 2048: pp regress → reverted  
- Per-token dual MMVQ multi-token: golden FAIL class  

## NEXT

1. lm_head (decode-weighted).  
2. Expert-loop dual+down (no glu materialize).  

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
