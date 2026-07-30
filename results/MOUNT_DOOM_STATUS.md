# Mount Doom status — 2026-07-30 (dual multi-token expert-batched GEMM tip)

## LIVE NOW

**Scored tip:** dual (decode MMVQ + **prefill expert-batched GEMM**) + hybrid mode8 + dense dual (cap32) + moe-down/integrated + mmid + RMS+MUL + ADD+ADD + softplus×mul + MUL_MAT+ADD + rope+set_rows + prefill two-step weighted reduce

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip dual multi-token GEMM** | **3167.7** | **128.7** | **+47.92%** |
| prior prefill moe-down weighted | 3148.4 | 128.9 | +47.88% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal: `results/20260730T092534Z/` · `notes/SHIP_20260730_dual_multitoken_gemm.md` · `patches/0026-*.patch`  
Kill prefill dual: `GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1`  
Kill all dual: `GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU=1`

## RESEARCH (not tip)

- Dense dual prefill cap 2048: golden OK, **pp −521** → reverted  
- Per-token dual MMVQ multi-token: golden FAIL class (opt-in MMVQ flag only)

## NEXT

1. lm_head.  
2. Dual+down multi-token epilogue.  

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
