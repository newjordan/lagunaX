# Ship note — dual+down **decode** (n_tokens=1) integrated (2026-07-30)

## Status: **SCORED TIP** (default ON)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior dense dual multi-col GEMM | 3396.6 | 129.1 | +50.89% | OK |
| **+ dual+down decode integrated** | **3422.7** | **128.8** | **+50.93%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T101935Z/`

Hit:
```
[lx-control-moe-dual] dual+down compose n_tokens=1 k=8 integrated=1
[lx-control-moe-dual] fuse hit (dual+down multi-token) tokens=1 k=8
# prefill unchanged:
[lx-control-moe-dual] multi-token dual+down EXPERT-LOOP n_tokens=512 k=8
```

## What

Extend dual+down fuse from multi-token only (`n_tokens>=2`) to **decode** (`n_tokens==1`):

1. Dual gate+up+SwiGLU (decode MMVQ) → real `glu`
2. **Integrated weighted-MMVQ down** (same as down-fuse tip) → residual  
   Fallback: stock `mul_mat_id` + weighted reduce if integrated rejects (alias/reorder)

First naive compose used two-step down for decode and **regressed tg** (~−0.5). Integrated restores tip-class decode path while one can_fuse still skips VIEW×k+ADD×(k−1) host dispatch.

Kill dual+down: `GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1`  
Kill integrated: `GGML_SYCL_DISABLE_MOE_DOWN_INTEGRATED=1`

## Why

Decode-weighted score: fuse full MoE FFN per sparse layer on tg path. Formal score **+0.04%** vs prior tip (noise-adjacent; pp +26 / tg −0.3).

## Tip stack

Prior tip + **decode dual+down with integrated down**.

## Next

1. lm_head (still largest decode BW).  
2. True dual+down one-kernel for decode (no glu store).
