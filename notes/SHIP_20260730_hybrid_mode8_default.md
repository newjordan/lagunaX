# Ship note — Hybrid mode8 default (2026-07-30)

## Status: **SCORED TIP** (default hybrid mode)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior tip mode7 + any-batch mm+add | 3006.4 | 128.1 | +45.49% | OK |
| **mode8 default** | **3015.2** | **128.6** | **+46.08%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T081402Z/`

Hit:
```
[lx-control-topk-moe] hybrid mode=8 stock-sum+fused-clamp+div-scale nelt=...
```

## What

Default `GGML_SYCL_TOPK_MOE_HYBRID_MODE` **7 → 8**:

- Stock: gather + SUM_ROWS
- Fused: CLAMP + DIV + SCALE in one kernel (clamp = fmin/fmax on stock sum)

Fallback mode7: `GGML_SYCL_TOPK_MOE_HYBRID_MODE=7`

## Why it wins

One fewer kernel (stock CLAMP) per MoE layer; golden-safe (clamp is pure fmin/fmax). ~**+0.5 tg** and slight pp lift on tip stack.

## Tip stack (default ON)

Prior stack with hybrid **mode8** (was mode7).

## Next

1. Multi-token dual/MMVQ.  
2. lm_head.  
3. Fuse stock SUM into gather (mode6-class still golden-fail without stock sum).
