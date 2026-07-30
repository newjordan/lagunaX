# Ship note — hybrid mode7 **skip no-op reshape** (2026-07-30)

## Status: **SCORED TIP** (default ON)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior tip k8 unroll reduce | 1139.5 | 121.4 | +9.65% | OK |
| **+ skip no-op reshape** | **1140.4** | **121.4** | **+9.71%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T065902Z/`

## What

In hybrid mode7/8 tail (after gather), skip `GGML_OP_RESHAPE` when `dst->data == src->data` (view-style no-op). Avoids empty `compute_forward` dispatch per MoE layer.

Still runs stock sum_rows/clamp; still fuses DIV+SCALE (mode7).

## Also tried this fire (not shipped)

| attempt | result |
|---------|--------|
| pair-embd weighted reduce (2 rows/thread) | golden OK, **tg regress** +9.06% |
| integrated down bitexact (prior) | still fail |

## Tip stack

1. MoE dual SwiGLU  
2. Hybrid mode7 + fused sigmoid+add + **noop reshape skip**  
3. Dense dual shexp  
4. MoE down weighted reduce k=8 unroll  
5. Device mmid sort/prefix/event  

## Next

1. Multi-token dual/MMVQ oracle.  
2. Integrated down (needs mmid buffer oracle).  
3. lm_head / residual.
