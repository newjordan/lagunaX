# Ship note — MoE down weighted-reduce **k=8 unroll** (2026-07-30)

## Status: **SCORED TIP** (default ON with moe-down weighted)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior tip hybrid mode7 | 1143.9 | 121.1 | +9.57% | OK |
| **+ k=8 unrolled weighted reduce** | **1139.5** | **121.4** | **+9.65%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T062910Z/`

## What

In `ggml_sycl_moe_weighted_reduce` (two-step MoE down after mul_mat_id):

- Prefetch row pointer + weight row
- **Fully unroll** the expert loop when `k==8` (Laguna)
- Same `volatile` MUL-then-ADD contract as before (bitexact class)

Kill (unchanged): `GGML_SYCL_DISABLE_MOE_DOWN_WEIGHTED=1`

## Why it wins

Decode-weighted score: ~**+0.25 tg** formal vs mode7 tip; floors OK.  
Compiler-friendly fixed-k=8 body on the hot embd×token reduce.

## Not in this ship

- Integrated weighted-MMVQ down still **golden FAIL** when `ENABLE_MOE_DOWN_INTEGRATED=1` (reconfirmed on tip stack).

## Tip stack (default ON)

1. MoE dual SwiGLU  
2. Hybrid mode7 + fused sigmoid+add  
3. Dense dual shexp  
4. MoE down weighted reduce (**k=8 unroll**)  
5. Device mmid sort/prefix/event-wait  

## Next

1. Bitexact integrated down (still fails).  
2. Multi-token dual/MMVQ oracle.  
3. lm_head / residual.
