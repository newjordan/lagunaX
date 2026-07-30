# Ship note — expert-loop **packed weighted reduce** (skip scatter) 2026-07-30

## Status: **SCORED TIP** (default ON)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior down sgs=8 tip | 3402.1 | 130.5 | +52.22% | OK |
| **packed reduce** | **3540.3** | **130.2** | **+53.41%** | **OK** |
| packed reduce rebench2 | 3512.4 | 130.1 | +53.07% | OK |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T115251Z/` (confirm `20260730T115337Z`)

Hit:
```
[lx-control-moe-dual] multi-token dual+down EXPERT-LOOP n_tokens=512 k=8 n_experts=256 packed_reduce=1
```

## What

On prefill dual+down **expert-loop**, after per-expert down GEMM into packed
`down_contiguous`:

1. Build `inv[token*k+slot] = packed_row` from `mmid_row_mapping`
2. Weighted reduce **from packed** with expert order 0..k-1 (same volatile MUL-then-ADD)
3. **Skip** `k_copy_dst_from_contiguous` scatter of embd×k×T

Decode path unchanged (integrated weighted-MMVQ). Fallback scatter+reduce if disabled.

Kill: `GGML_SYCL_DISABLE_MOE_PACKED_REDUCE=1`

## Why

Prefill drops a full scatter of ~2048×8×512 floats per MoE layer. Formal **~+1.2% score**
(pp **+138 t/s** primary; tg flat/noise).

## Tip stack

Prior down sgs=8 tip + **packed reduce** on expert-loop.

## Next

1. lm_head prune/mask with golden oracle.
2. Apply packed reduce to other multi-token mmid tails if any.
3. Expert-loop host counts wait still open (smaller).
