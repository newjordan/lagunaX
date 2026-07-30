# Ship note — MoE down weighted reduce (2026-07-30)

## Status: **SCORED TIP** (default ON)

| arm | pp512 | tg128 | score vs pin | golden |
|-----|------:|------:|-------------:|:------:|
| **tip + moe-down-weighted** | **1131.7** | **119.2** | **+7.99%** | **OK** |
| prior dual+hybrid+dense | 1143.8 | 113.3 | +4.26% | OK |
| baseline pin | 1139.2 | 107.4 | 0 | — |

Formal tip: `results/20260730T041927Z/` · `LATEST_SCORE.json`  
Earlier opt-in formal: `results/20260730T041746Z/` (pp1144 / tg118.0 / +7.44%)

## What

Surgical fuse of Laguna MoE **down projection tail**:

```
MUL_MAT_ID(down) → MUL(route_weights) → VIEW×8 → ADD×7
  ⇒  mul_mat_id + one weighted-reduce kernel
```

- Matches `build_moe_ffn` after dual SwiGLU (weight-after-FFN path)
- k=8, embd=2048, n_tokens≤32
- Volatile `value*weight` before ordered sum (bitexact-ish vs MUL+ADD chain)
- Hard-skip if weights/mmid/mul buffers overlap final dst (allocator reuse)
- Default **ON**; kill: `GGML_SYCL_DISABLE_MOE_DOWN_WEIGHTED=1` or `ENABLE_MOE_DOWN_WEIGHTED=0`

Hit: `[lx-control-moe-down] fuse hit (weighted reduce) embd=2048 k=8 tokens=1`

## Why it wins

Decode was paying **one MUL_MAT_ID + 1 MUL + 8 VIEW no-ops + 7 ADD launches** per MoE layer. Collapsing the glue into one reduce after MMVQ is pure host/submit + bandwidth win. ~**+6 tg128** formal vs prior tip.

## Full tip stack (default ON)

1. MoE dual SwiGLU (routed gate/up) — `DISABLE_MOE_DUAL_SWIGLU=1`
2. Hybrid router mode1 gather — `ENABLE_TOPK_MOE_BIAS=0`
3. Dense dual SwiGLU (shared expert) — `DISABLE_DENSE_DUAL_SWIGLU=1`
4. **MoE down weighted reduce** — `DISABLE_MOE_DOWN_WEIGHTED=1`

## Code

- `ggml-sycl.cpp`: `ggml_sycl_moe_weighted_reduce`, `ggml_sycl_fuse_moe_down_weighted`
- Wired in `ggml_sycl_fuse` after dense dual

## Next

1. Hybrid mode2 gather-norm bitexact
2. Device multi-token `mul_mat_id` (prefill host wait)
3. Optional: integrate weighted MMVQ (down+weight in one kernel) if more headroom
