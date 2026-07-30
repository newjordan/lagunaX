# Ship note — Dense dual SwiGLU for shared expert (2026-07-30)

## Status: **SCORED TIP** (default ON with dual MoE + hybrid m1)

| arm | pp512 | tg128 | score vs pin | golden |
|-----|------:|------:|-------------:|:------:|
| **tip dual+hybrid-m1+dense-dual** | **1143.8** | **113.3** | **+4.26%** | **OK** |
| prior tip dual+hybrid-m1 | 1142.6 | 110.2 | +2.09% | OK |
| baseline pin | 1139.2 | 107.4 | 0 | — |

Formal: `results/20260730T040839Z/` · `LATEST_SCORE.json`

## What

Port package dense dual gate+up+SwiGLU onto **control** for Laguna always-on shared expert (`build_ffn` shexp):

```
MUL_MAT(gate) + MUL_MAT(up) + GLU(swiglu)  →  one reorder-MMVQ dual kernel
```

- Types: Q4_K / Q5_K / Q6_K (Laguna shexp is Q4_K)
- `ncols_dst` 1..32 (decode + small prefill)
- Default **ON**; kill: `GGML_SYCL_DISABLE_DENSE_DUAL_SWIGLU=1`

Hit: `[lx-control-dense-dual] fuse hit (shared gate+up+swiglu)`

## Why it moved the needle

Shared expert is always-on every MoE layer (×~39 sparse layers). Dual MoE already fused routed experts; shexp was still two MMVQ + silu. Fusing shexp gate/up recovers ~**+3 tg128** formal on top of dual+hybrid-m1.

## Tip stack (all default ON)

1. MoE dual SwiGLU (routed experts) — kill `DISABLE_MOE_DUAL_SWIGLU=1`
2. Hybrid router mode1 (fused gather) — kill `ENABLE_TOPK_MOE_BIAS=0`
3. **Dense dual SwiGLU (shared expert)** — kill `DISABLE_DENSE_DUAL_SWIGLU=1`

## Code

- `mmvq.cpp` / `mmvq.hpp`: `ggml_sycl_mul_mat_vec_q_dense_dual_swiglu_reorder`
- `ggml-sycl.cpp`: fuse + reorder ensure
- `topk-moe.cpp` / `.hpp`: wire into `ggml_sycl_fuse` after MoE dual

## Next

1. Hybrid mode2 gather-norm bitexact (~+3 tg still parked)
2. MoE down fuse on control only
3. Device multi-token `mul_mat_id` for prefill
