# Ship note — MoE dual+down multi-token **graph tensors** (2026-07-30)

## Status: **SCORED TIP** (default ON)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior dual multi-token GEMM | 3167.7 | 128.7 | +47.92% | OK |
| **+ dual+down graph tensors** | **3161.0** | **129.0** | **+48.09%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T094742Z/`

Hit:
```
[lx-control-moe-dual] multi-token dual+down (graph tensors) n_tokens=512 k=8
[lx-control-moe-dual] fuse hit (dual+down multi-token) tokens=512 k=8
```

## What

Fuse full multi-token MoE FFN chain when pattern matches:

```
MMID(gate) + MMID(up) + GLU + MMID(down) + MUL(w) + VIEW×k + ADD×(k-1)
```

Implementation (**live graph tensors only** — stack shells segfaulted):

1. `ggml_sycl_mul_mat_id_dual_swiglu_fused` → real `glu`
2. stock `ggml_sycl_mul_mat_id` → real `down_mmid`
3. `moe_weighted_reduce` → real final residual (with alias scratch like two-step)

Default **ON**. Kill: `GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1`

## Why

Prior dual+down research crashed on pool/stack tensor shells for `mul_mat_id`.
Using the allocator's real node buffers is bitexact-class of separate dual + down
fuses while one `can_fuse` skips host dispatch of VIEW×8 + ADD×7 per sparse layer.

Formal: ~**+0.3 tg** / flat pp / **+0.17% score** vs prior tip (noise-adjacent but ≥ tip).

## Tip stack

Prior dual multi-token GEMM tip + **dual+down multi-token** (graph tensors).

## Next

1. True expert-loop dual+down (skip glu materialize + second sort) once stable.  
2. lm_head.  
3. Why first graph layer still takes dual-only then dual+down (can_fuse).
