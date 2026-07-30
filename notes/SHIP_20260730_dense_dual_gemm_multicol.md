# Ship note — Dense dual multi-col **GEMM** (2026-07-30)

## Status: **SCORED TIP** (default ON)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior dual+down all-layer | 3380.5 | 128.7 | +50.40% | OK |
| **+ dense dual multi-col GEMM** | **3396.6** | **129.1** | **+50.89%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T101259Z/`

Hit:
```
[lx-control-dense-dual] multi-col GEMM dual ncols_dst=512 nrows=8192
[lx-control-dense-dual] fuse hit (shared gate+up+swiglu) ncols_dst=512
# decode unchanged:
[lx-control-dense-dual] fuse hit (shared gate+up+swiglu) ncols_dst=1
```

## What

Dense dual (shared expert + leading dense FFN) for prefill:

| ncols_dst | path |
|----------:|------|
| 1..32 | reorder MMVQ dual (prior tip) |
| 33..2048 | **stock GEMM×2 + fused SwiGLU** into glu |

Prior raise of MMVQ cap to 2048 **regressed pp −521** (research). GEMM path is bitexact-class of stock matmul+swiglu and only elides gate/up graph stores + one fuse.

Kill all dense dual: `GGML_SYCL_DISABLE_DENSE_DUAL_SWIGLU=1`  
Kill prefill GEMM only: `GGML_SYCL_DISABLE_DENSE_DUAL_GEMM=1`

## Why

Hits leading dense block (`nrows=8192`) and shexp prefill. Formal ~**+16 pp** / ~**+0.4 tg** / **+0.49% score**.

## Tip stack

Prior dual+down expert-loop tip + **dense dual multi-col GEMM**.

## Next

1. lm_head (decode-weighted).  
2. Shexp down + residual fuse if not already mul_mat+add.
