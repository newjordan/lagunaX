# Ship note — MUL_MAT+ADD multi-col / any-batch (2026-07-30)

## Status: **SCORED TIP** (default ON)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior tip decode-only MUL_MAT+ADD | 1147.3 | 128.2 | +14.46% | OK |
| **+ any-batch MUL_MAT+ADD** | **3006.4** | **128.1** | **+45.49%** | **OK** |
| kill MUL_MAT+ADD | 1155.6 | 126.9 | +13.79% | OK |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T080609Z/`

Hit:
```
[lx-control-mm-add] fuse hit (mul_mat+add) ne0=2048 ne1=512 wtype=q4_K  # prefill
[lx-control-mm-add] fuse hit (mul_mat+add) ne0=2048 ne1=1 wtype=q4_K    # decode
```

## What

Extend decode-only MUL_MAT+ADD epilogue to **all batch sizes**:

1. Multi-col reorder MMVQ (ncols 2..8): `dst[j*stride+row] = gemv + residual[j*stride+row]`
2. Larger batches: per-column single-col loop with `addend = residual + i*ne0`
3. Fuse gate: no longer caps `ne11==1`

Hits both **o_proj+attn residual** and **shexp_down+moe** when graph-adjacent.

Kill: `GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1`

## Why the big prefill win

Prefill was paying a full embd×T ADD after every o_proj (and shexp_down+moe) across ~40 layers. Fusing into MMVQ store removes those launches + intermediate writes. Kill-switch A/B confirms prefill collapses back to ~1155 without the fuse.

## Tip stack (default ON)

Prior stack + **MUL_MAT+ADD any-batch**

## Next

1. Multi-token dual/MMVQ.  
2. lm_head.  
3. Further residual/elementwise epilogues.
