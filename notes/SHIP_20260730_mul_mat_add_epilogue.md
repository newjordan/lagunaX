# Ship note — MUL_MAT+ADD residual epilogue (2026-07-30)

## Status: **SCORED TIP** (default ON, decode-only)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior tip softplus×mul | 1158.8 | 127.2 | +14.07% | OK |
| **+ MUL_MAT+ADD** | **1147.3** | **128.2** | **+14.46%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T075349Z/`

Hit:
```
[lx-control-mm-add] fuse hit (mul_mat+add) ne0=2048 ne1=1 wtype=q4_K
```

## What

Fuse Laguna attention residual:

```
o = MUL_MAT(wo, gated_attn)   # decode GEMV Q4_K
ffn_inp = o + inpSA
```

into **one reorder-MMVQ launch** with row addend epilogue:

```
dst[row] = gemv(row) + residual[row]
```

- Decode only (`src1.ne[1]==1`); multi-col path not yet wired for addend
- Types: Q4_K / Q5_K / Q6_K reorder
- Default ON; kill: `GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1`

## Why it wins

~**+1 tg128** formal (decode-weighted). Elides a full embd ADD launch after every o_proj.

## Tip stack (default ON)

1. MoE dual SwiGLU  
2. Hybrid mode7 + sigmoid+add + noop reshape + skip DIV  
3. Dense dual shexp  
4. MoE down k8 + integrated decode-only  
5. Device mmid sort/prefix/event  
6. RMS_NORM+MUL  
7. ADD+ADD residual  
8. softplus×mul attn gate  
9. **MUL_MAT+ADD o_proj residual**

## Next

1. Multi-col MUL_MAT+ADD (prefill o_proj).  
2. shexp_down MUL_MAT+ADD(moe) if graph-adjacent.  
3. Multi-token dual/MMVQ · lm_head.
