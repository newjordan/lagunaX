# Ship — mul_mat+add+add double residual (decode) 2026-07-30

## Status: **SCORED TIP** (default ON; ne11≤32 for double-ADD)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior mm-add alias tip | 3734.7 | 129.7 | +55.10% | OK |
| **+ mul_mat+add+add (ne11≤32)** | **3711.2** | **131.6** | **+56.53%** | **OK** |
| double-ADD all batch (regressed) | 2483.2 | 131.3 | +41.30% | OK under |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T125600Z/`

Hit:
```
[lx-control-mm-add] fuse hit (mul_mat+add+add) ... mm='ffn_shexp-1' add='ffn_out-1' add2='l_out-1'
```

## What

Extend residual-alias `mul_mat+add` so Laguna MoE tail can fuse **three** nodes on decode:

```text
ffn_shexp = shexp_down          # Q6_K MMVQ
ffn_out   = add(moe_out, shexp) # often in-place on moe
l_out     = add(ffn_out, ffn_inp)
→ dst = gemv + moe + ffn_inp   (one MMVQ epilogue, two addends)
```

- `ggml_sycl_mmvq_set_row_addend2` + reorder MMVQ `dst = sum + a1 + a2`
- **Double-ADD only when `src1.ne[1] ≤ 32`** (decode / small batch). Uncapped formal
  crushed prefill (pp 2483) — large-N path not epilogue-safe/efficient.
- Prefill keeps single shexp+moe `mul_mat+add` (alias tip).

Kill: `GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1` (both single and double)

## Why win

Decode **+1.9 tg** vs alias tip (elide second residual ADD × ~39 MoE layers). Prefill
flat/noise vs alias tip. Composite **+1.4 pp**.

## Tip stack

Packed reduce + mul_mat+add residual-alias + **decode mul_mat+add+add**.

## Next

1. Optional tip rebench noise band.
2. Multi-col GEMM residual epilogue → unlock double-ADD for prefill.
3. Attn/FA remaining.
