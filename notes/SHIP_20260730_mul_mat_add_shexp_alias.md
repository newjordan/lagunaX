# Ship — mul_mat+add residual **alias allow** (Q6 shexp down fuse) 2026-07-30

## Status: **SCORED TIP** (default ON; prior mul_mat+add path)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior packed reduce tip | 3540.3 | 130.2 | +53.41% | OK |
| **+ residual-alias mul_mat+add** | **3734.7** | **129.7** | **+55.10%** | **OK** |
| dense dual+down residual opt-in (remeasure) | 3506.0 | 129.6 | +52.51% | OK under |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T124637Z/`

Hit:
```
[lx-control-mm-add] fuse hit (mul_mat+add) ... wtype=q4_K mm='attn_o_proj-0' add='ffn_inp-0'
[lx-control-mm-add] fuse hit (mul_mat+add) ... wtype=q6_K alias_res=1 mm='ffn_shexp-1' add='ffn_out-1'
```

## What

`ggml_sycl_fuse_mul_mat_add` previously **rejected** `residual->data == add->data`
(in-place ADD on residual). Laguna MoE layers do:

```text
ffn_shexp = shexp_down  # Q6_K MMVQ
ffn_out   = add(moe_out, ffn_shexp)  # often in-place on moe_out
```

So Q6 shexp down never fused; only Q4 o_proj hit.

**Fix:** allow residual alias. MMVQ epilogue `dst[row] = sum + addend[row]` is
row-parallel and correct when `addend == dst` (in-place residual).

Also: tolerate a few VIEW/NONE nodes between MUL_MAT and ADD; return skip span
through ADD.

Kill: `GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1`

## Why win

Prefill **+194 t/s** primary (shexp down+add elides intermediate write every MoE
layer × tokens). Decode flat/noise (−0.5 tg). Composite **+1.7 pp** vs packed tip.

## Tip stack

Packed reduce + **mul_mat+add residual-alias (Q6 shexp)**.

## Closed this fire

- Dense dual+down residual remeasure under tip: still under (opt-in only).
- ggml re-trace under tip → shexp Q6 down as remaining fuse gap.

## Next

1. Confirm rebench noise band if needed.
2. Attn / FA share from re-trace (rope already fused).
3. Avoid re-thrashing counts-sync / lm_head prune.
