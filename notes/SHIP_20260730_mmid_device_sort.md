# Ship note — multi-token mul_mat_id device counting-sort (2026-07-30)

## Status: **DEFAULT ON** (golden OK; score ≈ tip, no formal beat)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior tip hybrid mode2 | 1141.4 | 118.8 | **+7.94%** | OK |
| **+ device counting-sort** | **1144.8** | **118.6** | **+7.91%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal: `results/20260730T052508Z/` · hit  
`[lx-control-mmid] device counting-sort n_tokens=512 k=8 n_experts=256`

## What

For `mul_mat_id` when `ne12>1` (prefill), replace:

1. full ids D2H + `stream->wait()`
2. host counting-sort
3. H2D row mapping  

with:

1. device histogram (atomics over ids on GPU)
2. small counts D2H (`n_experts` ints) + wait
3. host exclusive scan → offsets
4. device fill of `mmid_row_mapping`
5. **same** contiguous copy → expert GEMM loop → scatter

Keeps stock multi-token **GEMM regroup** numerics (not MMVQ batch — that golden-fails).  
Kill: `GGML_SYCL_DISABLE_MMID_DEVICE_SORT=1`

## Why not a score tip bump

Prefill +~3 t/s and decode −0.2 within formal noise; composite score slightly below prior tip stamp.  
Still ships as default: golden-safe, removes full-ids host sort, opens further device-side mapping work.

## Relation to other multi-token research

| path | golden | note |
|------|:------:|------|
| `ENABLE_MMID_FUSED_BATCH` (per-token MMVQ) | FAIL | different numerics vs GEMM regroup |
| `ENABLE_MOE_DUAL_MULTITOKEN` | FAIL | same class |
| **device counting-sort (this)** | **OK** | GEMM path preserved |

## Tip stack (unchanged levers)

1. MoE dual SwiGLU (decode)  
2. Hybrid mode2  
3. Dense dual shexp  
4. MoE down weighted  
5. **Device mmid counting-sort (prefill)**  

## Next

1. Remove remaining counts wait via device prefix-sum + deferred expert dispatch.  
2. Bitexact multi-token dual/mmvq only after golden oracle vs GEMM rows.  
3. Fused sum/div router norm (still golden-fails).
