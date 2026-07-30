# Ship note — Hybrid router **mode7** (stock sum/clamp + fused DIV+SCALE) 2026-07-30

## Status: **SCORED TIP** (default hybrid mode)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior tip fused sigmoid+add (mode2) | 1148.9 | 120.2 | +9.09% | OK |
| **mode7 default** | **1143.9** | **121.1** | **+9.57%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T061617Z/` · probe (env=7): `results/20260730T061437Z/`

Hit:
```
[lx-control-topk-moe] hybrid mode=7 stock-sum+fused-div-scale nelt=8    # decode
[lx-control-topk-moe] hybrid mode=7 stock-sum+fused-div-scale nelt=4096 # prefill
```

## What

After fused sigmoid+add + gather (unchanged), replace stock **DIV + SCALE** with one kernel using the **stock CLAMP sum** as divisor:

```
q = get_rows_reshaped[i] / clamp_sum[token]
div_out[i] = q
scale_out[i] = q * scale + bias
```

Stock path kept for: reshape(get_rows) → sum_rows → clamp.  
Skipped: DIV, reshape(div), SCALE (fused).

Default: `GGML_SYCL_TOPK_MOE_HYBRID_MODE` unset → **7**.  
Fallback mode2: `GGML_SYCL_TOPK_MOE_HYBRID_MODE=2`.

## Why it wins

Decode-weighted score: ~**+0.9 tg128** vs prior tip with floors OK (prefill still ≥0.95× pin).  
One fewer kernel launch per MoE layer (DIV+SCALE → one fused) while divisor is stock-bitexact.

## What still fails (not shipped)

| mode | note |
|------|------|
| 6 warp-sum full norm | **golden FAIL** (even matching k_sum_rows butterfly) |
| pure sequential fused sum/div | golden FAIL (prior fires) |

## Tip stack (default ON)

1. MoE dual SwiGLU  
2. Hybrid **mode7** + fused sigmoid+add  
3. Dense dual shexp  
4. MoE down weighted reduce  
5. Device mmid counting-sort + prefix + event-wait  

## Next

1. Bitexact multi-token dual/MMVQ.  
2. Warp-sum full norm oracle (mode6 still fails — layout/broadcast?).  
3. lm_head / residual decode levers.
