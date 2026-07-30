# Research note — Hybrid mode8 (stock sum + fused clamp+div+scale) 2026-07-30

## Status: **opt-in research** — not scored tip (score ≈ tip, no formal beat)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| **tip mode7** | **1143.9** | **121.1** | **+9.57%** | **OK** |
| mode8 (env) | 1138.4 | 121.3 | +9.54% | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal: `results/20260730T062159Z/`

Enable:
```bash
export GGML_SYCL_TOPK_MOE_HYBRID_MODE=8
```

Hit: `[lx-control-topk-moe] hybrid mode=8 stock-sum+fused-clamp+div-scale nelt=...`

## What

Extends mode7 by also folding **stock CLAMP** into the fused DIV+SCALE kernel:

```
den = clamp(sum_rows[token], min, max)   // match stock op_clamp
div_out[i] = num[i] / den
scale_out[i] = div_out[i] * s + b
```

Stock still runs: reshape(get_rows) → sum_rows.  
Skipped: CLAMP, DIV, reshape(div), SCALE.

## Findings

- **Golden OK** (clamp is pure fmin/fmax on stock sum buffer).
- Decode ~flat/noise vs mode7 (+0.15 tg).
- Prefill ~−5 t/s vs mode7 → composite **does not beat tip**.
- Mode6 (warp full norm, no stock sum) still **golden FAIL**.

## Tip unchanged

Default remains **mode7**. Mode8 available via env for further A/B.

## Next

1. Bitexact multi-token dual/MMVQ.  
2. Integrated weighted-MMVQ down golden fix.  
3. Avoid promoting mode8 without prefill recovery.
