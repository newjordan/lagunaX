# Research note — hybrid mode9 fused sum (2026-07-30)

## Status: **opt-in research** — not tip (no score beat)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| **tip mode8** | **3015.2** | **128.6** | **+46.08%** | OK |
| mode9 fused sum | 3002.5 | 128.6 | +45.87% | **OK** |
| dual multi-token ON | — | — | — | **FAIL** |
| lm_head large-nrows pack | 2980 | 128.5 | +45.51% | OK (no tip beat) |

## mode9

```bash
export GGML_SYCL_TOPK_MOE_HYBRID_MODE=9
```

After gather: launch `router_sum_rows_kernel` matching `k_sum_rows_f32` (warp reduce), then mode8-style clamp+div+scale. Skips stock SUM_ROWS compute_forward.

**Golden OK** — proves fused sum bitexact vs stock when writing sum buffer. Score flat/slightly under mode8 (dispatch of stock sum already cheap for k=8).

## Also tried this fire

1. **Dual multi-token** (`ENABLE_MOE_DUAL_MULTITOKEN=1`): golden **FAIL** (still).
2. **lm_head MMVQ pack** (more subgroups for nrows≥16k): golden OK, **score regress** vs tip — reverted.

## Tip unchanged

mode8 default stack. mode9 available via env for further A/B.

## Next

1. lm_head design beyond pack (quant path / prune).  
2. Multi-token dual/MMVQ host-sort parity.  
3. Avoid re-thrashing mode6 full-norm / multi-sg dual without new theory.
