# Ship note — ROPE+VIEW+SET_ROWS fuse (2026-07-30)

## Status: **SCORED TIP** (default ON)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior tip mode8 | 3015.2 | 128.6 | +46.08% | OK |
| **+ rope+set_rows** | **3005.5** | **128.9** | **+46.18%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T084016Z/`

Hit:
```
[lx-control-rope] fuse hit (rope+view+set_rows) mode=2 ne0=128 ne1=8 type=f16
```

## What

1. **Wire** existing `ggml_sycl_rope_fused` into graph fuse (CUDA already had this).
2. **ISWA expand order fix** in `llama-graph.cpp`: expand `k_cur` **after** `v_cur` so `ROPE(k)` is adjacent to `VIEW+SET_ROWS` (was q→k→v, blocking fuse).

Pattern: `ROPE → VIEW → SET_ROWS` writes roped K straight into F16 KV cache.

Kill: `GGML_SYCL_DISABLE_ROPE_SET_ROWS_FUSE=1`

## Why it wins

Decode: one fewer full K rope intermediate write + set_rows scatter per layer. ~**+0.3 tg** formal.

## Tip stack (default ON)

Prior stack + **rope+set_rows KV fuse** + iswa k-last expand.

## Next

1. V path / other rope fuses if graph allows.  
2. Multi-token dual/MMVQ.  
3. lm_head.
