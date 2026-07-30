# Research — mode9 + dual MMVQ sgs re-probe on true top-k+gather tip (2026-07-30)

## Status: **no tip change** (both under tip; dual sgs default remains 1)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| **tip true top-k+gather** | **3412.8** | **129.9** | **+51.81%** | OK |
| hybrid mode9 (env) | 3377.1 | 130.1 | +51.54% | OK |
| dual MMVQ **sgs=4** default try | 3416.2 | 129.7 | +51.64% | OK |
| dual MMVQ **sgs=2** env | 3370.3 | 129.7 | +51.13% | (bench only) |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal: mode9 `results/20260730T111027Z/` · sgs4 `results/20260730T111333Z/` · sgs2 `results/20260730T111419Z/`

## What

1. **mode9** (`GGML_SYCL_TOPK_MOE_HYBRID_MODE=9`): fused SUM (warp order) + clamp+div+scale on tip stack with true top-k+gather. Golden OK; **pp −36** → score under tip. Keep mode8 default.
2. **dual MMVQ sgs=4** default probe: prefill now uses expert-loop GEMM (not dual MMVQ), so hoped decode packing free of pp tax. Hit `sgs=4`. Golden OK; slight pp lift, **tg −0.24**, score **−0.17%** vs tip. Reverted default to **sgs=1**.
3. **sgs=2** env: worse pp; no ship.

## Code

`mmvq.cpp` still accepts `GGML_SYCL_MOE_DUAL_SGS=2|4|8|16` (nrows divisibility fall-back). Default **1**.

## Tip unchanged

`+51.81%` true top-k+gather (`20260730T110508Z`).

## Next

1. lm_head prune/mask with golden oracle (packing + dual sgs closed).
2. MoE dual+down residual epilogue only if graph-local (shexp is parallel branch — hard).
3. Avoid re-defaulting dual multi-sg without tg lift ≥ tip.
