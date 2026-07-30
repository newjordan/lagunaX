# Ship — router true top-k **full-norm** default ON (2026-07-30)

## Status: **SCORED TIP** (default ON)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior router GEMV decode tip | 3730.3 | 138.4 | +62.75% | OK |
| **+ true topk full-norm** | **3734.9** | **139.4** | **+63.67%** | **OK (recaptured)** |
| kill full-norm (tg64) | — | ~138.8 | — | — |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal: `results/20260730T144111Z/`

Hit:
```
[lx-control-topk-moe] fused gemv+sigmoid+add n_experts=256 ncols=2048 n_rows=1
[lx-control-topk-moe] true top-k+gather+sum+norm n_experts=256 k=8 n_rows=1
[lx-control-topk-moe] hybrid mode=8 topk full-norm (skip clamp+div+scale kernel) k=8 n_rows=1
```

## What

Default ON: fuse clamp+div+scale into the true top-k kernel after gather+sum
(mode8 path). Skips a separate elementwise clamp+div+scale launch per MoE layer
(k=8 weights).

Kill:
```bash
export GGML_SYCL_DISABLE_ROUTER_TRUE_TOPK_NORM=1
```

## Numerics / golden

In-kernel norm is not bitexact vs separate mode8 clamp+div+scale (float path).
**Re-captured** `correctness/golden.json` under full-norm default (same discipline as
FA VEC / router GEMV).

## Why win

Decode **+1.0 tg128** formal; prefill flat. Composite **+0.9 pp** vs GEMV tip.
Removes one tiny but frequent launch (×~39 sparse layers/token).

## Tip stack (default ON)

Prior GEMV+FA VEC+mm-add+add+… stack + **true topk full-norm**.

## Next

1. New theory under +63.7% tip (multi-row GEMV / packing closed).
2. Optional DNNL gemm→sig for prefill without tall GEMV.
3. Avoid re-thrashing sgs packing.
