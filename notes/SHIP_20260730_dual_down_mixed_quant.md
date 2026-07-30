# Ship note — dual+down expert-loop **all layers** (mixed Q4/Q6) (2026-07-30)

## Status: **SCORED TIP** (default ON)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior expert-loop (partial layers) | 3266.0 | 128.9 | +49.29% | OK |
| **+ all-layer expert-loop (mixed quant)** | **3380.5** | **128.7** | **+50.40%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T100449Z/`

Hit (first MoE layer now expert-loop, no dual-only fallback):
```
[lx-control-moe-dual] multi-token dual+down EXPERT-LOOP n_tokens=512 k=8 n_experts=256
[lx-control-moe-dual] fuse hit (dual+down multi-token) tokens=512 k=8
```

## What

Laguna Q4_K_M uses **Q4_K gate/up** and **Q6_K down**. Dual+down fuse required
`down_w->type == gate_w->type`, so the first sparse layer (and any mixed layer)
fell back to dual-only + separate two-step down.

Fix:
1. Fuse `types_ok`: allow any K-quant for gate and down independently.
2. Expert-loop: require gate==up type; allow down different K-quant.

## Why it wins

~**+114 pp** formal by running expert-loop on **every** sparse layer (was missing
first MoE layer each prefill). Decode flat.

## Tip stack

Prior dual+down expert-loop + **mixed-quant all-layer** fix.

Kill switches unchanged.

## Next

1. lm_head (decode-weighted).  
2. Dense shexp multi-col GEMM dual.
