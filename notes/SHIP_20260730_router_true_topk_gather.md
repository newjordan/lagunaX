# Ship note — true top-k **+gather** fuse (2026-07-30)

## Status: **SCORED TIP** (default ON with true top-k)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior true top-k | 3403.3 | 129.7 | +51.48% | OK |
| **+ gather fuse** | **3407.5** | **129.7** | **+51.58%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T110313Z/`

Hit:
```
[lx-control-topk-moe] true top-k+gather n_experts=256 k=8 n_rows=512
[lx-control-topk-moe] true top-k+gather n_experts=256 k=8 n_rows=1
```

## What

Extend hybrid true top-k kernel so selection also writes the **get_rows** buffer
from unbiased sigmoid probs in the same launch (lane0: `ids[k]=e; gr[k]=probs[e]`).

Skips the separate `router_gather_kernel` launch when true top-k is active
(default hybrid mode8). Stock argsort fallback still uses standalone gather.

Kill true top-k (+gather): `GGML_SYCL_DISABLE_ROUTER_TRUE_TOPK=1`

## Why

One fewer kernel per MoE layer (×~39 sparse) on decode and prefill. Golden OK.
Formal **+0.10%** vs prior tip (both tg/pp slight lift).

## Tip stack

Prior true top-k tip + **fused gather** in selection kernel.

## Next

1. lm_head prune/mask with golden oracle (packing exhausted).
2. Prefill multi-token dual MMVQ host-sort parity (research golden-FAIL).
3. Optional: dual+down residual epilogue for MoE (not dense-only).
