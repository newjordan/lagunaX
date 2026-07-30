# Ship note — true top-k **+gather** fuse (2026-07-30)

## Status: **SCORED TIP** (default ON with true top-k)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior true top-k | 3403.3 | 129.7 | +51.48% | OK |
| gather shape-bug (no fuse) | 3407.5 | 129.7 | +51.58% | OK (noise) |
| **true top-k+gather (fixed)** | **3412.8** | **129.9** | **+51.81%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T110508Z/`

Hit:
```
[lx-control-topk-moe] true top-k+gather n_experts=256 k=8 n_rows=512
[lx-control-topk-moe] true top-k+gather n_experts=256 k=8 n_rows=1
```

## What

Extend hybrid true top-k kernel so selection also writes the **get_rows** buffer
from unbiased sigmoid probs in the same launch (lane0: `ids[k]=e; gr[k]=probs[e]`).

Skips the separate `router_gather_kernel` launch when true top-k is active
(default hybrid mode8).

**Shape fix:** get_rows is `[1,k,n_tokens]` — match `ne[2]==n_rows` and
`nelements==k*n_rows` (first formal used `ggml_nrows==n_rows` which never held;
fuse was dead → noise rebench only).

Kill: `GGML_SYCL_DISABLE_ROUTER_TRUE_TOPK=1`

## Why

One fewer kernel per MoE layer (×~39 sparse) on decode and prefill. Golden OK.
Formal **+0.33%** vs true top-k tip.

## Tip stack

Prior true top-k tip + **fused gather** in selection kernel.

## Next

1. lm_head prune/mask with golden oracle (packing exhausted).
2. Prefill multi-token dual MMVQ host-sort parity (research golden-FAIL).
3. Optional: dual+down residual epilogue for MoE (not dense-only).
