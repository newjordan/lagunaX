# Ship note — true top-k+gather+**sum** fuse (2026-07-30)

## Status: **SCORED TIP** (default ON with true top-k)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior true top-k+gather | 3412.8 | 129.9 | +51.81% | OK |
| **+ in-kernel sum** | **3409.5** | **130.1** | **+51.95%** | **OK** |
| mode9 separate sum (reprobe) | 3377.1 | 130.1 | +51.54% | OK under tip |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T112258Z/`

Hit:
```
[lx-control-topk-moe] true top-k+gather+sum n_experts=256 k=8 n_rows=512
[lx-control-topk-moe] hybrid mode=8 topk-sum+fused-clamp+div-scale ...
```

## What

Extend true top-k+gather kernel to also write **SUM_ROWS** with k_sum_rows order
(strided lane loads of selected probs + warp butterfly). Mode8 skips stock SUM_ROWS
launch when top-k wrote the sum buffer.

Unlike mode9 (extra sum kernel after gather), sum is free in the selection launch.

Kill: `GGML_SYCL_DISABLE_ROUTER_TRUE_TOPK=1` (reverts selection+gather+sum).

## Why

One fewer kernel per MoE layer vs mode8 stock sum. Golden OK.
Formal **+0.14%** score vs prior tip (~+0.2 tg, pp flat).

## Tip stack

Prior true top-k+gather + **fused sum** on mode8 path.

## Next

1. lm_head prune/mask with golden oracle.
2. Prefill multi-token dual MMVQ golden fix.
3. Avoid re-defaulting dual multi-sg (closed under tip).
