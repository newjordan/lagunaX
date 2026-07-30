# Ship note — hybrid router true top-k (not full argsort 256) 2026-07-30

## Status: **SCORED TIP** (default ON)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior dual+down decode tip | 3422.7 | 128.8 | +50.93% | OK |
| **+ true top-k selection** | **3403.3** | **129.7** | **+51.48%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T105726Z/`

Hit:
```
[lx-control-topk-moe] true top-k (not full argsort) n_experts=256 k=8 n_rows=512
[lx-control-topk-moe] true top-k (not full argsort) n_experts=256 k=8 n_rows=1
[lx-control-topk-moe] hybrid mode=8 stock-sum+fused-clamp+div-scale ...
```

## What

In hybrid Laguna bias path (`ggml_sycl_op_topk_moe_hybrid_bias`), replace stock full DESC
`k_argsort` of **256** experts with **true top-k** (k=8 iterative warp argmax):

1. One warp (sub-group, `WARP_SIZE=16`) per token
2. k iterations of subgroup max with **stable-DESC ties** (prefer lower expert index)
3. Write only `ids[row * n_experts + 0..k-1]` into the full ARGSORT buffer
4. VIEW / fused gather / mul_mat_id only consume those first-k slots

Bitexact aim: matches stable DESC bitonic top-k (equals never swap → lower index first).
Golden smoke **OK**.

Kill: `GGML_SYCL_DISABLE_ROUTER_TRUE_TOPK=1` → stock full argsort fallback.

## Why it wins

Decode-weighted score: skip bitonic of 256 (many local barriers / global passes) per MoE
layer × ~39 sparse layers. Formal **+0.55% score** vs prior tip (~**+0.9 tg**, ~−19 pp noise).

## Tip stack (default ON)

Prior dual+down decode tip stack + **true top-k** in hybrid mode8 router.

## Next

1. lm_head prune/mask only with golden oracle (packing exhausted).
2. Prefill multi-token dual/MMVQ host-sort parity (still golden-FAIL when forced).
3. Avoid re-thrashing mode6 full-norm without new theory.
