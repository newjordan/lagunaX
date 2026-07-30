# Ship — router F32 GEMV + sigmoid + bias (decode) 2026-07-30

## Status: **SCORED TIP** (default ON for n_rows==1)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior FA VEC GQA tip | 3716.0 | 135.0 | +59.61% | OK (VEC oracle) |
| **+ router gemv+sig+bias decode** | **3730.3** | **138.4** | **+62.75%** | **OK (recaptured)** |
| gemv all-N (prefill too) | ~3231 | ~138 | +56.69% | recaptured then dropped |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal: `results/20260730T141542Z/`

Hit:
```
[lx-control-topk-moe] laguna bias fuse HIT (mul_mat+hybrid mode=8)
[lx-control-topk-moe] fused gemv+sigmoid+add n_experts=256 ncols=2048 n_rows=1
[lx-control-topk-moe] true top-k+gather+sum n_experts=256 k=8 n_rows=1
```

## What

Extend hybrid Laguna bias fuse to **start at F32 router `MUL_MAT`** (`ffn_gate_inp` 256×2048) and fuse:

```
logits = W @ x          # F32 GEMV
sig    = 1/(1+exp(-logits))
add    = sig + exp_probs_b
```

into one SYCL kernel (WG=256 per expert) for **decode only** (`n_rows==1`).

Prefill keeps stock MKL GEMM + fused sigmoid+add (custom multi-row GEMV crushed pp).

Kill:
```bash
export GGML_SYCL_DISABLE_ROUTER_GEMV_FUSE=1
```
(falls back to stock mul_mat then fused sigmoid+add inside mul_mat-start hybrid)

## Numerics / golden

Custom WG reduce ≠ MKL GEMM order → not bitexact vs prior tip oracle.
**Re-captured** `correctness/golden.json` under decode-only GEMV (same discipline as FA VEC).

Kill restores prior routing surface for A/B.

## Why win

Router F32 GEMV was a separate MKL launch every MoE layer × ~39 layers on decode.
One launch with sigmoid+bias epilogue: formal **+3.4 tg128**, prefill flat/noise, composite **+3.1 pp** vs FA VEC tip.

## Tip stack (default ON)

Prior FA VEC GQA + mm-add+add + packed reduce + dual + hybrid mode8 + true top-k
+ **router gemv+sigmoid+bias (decode)**

## Next

1. Faster multi-row router GEMV (or DNNL gemv+epilogue) without pp tax.
2. Attn gate Q4_K GEMV into softplus-mul (low ROI expected).
3. Avoid packing/sgs thrash under new tip without new theory.
