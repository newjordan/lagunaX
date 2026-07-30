# Ship note — hybrid fused sigmoid+add (2026-07-30)

## Status: **SCORED TIP** (default ON)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior tip (device mmid sort) | 1144.8 | 118.6 | +7.91% | OK |
| **+ fused sigmoid+add** | **1148.9** | **120.2** | **+9.09%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal: `results/20260730T053204Z/` · `LATEST_SCORE.json`

Hit:
```
[lx-control-topk-moe] fused sigmoid+add n_experts=256 n_rows=1   # decode
[lx-control-topk-moe] fused sigmoid+add n_experts=256 n_rows=512 # prefill
```

## What

In hybrid Laguna bias path (mode2), replace separate stock `SIGMOID` + `ADD(bias)` launches with one kernel:

```
s = 1 / (1 + exp(-logits[i]))   // sycl::exp, matches stock op_sigmoid f32
sig_out[i] = s                  // unbiased probs → gather / get_rows
add_out[i] = s + bias[i % n_experts]  // selection scores → stock argsort
```

Stock `ARGSORT` unchanged. Gather + stock sum/div + fused scale unchanged.

Kill: `GGML_SYCL_DISABLE_ROUTER_SIGMOID_ADD=1` (falls back to stock sigmoid then add).

## Why it works

- Elementwise, same formulas as stock unary + binbcast ADD → golden OK.
- Saves one kernel launch per MoE layer (×~39 sparse layers × tokens).
- Decode-weighted score moves ~+1.2 pts formal.

## Tip stack (all default ON)

1. MoE dual SwiGLU  
2. Hybrid mode2 (gather + fused scale)  
3. **Fused sigmoid+add** (this)  
4. Dense dual shexp  
5. MoE down weighted  
6. Device mmid counting-sort (prefill)

## Next

1. Device prefix-sum / drop remaining counts wait on multi-token mmid.  
2. Bitexact multi-token dual/MMVQ (still golden-fail).  
3. Fused sum/div router norm (still golden-fail).
