# Ship note — MoE down **integrated decode-only** (2026-07-30)

## Status: **SCORED TIP** (default ON for n_tokens==1)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior tip (noop reshape skip) | 1140.4 | 121.4 | +9.71% | OK |
| same-binary tip (skip_div only) | 1139.9 | 121.7 | +9.88% | OK |
| **+ integrated decode-only** | **1147.6** | **122.4** | **+10.55%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T072139Z/`

Hit:
```
[lx-control-moe-down] fuse hit (integrated weighted-mmvq) embd=2048 k=8 tokens=1
[lx-control-topk-moe] hybrid mode=7 ... skip_div_store=1
```

## Root cause of prior golden FAIL

Integrated path was previously applied for **all** `n_tokens` including prefill (`ne12>1`).

- Tip multi-token `mul_mat_id` uses **host counting-sort** GEMM regroup (fused batch OFF).
- Integrated used **device fused reorder** for multi-token → different numerics → greedy golden FAIL.
- One-kernel math for **decode n_tokens==1** was never the real bug (oracle reorder+reduce also OK once gated).

## What shipped

### 1. Integrated weighted-MMVQ (decode only) — default ON

In `ggml_sycl_fuse_moe_down_weighted`:

- Gate: `n_tokens==1 && ne12==1` (+ Q4/5/6_K reorder, contiguous acts, etc.)
- Quantize SoA + one-kernel `mul_mat_vec_q_id_weighted_reorder` → scratch=`mmid`, dst=fuse out
- Prefill falls through to two-step `mul_mat_id` + k8 weighted reduce (unchanged)

Kill:
```bash
export GGML_SYCL_DISABLE_MOE_DOWN_INTEGRATED=1
# or
export GGML_SYCL_ENABLE_MOE_DOWN_INTEGRATED=0
```

Oracle (two-step GEMV into mmid + reduce, still decode-only):
```bash
export GGML_SYCL_MOE_DOWN_INTEGRATED_MODE=1
```

### 2. Hybrid mode7 skip DIV intermediate store

Fuse outputs are VIEW(ids) + SCALE(weights). DIV buffer is intermediate — skip store when
`div.data != scale.data` (saves bandwidth). Golden OK; alone ~noise vs tip.

## Tip stack (default ON)

1. MoE dual SwiGLU  
2. Hybrid mode7 + fused sigmoid+add + noop reshape skip + **skip DIV store**  
3. Dense dual shexp  
4. MoE down weighted reduce k=8 unroll  
5. **Integrated weighted-mmvq decode-only**  
6. Device mmid sort/prefix/event (prefill infra)

## Next

1. Multi-token dual/MMVQ bitexact (same class: must match host-sort multi-token mmid).  
2. lm_head / residual decode.  
3. Avoid re-enabling integrated for ne12>1 without matching stock multi-token path.
