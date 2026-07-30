# Research note — MoE dual+down multi-token fuse (2026-07-30)

## Status: **RESEARCH / opt-in only** — segfaults when enabled; tip unchanged

| Config | result |
|--------|--------|
| tip dual multi-token GEMM (default) | golden OK, **+47.92%** tip |
| `GGML_SYCL_ENABLE_MOE_DUAL_DOWN=1` | **SEGFAULT** mid-prefill |
| dual+down disabled (default) | stable |

## Goal

Fuse `gate+up dual + down mul_mat_id + weighted reduce` for multi-token so one
device-sort/pack serves the whole expert FFN (skip glu materialize + second sort).

## What was built

1. Graph match: `MMID×2 + GLU + MMID + MUL + VIEW×k + ADD×(k-1)` in `fuse_moe_dual_swiglu`.
2. **Compose path** (current): dual multitoken GEMM → temp glu → stock `mul_mat_id` down → weighted reduce.
3. Earlier **fused expert-loop** path (gate/up/swiglu/down per expert) also segfaulted.

Enable research:
```bash
export GGML_SYCL_ENABLE_MOE_DUAL_DOWN=1
```

## Findings

- Pattern match + can_fuse can succeed; entry into dual+down then dies during second
  dual multitoken / down setup (debug: `compose begin` then no further logs).
- With dual+down forced off, multi-token dual GEMM alone is stable (tip).
- Do **not** default ON. Needs root-cause (stack tensor validity for `mul_mat_id`,
  pool lifetime, or double-fuse interaction) before re-try.

## Tip unchanged

`+47.92%` dual multi-token expert-batched GEMM (`20260730T092534Z`).

## Next

1. Fix dual+down: use graph tensors (not stack shells) for down `mul_mat_id`, or
   true expert-loop without second dual call.
2. lm_head (decode-weighted).
3. Avoid thrashing dual+down without a fix theory.
