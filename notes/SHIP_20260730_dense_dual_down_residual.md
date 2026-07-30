# Ship note — Dense dual+down+residual (shexp/dense) — 2026-07-30

## Status: **OPT-IN only** (not tip)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| **tip** dual+down decode integrated | **3422.7** | **128.8** | **+50.93%** | OK |
| + dense dual+down+residual (this fire) | 3375.8 | 128.4 | **+50.06%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal (under tip): `results/20260730T103913Z/`

Enable: `GGML_SYCL_ENABLE_DENSE_DUAL_DOWN=1`  
Kill: `GGML_SYCL_DISABLE_DENSE_DUAL_DOWN=1`

Hit (decode only, ncols≤32):
```
[lx-control-dense-dual] fuse hit (dual+down+residual) ncols_dst=1 embd=2048
```

## What

Extend dense dual SwiGLU fuse to **gate+up+SwiGLU+down+ADD(residual)**:

1. Dual fused gate/up/SwiGLU → real `glu`
2. Down MMVQ with `mmvq_set_row_addend(residual)` into ADD buffer
3. `can_fuse` skips 5 nodes (2×MUL_MAT + GLU + MUL_MAT + ADD)

Constraints discovered this fire:
- Graph is contiguous `MUL_MAT,MUL_MAT,GLU,MUL_MAT,ADD` (Laguna shexp after MoE, and dense residual).
- Allocator often **aliases** `down->data == add->data` (in-place ADD on down) — must allow, not reject.
- Residual may also alias ADD (in-place on residual); MMVQ `dst[row]=sum+addend[row]` is safe.
- **Decode only** (`ncols_dst ≤ 32`): residual addend is MMVQ-only; multi-col GEMM has no epilogue.
- Shexp down weights are **Q6_K** (allowed).

## Why not tip

Formal score **−0.87 pp** vs tip (+50.06% vs +50.93%). Prefill path is dual-only (ncols>32 miss); slight pp/tg noise/regression not worth default ON. Golden OK so path is correctness-safe for future GEMM addend / prefill work.

## Tip unchanged

Keep dual+down decode integrated tip (`0031`, `20260730T101935Z`). Dense dual multi-col GEMM still default ON for shexp gate+up.

## Next

1. **lm_head** decode-weighted (Q6_K output.weight) — highest leverage if tip stuck.
2. GEMM multi-col residual addend → unlock dense dual+down for prefill.
3. Hybrid router sigmoid+add+norm + stock argsort (bitexact).
