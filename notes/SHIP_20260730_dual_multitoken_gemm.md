# Ship note — MoE dual multi-token **expert-batched GEMM** (2026-07-30)

## Status: **SCORED TIP** (default ON; extends dual to prefill)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior tip prefill moe-down weighted | 3148.4 | 128.9 | +47.88% | OK |
| **+ dual multi-token expert-batched GEMM** | **3167.7** | **128.7** | **+47.92%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T092534Z/`

Hit:
```
[lx-control-moe-dual] multi-token expert-batched dual GEMM n_tokens=512 k=8 n_experts=256
[lx-control-moe-dual] fuse hit (gate+up+swiglu)
# decode unchanged:
[lx-control-moe-dual] n_experts=8 nrows=512 ncols=2048 sgs=1
```

## What

Replace research per-token **MMVQ** multi-token dual (golden-fail class) with **expert-batched dual GEMM**:

1. One device counting-sort + activation pack (shared gate/up).
2. Per expert with rows: stock `mul_mat` gate + stock `mul_mat` up on packed rows.
3. Elementwise SwiGLU (`sycl::exp` silu, stock order) into contig buffer.
4. One scatter into glu layout `[ne0,k,T]`.

Decode (`ne12==1`) still uses reorder MMVQ dual (unchanged tip path).

Default **ON** for multi-token. Kill:
- Prefill dual only: `GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1`
- Full dual: `GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU=1`
- Research MMVQ multi-token: `GGML_SYCL_ENABLE_MOE_DUAL_MULTITOKEN_MMVQ=1`

## Why it is only a small score move

GEMMs still dominate; win is avoiding **duplicate** sort+pack for gate then up (and one scatter vs two + separate swiglu graph node). Formal ~**+19 pp** / flat tg / **+0.04% score** vs prior tip — noise-adjacent but golden-safe and unlocks further dual epilogue work.

## Tip stack (default ON)

Prior prefill moe-down weighted tip + **multi-token dual expert-batched GEMM**.

## Next

1. Stream swiglu / fuse down after dual for multi-token.  
2. lm_head.  
3. Avoid per-token MMVQ multi-token default (still research-only).
