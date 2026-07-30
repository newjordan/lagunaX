# Research — packing / norm re-probes on down-sgs8 tip (2026-07-30)

## Status: **no tip change**

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| **tip down sgs=8** | **3402.1** | **130.5** | **+52.22%** | OK |
| down sgs=16 | 3404.8 | 130.1 | +51.90% | OK under |
| dual gate/up sgs=2 | 3394.3 | 130.4 | +52.03% | (bench) under |
| dense dual sgs=8 | 3406.1 | 130.4 | +52.15% | OK under |
| dense dual sgs=4 | 3428.1 | 130.2 | +52.22% | OK flat/under (score) |
| topk full-norm (gr reload) | — | — | — | **FAIL** |

Formals: `20260730T114020Z` (down16), `20260730T114108Z` (dual2), `20260730T114501Z`/`114541Z` (dense dual).

## What

1. **MOE_DOWN_SGS=16** — golden OK, score under tip (keep default 8).
2. **MOE_DUAL_SGS=2** — still under tip with down-sgs8 stack.
3. **DENSE_DUAL_SGS** infrastructure added (default 1). sgs=4/8 golden OK, no tip beat.
4. **topk-norm reload fix** (barrier + load gr/sum from global) — still **golden FAIL**.

## Code retained

- `GGML_SYCL_MOE_DOWN_SGS` default 8 (tip)
- `GGML_SYCL_DENSE_DUAL_SGS` opt-in (default 1)
- `GGML_SYCL_ENABLE_ROUTER_TRUE_TOPK_NORM=1` still golden-unsafe

## Tip unchanged

`+52.22%` down sgs=8 (`20260730T113629Z`).

## Next

1. lm_head prune/mask design (packing class exhausted for MoE/dense dual).
2. Prefill expert-loop host-count wait reduction.
3. Avoid further multi-sg thrash without new theory.
