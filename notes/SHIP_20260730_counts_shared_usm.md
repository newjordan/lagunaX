# Research — shared-USM / copy-q counts for expert-loop (2026-07-30)

## Status: **no tip change** (both regressed; tip path restored)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| **tip packed reduce** | **3540.3** | **130.2** | **+53.41%** | OK |
| dual sgs=8 | 3538.9 | 129.9 | +53.20% | OK under |
| counts copy-q (prior fire) | 3367.4 | 129.8 | +51.22% | OK regress |
| **shared-USM counts default ON** | **3128.9** | **130.0** | **+48.64%** | OK **regress** |
| tip recheck after revert | 3511.4 | 130.5 | +53.40% | OK ~tip |

Formals: `20260730T121219Z` (shared), `20260730T121642Z` (recheck).

## What

1. **Shared USM hist** (`malloc_shared`) so host reads counts after `e_count` without D2H,
   hoping to build offsets while pack runs. **~−400 pp** — B70 shared-USM atomics/coherence
   tax on the hot hist path.
2. Pool LIFO bug when hoisting counts alloc across inv_alloc (crash) — fixed by reverting
   to scoped local pool allocs + D2H wait before destroy.
3. Code tip path: original-style device sort + D2H after fill (packed reduce unchanged).

## Tip unchanged

`+53.41%` packed reduce (`20260730T115251Z`).

## Next

1. lm_head prune/mask with golden oracle (prefill counts D2H experiments closed).
2. Avoid USM-shared for device atomics on B70 without microbench.
3. Compact active-expert host list (micro) only if profiled.
