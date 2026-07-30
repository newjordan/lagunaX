# Research — GEMM residual post-epilogue for prefill double-ADD (2026-07-30)

## Status: **no tip change** (regressed; code restored to tip)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| **tip mm-add+add decode** | **3711.2** | **131.6** | **+56.53%** | OK |
| tip rebench (after abort) | 3699.× / ~same | ~131 | **+56.33%** | OK |
| GEMM post + double-ADD all-N | **~1214** | — | — | (not formal) |
| prior uncapped double-ADD (no post) | 2483.2 | 131.3 | +41.30% | OK under |

## What was tried

MMVQ only applies in-kernel addends for `ne11 ≤ MMVQ_MAX_BATCH_SIZE` (**8**).
Larger batches take GEMM/MMQ and ignore `row_addend{,2}`.

Attempted unlock of prefill `mul_mat+add+add`:

1. GEMM (no addends) into dst or scratch
2. Post kernel: `dst = gemm + r1 [+ r2]` (alias-safe scratch when residual aliases dst)
3. Double-ADD allowed for all `ne11`

## Result

Prefill **collapsed** (~1214 vs tip ~3711 t/s). Even uncapped double-ADD without post
was already −15 pp score. Tip path restored (double-ADD only `ne11≤32`, MMVQ addends).

## Why (hypotheses)

1. Writing shexp GEMM into **l_out** (skipping intermediate + 2 ADDs) interacts badly with
   allocator/aliasing or MoE graph tail for large T.
2. Extra full-tensor post pass + optional scratch ≈ 2× embd traffic × 39 layers.
3. Not pursued further this fire — need isolated A/B (single POST only vs double-ADD only).

## Tip unchanged

`+56.53%` mul_mat+add+add decode (`20260730T125600Z`); rebench `20260730T130445Z` ~+56.33%.

## Next

1. Isolated: single-ADD GEMM post only (correctness) without double-ADD prefill.
2. Or raise MMVQ multi-col cap for residual path only (hard).
3. Attn/FA remaining.
