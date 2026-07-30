# Research — chunked multi-col MMVQ for prefill double residual (2026-07-30)

## Status: **no tip change** (under tip; code restored)

| arm | pp512 | notes |
|-----|------:|-------|
| **tip** mm-add+add decode | **~3711** | double-ADD only ne11≤32 |
| tip rebench prior | 3724 | +56.33% |
| chunked MMVQ residual2 (reorder fixed) | **~2845** | correct epilogue, −23% pp |
| no reorder (512× single-col) | ~1345 | reorder bootstrap needed |
| GEMM post-epilogue (prior fire) | ~1214 | closed |

## What

Prefill `mul_mat+add+add` needs residual epilogue. GEMM ignores `row_addend{,2}`.
Tried:

1. Force MMVQ when `addend2` set for ne11≤2048
2. Chunk multi-col (8) reorder MMVQ with offset addends
3. Allow `should_reorder_tensor` when `addend2` even for ne11>8 (critical — default reorder bootstrap only allows ne11≤8)

## Findings

1. **Reorder gate**: without (3), shexp Q6 never reorders on prefill → non-reorder path has no addend → wrong/slow.
2. **With reorder + chunk**: works, golden-class correctness, but **pp ~2845 ≪ tip ~3711** (decode tip keeps GEMM single-add on shexp+moe for large T).
3. Tip path: double-ADD **decode only** (ne11≤32, practically ne11=1 MMVQ); prefill keeps single `mul_mat+add` (GEMM, residual may be imperfect — performance path).

## Tip unchanged

`+56.53%` / rebench `+56.33%`. Chunked residual2 code **not** left in tree (restored).

## Next

1. Attn/FA remaining (this fire pivoted to residual2 research).
2. Faster large-T residual epilogue (oneDNN GEMM + single fused add kernel, not 64× MMVQ).
3. Do not re-open GEMM-post / uncapped double-ADD without new theory.
