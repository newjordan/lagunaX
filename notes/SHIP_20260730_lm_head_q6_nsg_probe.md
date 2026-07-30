# Research — lm_head Q6_K MMVQ launch packing (2026-07-30)

## Status: **no tip change** (flat vs tip; code reverted)

| arm | note | tg64 (r=2) |
|-----|------|------------:|
| tip (WARP_SIZE=16 sgs on B70) | reorder Q6_K MMVQ | ~129.0–129.2 |
| `nsg=1/2/4/8/16` | large-nrows only | all within ±0.3 of tip |
| `GGML_SYCL_PRIORITIZE_DMMV=1` | global | **~116** (regress) |

`output.weight` is **Q6_K** `[2048 × 100352]` (~168 MB/token GEMV). Already on reorder MMVQ (`reorder_mul_mat_vec_q6_k_q8_1_sycl`).

## What was tried

1. Adaptive fewer subgroups for `nrows ≥ 16k` (opposite of prior “more sgs pack” that also regressed formal).
2. Full A/B via env; **WARP_SIZE on this build is 16** (Intel), not 32 — prior notes saying “32 sgs” were wrong for B70.
3. Global DMMV priority: large decode regress.

## Why no ship

Launch packing does not move serial decode on B70 for this lm_head shape. Bandwidth-bound GEMV is not occupancy-limited by WG packing in the tested range.

## Next (lm_head / decode)

1. **True top-k selection** for router (replace full argsort 256) — bitexact/tie-break design; mode8 still full `k_argsort`.
2. lm_head **prune / candidate mask** only with golden oracle (high risk).
3. Quant/path change for output only (not packing).

## Tip unchanged

`+50.93%` dual+down decode (`20260730T101935Z`).
