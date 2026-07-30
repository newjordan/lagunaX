# Research note — MoE/dual MMVQ multi-subgroup packing (2026-07-30)

## Status: **NOT SHIPPED** (reverted; golden OK, no tip beat)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| tip fused sigmoid+add | 1148.9 | 120.2 | **+9.09%** | OK |
| all MoE/dual multi-sg sgs=16 | 1135.2 | 120.4 | +8.92% | OK |
| dual-only multi-sg sgs=16 | 1133.9 | 120.3 | +8.77% | OK |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal: `results/20260730T054814Z/` (all), `results/20260730T055106Z/` (dual-only)

## What tried

Dense Q4_0 reorder packs **WARP_SIZE subgroups per WG**. Ported that geometry to:

1. MoE dual SwiGLU (gate+up hot path, nrows=512)
2. MoE reorder mul_mat_id (down)
3. Dense dual shexp
4. Integrated weighted down (opt-in)

Launch: `block_dims(1,1,16*WARP_SIZE)`, row = `group(2)*sgs + sg_id`.

## Findings

- **Golden OK** (integer launch only; same reduce_over_group math).
- Decode flat/noise (~+0.2 tg).
- **Prefill −10…−15 t/s** vs tip → composite score loses.
- Dual-only pack still regresses pp (not just broad reorder).
- **Keep 1-warp/row** for MoE dual / reorder / dense dual on B70.

Comment left in `mmvq.cpp` dual launch so next fire does not re-thrash.

## Tip unchanged

Scored tip remains fused sigmoid+add (`20260730T053204Z`, +9.09%).  
Default stack still includes device mmid prefix-sum infra (`SHIP_20260730_mmid_prefix_sum.md`).

## Next

1. Zero-wait expert dispatch without pack→GEMM host bubble (USM counts / compact non-empty).  
2. Bitexact multi-token dual/MMVQ oracle.  
3. Fused router sum/div (still golden-fail).  
4. Avoid re-trying multi-sg dual without new theory (occupancy trace / different sgs count with A/B).
