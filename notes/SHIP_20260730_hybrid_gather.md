# Ship note — Hybrid router mode1 gather fuse (2026-07-30)

## Status: **SCORED TIP** (default ON with dual SwiGLU)

| arm | pp512 | tg128 | score vs pin | golden |
|-----|------:|------:|-------------:|:------:|
| **dual + hybrid m1 (default ON)** | **1142.6** | **110.2** | **+2.09%** | **OK** |
| dual + hybrid m1 (earlier formal) | 1146.7 | 110.3 | +2.25% | OK |
| dual only (same binary A/B) | 1146.7 | 109.9 | +1.97% | OK |
| baseline pin | 1139.2 | 107.4 | 0 | — |

Formal tip artifact: `results/20260730T040034Z/` · `LATEST_SCORE.json`

## What shipped

Hybrid Laguna router under pattern  
`SIGMOID → RESHAPE → ADD(bias) → ARGSORT → VIEW → GET_ROWS → [norm] → [scale]`:

| Stage | Implementation |
|-------|----------------|
| sigmoid / add / argsort | **stock** `ggml_sycl_compute_forward` (bitexact selection) |
| get_rows (gather) | **fused** `router_gather_kernel` |
| sum-norm / scale | **stock** ops |

Default: **ON**. Modes via `GGML_SYCL_TOPK_MOE_HYBRID_MODE`:
- `0` = full stock-oracle
- `1` = fused gather + stock norm (**default**, golden OK)
- `2` = fused gather-norm (~+3 tg probe, **golden FAIL** — research only)

Kill:
- `GGML_SYCL_ENABLE_TOPK_MOE_BIAS=0` — disable hybrid bias path
- `GGML_SYCL_DISABLE_TOPK_MOE=1` — hard kill all topk-moe fuse
- `GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU=1` — dual off

Hit log: `[lx-control-topk-moe] laguna bias fuse HIT (hybrid mode=1: ...)`

## Isolation results

| Experiment | golden | note |
|------------|:------:|------|
| Full stock-oracle | OK | no kernel win |
| Fused gather only + stock norm | **OK** | **shipped** |
| Fused gather-norm (sequential sum, true div, scale s/b) | FAIL | params correct (k=8, scale=2.5, clamp=6.1e-5); still wrong tokens |
| Prior bitonic full fuse | FAIL | + slower |

**Lesson:** gather is bitexact; fused **norm** remains a golden surface (not selection). Small formal tg lift from gather fuse alone (~0.3–0.4 tok/s).

## Next fire

1. Fix mode2 gather-norm bitexact (real ~+3 tg headroom) — compare intermediate tensors vs stock sum/div.
2. Or **MoE down fuse / dense dual shared expert** on control (independent).

## Code / patches

- Control: `treebeard-base-control-latest` `ggml/src/ggml-sycl/topk-moe.cpp`, `ggml-sycl.cpp`
- `patches/0001-control-q4k-moe-dual-swiglu.patch`, `patches/0003-control-topk-moe-bias-optin.patch`
