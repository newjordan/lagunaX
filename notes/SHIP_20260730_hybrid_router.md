# Ship note — Hybrid Laguna router (stock-oracle stage) 2026-07-30

> **Superseded for tip by** `SHIP_20260730_hybrid_gather.md` (mode1 default ON).

## Status: **research / opt-in only** — not scored default

| Config | tg32 probe | golden | notes |
|--------|-----------:|:------:|-------|
| dual ON, bias **OFF** (default) | ~110–111 | **OK** | scored tip |
| dual ON, hybrid bias **ON** (stock-oracle) | ~111 | **OK** | fuse hits; **no kernel win** |
| dual ON, custom sigmoid+add+gather (tried) | ~115 | **FAIL** | faster but wrong tokens |
| dual ON, stock-sel + fused gather-norm (tried) | ~114 | **FAIL** | selection OK still golden-fail |
| prior bitonic full fuse | ~97 | **FAIL** | slower + wrong |

## What this fire built

1. **Laguna graph pattern** (confirmed again):
   ```
   SIGMOID → RESHAPE → ADD(exp_probs_b) → ARGSORT → VIEW → GET_ROWS → [norm] → [scale]
   ```
2. **Hybrid fuse entry** under `GGML_SYCL_ENABLE_TOPK_MOE_BIAS=1`:
   - Match full router subgraph (incl. norm/scale)
   - Re-dispatch every real op via **stock** `ggml_sycl_compute_forward` (includes stock `k_argsort`)
   - Hit log: `[lx-control-topk-moe] laguna bias fuse HIT (hybrid stock-oracle)`
3. Exported helpers for next fire:
   - `ggml_sycl_argsort_f32_i32` (stock bitonic path)
   - non-static `ggml_sycl_compute_forward`

## Findings (actionable)

| Experiment | Result |
|------------|--------|
| In-fuse bitonic selection (prior) | golden FAIL + tg regress |
| Fused sigmoid+add + stock argsort + fused gather-norm | ~**+4 tg** probe, golden **FAIL** |
| Stock sigmoid/add/argsort + fused gather-norm | ~+3 tg probe, golden **FAIL** |
| Full stock-oracle hybrid | golden **OK**, tg ≈ dual tip (no win) |

**Conclusion:** Pattern match + fuse skip wiring is correct and bitexact when stock ops run.
The **golden surface is post-argsort weight path** (gather-norm / layout / float), not only selection algorithm.
There is real headroom (~3–5 tg probe) if gather-norm can be made bitexact.

## Scored tip remains

**Control dual SwiGLU only** — formal ~**+1.9%** vs pin (`LATEST_SCORE.json`), golden OK.
Kill: `GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU=1`
Bias hybrid opt-in: `GGML_SYCL_ENABLE_TOPK_MOE_BIAS=1`
Hard kill topk: `GGML_SYCL_DISABLE_TOPK_MOE=1`

## Next fire should pick

1. **Debug gather-norm bitexact** vs stock `get_rows`+`sum_rows`+`clamp`+`div`+`scale` (ids already stock).
2. Or **MoE down fuse** / **dense dual shared expert** on control (independent of router).
3. Do **not** default hybrid ON until golden + formal floors + score ≥ tip.

## Artifacts

- Control tree: `treebeard-base-control-latest` (`topk-moe.cpp`, `ggml-sycl.cpp`)
- Patches: `patches/0001-control-q4k-moe-dual-swiglu.patch`, `patches/0003-control-topk-moe-bias-optin.patch`
- Results: `results/hybrid-final-20260730T034307Z/`, earlier probes under `results/hybrid-*`
