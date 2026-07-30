# Ship note — Laguna topk-moe bias fuse (2026-07-30)

## Status: **opt-in only** (not scored default)

| Config | pp512 | tg128 | golden | score vs pin |
|--------|------:|------:|:------:|-------------:|
| dual ON, topk bias **OFF** (default tip) | ~1135–1145 | ~110 | **OK** | **~+1.6–2.0%** |
| dual ON, topk bias **ON** (A/B ub4k) | 1140 | **115.1** | **FAIL** | ~+5.4% raw |
| topk OFF (same binary) | 1145 | 110.1 | OK | ~+2.0% |

Absolute-speed probe (not golden-safe): **+~5 tg** when bias fuse enabled.

## What it does

Fuses Laguna router:

```
SIGMOID → ADD(exp_probs_b) → ARGSORT → VIEW → RESHAPE(probs) → GET_ROWS
[+ weight norm + scale]
```

into one warp top-k kernel (CUDA `has_bias` semantics: **select** with `wt+bias`, **emit** unbiased `wt`).

Patch: `patches/0003-control-topk-moe-bias-optin.patch`  
Also rolled into refreshed `patches/0001-control-q4k-moe-dual-swiglu.patch` (dual + topk).

## Enable

```bash
export GGML_SYCL_ENABLE_TOPK_MOE_BIAS=1
# hard kill:
export GGML_SYCL_DISABLE_TOPK_MOE=1
```

## Why not default ON

Teacher-forced golden **mismatches** with fuse ON. Iterative warp argmax (tie-break: lower expert id) ≠ full `ggml_argsort` DESC on some layers → different expert set → different tokens.

Matches the M5 note’s lesson: restructured selection is transfer-risky even when “equivalent” in spirit.

## Next for this lever

1. Bitexact path: implement top-k that matches ggml_argsort total order (or use a device full-sort of 256 + view — still cheaper than unfused multi-op).  
2. Or recapture golden under ENABLE=1 as a **separate** absolute-limit track (not scored vs pin without contract bump).  
3. Keep default tip = dual only for mlx.fast-shaped score.

## Artifacts

- A/B: `results/topk-bias-fuse-20260730T024717Z/`  
- Formal dual tip: `results/LATEST_SCORE.json` / `results/20260730T025131Z/`
