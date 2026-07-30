# Research — true top-k full-norm (clamp+div+scale in-kernel) 2026-07-30

## Status: **opt-in research only** — golden FAIL when defaulted; tip unchanged

| arm | golden | note |
|-----|:------:|------|
| tip true top-k+gather+sum (mode8 elementwise clamp+div+scale) | **OK** | +51.95% formal tip |
| **+ in-kernel norm** (default try) | **FAIL** | mode6-class weight path |
| in-kernel norm opt-in | (not rebench) | code retained |

Enable (research only):
```bash
export GGML_SYCL_ENABLE_ROUTER_TRUE_TOPK_NORM=1
```

## What

Extended true top-k+gather+sum kernel to also write final route weights:

```
den = clamp(warp_sum(sel), min, max)
weights[i] = (sel[i] / den) * scale + bias
```

Skip the separate mode8 elementwise clamp+div+scale launch.

## Findings

- **Golden FAIL** on greedy smoke (token stream diverges) — same class as hybrid mode6
  full fused norm (sum bitexact OK; post-sum weight path sensitive).
- Tip remains mode8 with **topk-sum** + **stock-order elementwise** clamp+div+scale.
- Code path kept opt-in for future bitexact debug (e.g. force write via get_rows reload).

## Tip unchanged

`+51.95%` true top-k+gather+sum (`20260730T112258Z`).

## Next

1. lm_head prune/mask with golden oracle (router tail mostly closed).
2. Debug topk-norm float path vs elementwise (reload gr_row after barrier?) before re-default.
3. Prefill multi-token dual MMVQ golden fix.
