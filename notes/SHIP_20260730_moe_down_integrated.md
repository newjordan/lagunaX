# Research note — integrated weighted-MMVQ MoE down (2026-07-30)

## Status: **research / opt-in only** — not scored tip

| Config | tg128 formal | score | golden |
|--------|-------------:|------:|:------:|
| two-step moe-down (tip) | ~118–119 | ~+7.4–8.2% | **OK** |
| integrated weighted-mmvq | ~119.8 | ~+8.6% | **FAIL** |

## What

One-kernel MoE down for Laguna decode:

```
for each embd row:
  sum_e GEMV(down_expert[e], act[e]) * route_weight[e]  → dst
```

- Q4_K/Q5_K/Q6_K reorder MMVQ, k≤16 (unrolled k=8)
- Ordered `volatile` weight multiply (same contract as two-step)
- Wired into existing `fuse_moe_down_weighted` before two-step fallback

Enable research:
```bash
export GGML_SYCL_ENABLE_MOE_DOWN_INTEGRATED=1
```

Hit when ON: `[lx-control-moe-down] fuse hit (integrated weighted-mmvq) embd=2048 k=8 tokens=1`

## Findings

- Integrated **hits** and is slightly faster (~+1–1.5 tg probe/formal) but **golden diverges**.
- Two-step (`mul_mat_id` + weighted reduce) remains the golden-safe tip.
- Float surface likely in reduce_over_group / weight broadcast vs separate GEMV then reduce.

## Tip unchanged

dual + hybrid m1 + dense dual + **two-step moe-down** (defaults ON).

## Next

1. Bitexact integrated: compare per-expert GEMV vs stock mmid row-by-row before weight sum.
2. Hybrid mode2 gather-norm.
3. Multi-token mmid oracle fix.
