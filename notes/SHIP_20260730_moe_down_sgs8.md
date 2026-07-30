# Ship note — MoE down weighted MMVQ **sgs=8** (2026-07-30)

## Status: **SCORED TIP** (default ON)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior true top-k+gather+sum | 3409.5 | 130.1 | +51.95% | OK |
| down sgs=4 env | 3407.5 | 129.7 | +51.54% | OK under |
| down sgs=8 env | 3402.5 | 130.4 | +52.15% | OK |
| down sgs=8 default formal1 | 3419.6 | 129.8 | +51.78% | OK (noise under) |
| **down sgs=8 default formal2** | **3402.1** | **130.5** | **+52.22%** | **OK** |
| dense dual+down residual env | 3389.4 | 129.8 | +51.47% | OK under |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T113629Z/` (confirming rebench; env probe `20260730T113349Z`)

Hit:
```
[lx-control-moe-down] weighted-mmvq n_experts=8 nrows=2048 ncols=512 sgs=8 n_tokens=1
```

## What

Multi-subgroup packing for **integrated weighted MoE down** MMVQ only:

- `sgs=1`: legacy `block_dims(1, MMV_Y, WARP_SIZE)`
- `sgs>1`: `block_dims(1, 1, sgs*WARP_SIZE)`, row = `group*sgs + sg_id`

Default **sgs=8** (nrows=2048 embd divides evenly). Kill: `GGML_SYCL_MOE_DOWN_SGS=1`.

Distinct from dual gate/up multi-sg (`MOE_DUAL_SGS`) which stayed under tip.

## Why

Decode-weighted: down is embd=2048-row GEMV×k experts per sparse layer. Packing 8
subgroups/WG lifts occupancy. Formal **+0.27%** vs prior tip (tg lift primary).

## Tip stack

Prior true top-k+gather+sum + **down MMVQ sgs=8**.

## Also this fire

- Dense dual+down residual remeasure: still under tip (opt-in only).
- Router full-norm: already closed golden-fail.

## Next

1. lm_head prune/mask with golden oracle.
2. Avoid dual multi-sg default (closed).
3. Optional A/B sgs=16 for down only.
