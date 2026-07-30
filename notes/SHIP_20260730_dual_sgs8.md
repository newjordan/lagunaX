# Research note — MoE dual multi-sg **sgs=8** (2026-07-30)

## Status: **NOT default** (golden OK; score under tip)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| **tip k8 unroll reduce** | **1139.5** | **121.4** | **+9.65%** | OK |
| dual sgs=8 | 1124.5 | 121.4 | +9.32% | **OK** |
| prior dual sgs=16 (earlier fire) | ~1134 | ~120.3 | ~+8.8% | OK |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal: `results/20260730T063711Z/`

Enable research:
```bash
export GGML_SYCL_MOE_DUAL_SGS=8   # also 2,4,16; default 1 = legacy 1-warp/row
```

Hit: `[lx-control-moe-dual] n_experts=8 nrows=512 ncols=2048 sgs=8 (first entry)`

## What

Configurable multi-subgroup packing for MoE dual SwiGLU only:

- `sgs=1`: legacy `block_dims(1, MMV_Y, WARP_SIZE)`, one row per WG  
- `sgs>1`: `block_dims(1, 1, sgs*WARP_SIZE)`, row = `group*sgs + sg_id`

## Findings

- **Golden OK** (integer launch geometry).  
- Decode ~flat vs tip.  
- Prefill **−15 t/s** → composite loses. Same class as sgs=16.  
- **Default remains sgs=1.**

## Tip unchanged

Scored tip remains k8-unroll moe-down + mode7 stack (`20260730T062910Z`, +9.65%).

## Next

1. Bitexact integrated weighted-MMVQ down.  
2. Multi-token dual/MMVQ oracle.  
3. Avoid dual multi-sg default without prefill recovery (maybe sgs only when ne12==1 and not prefill — dual is decode-only already; pp still hurt somehow via shared paths or noise).
