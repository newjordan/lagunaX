# Mount Doom status — 2026-07-30 (dual+down expert-loop tip)

## LIVE NOW

**Scored tip:** dual decode MMVQ + prefill **expert-loop dual+down** + hybrid mode8 + dense dual + moe-down/integrated + mmid + RMS+MUL + ADD+ADD + softplus×mul + MUL_MAT+ADD + rope+set_rows + prefill two-step weighted reduce

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip dual+down expert-loop** | **3266.0** | **128.9** | **+49.29%** |
| prior dual+down graph compose | 3161.0 | 129.0 | +48.09% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal: `results/20260730T095249Z/` · `notes/SHIP_20260730_dual_down_expert_loop.md` · `patches/0028-*.patch`  
Kill expert-loop: `GGML_SYCL_DISABLE_MOE_DUAL_DOWN_EXPERT_LOOP=1`  
Kill dual+down: `GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1`  
Kill all dual: `GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU=1`

## NEXT

1. lm_head (decode-weighted).  
2. First-layer dual+down can_fuse.  

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
