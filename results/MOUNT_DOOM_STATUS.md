# Mount Doom status — 2026-07-30 (FA VEC GQA decode tip)

## LIVE NOW

**Scored tip:** packed reduce + mm-add+add decode + **FA VEC for GQA decode**

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip FA VEC GQA** | **3716.0** | **135.0** | **+59.61%** |
| tip rebench | 3729.7 | 134.8 | **+59.59%** |
| prior mm-add+add | 3711.2 | 131.6 | +56.53% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal: `results/20260730T134432Z/` · rebench `20260730T135411Z/`  
`notes/SHIP_20260730_fattn_vec_gqa_default.md` · `patches/0046-*`  
Kill VEC→TILE: `GGML_SYCL_FATTN_FORCE_TILE=1`  
**Note:** golden re-captured under VEC (not bitexact vs TILE).

## NEXT

1. New theory (packing / dual sgs / dense dual-down **closed** under FA VEC tip — `notes/SHIP_20260730_packing_under_fa_vec.md`).
2. Prefill residual2 / GEMM-post remain closed.
3. Optional low-ROI: attn gate GEMV+softplus.

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
