# Mount Doom status — 2026-07-30 (topk full-norm tip)

## LIVE NOW

**Scored tip:** packed reduce + mm-add+add + FA VEC GQA + router GEMV decode + **true topk full-norm**

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip topk full-norm** | **3734.9** | **139.4** | **+63.67%** |
| prior router GEMV | 3730.3 | 138.4 | +62.75% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal: `results/20260730T144111Z/`  
`notes/SHIP_20260730_router_true_topk_norm_default.md` · `patches/0048-*`  
Kill full-norm: `GGML_SYCL_DISABLE_ROUTER_TRUE_TOPK_NORM=1`  
Kill GEMV: `GGML_SYCL_DISABLE_ROUTER_GEMV_FUSE=1`  
**Note:** golden re-captured under full-norm.

## NEXT

1. New theory under +63.7% tip (multi-row GEMV / packing closed).
2. Prefill residual2 / GEMM-post remain closed.
3. Optional DNNL gemm→sig epilogue for prefill.

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
