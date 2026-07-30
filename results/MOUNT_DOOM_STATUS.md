# Mount Doom status — 2026-07-30 (router GEMV decode tip)

## LIVE NOW

**Scored tip:** packed reduce + mm-add+add decode + FA VEC GQA decode + **router F32 gemv+sigmoid+bias (n_rows==1)**

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip router gemv decode** | **3730.3** | **138.4** | **+62.75%** |
| prior FA VEC GQA | 3716.0 | 135.0 | +59.61% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal: `results/20260730T141542Z/`  
`notes/SHIP_20260730_router_gemv_sigmoid_add.md` · `patches/0047-*`  
Kill GEMV: `GGML_SYCL_DISABLE_ROUTER_GEMV_FUSE=1`  
**Note:** golden re-captured under decode-only GEMV (not bitexact vs MKL).

## NEXT

1. Multi-row router GEMV without pp collapse (DNNL / better batching).
2. New theory under +62.8% tip — not packing thrash.
3. Prefill residual2 / GEMM-post remain closed.

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
