# Mount Doom status — 2026-07-30 (dual+down decode tip; dense dual+down residual parked)

## LIVE NOW

**Scored tip:** MoE dual+down expert-loop (prefill) + **dual+down decode integrated** + dense dual multi-col GEMM + hybrid mode8 + rest of stack

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip dual+down decode** | **3422.7** | **128.8** | **+50.93%** |
| dense dual+down residual (opt-in) | 3375.8 | 128.4 | +50.06% golden OK, not tip |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal tip: `results/20260730T101935Z/` · `notes/SHIP_20260730_dual_down_decode.md` · `patches/0031-*.patch`  
Parked: `notes/SHIP_20260730_dense_dual_down_residual.md` · `results/20260730T103913Z/`  
Opt-in dense dual+down residual: `GGML_SYCL_ENABLE_DENSE_DUAL_DOWN=1`

## NEXT

1. **Hybrid true TOP_K** (replace full argsort 256→k=8; bitexact/ties) — mode8 still full sort.  
2. lm_head prune/mask only with golden oracle (packing A/B exhausted — see `SHIP_20260730_lm_head_q6_nsg_probe.md`).  
3. GEMM residual addend for prefill dense dual+down (if revisiting).

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
