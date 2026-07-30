# Mount Doom status — 2026-07-30 (mul_mat+add+add decode tip)

## LIVE NOW

**Scored tip:** packed reduce + mm-add residual-alias + **mul_mat+add+add (decode ne11≤32)**

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip mm-add+add decode** | **3711.2** | **131.6** | **+56.53%** |
| prior mm-add alias | 3734.7 | 129.7 | +55.10% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal: `results/20260730T125600Z/` · `notes/SHIP_20260730_mul_mat_add_add_decode.md` · `patches/0044-*`  
Kill mm-add: `GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1`

## NEXT

1. Optional rebench noise band.
2. GEMM residual epilogue → prefill double-ADD.
3. Attn/FA remaining.

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
