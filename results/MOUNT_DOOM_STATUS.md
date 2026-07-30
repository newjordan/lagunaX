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

1. **FA VEC for GQA decode** — ~+3.6 tg64 but golden FAIL; opt-in `GGML_SYCL_FATTN_FORCE_VEC=1` (`notes/SHIP_20260730_fattn_vec_gqa.md`). Bitexact VEC is high leverage.
2. Prefill double-ADD / GEMM post / chunked residual2 **closed**.
3. Other attn fuses low ROI vs FA numerics.

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
