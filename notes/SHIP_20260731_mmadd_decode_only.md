# Ship — MUL_MAT+ADD residual fuse **decode-only** reclaim (2026-07-31)

## Status: **QUALITY-SAFE SCORED TIP** (default ON via patched tip binary)

| arm | pp512 | tg128 | score | golden | wikitext-2 PPL (2×512) |
|-----|------:|------:|------:|:------:|----------------------:|
| pin | 1139 | 107.35 | 0 | — | — |
| quality-safe tip (mm-add killed) | ~1187 | ~136.4 | **1.209** | OK | ~12.8 |
| **+ decode-only mm-add** r1 | **1177.2** | **138.89** | **1.223** | **OK** | **12.60** |
| **+ decode-only mm-add** r2 | **1177.8** | **138.46** | **1.220** | — | — |

Formal: `results/20260731T113913Z/` (confirm `20260731T114012Z/`)  
Beat-this bar freeze: `results/tip-freeze-20260731Tgoal/` (score ≈ 1.209)

## What

Reclaim `ggml_sycl_fuse_mul_mat_add` (and decode double residual `mul_mat+add+add`) for
**serial decode only** (`src1.ne[1] == 1`).

Any-batch prefill fuse is the quality break: large-N paths can write GEMV into the
ADD destination while residual addends are not applied, eliding graph ADDs → PPL
1e5–1e6. Decode reorder-MMVQ epilogue with `g_mmvq_row_addend{,2}` stays correct.

### Delivery (tip source incomplete)

Control tip **source tree is missing** `fuse_mul_mat_add` / addend epilogue bodies
(see `SHIP_20260730_tip_source_regression.md`). Live tip `libggml-sycl.so` still
has them. Champion ships a **binary patch** on that tip lib:

```text
ggml_sycl_fuse_mul_mat_add @ 0x22952f
  was: cmp $0x20, %rax; ja single_ADD_path   # ne[1]>32 still fuses
  now: cmp $0x1,  %rax; ja return_0          # ne[1]>1 rejects fuse
```

```bash
# rebuild path from unpatched tip lib:
cp -a build-base-control/bin/libggml-sycl.so.0.17.0 build-mmadd-decode/bin/
python3 /home/frosty40/turbo/lx/scripts/patch-mmadd-decode-only.py \
  build-mmadd-decode/bin/libggml-sycl.so.0.17.0
```

Source intent (when fuse is restored): `patches/0044-...fullsnippet.cpp` now defaults
to `ne11==1`; opt-in research `GGML_SYCL_ENABLE_MUL_MAT_ADD_ANY_BATCH=1` (not quality-safe).

## Env (champion)

```bash
export LX_BIN=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-mmadd-decode/bin
export GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=0   # ON — decode-only via patch
export GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1      # still broken alone
export GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1
export GGML_SYCL_DISABLE_GRAPH=1
```

Kill mm-add (restore prior quality-safe tip): `GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1`

## Why win

Decode **+2.1–2.5 tg128** vs quality-safe tip (elide residual ADD launches × MoE layers).
Prefill flat/noise. Composite **+1.1–1.4 pp** formal score (~+22.1% vs pin vs tip +20.9%).

## Gates

| gate | result |
|------|:------:|
| Golden greedy | OK (same `correctness/golden.json`) |
| wikitext-2 PPL 2×512 | ~12.6 (matches QS ~12.8; not 1e5+) |
| floors_ok | true both rebenches |
| dual_down / dual_multitoken | remain disabled |

## Tip stack (quality-safe)

Prior quality-safe defaults (router GEMV, true top-k full-norm, FA VEC, dual SwiGLU,
dense dual, moe-down weighted, add+add, rms/rope/softplus, …) + **decode-only
mul_mat+add(+add)**. Dual_down and dual_multitoken still off.

## Next

1. Restore full fuse source into control tree; land decode-only gate in C++ (drop binary patch).
2. Fix dual_down bitexact/PPL (expert-loop multitoken) for prefill reclaim.
3. Prefill residual epilogue that is GEMM-correct (not any-batch MMVQ drop).
4. Rapid wave under new tip (lm_head / packing closed for small ROI).
