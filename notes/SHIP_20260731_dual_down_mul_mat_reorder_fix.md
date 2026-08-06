# Ship — dual_down expert-loop PPL root cause + fix (2026-07-31)

## Status: **ROOT CAUSE FIXED in source** (not yet tip-champion score)

| arm | PPL 2×512 | golden | score | notes |
|-----|----------:|:------:|------:|-------|
| tip binary dual_down ON | **-nan** | OK | 1.231 invalid | expert-loop + reorder GEMM |
| tip binary dual_down OFF (champ) | ~12.5 | OK | **1.227** | quality-safe tip |
| tip binary dual_down + `ENABLE_OPT=0` | **12.65** | garbage | floors fail | proves reorder interaction |
| **source + mul_mat reorder-MMVQ chunk + dual_down+packed** | **13.07** | **OK** | **1.182** | incomplete tip stack |

Artifacts: `results/src-dual-down-fix-20260731T165656Z/`, `results/src-dual-down-packed-20260731T170108Z/`

## Root cause

`opt_for_reorder` / `opt_for_reorder_id` rewrite Q4/Q5/Q6_K expert weights to SoA for decode MMVQ.

Expert-loop (and multi-token dual GEMM) then call `ggml_sycl_mul_mat` with **N > MMVQ_MAX_BATCH_SIZE (8)** packed expert rows. Dispatch falls through to **MMQ/oneDNN**, which assumes **linear** block layout → garbage activations → PPL -nan.

Confirmed: `GGML_SYCL_ENABLE_OPT=0` makes dual_down PPL healthy (but kills decode reorder + golden).

## Fix (source)

In `ggml_sycl_mul_mat` (`ggml-sycl.cpp`):

- If `src0->extra->optimized_feature.reorder` and quant + F32 multi-col:
  - **Chunk** columns into groups of `MMVQ_MAX_BATCH_SIZE`
  - Dispatch **reorder-MMVQ** only (`quantize_and_reorder_q8_1_soa`)
  - Never fall through to MMQ/oneDNN for reordered weights

Also restored into source:

- dual_down multi-token fuse + expert-loop (patch 0028)
- packed weighted reduce (skip scatter; patch 0039 intent)
- FA VEC GQA default, decode-only mm-add (prior)

## Why score not yet > 1.227

Source `build-base-control` still lacks full tip residual stack (rms/softplus/rope/add_add parity with tip binary). Formal ~**1.18** vs tip champ **1.227**.

dual_down expert-loop **hits** on pp512 (`packed_reduce=1`) and is **quality-safe** on this stack.

## Next

1. Restore residual fuses (rms/softplus/add_add/rope) into source so dual_down+fix tip can beat 1.227.
2. Or rebuild tip binary with mul_mat chunk fix once full tip source is coherent.
3. Keep tip `env.sh` dual_down **OFF** until tip binary carries the mul_mat fix (or source tip-parity ships).

## Env (source candidate research)

```bash
export LX_BIN=.../build-base-control/bin
export GGML_SYCL_DISABLE_MOE_DUAL_DOWN=0          # ON — quality OK with fix
export GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1
export GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=0       # decode-only
# do NOT set ENABLE_OPT=0
```
