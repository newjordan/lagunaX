# Research note — multi-token device mul_mat_id (2026-07-30)

## Status: **research / opt-in only** — not scored tip

| Config | golden | note |
|--------|:------:|------|
| tip stack, multi-token batch **OFF** (default) | **OK** | formal ~tg119 / +8.1% |
| `GGML_SYCL_ENABLE_MMID_FUSED_BATCH=1` | **FAIL** | even per-token device path |

## Goal

Drop D2H + host counting-sort for MoE `mul_mat_id` when `ne12>1` (prefill). Decode (`ne12==1`) already fused.

## What we built

1. Extended `ggml_sycl_mul_mat_vec_q_id[_reorder]` with multi-token args (token grid / strides).
2. `ggml_sycl_mul_mat_id_mmvq_fused` accepts `ne12` 1..64 when batch enabled.
3. Safer multi-token attempt: **per-token single-row launches** (same kernel as decode, no D2H) still golden-fails.

Enable research:
```bash
export GGML_SYCL_ENABLE_MMID_FUSED_BATCH=1
```
Hard-off always: `GGML_SYCL_DISABLE_MMID_FUSED_BATCH=1`

## Findings

- Prefill on this harness often packs tokens as `ne11=T, ne12=1` for some ops (single-token fused still applies when `ne11==1||ne11==k`); true `ne12>1` appears in MoE `reshape_3d(..., 1, n_tokens)` path during golden prefill.
- Multi-token device path is a **correctness surface** (not just perf) — do not default ON.
- Tip pp still ~pin (1130–1140); prefill win remains open after golden-safe multi-token.

## Tip unchanged

Full stack default ON (dual + hybrid m1 + dense dual + moe-down). Multi-token mmid **OFF**.

## Next

1. Diff multi-token mmid outputs vs host counting-sort on a tiny synthetic graph.
2. Or hybrid mode2 bitexact / integrated weighted-MMVQ down.
