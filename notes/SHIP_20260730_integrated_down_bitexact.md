# Research note — integrated MoE-down bitexact attempts (2026-07-30)

## Status: **RESOLVED → shipped decode-only** (see `SHIP_20260730_moe_down_integrated_decode.md`)

| attempt | golden | note |
|---------|:------:|------|
| one-kernel for **all** n_tokens (incl prefill) | **FAIL** | pre-existing |
| one-kernel + scratch (all tokens) | **FAIL** | multi-token path |
| direct reorder + reduce (all tokens) | **FAIL** | multi-token path |
| exact `mul_mat_id` + reduce (all tokens) | **OK** | uses host-sort for ne12>1 |
| **one-kernel n_tokens==1 only** | **OK** | **shipped default** |
| **direct reorder+reduce n_tokens==1** | **OK** | oracle MODE=1 |

## Finding (final)

Divergence was **not** one-kernel float math on decode. Prefill `ne12>1` tip uses **host counting-sort** `mul_mat_id`; fused reorder multi-token differs → golden FAIL. Gate integrated to decode only.

## Env (current tip)

```bash
# default ON for decode; kill:
export GGML_SYCL_DISABLE_MOE_DOWN_INTEGRATED=1
# oracle two-step GEMV+reduce (decode-only):
export GGML_SYCL_MOE_DOWN_INTEGRATED_MODE=1
```
