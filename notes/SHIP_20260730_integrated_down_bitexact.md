# Research note — integrated MoE-down bitexact attempts (2026-07-30)

## Status: **still research / not default**

| attempt | golden | note |
|---------|:------:|------|
| one-kernel register accumulate (original) | **FAIL** | pre-existing |
| one-kernel + store experts to mmid scratch then reduce | **FAIL** | this fire |
| direct reorder MMVQ (mmid nb strides) + k8 reduce | **FAIL** | this fire |
| **exact** `mul_mat_id` + k8 reduce inside ENABLE path | **OK** | proves flag wiring fine |
| weighted reduce WG=512 (default two-step) | **OK** | +9.62% < tip +9.65% |

Tip remains two-step mul_mat_id + k8-unroll weighted reduce (`20260730T062910Z`, +9.65%).

## Finding

Divergence is **not** just “skip intermediate store”: even launches that should mirror `mul_mat_id_mmvq_fused` + reduce still golden-fail, while calling `ggml_sycl_mul_mat_id` + reduce from the same fuse branch is OK. Likely subtle stride/quantize/path differences vs the full `mul_mat_id` entry.

One-kernel integrated remains blocked until a line-by-line oracle vs `mul_mat_id` outputs.

## Env

```bash
# still does not enable a correct faster path (falls through to two-step after this fire)
export GGML_SYCL_ENABLE_MOE_DOWN_INTEGRATED=1
```

## Next

1. Diff mmid buffer after `mul_mat_id` vs after direct reorder on a single layer.  
2. Multi-token dual/MMVQ oracle.  
3. Other decode levers (lm_head).
