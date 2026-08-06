# FRONTIER: ggml-level SYCL dispatch — dual matmul code paths + activation-quantization tax

## Direction
The ggml SYCL op dispatch layer (`[SYCL][OP]` trace) splits `mul_mat` into **two completely
separate execution paths** that the oneDNN primitive trace (all 40 prior findings) cannot see.
This is the ggml-backend dispatch axis, not the oneDNN primitive axis (dir 2,3,7), not the
SYCL graph-capture axis (dir 15), not the fpmath axis (dir 14).

## Evidence

Source: `results/ktrace-tip-20260730/decode-ggml/trace.log` (1,677,909 lines, ggml-level
SYCL op instrumentation — distinct from the oneDNN primitive trace
`ktrace-post-brownout-20260731/trace.log`).

### Finding 1: Two mul_mat code paths at the ggml dispatch layer

Total `ggml_sycl_op_mul_mat` dispatches = **288,266**. These split into:
- **Native quantized path** (`op_mul_mat/quantize_row_q8_1_sycl`): **255,586 calls (88.7%)**
  — quantizes activations to Q8_1 on-GPU, then calls native ggml GEMV kernels
- **oneDNN path** (`op_mul_mat_sycl/to_fp16_sycl`): **32,680 calls (11.3%)**
  — converts weights/activations to fp16, then dispatches through oneDNN

All 40 prior findings analyzed ONLY the oneDNN path (the `ktrace-post-brownout-20260731`
oneDNN primitive trace). The dominant 88.7% native-quantized path was never characterized.

### Finding 2: Expert stream uses `reorder_mul_mat_vec_q*_k_q8_1_sycl` GEMV kernels

The dominant expert matmul kernels by call count:
- `reorder_mul_mat_vec_q4_k_q8_1_sycl`: **105,600 calls** (gate/up experts, Q4_K weights)
- `reorder_mul_mat_vec_q6_k_q8_1_sycl`: **21,680 calls** (down experts, Q6_K weights)
- Total reorder-GEMV = **127,280 calls** = the expert matmul stream

The `reorder_` prefix indicates a **weight-reorder/relayout kernel precedes each expert
GEMV dispatch** — a per-call host-side reorder operation invisible to the oneDNN trace.
The Q4_K:Q6_K ratio is 4.87:1 (gate+up combined vs down), consistent with 2 gate/up
families sharing Q4_K weights vs 1 down family in Q6_K.

### Finding 3: Activation-quantization tax (`quantize_row_q8_1_sycl`)

Every one of the 255,586 native-path mul_mat dispatches triggers a **separate
`quantize_row_q8_1_sycl` GPU kernel** to convert f32 activations → Q8_1 before the GEMV.
This is a full GPU kernel launch per matmul, not a fused epilogue — 255,586 additional
kernel submissions on the critical path that the oneDNN trace never records (oneDNN
receives pre-converted fp16 inputs).

### Finding 4: The small-batch expert fallback path

429 calls use the non-reordered `mul_mat_vec_q4_K_q8_1_sycl_switch_ncols` /
`mul_mat_vec_q6_K_q8_1_sycl_switch_ncols` variants with ncols=2–8 (data-dependent token
routing counts), plus 2 calls each at ncols=512 and ncols=2048 (shared-expert / dense
paths). These are a distinct dispatch branch from the reordered expert stream.

### Finding 5: to_fp16 conversion tax on the oneDNN path

The 32,680 oneDNN-path calls each trigger `to_fp16_sycl` conversions (26,484 conversion
events) — converting Q4_K/Q6_K weights OR f32 activations to fp16 before oneDNN dispatch.
This is a precision-conversion kernel tax on the oneDNN path that the native quantized
path avoids entirely.

## Why this is new

| Prior directions | What they analyzed | What this finds |
|---|---|---|
| Dir 2,3,7 (oneDNN primitive) | `primitive,exec,gpu,matmul` + `create:cache_hit` | 88.7% of dispatches never touch oneDNN |
| Dir 9 (weight dequant BW) | f16-dequant amplification in oneDNN | Native path uses Q8_1 activation quant instead |
| Dir 14 (fpmath mode) | `attr-fpmath:f16` in oneDNN | Native path has no fpmath attr — it's a fixed Q8_1×Q4_K integer kernel |
| Dir 15 (SYCL graph capture) | Per-kernel dispatch overhead | The `reorder_` prefix + `quantize_row` are ADDITIONAL kernels per expert |

## Optimization levers this opens

1. **Fuse `quantize_row_q8_1_sycl` into the GEMV kernel epilogue/prologue** — eliminates
   255,586 kernel launches (one per native mul_mat) by making the Q8_1 quant a preamble
   inside the mat-vec kernel rather than a separate submission.

2. **Eliminate the `reorder_` prefix kernel** — if the weight reorder can be pre-computed
   at load time (one-time cost) rather than per-call, 127,280 reorder launches are removed.

3. **These are kernel-LAUNCH-count reductions on the dominant 88.7% path** — structurally
   the same class of optimization as dir 1 (dispatch coalescing) but on a completely
   different code path that dir 1 never examined because it only saw the oneDNN trace.
