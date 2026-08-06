# FRONTIER 20260802 — matmul API unit-batch descriptor tax (abc/cab vs ab/ba)

New direction, distinct from all 7 tried (launch-count, cache-tax, N-tile,
output-precision, post-op, kernel-selection, 3-tier-cache-nesting).

## Core observation

Every one of the 190,211 `primitive,exec,gpu,matmul` calls carries a **3D batched
blocked memory descriptor** with a *unit* batch dimension:

  `src:f16::blocked:abc::f0  wei:f16::blocked:cab::f0  dst:f32::blocked:cab::f0`
  problem: `1xMxK : 1xKxN`   (e.g. `1x512x2048:1x2048x9`)

while the underlying `gemm` kernel it resolves to (`jit:gemm:any`) is described in
**2D** layout at create time:

  `src_a:f16::blocked:ab::f0  src_b:f16::blocked:ba::f0  dst:f32::blocked:ba::f0`
  problem: `MxK:KxN`         (e.g. `512x2048:2048x256`)

So the matmul-primitive API path carries a redundant batch axis (always `1x`)
through every memory descriptor AND every cache lookup, while the actual JIT
kernel computes a 2D GEMM.

## Measured (post-brownout trace, results/ktrace-post-brownout-20260731/trace.log)

- 3D blocked descriptors on the matmul path:
  `blocked:cab::f0` = 760,844 occurrences; `blocked:abc::f0` = 380,422.
  These factor exactly as (189,707 cache_hit + 190,211 exec) lines × {1 src:abc
  + 2 (wei,dst):cab} = 379,918 abc ≈ 380,422; 759,836 cab ≈ 760,844.
  → 100% of matmul-path lines use the batched abc/cab format. [trace.log, awk/grep]
- 2D blocked descriptors on the gemm create path:
  `blocked:ba::f0` = 2,016; `blocked:ab::f0` = 1,008.
  These factor as (11 cache_miss + 997 kernel_cache_hit) = 1,008 gemm lines ×
  {1 src_a:ab + 2 (src_b,dst):ba} = 1,008 ab + 2,016 ba. Exactly consistent.
  → 0% of gemm-create lines use abc/cab; the batch axis exists only on the
  matmul wrapper, never on the kernel descriptor. [trace.log, awk/grep]
- The gemm→matmul bridge is exactly `nested_primitive_cache_hit` (504 events,
  6.49 ms): each matmul descriptor must be matched to a cached 2D gemm primitive
  through this nested lookup. The unit-batch abc/cab → ab/ba match is the
  descriptor work that 504-event tier performs. [trace.log lines 16-22, 60-95]

## Why it is a new axis

- NOT launch-count (dir 1): keeps the same 190k calls; concerns the *descriptor
  format* carried per call, not how many calls fire.
- NOT cache-tax (dir 2): the 224.8 ms `create:cache_hit` is a hash lookup; this
  is the *layout reconciliation* done by the distinct `nested_primitive_cache_hit`
  tier (6.49 ms / 504) plus per-exec abc/cab descriptor construction.
- NOT N-tile / output-precision / post-op / kernel-selection: orthogonal to N,
  to dst dtype, to post-ops, and to `jit:gemm:any` selection — it is purely the
  memory-descriptor rank/format the matmul API imposes.

## Hypotheses (untested — need backend code change)

- If the ggml SYCL backend called the 2D `gemm` primitive directly (ab/ba
  descriptors, no unit batch axis) instead of the `matmul` wrapper, the abc/cab
  descriptor construction per exec and the 504 nested-cache reconciliations are
  eliminated. This is open lead #15, now with a concrete *layout* mechanism:
  the cost is the rank-3 descriptor, not just "API choice".
- The abc/cab blocked format may force a strided/reshape on the src tensor
  (ggml stores activations row-major 2D) per exec; a 2D ab path would be a
  zero-copy view. Needs the backend matmul construction code to confirm.
- `wei:cab` (weights transposed+batched) vs `src_b:ba` (weights transposed 2D):
  if weights are laid out as 2D ba in ggml, the cab view is a synthetic batched
  reinterpretation paid every lookup.

## Next action

Locate the ggml SYCL backend matmul construction (ggml-sycl/mmq.cpp or
mul-mat.cpp) and check whether it builds a `dnnl::matmul::primitive_desc` with a
batched memory::desc (rank-3 abc/cab) where a rank-2 `dnnl::gemm` would suffice.
A 2D-gemm-direct path removes the unit-batch descriptor from the 190k-call
critical path quality-neutrally (identical 1×M×K×N math).
