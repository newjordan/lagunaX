# FRONTIER: SYCL In-Order Queue Serialization

## Direction
All SYCL kernel submissions on the compute stream serialize device-side
via `sycl::property::queue::in_order()`, preventing GPU pipeline overlap
of independent kernels (activation-quant ↔ GEMV, gate/up ↔ down).

This is materially distinct from all 24 prior directions: not oneDNN
primitive overhead, not graph capture, not precision, not fusion knobs,
not memory-pool, not device_supports_op — it is the **queue-ordering
model** that governs device-side execution ordering of every submitted
kernel.

## Evidence

### 1. default_queue() IS the in-order queue
```cpp
// helper.hpp:723
sycl::queue &default_queue() { return in_order_queue(); }
```
[evidence: llama.cpp/ggml/src/ggml-sycl/dpct/helper.hpp:723]

### 2. In-order queue created with explicit in_order property
```cpp
// helper.hpp:786-788
_q_in_order = create_queue_impl(true, sycl::property::queue::in_order());
_q_out_of_order = create_queue_impl(true);
_saved_queue = default_queue();
```
[evidence: llama.cpp/ggml/src/ggml-sycl/dpct/helper.hpp:786-788]

### 3. Compute stream = in-order queue (the ONLY queue used)
```cpp
// ggml-sycl.cpp:1042
ctx->streams.push_back(&(dpct::get_current_device().default_queue()));
```
[evidence: llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp:1042]

### 4. out_of_order_queue NEVER referenced in ggml-sycl.cpp
grep for `out_of_order` in ggml-sycl.cpp returns ZERO matches.
The out-of-order queue exists in helper.hpp:721 but is dead code.
[evidence: grep result — zero hits in ggml-sycl.cpp]

### 5. 13+ submit sites all go through the in-order stream
Lines 2154, 2169, 2201, 2298, 5257, 5575, 5768, 5803, 5839, 5875,
6114, 6416, 6466 — all `stream->submit(...)` where stream is the
in-order queue.
[evidence: llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp]

## Why This Matters

On an in-order SYCL queue, the device executes commands strictly in
submission order. The GPU cannot start kernel N+1 until kernel N
completes — even when they are data-independent and could overlap.

In the MoE decode path per layer (39 layers), the typical sequence is:
1. quantize_row_q8_1_sycl (activation → Q8_1)     ~0.01 ms
2. gate/up expert GEMV × 8 experts                 ~0.15 ms
3. quantize_row_q8_1_sycl (swiglu output → Q8_1)   ~0.01 ms
4. down expert GEMV × 8 experts                    ~0.15 ms

On an out-of-order queue, the activation-quant (step 1) of layer N+1
could overlap with the down-GEMV (step 4) of layer N — a pipeline
that fills GPU idle gaps between the tiny quantize kernels and the
larger GEMV dispatches. On the in-order queue, this overlap is
structurally impossible.

## Impact Estimate (hypothesis)

The decode host-overhead fraction was reported as ~73.3% of wall time
(direction 11). The in-order queue does not increase *host* overhead,
but it prevents the *device* from absorbing the inter-kernel gaps
(kernel launch latency, ~3-5 µs each on Level Zero) by overlapping
independent work. With ~5,200 kernel submissions per decode step
(finding #39), at ~4 µs launch gap each, the serialized gap budget
is ~20.8 ms/step × 128 steps ≈ 2,662 ms — roughly 72% of the 3,705 ms
decode matmul budget, consistent with the reported overhead fraction.

## Experiment

Switch the compute stream to the out-of-order queue. This is a
one-line source change in ggml-sycl.cpp:1042:

```cpp
// Before:
ctx->streams.push_back(&(dpct::get_current_device().default_queue()));
// After:
ctx->streams.push_back(&(dpct::get_current_device().out_of_order_queue()));
```

The out-of-order queue is already created at init (helper.hpp:788)
and already has the exception handler attached. SYCL guarantees that
on an out-of-order queue, data dependencies are tracked via accessor
requirements — but dpct's submit-with-capture pattern uses USM
pointers, which have NO implicit dependency tracking. So on an
out-of-order queue, independent USM kernels WILL overlap (correct
behavior), but dependent kernels that currently rely on in-order
ordering for correctness may break.

**Correctness risk:** MEDIUM. The MoE dispatch path may rely on
implicit in-order ordering between the activation-quant and the GEMV
that consumes it (both use raw USM pointers). Needs: (a) golden-smoke
correctness check, (b) tg128 bench. If correctness breaks, explicit
`event` dependencies must be threaded through the submit calls.

**Alternative zero-code experiment:** SYCL Level Zero has no env var
for queue ordering. This must be a source patch.

## Status
Documented. Needs source patch + bench run.
