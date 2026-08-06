# Frontier — Per-decode-step host↔device leaf/mask/logits traffic (2026-08-01)

## Direction (new; not dirs 1–24)

**Host↔device per-token traffic + blocking `buffer_set_tensor` barriers** — the D2H full-vocab logits pull and the H2D embd/mask/position-leaf uploads that bracket every decode step, each forced through a full-queue drain. Distinct from:
- dir 23 (`device_supports_op` query volume)
- dir 22 (memory-pool allocator)
- dir 24 (in-order kernel serialization)
- dir 9 (oneDNN create:cache_hit host serialization)
- reorder-path `event.wait()` (open lead under dir 24)

## Evidence source

- Trace: `results/ktrace-tip-20260730/decode-ggml/trace.log`
- Code: `/home/frosty40/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp`

## Findings

1. **All 544 `get_tensor_async` calls are exclusively `result_output` f32 logits** of fixed shape `ne=[100352, 1, 1, 1]`, size **401,408 bytes** (vocab=100352 × 4). No other tensor is ever D2H'd via this path. [trace + code 6953–6966]

2. **Each decode step ends with `get_tensor_async(result_output)` followed by multiple `synchronize` → `stream->wait()`** (observed triple-wait immediately after logits in mid-trace window at ~L338868). The async memcpy is non-blocking, but the subsequent `ggml_backend_sycl_synchronize` fully drains the compute queue before sampling can proceed. [trace L338868–338871; code 7004–7008]

3. **Per single-token decode step the host uploads (H2D) approximately**:
   - `embd` f32 `[2048,1]` = 8,192 B (528 calls)
   - `attn_inp_kq_mask` f16 `[256,1]` = 512 B × **2 uploads/step** (1,056 calls / ~528 steps)
   - scalar leaves: `leaf_6` i32 (4 B), `leaf_10/12/27/29` i64 (8 B each), `leaf_757` i32 (4 B)
   - **H2D payload ≈ 9.3 KB/step** vs **D2H logits ≈ 401 KB/step** → logits dominate byte volume ~43×
   [trace set_tensor/get_tensor aggregates]

4. **`ggml_backend_sycl_buffer_set_tensor` is a triple host barrier on Linux** for EVERY upload (including the 9 KB of leaves):
   1. `queues_wait_and_throw()` — drains **all** device queues
   2. `malloc(size)` + host `memcpy` bounce (PVC mmap workaround, non-Windows path)
   3. `stream.memcpy(...).wait()` — blocking H2D
   So ~7 leaf/mask/embd uploads × full-device drain = **~7 forced device idle points per decode step before the graph even runs**, independent of kernel-launch tax. [code 564–584]

5. **Prefill-shaped H2D is rare in this capture**: 16× embd `[2048,256]` (2 MiB) and 32× mask `[256,256]` (128 KiB) vs 528 single-token embd uploads — the scored-like workload is overwhelmingly decode-shaped host traffic. [trace]

## Collateral (same pass; not the direction's core)

- `attn_v.weight` is **not** uniformly q6_K: **20 layers q4_K + 20 layers q6_K** (finding #15 claimed all-q6_K — disproved). Layers 5,6,8,9,11,12,14,15,17,18,20,21,23,24,26,27,29,30,32,33 are q4_K.
- Q-head count / attn_gate width cycle every 4 layers: 10 layers at 48 heads (ne1=6144 / gate=48), 30 at 64 heads (ne1=8192 / gate=64).

## Hypotheses

1. Replacing the Linux `set_tensor` bounce (`malloc`+host copy+`.wait()`) with a pinned-host staging buffer and event-based H2D (no `queues_wait_and_throw` per leaf) could remove ~7 full-device drains per decode step.
2. Sampling only needs argmax / top-k of logits — a device-side argmax (or partial D2H of top-K candidates) would cut the 401 KB D2H + post-logits `stream->wait()` chain; lm_head ROI notes already put lm_head GEMV at ~0.3 ms / ~4% of decode, but the **host round-trip tax after it** was never budgeted.
3. Double mask upload (2×/step) may be a graph-rebuild artifact (scheduler re-sets the same leaf twice); collapsing to one set_tensor would cut two drains.

## Not claimed

Wall-time share of this host traffic vs the 7.2 ms/token decode budget is unmeasured this iteration (needs host-side timing around set/get/sync, not just counts).
