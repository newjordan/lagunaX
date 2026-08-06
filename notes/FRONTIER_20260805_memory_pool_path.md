# FRONTIER: SYCL memory-pool allocation path (direction #24)

New axis: the SYCL **memory allocator** substrate beneath every per-step
`pool_alloc` temporary (activation-quant, reordered-Q8, gate-up-handoff,
down-input). No prior direction (1–23) examined the pool implementation;
all analyzed kernel dispatch, fusion, graph capture, precision, or oneDNN.

## Pool selection (runtime, hardware-dependent)
`new_pool_for_device()` (line 1737) selects:
- **VMM** (`ggml_sycl_pool_vmm`, line 1545): O(1) bump allocator, 32 GB
  virtual reserve, 2 MiB page granularity — when
  `g_ggml_sycl_enable_vmm && device.has(ext_oneapi_virtual_mem)`.
- **Legacy** (`ggml_sycl_pool_leg`, line 1406): O(MAX_SYCL_BUFFERS=256)
  best-fit linear scan on every alloc AND free — otherwise.
Default: `g_ggml_sycl_enable_vmm = 1` (line 79), env override
`GGML_SYCL_ENABLE_VMM` (line 275).

## VMM constraints
- `free()` requires **strict LIFO**: `GGML_ASSERT(ptr == (void*)(pool_addr
  + pool_used))`. Any free-order violation by the dozens of simultaneously-
  live per-step temporaries → assert (debug) or silent bump-pointer
  corruption (release).
- Physical commits round to `granularity` (≥ 2 MiB, line 137), wasting up
  to ~2 MiB per pool extension.
- `SYCL_POOL_VMM_MAX_SIZE = 1ull << 35` (32 GB virtual reserve).

## Legacy constraints
- O(256) scan on alloc (best-fit search) and free (find empty slot), no
  free-list or size-bucket index.
- 5% over-allocation: `look_ahead_size = (size_t)(1.05 * size)` (line 1508).

## Async-malloc interaction
`sycl_ext_malloc_device()` (line 3650) branches on
`g_ggml_sycl_use_async_mem_op`: async `syclex::async_malloc` when true,
synchronous `ggml_sycl_malloc_device` when false. Finding #31 proved the
flag is false, so every `sycl_reorder_temp_buffer` allocation (weight
reorder path) is synchronous. Code comment at line 392 confirms async USM
"avoids the host waits in the reorder."

## Connection to known findings
- Pool pressure is directly reduced by `MOE_ACT_Q8_CACHE` (open lead #23):
  fewer `pool_alloc<char> src1_q8_alloc` temporaries → less LIFO nesting
  depth → lower peak VRAM + fewer alloc/free cycles.
- The LIFO constraint explains why operation reordering across MoE sub-stages
  is structurally limited: temporaries must nest like a stack, preventing
  arbitrary overlap of gate/up and down stages.
