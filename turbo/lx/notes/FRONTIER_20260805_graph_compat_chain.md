# FRONTIER 20260805 — SYCL Graph-Compatibility Block Chain

## Direction

The three-layer nesting of blockers that prevents SYCL command-graph capture
on the expert stream — a root-cause chain direction #15 (env var only) never
reached.

## Summary

Direction #15 identified `GGML_SYCL_DISABLE_GRAPH=1` as the env var forcing
per-kernel dispatch. But that is only the **outermost** of three nested gates.
Even with the env var unset, two more blockers exist — a compile-time macro
and a runtime op-type rejection — and the runtime rejection is caused by a
host-sync pattern whose elimination (via the fused path) does NOT unblock
graph capture, because the gate is op-type-based not dispatch-path-based.

## The three gates (outer → inner)

### Gate 1: `GGML_SYCL_DISABLE_GRAPH` env var (direction #15)

Runtime env var. Already documented.

### Gate 2: `#ifdef GGML_SYCL_GRAPH` compile-time macro

The ENTIRE graph-capture code path — `check_graph_compatibility()` and the
graph record/replay dispatch — is wrapped in `#ifdef GGML_SYCL_GRAPH`. If
the build does not define this macro, the code is absent and the non-graph
`ggml_backend_sycl_graph_compute_impl()` is always called, regardless of
any env var.

Source: `ggml/src/ggml-sycl/ggml-sycl.cpp`:
- Line ~7161: `#ifdef GGML_SYCL_GRAPH` guards `check_graph_compatibility()`
- Line ~7224: `model_sycl_graph` creation is inside `#ifdef GGML_SYCL_GRAPH`
- The else-branch calls `ggml_backend_sycl_graph_compute_impl()` directly

**Status of this build: UNKNOWN** — whether `GGML_SYCL_GRAPH` is defined in
the CMake build configuration has not been checked. If it is NOT defined,
then unsetting the env var (gate 1) has zero effect because the graph code
is compiled out entirely.

### Gate 3: `check_graph_compatibility()` runtime op-type rejection

Even with gates 1 and 2 open, `check_graph_compatibility()` walks every
cgraph node and **returns false (disabling graphs) if ANY node is
`GGML_OP_MUL_MAT_ID`**:

```cpp
case GGML_OP_MUL_MAT_ID:
    // ggml_sycl_mul_mat_id() does a blocking host wait on the sycl queue after
    // submitting a memcpy operation, but wait() can't be called on a queue that
    // is recording to a graph.
    GGML_LOG_INFO("...disabling SYCL graphs due to unsupported node type...");
    return false;
```

Source: `ggml/src/ggml-sycl/ggml-sycl.cpp` line ~7182.

`GGML_OP_MUL_MAT_ID` is the MoE expert-dispatch op — every expert GEMV in
the decode stream is a `mul_mat_id` node. So graphs are disabled for the
entire cgraph whenever ANY layer uses MoE (layers 1–39 in this model).

The comment cites the root cause: `ggml_sycl_mul_mat_id()` does
`stream->memcpy(D2H)` of the routing `ids` tensor, then `stream->wait()`,
then dispatches per-expert GEMVs in a host loop. A blocking `wait()` on a
graph-recording queue is illegal.

Source: `ggml/src/ggml-sycl/ggml-sycl.cpp` lines ~5647–5651:
```cpp
std::vector<char> ids_host(ggml_nbytes(ids));
SYCL_CHECK(CHECK_TRY_ERROR(
    stream->memcpy(ids_host.data(), ids_dev, ggml_nbytes(ids))));
// also ensures ctx.mmid_row_mapping_host is drained before we use it again
SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
```

`GGML_OP_MUL_MAT` is also conditionally rejected — only when
`!g_ggml_sycl_use_async_mem_op` (the oneAPI async memory extension is
unavailable), because the reordering path uses SYCL malloc/free and host
waits that can't be recorded into a graph. Source: line ~7186.

## Critical insight: the gate is op-type-based, not dispatch-path-based

`ggml_sycl_mul_mat_id()` tries the **device-routed fused path** FIRST
(line ~5635: `ggml_sycl_mul_mat_id_mmvq_fused`), which has NO host sync.
Only if that returns false does the memcpy→wait()→host-loop fallback run.

BUT: `check_graph_compatibility()` checks the **node type**
(`GGML_OP_MUL_MAT_ID`), NOT the dispatch path. So even if the fused path
eliminates all host-sync from expert dispatch (which the mmvq_fused patches
do for decode), graph capture remains blocked — the op-type gate rejects
the node regardless of how it will be dispatched.

**This means:** the patches that landed the `mmvq_fused` device-routed path
(fixing the host-sync inside mul_mat_id dispatch) are NECESSARY but NOT
SUFFICIENT to enable graph capture. A separate fix to
`check_graph_compatibility()` is required: it must allow `MUL_MAT_ID` when
the fused (host-sync-free) path is active, rather than blanket-rejecting
all `MUL_MAT_ID` nodes.

## Leverage estimate

If all three gates are opened:
- The 127,280 expert GEMV dispatches + 255,586 quantize_row dispatches +
  reorder dispatches (open leads #4, #5, #8, #9) would be captured into a
  single recorded command list, replayed with one `ext_oneapi_graph()` call
  per decode step instead of ~400K individual `queue::submit()` calls.
- The per-dispatch host-side overhead (finding #16: 73.3% of expert
  wall-time is overhead) would collapse to a single graph-replay overhead.
- The serialization barrier (direction #11: interleaved cache_hit lookups)
  would not exist inside a captured graph.

## Required next actions

1. Check CMake: is `GGML_SYCL_GRAPH` defined in the build? (`grep -r
   GGML_SYCL_GRAPH CMakeLists.txt cmake/ ggml/CMakeLists.txt`)
2. If not defined → the env var is a no-op; the macro must be added to the
   build first.
3. If defined → verify whether `check_graph_compatibility()` is reached
   (add a debug print) and confirm it rejects on `MUL_MAT_ID`.
4. To unblock: update `check_graph_compatibility()` to allow `MUL_MAT_ID`
   when the fused dispatch path is guaranteed (decode-only, mmvq_fused
   active), then test with `GGML_SYCL_DISABLE_GRAPH` unset.
5. Quality gate: graph capture must produce identical output (the MoE
   routing is data-dependent but deterministic per-step, so a captured
   graph per-step is correct if the routing ids are computed BEFORE graph
   replay, not inside it).
