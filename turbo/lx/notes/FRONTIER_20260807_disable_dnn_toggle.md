# FRONTIER: GGML_SYCL_DISABLE_DNN — the untested oneDNN kill-switch env var

## Direction
A **runtime env-var toggle** (`GGML_SYCL_DISABLE_DNN`) that forces ALL mul_mat dispatches
onto the native quantized Q8_1×Q4_K path, eliminating the oneDNN primitive-API path entirely.
Distinct from `GGML_SYCL_DISABLE_GRAPH` (dir 15/17 — graph capture), from the oneDNN
primitive-overhead analyses (dirs 2,7,11 — which measured the cost but never proposed the
switch), and from the f16-dequant BW axis (dir 9 — which quantified the amplification but
not the toggle that avoids it). This is the **dispatch-routing axis**: which of the two
code paths (open lead 2's dual-path split) actually runs.

## Evidence

### Finding 1: GGML_SYCL_DISABLE_DNN is unset (null) on every scored run

LATEST_SCORE.json's candidate env block sets `GGML_SYCL_DISABLE_GRAPH` to `"1"` but
`GGML_SYCL_DISABLE_DNN` to `null` (absent → unset → defaults to 0 → **oneDNN ENABLED**).
[evidence: file:results/LATEST_SCORE.json:16-21]

### Finding 2: The toggle is a runtime env var, not a compile flag

In ggml-sycl.cpp: `g_ggml_sycl_disable_dnn` is initialized to `0` (line 78), read from
the `GGML_SYCL_DISABLE_DNN` env at init (line 274), and gates every oneDNN dispatch site
via `if (!g_ggml_sycl_disable_dnn)` at lines 2474, 2527, 2577, and 3436. Setting the env
var to `1` at runtime (no recompile) routes all matmuls through the native path.
[evidence: file:ggml-sycl.cpp:78,274,2474,2527,2577,3436 — via grep of llama.cpp source]

### Finding 3: Disabling oneDNN eliminates 32,680 dispatches + 26,484 conversion kernels

The dual-path note established the split: 255,586 native-path (88.7%) vs 32,680 oneDNN-path
(11.3%) dispatches, plus 26,484 `to_fp16_sycl` conversion events unique to the oneDNN
path. With DNN disabled, the 32,680 oneDNN dispatches reroute to native quantized GEMV —
removing the oneDNN primitive-cache-lookup serialization tax (dir 2's 1.185 µs/call ×
~33K oneDNN create events), the f16 weight-dequant BW amplification (dir 9's 3.6×), and
the to_fp16 conversion kernel launches entirely.
[evidence: file:turbo/lx/notes/FRONTIER_20260805_ggml_dispatch_dual_path.md]

### Finding 4: The scored env already demonstrates the toggle is safe to flip

`GGML_SYCL_DISABLE_DNN` is the ONLY env var in the candidate block set to `null` while
the sibling `GGML_SYCL_DISABLE_GRAPH` is `"1"` — the baseline does NOT set it either,
so flipping it changes candidate-only behavior without touching the pinned baseline,
making it a valid A/B axis under the current scoring harness.
[evidence: file:results/LATEST_SCORE.json]

## Why this is distinct

| Prior direction | What it analyzed | What this direction adds |
|---|---|---|
| Dir 2 (primitive-cache tax) | 1.185 µs/call on oneDNN path | The env var that ELIMINATES the path |
| Dir 7 (per-call overhead) | 3-tier cache nesting decomposition | The single toggle removing all 3 tiers |
| Dir 9 (f16 dequant BW) | 3.6× amplification in oneDNN | The native path avoids f16 entirely |
| Dir 11 (serialization barrier) | cache_hit host lookups | Removing the lookups by removing the path |
| Dir 15/17 (DISABLE_GRAPH) | Graph capture env var | A DIFFERENT env var, orthogonal mechanism |

## Optimization lever this opens

1. **Set `GGML_SYCL_DISABLE_DNN=1` in the candidate env** — zero-recompile A/B test:
   forces 100% of matmul dispatches onto the native quantized path, potentially
   eliminating ~33K oneDNN dispatches + ~26K conversion kernels from the decode critical
   path. The risk axis is whether the native Q8_1×Q4_K GEMV kernel is slower than oneDNN's
   f16 GEMM for the shapes currently routed to oneDNN (the 32,680 non-expert dispatches) —
   open lead 1's unmeasured question, now actionable via this toggle.

2. **The toggle also removes the f16-dequant BW penalty (dir 9)** for the oneDNN-routed
   calls: native path reads Q4_K weights directly (144 B/256elem) instead of f16-dequanted
   (512 B/256elem = 3.6×), cutting BW on those 32,680 dispatches by ~3.6×.
