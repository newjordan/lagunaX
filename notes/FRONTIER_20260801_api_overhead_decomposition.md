# FRONTIER: Per-call primitive-API overhead decomposition

## Direction (distinct from all 6 tried)

Prior directions attacked: (1) launch count, (2) create:cache_hit host tax,
(3) N-tile padding, (4) dst precision f32→f16, (5) post-op epilogue fusion,
(6) JIT kernel selection. NONE decomposed the **per-exec API call-site cost**
into its constituent host-side components, or measured **achieved bandwidth**
per shape class to locate the GPU utilization floor.

## New findings (from ktrace-post-brownout-20260731/trace.log)

### 1. gemm→matmul primitive NESTING (create path)

Every matmul exec is reached through a TWO-LEVEL primitive hierarchy:
- CREATE events use primitive type **`gemm`**: `primitive,create:*,gpu,gemm,jit:gemm:any`
- EXEC events use primitive type **`matmul`**: `primitive,exec,gpu,matmul,jit:gemm:any`
- The bridge is `primitive,create:nested_primitive_cache_hit` — 504 events, 6.49 ms

So ggml requests a `gemm` primitive; oneDNN internally instantiates a nested
`matmul` primitive; the `nested_primitive_cache_hit` is the inner-level cache
lookup. This nesting is a structural host-side cost layer distinct from:
  - cache_hit (189,707 events / 224.8 ms) = outer primitive-descriptor cache
  - nested_primitive_cache_hit (504 events / 6.49 ms) = inner matmul→gemm bridge
  - kernel_cache_hit (997 events / 1.53 ms) = JIT binary cache

The three cache tiers sum to 232.8 ms of pure host-side lookup overhead —
and they are ADDITIVE on the per-call critical path (each create fires before
its exec). This was never separated: direction #2 lumped them as "cache_hit tax."

### 2. scratchpad:user on EVERY exec — workspace management per call

ALL 190,211 primitive,exec,gpu,matmul calls carry `attr-scratchpad:user`.
The graph-exec sdp (attention) partitions carry `fpm:strict` with NO scratchpad.
This means the primitive (expert matmul) path requires the caller to provide
a workspace memory object to every single exec() call, while the graph
(attention) path does not — a per-exec workspace-management cost that is
distinct from cache lookup, kernel selection, and launch overhead.

### 3. Achieved-bandwidth measurement per shape class (NEW quantification)

| Shape family | Calls | avgBW | ms/call |
|---|---|---|---|
| 8192x2048x256 (dense) | 1024 | 444 GB/s | 0.097 |
| 6144x2048x256 (dense) | 320 | 436 GB/s | 0.076 |
| 2048x8192x256 (dense) | 992 | 345 GB/s | 0.116 |
| 1024x2048x256 (mid) | 2560 | 177 GB/s | 0.036 |
| 512x2048xN (expert gate/up) | 121666 | 122-125 GB/s | 0.018 |
| 2048x512xN (expert down) | 60833 | 131-132 GB/s | 0.017 |
| 256x2048x256 (router f32) | 1216 | 61 GB/s | 0.039 |

Key: the BW **doubles** at each M stepup (512→1024→2048), then plateaus near
peak at M≥6148. The GPU's own dense calls prove 340-444 GB/s is achievable on
this device; the expert path runs at 25-35% of that. This is the first
quantitative BW-per-shape measurement — prior directions cited call count
and per-call latency but never computed achieved GB/s.

The 1024x2048x256 mid-tier at 177 GB/s proves the underutilization extends
BEYOND tiny-N into mid-sized shapes — it's not purely a launch-floor artifact.

## What this direction implies (hypotheses — untested)

- If the gemm→matmul nesting is avoidable (calling the matmul primitive API
  directly instead of through the gemm wrapper), the 504 nested cache lookups
  and the gemm-descriptor-to-matmul-descriptor translation are eliminated.
- If scratchpad:user were replaced with scratchpad:library (oneDNN-managed
  persistent workspace), the per-exec workspace argument passing is removed
  from the 190k-call critical path.
- The BW plateau at M≥2048 (340-444 GB/s) vs floor at M≤512 (122 GB/s)
  suggests the GPU's L2 cache / memory subsystem can sustain the dense rate
  but the expert path is starved by per-call overhead — consistent with the
  launch-overhead-bound regime but now quantified as a BW-utilization gap.
