# FRONTIER: jit:gemm:any — uniform single-impl kernel selection

## Distinctness claim
This direction is materially distinct from all five tried:
- NOT launch-coalescing (dir 1)
- NOT create:cache_hit host tax (dir 2)
- NOT N-tile cost ladder / graph-bypass (dir 3)
- NOT dst-precision f32-vs-f16 (dir 4)
- NOT primitive post-op epilogue fusion (dir 5)

It targets the **JIT kernel-selection path** — *which* implementation oneDNN's
matmul heuristic chooses — an axis no prior direction examined.

## Core finding (this iteration)
ALL 190,211 `primitive,exec,gpu,matmul` calls in the 382,797-line post-brownout
trace select exactly ONE implementation variant:

    $ grep 'primitive,exec' trace.log | sed -E 's/^.*matmul,([^,]+),.*/\1/' | sort | uniq -c
    190211 jit:gemm:any

i.e. 100% of calls — from the smallest expert shape (1x512x2048:1x2048x9) to the
largest dense (1x6144x2048:1x2048x256) — resolve to the SAME generic GEMM
micro-kernel `jit:gemm:any`. The heuristic never routes any workload to a
specialized path.

[evidence: file:results/ktrace-post-brownout-20260731/trace.log — `jit:gemm:any` = 190211/190211]

## Secondary signal
The `auxiliary` field (field 8 of the exec template) is `undef` for all 190,211
calls — no micro-kernel/isa/algorithm hint is ever attached:

    $ grep 'primitive,exec' trace.log | awk -F, '{print $8}' | sort | uniq -c
    190211 undef

This is consistent with the heuristic selecting the generic fallback: a bare
`jit:gemm:any` with no auxiliary specialization flag.

[evidence: file:results/ktrace-post-brownout-20260731/trace.log]

## Why this is a new speed lever (not a restatement)
The five prior directions all assume the *selected kernel is optimal for its
shape* and attack orthogonal overhead (launch count, host cache, tile padding,
output width, epilogue fusion). The kernel-selection angle asks the opposite:
is `jit:gemm:any` the right kernel for the 121,666 expert up-proj calls
(1x512x2048, M=512) and the 60,833 down-proj calls (1x2048x512, M=2048)?

oneDNN's matmul JIT dispatch contains shape-specialized micro-kernels (tall-skinny,
small-N, power-of-two). If the heuristic is collapsing every shape to the generic
`gemm:any` path because the dynamic-N expert stream defeats the heuristic's shape
classification, the M=512×K=2048×N=9..16 calls — which are individually
launch-overhead-bound (finding #10: 17.6 µs floor, ≥97% overhead) — may be paying
extra compute/BW work inside a general GEMM kernel that a small-N tile-specialized
kernel would skip.

## Problem-desc → impl map (measured this iteration)
M=512  K=2048 (gate/up expert):  121,666 calls  → all jit:gemm:any
M=2048 K=512  (down expert):      60,833 calls  → all jit:gemm:any
M=1024 K=2048 (dense):             2,560 calls  → all jit:gemm:any
M=256  K=2048 (dense):             1,216 calls  → all jit:gemm:any  (these are src:f32!)
M=8192 K=2048:                     1,024 calls  → all jit:gemm:any
M=64   K=2048:                       960 calls  → all jit:gemm:any

8 distinct M values, 1 implementation. The heuristic is flat.

## Open question (untested — do NOT ship on this)
Whether forcing a shape-specific matmul algorithm hint (via oneDNN's
`primitive_attr` / algorithm-flags or a reordering hint) would move the tiny-N
expert floor below 17.6 µs/call. This requires a code change to the ggml SYCL
backend's matmul primitive construction to (a) pass an aux hint, or (b) test an
explicit `jit:gemv`/small-N path — not yet attempted.

## What would confirm/kill it
1. Check oneDNN's heuristic decision log — does `primitive,exec` ever log a
   *rejected* impl alongside the chosen one? (Current trace logs only the winner.)
2. Build a one-call microbench: same shape (1x512x2048:1x2048x9) with an explicit
   algorithm hint vs default; compare per-call wall. If identical → heuristic is
   already optimal and this direction dies. If lower → the flat-selection is real
   overhead.
