# FRONTIER: SYCL Queue Ordering Property — out-of-order submission to break host-exec serialization

## Direction (distinct from all 16 tried)
All 16 prior directions modified **ggml SYCL backend code** (patches 0001–0048:
kernels, fuses, MoE restructure, descriptors, precision, epilogues). NONE
touched the **SYCL queue configuration** that controls how the host submits
work to the GPU command queue.

The SYCL `queue` has a fundamental property:
- **in_order**: submissions are serialized — the host cannot enqueue op[n+1]
  until op[n] is submitted AND (depending on impl) the queue accepts it.
- **out_of_order** (default): the host can enqueue many operations into the
  command buffer without waiting; the GPU scheduler reorders based on
  dependencies.

Finding #13 (direction 11) proved the expert stream alternates
`exec → create:cache_hit → exec` on a synchronous host path: the host does a
1.185 µs cache_hit lookup, then enqueues an exec, then does the next cache_hit.
**If the queue is in_order, the exec submission blocks until the GPU accepts
it; if out_of_order, the host can fire-and-forget the exec and immediately
begin cache_hit[n+1], overlapping host lookup with GPU compute.**

This is NOT:
- graph capture (dir 15: captures a whole command list; this changes the live
  submission mode)
- launch coalescing (dir 1: reduces call count; this keeps the same calls but
  removes the submission barrier between them)
- cache pre-warming (open lead #7: eliminates the lookup; this overlaps it)

## Evidence from existing findings
- Finding #13: "the host cannot enqueue exec[n+1] until cache_hit[n+1]
  completes" — assumes in_order semantics. An out_of_order queue would let
  the host enqueue exec[n] then immediately start cache_hit[n+1].
- Finding #15: expert matmul p99/p50 = 1.48×, max/p50 = 12.5× — the wide
  variance is "host-side queue stalls." An out_of_order queue decouples
  submission from completion, which should narrow this distribution.
- Finding #20: attention graph-exec p99/p50 = 1.16×, max/p50 = 1.69× — the
  graph path already runs through a captured command list (effectively
  out_of_order within the graph). The primitive path does not.
- Finding #33: M=512 and M=2048 have nearly identical per-call time (0.0192 vs
  0.0194 ms) despite different M — cost is overhead-floor-dominated. The
  overhead floor includes the submission barrier.

## Hypothesis
If ggml-sycl creates its SYCL queue with `property::queue::in_order()`,
removing that property (switching to the default out_of_order) would allow the
host to overlap the 1.185 µs cache_hit lookup with GPU exec time on all
184,345 decode matmul dispatches. Even partial overlap (say 50%) would save
184,345 × 0.59 µs ≈ 109 ms = 2.9% of decode matmul time — but the larger win
is narrowing the max/p50 tail (finding #15: max is 12.5× p50).

## What to check
1. Does ggml-sycl use `sycl::property::queue::in_order()`? Grep for
   `in_order` in `ggml/src/ggml-sycl/`.
2. If yes, does removing it (and adding explicit dependencies via events
   where needed) preserve correctness?
3. Benchmark: `GGML_SYCL_DISABLE_GRAPH=1` + out_of_order queue vs in_order.

## Secondary lever: Level Zero immediate command lists
Level Zero offers `zeCommandListCreateImmediate()` which bypasses the
close+execute cycle (4 steps → 2 steps per submission). If ggml-sycl uses
deferred command lists, the per-launch overhead includes an extra host→driver
round-trip. This compounds with the queue ordering: immediate + out_of_order
maximizes submission throughput.
