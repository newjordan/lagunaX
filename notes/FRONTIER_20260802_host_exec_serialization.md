# FRONTIER: Host-exec serialization barrier from interleaved create:cache_hit

## Direction
Host-GPU pipeline serialization: every expert GEMM exec is preceded by an interleaved
`create:cache_hit` host lookup on the synchronous critical path, creating per-call
host-side barriers that prevent GPU command pipelining / overlap.

This is structurally distinct from all 10 tried directions:
- NOT direction 2 (cache_hit tax amount — 224.8ms) — this is about the *pattern*
  of interleaving preventing overlap, not the raw tax size
- NOT direction 1 (launch count) — this is about when the host work happens relative
  to exec submission, not how many calls there are
- NOT graph capture (lead 3) — graph capture is one *fix*; the *finding* is that the
  barrier pattern exists and compounds the tax with lost overlap

## Key evidence (from ktrace-post-brownout-20260731/trace.log)

1. **Temporal contiguity**: Between [lx-control] markers, oneDNN event counts are
   20, 6, 10, 18, 4, 264, 162, 10, then **11982, 12034, 358212** — the expert
   matmul stream runs as a single contiguous block of ~382K oneDNN events with
   essentially no control-kernel interleaving. This means the create→exec stream is
   a monolithic host-driven loop, not broken up by independent GPU work.

2. **Interleaving pattern**: In the dense region, the event sequence is
   `exec → create:cache_hit → exec → create:cache_hit → exec → ...` — the cache_hit
   fires between consecutive exec submissions, on the synchronous host path. The host
   cannot enqueue exec[n+1] until cache_hit[n+1] completes.

3. **create:cache_hit cost distribution**: 189,707 events, mean=1.185 µs,
   max=24.17 µs (20× mean), min=0.0 — the max tail shows occasional hash stall /
   eviction events that are far more costly than the mean, compounding serialization.

4. **Same-shape exec_time variance**: the dominant gate/up shape (M=512×K=2048)
   shows min=0.0159 ms, p50=0.0188 ms, p99=0.0278 ms, max=0.200 ms — the p99 is
   1.75× the p50 and the max is 12.5× the p50, consistent with host-side queue
   stalls from the serialization barrier pattern (not pure GPU compute variance,
   which would be tight for a fixed shape).

## Why this is a new lever
The 224.8ms cache_hit tax (direction 2) was measured as an additive sum. But if
the cache_hit is interleaved with exec on the synchronous host path, the GPU is
idle during each cache_hit lookup — the real cost is not 224.8ms of host time but
potentially 224.8ms of GPU idle time (lost overlap), which could be much worse
if the GPU finishes the prior exec before the host submits the next one.

## Precondition for the fix
The temporal contiguity (finding 1) means the entire expert stream IS eligible
for SYCL graph capture as a single recorded unit — there are no control-kernel
boundaries to break it at, unlike what prior directions assumed.
