# oneDNN per-launch primitive-cache tax (NEW angle — distinct from launch-coalescing)

Discovered 2026-07-31 via the never-summarized `results/ktrace-post-brownout-20260731/trace.log`
(382,797 lines, champion tip = treebeard-base-control-latest/build-mmadd-decode).

## The new angle

Every matmul launch pays a **oneDNN primitive `create:cache_hit`** host-side lookup
even when the JIT kernel is already cached. This is a framework tax per launch,
**separate** from (and additive to) the level-zero launch overhead the dispatch-
coalescing direction targeted.

## Measured (post-brownout champion tip)

- `primitive,create:cache_hit`: **224.80 ms over 189,707 events** = **1.18 µs/call**
  [evidence: file:results/ktrace-post-brownout-20260731/trace.log — awk aggregate]
- = **5.76% of the 3902 ms primitive exec total** (matmul exec = 3828 ms / 190,211 calls)
- For contrast: actual JIT `create:cache_miss` = only 66.7 ms over 11 events (one-time);
  `create:kernel_cache_hit` 1.5 ms/997; `create:nested_primitive_cache_hit` 6.5 ms/504.
  The cost is NOT compilation — it is the per-call cache-hit *lookup*.
- matmul exec per-call floor is **stable**: hottest shape `1x512x2048:1x2048x9`
  = 0.01764 ms/call now vs 0.01761 ms/call in the 29T210618Z trace → champion
  patches did NOT move the dispatch floor (kills open-lead #3's "did it change?").

## Structural reason grouped launch is hard (why direction-1 stalls)

The expert-up matmul has **248 distinct N values (N=9..256)** — N is the dynamic
top-k expert count, fully continuous. So there is no single fixed batch-N to group
on; a grouped/padded scheme must handle 248 distinct effective-N per token-stream,
which is why naive padding wastes BW. This is the concrete blocker for the
launch-coalescing direction.

## Lever implication

- The `create:cache_hit` 224.8 ms is ~**8% of a ~2850 ms decode wall** if exec total
  ≈ decode, and it is **host-side**, so a SYCL graph capture (replay the 190k-call
  stream as one recorded graph) would amortize it to ~0 without changing math.
- SYCL graph is currently KILLED (`GGML_SYCL_DISABLE_GRAPH=1`, env.sh). Only data
  point on record is pp −2.6% (ABSOLUTE_LIMIT.md:33); **the decode-only / graph-
  capture-of-MoE-stream case was never measured**.

## Next concrete probe
Re-enable graph capture on the decode (tg128) path only and re-bench: if pp holds
within floor, the 224.8 ms cache-hit tax + per-call launch overhead collapses for
decode → quality-neutral tg gain. (Keep prefill on the non-graph path that was
measured at −2.6%.)
