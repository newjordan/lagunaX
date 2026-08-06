# FRONTIER 20260731 — N-tile ladder + MoE bypasses the graph compiler

**Direction (new):** the expert-matmul per-call cost is not flat — it is a
**16-rung tile ladder** keyed on `ceil(N/16)`, and the entire MoE expert matmul
stream runs through the oneDNN **primitive** API, never the **graph** compiler.
This reframes the "248 distinct N" blocker (finding #7) and gives a concrete
launch-overhead-bound proof (lead #1).

All numbers from `results/ktrace-post-brownout-20260731/trace.log`
(champion tip, 1.227). Field layout: CSV col3=event, col4=gpu, col5/6=kind,
col12=shape, NF=exec_time_ms.

## 1. Marginal tile cost proves launch-overhead-bound (lead #1 → CONFIRMED)

Per-call avg for gate/up family `1x512x2048:1x2048xN`, grouped by `ceil(N/16)`:

| tile (ceil N/16) | N range   | calls   | avg µs/call |
|-----------------:|-----------|--------:|------------:|
| 1                | 9–16      | 58406   | **17.61**   |
| 2                | 17–32     | 30226   | 18.89       |
| 3                | 33–48     | 11248   | 20.84       |
| 4                | 49–64     | 5440    | 21.27       |
| 8                | 113–128   | 1200    | 22.13       |
| 16               | 241–256   | 3336    | 26.28       |

[evidence: file:results/ktrace-post-brownout-20260731/trace.log — awk aggregation
of 121666 gate/up `primitive,exec` lines]

- Launch + first-tile floor = **~17.6 µs**.
- Going N=9 → N=256 is **+2744% FLOPs for only +49% wall (17.6→26.3 µs)**.
- Marginal cost of each additional 16-column tile = (26.28−17.61)/15 ≈
  **0.58 µs per tile** — i.e. real GEMV compute is ~3.3% of the per-call floor.
- ⇒ the per-call floor (open lead #1) is **launch/scheduling overhead**, not
  compute or bandwidth. Confirmed with a hard cost curve.

## 2. "248 distinct N" is really only 16 tile-cost classes

The 248 distinct N values (finding #7) collapse to **16 meaningful cost
buckets** under `ceil(N/16)` — verified: padded-N set =
`{16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256}`
(16 values, from both gate/up and down families).
[evidence: file:results/ktrace-post-brownout-20260731/trace.log]

- Within a tile bucket, cost is **flat** (e.g. N=9..16 all ≈17.6 µs, ±0.05).
- ⇒ padding N up to `ceil(N/16)*16` is **free in latency** (you already paid
  for the tile). 248 primitive-cache entries → 16, killing 232/248 of the
  `create:cache_hit` primitive lookups (distinct-shape pressure).
- Caveat: padding the N (expert-count) axis does extra expert math and is NOT
  bit-exact without a mask/zero-weights — so padding alone is not
  quality-neutral. But it turns a 248-way cache/launch problem into a 16-way
  one, enabling per-bucket grouped launches.

## 3. MoE stream bypasses the oneDNN graph compiler (primitive-only)

The oneDNN **graph** backend is live but captures **only SDP (attention)**
partitions: 74.10 ms across 1280 graph-exec partitions, partition_kind
`100002`/`100008` (both `sdp`, op_names `bmm1;scale_div;mask_add;softmax;bmm2`).
[evidence: file:results/ktrace-post-brownout-20260731/trace.log — graph lines]

- **Zero** graph-exec partitions contain `matmul`/`gemm`
  (`grep -ciE 'matmul|gemm'` on graph-exec lines = 0).
- All **190211** expert/dense matmuls run via the **primitive** API
  (`primitive,exec,gpu,matmul`), which is why every one of the 190k calls pays
  the host-side `create:cache_hit` lookup (189707 events, 224.8 ms — finding #6).
- ⇒ the SYCL-graph-capture angle (lead #4) and the cache-hit tax both attack a
  path the oneDNN **graph compiler never sees**. Routing the MoE expert matmul
  stream into the graph API (or into a ggml-level cuBLAS-style grouped
  primitive) is the structural unlock — the primitive API is the bottleneck by
  construction here.

## 4. Expert families = 86.3% of all matmul exec time

| family                | calls   | total ms | % of 3828.33 |
|-----------------------|--------:|---------:|-------------:|
| gate/up tiny-N (N<256)| 119234  | 2269.53  | 59.3%        |
| down   tiny-N (N<256) | 59617   | 1035.77  | 27.0%        |
| **expert combined**   | **178851** | **3305.30** | **86.3%** |
| dense (N=256)         | 11360   | 523.03   | 13.7%        |

[evidence: file:results/ktrace-post-brownout-20260731/trace.log]

Down-family per-call (17.49 µs) ≈ gate/up (19.18 µs): both are at the same
launch floor despite M/K differing (2048×512 vs 512×2048) — further confirms
overhead-bound, not arithmetic-bound.

## Hypotheses (untested)

- A **grouped matmul** that batches all expert calls of one layer-step into a
  single launch (M=batched, variable K via masking, or `ceil(N/16)`-padded)
  would amortize the 17.6 µs floor across ~128 experts/step. Theoretical: gate/up
  121666 calls → ~数千 calls ⇒ recovers the bulk of the 2269 ms gate/up time,
  since marginal compute/tile is only 0.58 µs.
- Re-enabling SYCL graph capture **on the decode MoE path only** (lead #4) would
  collapse both the per-launch floor and the 224.8 ms cache-hit tax, but only if
  ggml routes those ops through a capturable (non-primitive-API) path — the
  primitive-API finding above is the precondition to verify, not assume.
- `ceil(N/16)` padding as a **cache-coalescing** move (248→16 primitives) alone
  may cut the `create:cache_hit` 1.18 µs/event tax proportionally even without
  grouping, since cache_hit cost likely scales with cache-table occupancy.

## What this changes vs prior directions

- Prior "dispatch-overhead coalescing" (dir #1) and "cache-hit tax" (dir #2)
  treated the 248-N variety as a fixed blocker. It is a **16-bucket** problem,
  and the cost is **provably launch-bound** (0.58 µs/tile vs 17.6 µs floor).
- The concrete next experiment is **per-bucket grouping** (16 padded-N classes)
  rather than global fixed-N padding — bit-exactness is salvageable with a
  post-mask on the over-counted experts, and latency is free within a bucket.
