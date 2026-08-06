# FRONTIER: SYCL command-graph capture disabled — env-var root cause + structural incompatibility

## Direction
`GGML_SYCL_DISABLE_GRAPH=1` is explicitly set in the scored benchmark env for both
candidate and baseline. This is the upstream causal lever for the per-kernel dispatch
overhead that directions 1, 2, 7, and 11 have been measuring downstream effects of.
This note identifies the env var, characterizes why it exists (dynamic MoE routing),
and measures the per-call create-to-exec coupling it forces.

## Why this is distinct from all 14 tried directions
- Direction 3 examined the **oneDNN graph COMPILER** (op fusion into single kernels) —
  this is **SYCL command-graph capture** (batching kernel submissions into a single
  command-list replay). Different mechanism: fusion reduces kernel count; capture
  reduces submission overhead without changing kernel count.
- Direction 11 measured the **serialization pattern** (interleaved create:cache_hit on
  the critical path) — this identifies the **root-cause env var** that forces individual
  dispatch in the first place.
- Direction 1 proposed **ggml-level call coalescing** — this identifies that SYCL-level
  graph capture is the native hardware mechanism for the same goal, but it's disabled.

## Measured facts (all from this iteration's direct awk/grep on the trace)

### Env var
Both scored candidate and pinned baseline carry `"GGML_SYCL_DISABLE_GRAPH": "1"`.
[evidence: results/LATEST_SCORE.json candidate_meta.env + baseline_meta.env]

### Dynamic expert routing (the structural blocker for graph replay)
- Decode moe-dual fires with `n_experts=8 nrows=512 ncols=2048 sgs=1` (line 12495)
- M=512 (gate/up family) shows at least 20 distinct N values: 9,10,11,...,26,28,256
  with call counts 11834,9876,8388,...,1304,2356 — a power-law distribution
  of per-expert token assignments
- This proves the expert op sequence is data-dependent: different experts are active
  each step, different N per expert, so the recorded graph from step 1 is stale at step 2
[evidence: results/ktrace-post-brownout-20260731/trace.log:12494-12497 + awk N-dist]

### Create-to-exec coupling (the overhead graph capture would eliminate)
- Decode region: 184,661 primitive create events for 184,345 matmul execs = **1.0017:1**
  ratio — every single exec triggers a fresh create lookup
- Create breakdown: 184,187 cache_hit (99.92%), 316 kernel_cache_hit (0.17%),
  158 nested_primitive_cache_hit (0.09%)
- The cache_hit is the UNIVERSAL per-exec path (99.92%), not an occasional miss
[evidence: results/ktrace-post-brownout-20260731/trace.log:12494-382797]

### Per-M-class decode time decomposition (more granular than family split)
| M    | calls  | total ms | mean ms | pct of 3705 ms |
|------|--------|----------|---------|----------------|
| 512  | 117916 | 2259.01  | 0.0192  | 61.0%          |
| 2048 | 60229  | 1167.98  | 0.0194  | 31.5%          |
| 8192 | 992    | 96.14    | 0.0969  | 2.6%           |
| 1024 | 2480   | 88.17    | 0.0356  | 2.4%           |
| 256  | 1178   | 45.76    | 0.0388  | 1.2%           |
| 64   | 930    | 18.27    | 0.0196  | 0.5%           |
| 6144 | 310    | 23.18    | 0.0748  | 0.6%           |
| 48   | 310    | 6.20     | 0.0200  | 0.2%           |

The M=1024 class (88 ms, 2.4%) was previously uncharacterized — not gate/up (512),
not down (2048), not dense (8192). Its mean time (0.0356 ms) is 1.85× the expert
mean, suggesting a distinct computation (possibly concatenated shexp gate+up).

### Per-call time: M=512 vs M=2048 are nearly identical (0.0192 vs 0.0194 ms)
despite different M values — because M×K is identical (512×2048 = 2048×512 =
1,048,576), so FLOPs per call are equal. This confirms the per-call cost is dominated
by the overhead floor, not by M-dependent compute.

## Hypotheses (untested)
1. **Root cause:** GGML_SYCL_DISABLE_GRAPH=1 is set because SYCL command-graph capture
   records the op sequence during the first decode step and replays it — but MoE routing
   changes the active expert set per token, making the graph stale. Needs the ggml SYCL
   backend source to confirm the graph-replay semantics.

2. **Hybrid graph-capture:** Capturing only the FIXED portion of the decode graph (dense
   layers, attention, control kernels — ~12 of ~36 matmuls/layer) while leaving the
   dynamic expert loop on individual dispatch could eliminate ~33% of launch overhead
   without correctness risk. Needs implementation.

3. **oneDNN graph participation:** Whether oneDNN primitive submissions participate in
   SYCL ext::oneapi::experimental::command_graph capture is unknown — if oneDNN uses
   its own internal queue, graph capture wouldn't help the expert matmul stream at all.
   Needs a spike test: enable GGML_SYCL_DISABLE_GRAPH=0 and observe whether the expert
   stream is captured or falls back.

4. **Per-token budget:** The trace likely includes both tg32 and tg128 phases in the
   "decode region" (lines 12494+), so the per-token matmul count should be computed as
   184,345 / (32+128) = 1,152 per token, not 184,345 / 128 = 1,440.
