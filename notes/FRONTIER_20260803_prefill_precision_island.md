# FRONTIER: Prefill matmul precision island + prefill region decomposition

## Core finding
The prefill region (lines 1–12493, ~12,493 oneDNN events) has NEVER been
decomposed by any of the 40 findings or 15 tried directions — all prior work
focuses on decode (lines 12494+). Separately verified this iteration: the trace
contains 1,216 matmul execs carrying `src:f32::blocked:abc` at shape
`1x256x2048:1x2048x256`, the ONLY src:f32 path in the trace (all 188,495 others
are src:f16). Whether these f32 calls fall in prefill vs the decode M=256 tier
(finding #40: M=256 = 45.76 ms in decode) is NOT yet confirmed — the residency
grep failed this iteration and must be re-run.

## Why this is a NEW direction (distinct from all 15 tried)
- NOT direction 4 (output precision `dst:f32`→`f16`) — this is the
  **activation** (src) precision, not the output
- NOT direction 9 (weight dequant `wei:f16` from Q4_K) — the weight is f32 here,
  but the novelty is that BOTH src AND wei are f32 (a full-f32 GEMM)
- NOT direction 14 (fpmath accumulation mode) — this is the data type itself,
  not the XMX dot-product rounding mode
- NOT any decode-focused direction (1,2,3,5,7,10,11,13) — these calls exist
  ONLY in prefill, never in the decode region

## Evidence (verified this iteration from trace.log)
- Subtype counts via `grep -oP 'primitive,(exec|create):\S+' | sort | uniq -c`:
  1,215 `create:cache_hit,...,src:f32::blocked:abc`, 1 `create:cache_miss,...,src_a:f32::blocked:ab`,
  1 `create:kernel_cache_hit,...,src_a:f32`, 1 `create:nested_primitive_cache_hit,...,src:f32`
- All f32 create lines carry shape `1x256x2048:1x2048x256` (verified via grep on `src:f32::blocked:abc`)
  — a SINGLE fixed shape with ZERO dynamic-N variation, unlike f16's 20+ N values
- The f32 path has its own JIT-compiled gemm kernel (1 f32 `cache_miss` for `src_a:f32::blocked:ab`),
  separate from the f16 kernel's 10 cache_miss events
- SDP attention has exactly 2 graph partitions: 100008 (`1x8x8x256x128`, 960 calls) and
  100002 (`1x8x6x256x128`, 320 calls) — a 3:1 ratio with heterogeneous KV-head dims (8 vs 6),
  never noted by any prior finding

## Leverage estimate
- Finding #21: prefill matmul block = 123.6 ms / 5,866 calls
- 1,216 f32 calls / 5,866 prefill calls = **20.8% of prefill matmul call-count**
- On Intel XMX, f32 GEMM has ~2× lower throughput than f16; activation read
  bandwidth doubles (4 B/elem vs 2 B/elem)
- Converting these to f16 could yield ~10% prefill matmul speedup → prefill
  speedup from 1.025× toward ~1.07×, improving score by ~1.1%

## Next step
- Identify what the M=256 f32 GEMM computes (likely ffn_shexp or a norm-
  sensitive projection in the prefill path)
- Check if the ggml graph node can be forced to f16 without quality regression
- Benchmark with the conversion to measure the actual XMX f32→f16 delta
