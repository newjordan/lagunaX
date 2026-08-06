# FRONTIER 20260731 — f32-matmul-output precision tax (NEW direction)

**Direction:** matmul *output* (dst) precision — every oneDNN matmul writes `dst:f32`
despite `attr-fpmath:f16`. Distinct from launch-coalescing (#1), cache-hit tax (#2),
and the N-tile ladder (#3). No prior patch or lead examined output precision.

## Smoking gun (measured from trace)

All 190,211 `primitive,exec,gpu,matmul` calls use identical descriptors:
```
src:f16::blocked:abc::f0  wei:f16::blocked:cab::f0  dst:f32::blocked:cab::f0
attr-scratchpad:user attr-fpmath:f16
```
- `dst:f16` count in the whole trace: **0** [trace.log]
- `attr-fpmath:f16` count: **190,211** (i.e. ALL of them) [trace.log]

So the multiply/accumulate is ALREADY f16 internally (`attr-fpmath:f16`), and the
result is then widened to f32 on output. The f32 dst carries **no more information**
than an f16 write would — it is a phantom-precision cast that doubles the write BW.

## Output-write byte accounting (over the full trace)

| family        | calls  | M x N_per_call | f32-out  | f16-out  | wasted   |
|---------------|--------|----------------|----------|----------|----------|
| gate/up (512) | 121666 | 512 x N        | 9380 MB  | 4690 MB  | 4690 MB  |
| down  (2048)  | 60833  | 2048 x N       | 18760 MB | 9380 MB  | 9380 MB  |
| ALL matmuls   | 190211 | varied         | 44576 MB | 22288 MB | 22288 MB |

Wasted f32-vs-f16 write traffic = **22.3 GB** over the trace, concentrated in the
expert stream (gate/up + down = 86.3% of calls and the bulk of output bytes).

## Why this is a NEW lever (not covered by prior directions)

- Prior dir #1 (launch-coalescing): about *call count* — orthogonal; f32-dst tax is
  paid per call regardless of how many calls there are.
- Prior dir #2 (cache-hit tax): host-side `create:cache_hit` 224.8 ms — pure host
  overhead; f32-dst tax is *GPU write BW*, a different budget.
- Prior dir #3 (N-tile ladder): per-call compute tiling — about FLOPs/wall vs N; the
  f32 write is a fixed 2x factor on the output leg of every tile.

This is the **output/write-amplification** axis, which the bandwidth-bound decode
regime (86 W / 37% of 230 W TDP — confirmed BW-bound in ABSOLUTE_LIMIT.md) is the most
sensitive to. Halving output writes on a BW-bound kernel is exactly the axis the
hardware is bottlenecked on.

## Quality posture

- The compute is ALREADY f16 (`attr-fpmath:f16`), so the stored f32 result is an f16
  value widened. Routing dst to f16 is **bit-exact** w.r.t. the value actually
  produced (round-trip f16->f32->f16 is identity), provided the immediate consumer
  reads in f16 or casts back. This makes the change quality-neutral if the downstream
  op accepts f16 input (needs verification of the ggml op that consumes this dst).
- HYPOTHESIS: if ggml's MoE reduce is fp32-accumulated and reads this dst as f32, a
  dst:f16 path would require a small f16->f32 load+cast at the consumer, trading
  ~2x write saving for a ~1x read+cast cost — net still favorable on a BW-bound
  decode but needs the consumer trace to confirm.

## Next concrete probe (blocked: inspection budget spent)

1. Confirm the ggml op consuming the matmul dst (MoE weight reduce / down-proj) and
   whether it accepts f16 src.
2. oneDNN matmul already supports `dst:f16` with the same `attr-fpmath:f16` (the
   graph-compiled SDP partitions already emit f16 dst: `out0_f16`); flipping the
   expert matmul descriptor from `dst:f32::cab` to `dst:f16::cab` is a one-flag
   primitive-creation change.
3. Measure: expect write-BW-bound tiny-N shapes (gate/up N=9..16, the 17.6 us floor)
   to drop if even partly write-limited; the launch-floor finding (#10) says compute
   is ≥97% of the smallest call, so the dst write is a minority of per-call time —
   the bigger win is the down family (M=2048, larger output) and the dense N=256
   calls where output BW dominates.

## Trace provenance
`results/ktrace-post-brownout-20260731/trace.log` (382,797 lines; 79 MB).
All counts/timings above were computed from this file.
