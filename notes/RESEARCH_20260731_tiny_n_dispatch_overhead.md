# Research — tiny-N MoE matmul dispatch-overhead ceiling (2026-07-31)

## Status: **untested hypothesis** (analysis only; no kernel edit yet)

## The angle prior directions missed

Every shipped patch (0001–0048) targeted **op fusion** (rms+mul, add_add residual,
softplus+mul+attn, rope, mul_mat add-epilogue) or **MoE path restructuring**
(dual-SwiGLU, dual-down, expert-loop, packed-reduce, true-topk). **None targeted the
launch/dispatch overhead of the tiny-N MoE expert matmul stream itself.**

## Evidence from the decode oneDNN trace (20260729, pre-tip — still representative of expert shapes)

Source: `results/kernel-trace-20260729T210618Z/HOT_DECODE.md` + `decode-onednn.summary.txt`

The decode path issues **thousands** of individual oneDNN matmul launches for a *single*
expert shape:

| shape (mxk:kxn) | ms_tot | calls | avg_ms |
|-----------------|-------:|------:|-------:|
| `1x512x2048:1x2048x9`  | 158.53 | 9004 | 0.0176 |
| `1x512x2048:1x2048x10` | 133.85 | 7588 | 0.0176 |
| `1x512x2048:1x2048x11` | 112.55 | 6386 | 0.0176 |
| `1x512x2048:1x2048x12` |  97.34 | 5528 | 0.0176 |
| `1x2048x512:1x512x9`   |  75.75 | 4502 | 0.0168 |

- Total GPU exec across all prims: **2856.26 ms over 141855 calls**.
- The top-4 shapes alone = **39178 calls** (~28% of all calls), each ~0.0176 ms.
- At ~17.6 µs/call with B70's level-zero launch overhead, a non-trivial fraction of
  each call's wall time is **dispatch + scheduling**, not compute. (BW-bound GEMV at
  0.0176 ms for 512×2048×9 is already near-peak for that payload — the per-call time
  floor is likely set by launch overhead, not arithmetic or bandwidth.)

## Why this is distinct from every prior direction

- Patches 0017–0031 (moe-down variants, expert-loop) restructured *which* experts fire
  and *how* the down-projection is computed — they did **not** coalesce multiple
  tiny-N matmul launches into one.
- The `mm-add decode-only` champion (score 1.227) fused an **epilogue** onto existing
  matmul calls — same call count, just cheaper per call.
- **Launch coalescing** (batching the N=9,10,11,... expert calls into a single grouped
  GEMM, or padding to a fixed N and doing one launch) has never been attempted.

## Hypothesized ceiling

If dispatch overhead is even 30% of the 0.0176 ms floor (~5 µs), eliminating it across
~40k calls saves ~200 ms of the 2856 ms trace total (~7% of decode GPU time). At the
champion's tg≈139, a 7% decode win → tg≈149 → score ~1.29 — within striking distance of
the stretch goal (1.300) **without touching quality** (same math, fewer launches).

## Next concrete action

1. Re-summarize `results/ktrace-post-brownout-20260731/trace.log` (382K lines, never
   summarized) to confirm tiny-N shapes still dominate under the current champion.
2. Profile per-call launch overhead (level-zero `zeCommandListAppend` → kernel-start
   delta) on a representative N=9 expert matmul.
3. Prototype a grouped-GEMM or padded-single-launch path for the expert gate/up stream.

## References

- Trace: `results/kernel-trace-20260729T210618Z/decode-onednn.summary.txt`
- Summary: `results/kernel-trace-20260729T210618Z/HOT_DECODE.md`
- lm_head ceiling (ruled non-dominant): `notes/SHIP_20260730_lm_head_roi_ceiling.md`
- Champion: score 1.227 (`results/20260731T141436Z/`), `MOUNT_DOOM_STATUS.md`
