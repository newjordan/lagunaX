# lmhead-q8 LEDGER — q6_K -> q8_0 lm_head pre-convert A/B (2026-08-06)

## Result: NULL, axis closed with data on the real fused kernel
Official geometry (pp512/tg128, r=5), same-window 3-arm sandwich:
- ctrl-a tg128 = 138.267 (+-0.231), ctrl-b = 138.130 (+-0.209) -> ctrl mean 138.198
- cand (GGML_SYCL_LMHEAD_Q8=1) tg128 = 138.147 (+-0.189) -> **-0.037%** vs ctrl mean
- pp512: ctrl 1169.753 / 1158.813 (mean 1164.283) vs cand 1155.844 -> -0.725%
  (pp ctrl spread is 0.94% itself; lm_head is decode-only, gate fires only under
  g_mmvq_row_addend, so pp delta is ambient, not the payload)
- GOLDEN OK with the gate ON: the lossy q6->q8_0 conversion (per-32-quant requant)
  is greedy-smoke-invisible.
Evidence: results/lmhead-q8-20260806T174203Z/{ctrl-a,cand,ctrl-b}.log

## What the payload did
One-time (cached, per-src0) device conversion of the fused lm_head weight from the
reordered q6_K layout to the reordered Q8_0 layout (dequantize_row_q6_K semantics:
value = d * sc[v/16] * (6bit-32); 16 scales of 16 values; q8_0: d8=max|w|/127,
q8=round(w/d8) clamped) then dispatch the existing Q8_0 reorder MMVQ entry
(reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>: pure dp4a int8 dot, no bit-unpack).
Env-gated GGML_SYCL_LMHEAD_Q8=1 AND g_mmvq_row_addend!=nullptr (fused path only);
default OFF = champion path. Patch: results/lmhead-q8/q6k-q8.patch.

## Why this axis was worth measuring
Decode lm_head GEMV = 353.6 us/token (~4.8% of the 7.3 ms decode iteration;
lmhead-probe ledger). Effective BW ~475 GB/s vs ~2 TB/s card capability. Prior
nulls: VDR 2/4/8 (thread partition), kpath (kernel selection), prefetch (load
order — see caveat below). The last untouched lm_head dimension was load FORMAT
(q6_K bit-unpack ALU vs pre-converted int8). Now also null -> the fused lm_head
GEMV is DRAM-latency/occupancy bound, not ALU/format bound. Decode-side lm_head
axis is CLOSED: no source-level lever measured better than champion.

## Harness finding: the lmhead-prefetch A/B candidate never contained the payload
probe-build.sh compiles ONLY ggml-sycl.cpp; the prefetch payload lives in
mmvq.cpp (a separate TU). The 15:36Z prefetch cycle swapped only ggml-sycl.cpp.o,
so the measured "candidate" was the champion mmvq object -> the -0.031% null was
champion-vs-champion, NOT a measurement of the prefetch feature. Evidence:
- probe-build.sh target list (only ggml-sycl.cpp)
- mmvq.cpp.o mtime 10:08 (pre-cycle) unchanged by the 15:36Z cycle
- mmvq.cpp.orig (10:35Z) contains the PREFETCH machinery and does NOT compile
  (21 errors: "function definition is not allowed here" at line 1813+; the
  prefetch if/else at ~1805 leaves the else branch unclosed) -> the PREFETCH
  source state was never buildable, hence never measured.
The load-order (prefetch) axis is therefore UNMEASURED and remains open for a
future cycle using the fixed two-TU probe harness below.

## Harness fix (this cycle): two-TU probe build
lmhead-q8-cycle.sh compiles BOTH ggml-sycl.cpp (probe-build.sh) and mmvq.cpp
(icpx with the champion DEFS/INCS from vdrN-cycle.sh) into probe objects, swaps
both into the build tree, relinks via link.txt. This is the correct pattern for
any mmvq.cpp payload (vdr2/vdrN already had it; lmhead-prefetch did not).

## Champion source recovery (this cycle)
The worktree mmvq.cpp had been left with broken, never-compiled PREFETCH
machinery (see above). `git show HEAD:mmvq.cpp` is the base control WITHOUT the
champion addend machinery (g_mmvq_row_addend=0) — restoring from HEAD destroyed
the champion working state; recovered from results/lmhead-prefetch/q6k.patch by
reverse-applying it: PREFETCH count 0, g_mmvq_row_addend=30, reorder switch at
1798 (matches every pre-cycle read). A fresh compile of the recovered source has
function-identical symbol sizes vs the champion mmvq.cpp.o (all T-symbol sizes
equal; 48-byte rodata-only delta) -> behaviorally champion.

## State after cycle
- bin/libggml-sycl.so.0.17.0 restored to pristine champion (md5 2361042a...)
- build-dir mmvq.cpp.o restored to a champion-source compile (01bc7bb4...)
- source reverted (grep lmhead_q6_to_q8_kernel = 0)
- worktree clean for results/scripts; commit d5fbcf5
