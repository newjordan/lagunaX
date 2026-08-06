# lmhead-prefetch LEDGER — first REAL measurement of the load-order axis (2026-08-06)

## Result: NULL on the real kernel (the 15:36Z A/B never measured the payload)
Official geometry (pp512/tg128, r=5), same-window 3-arm sandwich, gate
GGML_SYCL_LMHEAD_PREFETCH (default OFF = champion path):
- ctrl-a tg128 = 138.205 (+-0.219), ctrl-b = 137.683 (+-0.255) -> ctrl mean 137.944
- cand (GGML_SYCL_LMHEAD_PREFETCH=1) tg128 = 138.150 (+-0.226) -> **+0.15% vs ctrl mean**
  (ctrl spread 0.38% > cand delta; inside the ±0.68% between-run drift bound -> null)
- pp512 (gate is lm_head/decode-only; prefill has no row addends): ctrl 1166.599 /
  1163.806 (mean 1165.203) vs cand 1167.970 -> +0.24% ambient
- GOLDEN OK with the gate ON: the prefetch payload is greedy-smoke-invisible.
Evidence: results/lmhead-prefetch-20260806T182157Z/{ctrl-a,cand,ctrl-b}.log

## Why this run is the FIRST valid measurement of this axis
The 15:36Z "prefetch" A/B (prior finding 20, -0.031%) was invalid: probe-build.sh
compiles ONLY ggml-sycl.cpp, the payload lives in mmvq.cpp (separate TU), so that
cycle swapped only ggml-sycl.cpp.o and measured champion-vs-champion.

## ROOT CAUSE of the never-compiled prefetch machinery (why it never built)
results/lmhead-prefetch/q6k.patch hunk 2 header is `-1778,19 +1778,29` but the hunk
carries 31 new lines (16 context + 15 added) — a **miscounted + count (29 vs 31)**.
patch(1) consumes exactly the declared count, so it DROPPED the last two added
lines: `    }` (close else) and `}` (close dispatcher fn). The applied file therefore
had the env-gated dispatcher left UNCLOSED, swallowing every following function
-> "function definition is not allowed here" (20 errors) -> the source state was
never buildable, hence never measured. The mmvq.cpp.orig (10:35Z) carried exactly
this broken state.
Fix: regenerated the patch from a diff of the corrected file (dispatcher relocated
to AFTER the impl's closing `}` AND the two closing braces restored):
results/lmhead-prefetch/q6k-fixed.patch (77 lines, `git apply --check` OK on the
champion file; the first hunk also offsets at +32 in the original due to the same
miscount family). The fixed patch applied cleanly via git apply, compiled under
icpx (both TUs), relinked, goldened with the gate ON.

## Decode-side lm_head axis is now CLOSED on all four measured dimensions
- kernel path (dmmv|mmvq|mmq): forced mmq catastrophic -18.5%, mmvq default optimal
- thread partition (VDR 2/4/8): all null
- load format (one-time q6_K->q8_0 pre-convert): -0.037% null
- load order (next-block prefetch, THIS run): +0.15% vs ctrl mean null
Conclusion: the fused lm_head GEMV is DRAM-latency/occupancy-bound at this
occupancy (475 GB/s effective BW), not ALU/format/scheduling bound. Any further
decode-side lm_head work has no measured ceiling to chase.

## Harness: scripts/mmvq-payload-cycle.sh (parameterized, committed with this cycle)
Clones the proven lmhead-q8-cycle.sh pattern for ANY mmvq.cpp payload:
<payload.patch> <label> [gate_env]; builds BOTH TUs, swaps both objects, relinks,
goldens with the gate ON, runs the same-window 3-arm sandwich. NEW this cycle:
the EXIT trap now also restores the proven pristine champion .so (md5
2361042a185a7562c6ba5087eeaa89a0 from results/src-repro-20260806T035656Z/bin)
and recompiles the build-dir mmvq.cpp.o from champion source (md5 01bc7bb4...)
after the cycle — the lmhead-q8 cycle had to do this by hand.

## State after cycle (verified)
- source reverted (grep PREFETCH mmvq.cpp = 0)
- src-lmhead-build/bin/libggml-sycl.so.0.17.0 = 2361042a185a7562c6ba5087eeaa89a0 (pristine)
- build-dir mmvq.cpp.o = 01bc7bb458b9876725c01aff88f1f767 (champion-source compile)
- llabench binary untouched (bff3495b... matches src-repro tree)
- worktree clean except new harness/ledger/patch/results (committed with this cycle)

## Open after this cycle
- The ONLY remaining structural lever is prefill-side: the fused 512-token ffn_out
  down-GEMM (q6_K) q8_0/dp4a pre-convert — n=512 makes dp4a/mmq the correct kernel
  there (unlike lm_head n=1 where mmq was proven catastrophic). Requires the same
  two-TU source cycle gated on golden + proof-suite; mmvq-payload-cycle.sh is the
  vehicle once the ffn_out dispatch site is authored.
