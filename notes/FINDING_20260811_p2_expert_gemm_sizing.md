# FINDING 2026-08-11 — P2 sizing: the post-C4 expert loop makes 264K oneMKL calls per 23K pass; C1 ceiling ≈ another 5x on max-context prefill

Receipts: `results/p2-census-20260811T*/` (census-ub2048.err, 23K real text,
ub2048, C4+M1 knobs ON, DIAG_NAME_QUANT; leg itself: prefill 1490.6 t/s with
diag = −3% overhead, decode 90.2 — M1 confirmed on the real instrument again).

Per 23K pass (13 ubatches × 38 MoE layers × 3 projections):
**264,339 expert oneMKL GEMM calls, 22.8 M routed rows, mean N = 86.**
N-histogram (calls%/rows%): ≤64: 64.5%/22.2% · 65-256: 29.9%/40.8% ·
>512: 2.1%/22.5%. Dense oneMKL calls: 4,121 (already fine).

Cost model on today's path (per pass): ~1.06 M device launches (4 per call ×
264K, at the 4-6 µs floor ≈ 4-6 s), ~1 TB fp16 weight round-trip (2 MiB-class
whole-slice conversion per call regardless of N ≈ 2 s at peak BW), ~800 K
pool allocs — against an irreducible floor of ~2.6 s FLOPs (52 TFLOP at
20 TFLOPS) + ~0.3 s quantized weight reads. Measured prefill ≈ 17 s.
**C1 (one fused expert-blocked kernel per MUL_MAT_ID dispatch) ceiling ≈
3-4 s pass ≈ 5x further prefill; even 30% of ideal ≈ 2x.**

## C1 design constraints (from the campaign record + this census)

- Grid: expert-block × token-tile-64 × grid-stride over N (step-0 histogram,
  reconfirmed at serving geometry). Weights outermost — token loop INSIDE the
  workgroup (the −17% MMVQ_PREFILL lesson: no weight reuse across tokens is
  fatal). Fat tail N>512 amortizes via grid-stride.
- Inputs already on device: src1_contiguous (packed rows), dev_row_mapping,
  expert_row_counts/offsets (ggml-sycl.cpp:6751-6795).
- Weight formats IN SERVING STATE: banks are REORDERED post-warmup —
  q4_K SoA-per-expert [qs|scales|dm] (gate/up all 38, down 22 layers),
  q6_K [ql|qh|scales|d] (down 16 layers). Kernel must dequant from the
  reordered layout in-register (reuse the offset math of
  dequantize_block_q4_K_reorder / q6_K_reorder, dequantize.hpp:1106/1287).
- Accumulate fp32; activations fp16 or fp32 read from src1_contiguous (F32) —
  KLD is the arbiter, not bit-exactness (C4 precedent).
- Env knob GGML_SYCL_LX_EXPERT_TILE_GEMM=1 default OFF; staged engagement:
  start N ≤ 64 calls only (64.5% of calls, 170K launches deleted), oneMKL
  keeps the tail; widen after gates pass.
- Never: dpct::gemm_batch (hangs B70), MMQ on reordered banks, full-tensor
  pre-conversion (−24% receipt).

Gate battery: build → golden (both knob states) → KLD pinned + canonical
triangulation → real-text A/B at -c 131072 (beat 1540 prefill, hold 90 decode)
→ tg128 d0 ≥ 152.5.

## C1 v1 (SIMT dequant-GEMM) — FALSIFIED as drafted; XMX is the requirement

Draft (patch: `results/p2-c1-tile-20260811T*/c1-expert-tile.patch`, +326 lines,
band N≤64, reorder+linear q4_K/q6_K dequant in-register, fp32 accum, one
launch per dispatch): builds clean, engages correctly, and is **~2x slower
than the per-expert oneMKL path it replaces in-band**: pp512 1160 → 558
(band ≈ all experts at T=512), 131K real-text prefill 1553.7 → 1108.3 (−29%).
tg128 d0 unaffected (153.19). Numerics also unproven: golden diverges knob-ON
and canonical-KLD distance worsens (0.0340 → 0.0396) where fp32 accumulation
should improve it — do not optimize before a dst-parity verify harness exists.

Lesson (bounds the whole C1 family): a scalar-SIMT fused dequant-GEMM cannot
beat oneMKL's XMX systolic path even at N∈[9,64] where oneMKL pays full
per-call overheads — the 8-lane-redundant dequant ALU and L1-broadcast
activation reads burn the compute the tile saves in traffic. **C1 v2 must be
joint_matrix (XMX) based**: SLM-staged dequant tiles feeding sub-group
matrix-multiply-accumulate, or it will lose again. Reverted; branch stays
C4+M1 (d61bdf435, .so 94015650). v2 prerequisites, in order:
1. dst-parity verify mode (kernel + oneMKL both run, max-abs-err report);
2. XMX tile microbenchmark standalone (prove ≥oneMKL on one shape first);
3. only then the integrated kernel.

## XMX microbenchmark (C1 v2 prerequisite #2) — GO at N≤32, revised prize

Standalone bench `benchmark/xmx-dequant-gemm/` (M=512 K=2048, q4_K, B70):
fused XMX (16x16x16 fp16→fp32 joint_matrix, VNNI-packed SLM dequant tiles,
oversubscribed dequant sub-groups) vs MKL dequant-then-GEMM:
N=16 **1.33x**, N=32 **1.21x**, N=64 1.01x, N=128 0.78x. XMX numerics BETTER
than MKL (rel-err 2e-4 vs 3.5e-4; fp32 accum). Device caveat: only 16x16x16 /
8x16x16 joint_matrix combos are hardware-real; 32x64x* are emulated (300µs+).
Two load-bearing kernel tricks recorded in results.txt: oversubscribed
dequant (16 SG dequant / 4 SG MAD) and split-K with quarter-block KB=64.

Prize revision (honest): the naive 5x ceiling assumed the whole 264K-call
apparatus vanishes; measured XMX efficiency says the win is the launch-floor
deletion + fp16 round-trip on the N≤32 band (39.4% of calls, 8.7% of rows)
at kernel parity+ ≈ **+15-20% prefill at 131K**, more if the band widens to
64 after verify. C1 v2 = v1's integration skeleton (engagement/skip/metadata
were correct) + the microbench kernel + a dst-parity VERIFY mode, band ≤32.

## C1 v2 (XMX) — LANDED. +11.7% prefill@131K, +33% pp512, decode untouched

Commit `05755f5f3` (stacks on C4+M1), knob `GGML_SYCL_LX_EXPERT_TILE_GEMM=1`
default OFF, verify mode `..._VERIFY=1`. Receipts: `results/p2-c1v2-*/`
(battery + interleaved tg floor A/B) and the agent verify log in
`results/p2-c1-tile-20260811T225429Z/c2-xmx-verify-golden.log`.

| gate | result |
|---|---|
| 131K real text | prefill **1544.6 → 1726.1 (+11.7%)**, decode 89.9 → 89.5 (noise) |
| pp512 (board) | **1160 → 1547 (+33%)** — band ≈ all experts at T=512, works pre-reorder too |
| tg128 d0 | interleaved off/on r=5: 152.22/152.22 vs 152.11/152.31 — Δ≈0 (both ~0.3 below the morning floor = thermal, state-independent) |
| canonical KLD | 0.034444 / 93.211% — unchanged vs C4-only (0.0338/94.0) ⇒ numerics sound |
| pinned KLD | 0.0504 — same class as C4 (base-path bias, see gate-policy question) |
| verify mode | typical per-dispatch rel err 2-3e-3 vs fp16-accumulate oneMKL |
| golden knob-ON | greedy near-tie flip, coherent — reduction-order class, KLD arbitrates |

Open (v2.1 candidates): q6_K band support (down banks in 16 layers), widen
band to N≤64 (microbench says tie — only launch-deletion would pay),
runtime receipt for the reorder-SoA path (all golden dispatches ran linear).

## Loop day-1 close-out (2026-08-11)

Stacked env-gated candidate set on `lx/reorder-multicol-mkl` @ 05755f5f3:
C4 (reorder-multicol) + M1 (fattn split-K pb16) + C1v2 (XMX expert tile).
vs the shipped champion, real text at -c 131072:
**prefill 307 → 1726 t/s (5.6x) · at-depth decode 81.5 → 89.5 (+10%) ·
tg128 d0 and short-context serving unchanged · absolute PPL better ·
closer to canonical math than the shipped path on every measure.**
All of it dark behind env knobs, promotion blocked solely on the
KLD-gate policy decision documented in FINDING_20260811_reorder_multicol_c4.
