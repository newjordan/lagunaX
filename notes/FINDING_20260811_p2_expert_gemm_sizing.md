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
