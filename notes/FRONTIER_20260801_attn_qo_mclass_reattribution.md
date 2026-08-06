# Frontier — Attention Q/O + attn_gate + f32-router M-class re-attribution (2026-08-01)

## Direction (new; not dirs 1–24)

**Non-MoE projection budget: gated-GQA Q/O M-class conflation with dense FFN, attn_gate as M=48/64, f32 path = MoE router.**

Distinct from dir 17 (tensor metadata inventory without M-class re-map), dir 14 (f32 "prefill island"), finding #14 (M=8192 = blk.0 only), finding #4 (unlabeled M tiers).

## Kill / correct

Finding #14 claim that decode M=8192 is "exactly ONE layer (blk.0 dense FFN)" is **false by call-count identity**:
- M=8192 execs = 1024 = 32 ubatch-windows × (30×Q64 + 2×dense gate/up)
- Share: **93.75% attention Q (64-head layers), 6.25% dense FFN gate+up**

## Architecture facts (Laguna-XS-2.1)

- Q-head pattern period-4: layers 0,4,8,...,36 → 48 Q heads (GQA 48:8); all others → 64 Q heads (GQA 64:8). Counts: 10×48 + 30×64.
- `attn_gate.weight` ne[1] equals Q-head count on every layer (0 mismatches): per-head attention gate, not MoE.
- SDP partition asymmetry (finding #8: 1x8x6 vs 1x8x8) is the Q-group view of the same pattern (48/8=6, 64/8=8).

## oneDNN M-class map (ktrace-post-brownout, 190211 execs)

| M | calls | ms | identity |
|---|------:|---:|----------|
| 8192 | 1024 | 99.23 | 30×Q64 + 2×blk0 gate/up |
| 6144 | 320 | 24.17 | 10×Q48 only |
| 64 | 960 | 18.88 | 30×attn_gate (64-head) |
| 48 | 320 | 6.41 | 10×attn_gate (48-head) |
| 256 f32 | 1216 | 47.82 | MoE router `ffn_gate_inp` (f32), not generic prefill island |
| 2048×8192 (as K) | 992 | 114.70 | 30×O64 + 1×blk0 down |

## Weight-byte floor per decode step (N=1, full tensor reads)

Attention Q+O alone ≈ 707.8 MB/step vs MoE 8-expert×39L ≈ 636.4 MB/step — **Q+O exceeds expert weight traffic**, while oneDNN *time* still concentrates on tiny-N experts (launch-bound). LM head Q6_K is 168.6 MB and **absent from oneDNN shapes** (native GEMV only).

## Levers

1. Re-target M=8192/6144 optimization as **attention Q projection** (not dense FFN).
2. Treat M=48/64 as **attn_gate** GEMVs (already softplus-mul fuse candidate).
3. Route f32 disable-DNN experiment expectations: killing DNN also moves **router** off oneDNN f32 GEMM.
4. Non-MoE projection BW (Q+O+K+V+lm+router) is majority of weight bytes — fusion/graph work that only touches experts leaves this axis cold.
