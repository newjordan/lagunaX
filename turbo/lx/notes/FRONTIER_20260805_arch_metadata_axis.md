# FRONTIER 20260805 — Model-architecture tensor-metadata axis

**Direction 19 (NEW):** Extract the model topology from `buffer_init_tensor`
declarations — a metadata axis NO prior direction opened. All 18 prior
directions analyzed oneDNN primitive-execution traces (timing) or ggml op-dispatch
counts (launch tax). None read the tensor shape/quant declarations that define
the actual compute graph.

## Findings (all from `results/ktrace-tip-20260730/decode-ggml/trace.log` init lines)

### F1: Layer 0 is a DENSE FFN; layers 1–39 are MoE
- `blk.0.ffn_gate.weight` type=q4_K ne=[2048, 8192] — dense intermediate=8192
- `blk.0.ffn_up.weight`   type=q4_K ne=[2048, 8192]
- `blk.0.ffn_down.weight` type=q6_K ne=[8192, 2048]
- blk.1+ have NO ffn_gate/ffn_up/ffn_down; instead they have
  ffn_gate_exps / ffn_up_exps / ffn_down_exps (the 256-expert MoE) +
  ffn_gate_shexp / ffn_up_shexp / ffn_down_shexp (the shared expert).
- This confirms the single M=8192 dense GEMM tier in decode (finding #31:
  96.14 ms, 2.6%) is exactly ONE layer (blk.0), not a generic dense path.

### F2: Per-layer alternating attention Q-head count (48 vs 64)
- blk.0, blk.4, blk.8, blk.12, blk.16, blk.20, blk.24, blk.28, blk.32, blk.36:
  attn_q ne=[2048, 6144] → 6144/128 = **48 query heads**
- blk.1-3, blk.5-7, etc.: attn_q ne=[2048, 8192] → 8192/128 = **64 query heads**
- Pattern: every 4th block (0 mod 4) has 48 Q-heads; the other 3 have 64.
- This maps to finding #35's two attention partitions (1x8x8x256x128 vs
  1x8x6x256x128): the "8" and "6" are the group counts, not KV-head counts.

### F3: attn_gate.weight — a non-standard attention gate
- blk.0: attn_gate.weight type=q4_K ne=[2048, 48]
- blk.1: attn_gate.weight type=q4_K ne=[2048, 64]
- The output dim (48/64) matches the query-head count (F2) — this is a
  per-head learned gate applied to attention output. Standard llama has no
  attn_gate; this is a Laguna-specific architectural addition adding a tiny
  GEMV per attention layer.

### F4: K and V use asymmetric quantization
- attn_k.weight: type=q4_K ne=[2048, 1024] (every layer)
- attn_v.weight: type=q6_K ne=[2048, 1024] (every layer)
- V is stored at higher precision (Q6_K = 210 B/256elem) than K (Q4_K = 144 B/256elem)
- 8 KV heads constant across all 40 layers (1024/128 = 8) — GQA ratio alternates
  48:8=6:1 and 64:8=8:1 per F2.

### F5: Shared-expert (shexp) is a fixed single-expert FFN computed per MoE layer
- ffn_gate_shexp type=q4_K ne=[2048, 512]  — same shape as one expert gate
- ffn_up_shexp   type=q4_K ne=[2048, 512]
- ffn_down_shexp type=q6_K ne=[512, 2048]
- Computed on ALL 39 MoE layers alongside the routed experts — this is the
  DeepSeek-V2 shared-expert pattern.
- shexp gate+up concatenated = [2048, 1024], the likely source of the M=1024
  oneDNN tier (finding #27: 2480 calls, 88 ms). 39 layers × ~63 decode steps
  ≈ 2457, within 1% of 2480.

### F6: Expert weight-precision split — gate/up Q4_K, down Q6_K
- ffn_gate_exps / ffn_up_exps: q4_K, ne=[2048, 512, 256]
  → per-expert: 512 × 1152 B = 576 KB
- ffn_down_exps: q6_K, ne=[512, 2048, 256]
  → per-expert: 2048 × 420 B = 840 KB
- Down experts read **1.46× more bytes** per call than gate/up experts despite
  having the same MAC count (512×2048 = 2048×512). This byte asymmetry means
  the down family's BW-floor is 46% higher per call — a precision-driven cost
  differential, not a shape-driven one.

### F7: Total model weight footprint from metadata
- Dense blk.0 FFN: gate+up+down ≈ 9.4+9.4+13.8 MB = 32.6 MB
- Per MoE layer: 256×(576+576+840) KB + shexp(576+576+840) KB
  = 256×1992 KB + 1992 KB = 511.3 MB per layer
- 39 MoE layers × 511.3 MB ≈ 19.9 GB total expert weights
  (but only 1.6 GB working set per finding #23 — the 238× re-read factor)
