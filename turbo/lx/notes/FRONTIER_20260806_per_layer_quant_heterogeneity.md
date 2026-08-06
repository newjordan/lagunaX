# FRONTIER: Per-layer shexp quantization heterogeneity — selective precision downgrade lever

## Direction
The GGUF model already assigns **non-uniform per-layer precision** to the shared-expert
(shexp) down-projection weights. This is the per-tensor calibration-sensitivity axis —
distinct from direction 9 (blanket f16-dequant BW amplification), open lead 19 (blanket
Q6_K→Q4_K re-quant of all down experts), finding 40 (per-family precision stratification),
and direction 19 (static tensor metadata). The model's own calibration pipeline proves
that individual layers tolerate different precision levels, opening a **targeted**
downgrade path that avoids the blanket-quality risk of lead 19.

## Evidence

Source: `results/ktrace-tip-20260730/decode-ggml/trace.log` — `buffer_init_tensor`
declarations showing actual GGUF weight types per layer.

### Finding 1: blk.9 shexp down is Q4_K, not Q6_K — per-layer precision anomaly

The shexp down-projection (`ffn_down_shexp`) is Q6_K on blk.1, blk.2, blk.3, blk.10 but
**Q4_K on blk.9**:

| Layer | Tensor | Type | nb[0] (B/256elem) |
|-------|--------|------|---------------------|
| blk.1 | ffn_down_shexp.weight | q6_K | 210 |
| blk.2 | ffn_down_shexp.weight | q6_K | 210 |
| blk.3 | ffn_down_shexp.weight | q6_K | 210 |
| **blk.9** | **ffn_down_shexp.weight** | **q4_K** | **144** |
| blk.10 | ffn_down_shexp.weight | q6_K | 210 |

This contradicts open lead 24's claim that "shexp is Q6_K for down" — the assignment is
**layer-dependent**, not family-uniform. The Q4_K assignment at blk.9 reads 31% fewer
bytes per row (144 vs 210 B/256elem).

[evidence: file:results/ktrace-tip-20260730/decode-ggml/trace.log:33,50,67,169,186]

### Finding 2: Routed-expert precision IS uniform across observed layers

In contrast to the shexp anomaly, the routed expert weights are consistent across all 5
observed layers:
- `ffn_gate_exps` / `ffn_up_exps`: always Q4_K (nb[0]=144) on blk.1,2,3,10
- `ffn_down_exps`: always Q6_K (nb[0]=210) on blk.1,2,3,10
- `ffn_gate_shexp` / `ffn_up_shexp`: always Q4_K (nb[0]=144) on blk.1,2,3,9,10

So the heterogeneity is specific to `ffn_down_shexp` — the calibration pipeline singled
out this one tensor-type on (at least) blk.9 for lower precision.

[evidence: file:results/ktrace-tip-20260730/decode-ggml/trace.log:28-33,45-50,62-67,168-169,181-186]

### Finding 3: The shexp down-projection is finding 39's M=1024 decode tier

Finding 39 identified the shexp gate+up stacked as [2048,1024] → M=1024, 2480 calls, 88 ms
(2.4% of decode matmul time). The shexp down-projection (`ffn_down_shexp`,
ne=[512,2048]) runs as M=2048 in the decode budget (finding 27: 1167.98 ms, 31.5% — but
this includes ALL M=2048 including routed down experts). The shexp-only portion is
39 layers × ~63 steps ≈ 2457 calls out of 60,229 total M=2048 calls = **4.1% of the
M=2048 tier**.

[evidence: file:results/ktrace-tip-20260730/decode-ggml/trace.log:33]

## Why this is new

- **Finding 40** established the per-FAMILY split (gate/up=Q4_K, down=Q6_K) but assumed
  it was uniform across all layers. This finding proves it is NOT uniform — at least one
  layer (blk.9) deviates.
- **Open lead 19** proposed blanket Q6_K→Q4_K re-quantization of all down experts with
  a quality-cost caveat. This finding reframes it: the model's own calibrator already
  determined that blk.9's shexp-down tolerates Q4_K. A **per-layer sensitivity sweep**
  could identify which other layers' shexp-down (and potentially routed-down) weights
  can safely drop to Q4_K, cutting bandwidth without the blanket risk.
- **Direction 9** analyzed the f16-dequant amplification factor but never examined
  per-layer precision assignment heterogeneity within the same tensor family.

## Leverage estimate

The shexp down-projection is a small fraction of decode (4.1% of M=2048 tier ≈ ~48 ms
across all 39 MoE layers). The direct BW saving from Q6_K→Q4_K on shexp-down is ~31%
of ~48 ms ≈ **15 ms** (~0.4% of decode matmul time) — marginal.

The **strategic** value is proving the calibration pipeline supports per-layer precision
assignment. If the same sensitivity analysis extends to the routed `ffn_down_exps`
(31.5% of decode, all currently Q6_K), even a partial downgrade (say 50% of layers
tolerate Q4_K) would save ~180 ms (~4.9% of decode matmul time) — the largest known
single-axis BW lever on the down family.

## Caveats

- Only 5 of 40 layers were inspected (blk.1,2,3,9,10). The full scope of the anomaly
  (how many layers have Q4_K shexp-down) requires a complete scan of all
  `ffn_down_shexp` init_tensor declarations.
- The shexp-down M=1024 attribution is from finding 39's architectural inference, not a
  direct timing measurement of shexp-only calls.
- Any precision downgrade requires a perplexity regression check (the Q4_K→quality
  tradeoff that killed the any-batch mul_mat_add fuse per SHIP_20260731 notes).
