# FINDING: B70 has ~2x decode BW headroom (2026-08-02)

## Measured
- B70 peak device BW (SYCL read-modify-write probe, /tmp/opencode/bw_probe): **567 GB/s**.
  Pure copy engine: 1362 GB/s.
- Laguna decode active read: ~1.69 GB/token (3B active @ Q4: 40 layers, n_embd=2048,
  256 experts k=8 expert_ffn=512, GQA 8 KV heads, head_dim 128).
- Champion decode 138 tok/s → effective BW = 1.69 * 138 = **~233 GB/s = 41% of peak.**

## Implication
Decode is BW-bound BUT only at 41% utilization. The current reorder-MMVQ kernels
leave ~half the memory BW unused (scattered expert selection 8/256 + tiny M=1 GEMVs
under-saturate the memory controllers). A BW-optimized MoE decode kernel (prefetch /
software-pipeline the next expert's weights, larger transactions, better scheduling)
that reaches 60-80% of peak → decode 200-270 tok/s → **score ~1.6-1.9**.

This vindicates the "faster MMVQ" path: there IS real headroom, not a ceiling.
Prior "all levers exhausted" conclusion was WRONG for the decode-kernel lever — it
assumed near-peak BW, which the probe disproves.

## Also found (MoE down "wiring bug")
The integrated down kernel divergence vs reference is **fp reduction-order noise**
(~1e-4 uniform, 3995/4096 elems), NOT a correctness bug (kernel is bit-exact vs its
own reference, 152/152 max_diff=0 on server workload). Golden flips on a near-tie.
Solvable by matching reduction order or recapturing golden — but the down-fusion
gives no decode speedup anyway (decode is 1 down launch already). The real lever is
the BW headroom above.
