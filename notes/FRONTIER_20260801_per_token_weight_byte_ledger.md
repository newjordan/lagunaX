# Frontier — Per-token weight-byte traffic ledger (2026-08-01)

## Direction (PIVOT — not dirs 1–24)

**Invert the importance ranking axis.** All prior MoE-dominated ceilings came from
*oneDNN/exec-time* or *launch-count* ledgers (dir 5/10/11/16). Rebuild the budget as
**quantized on-device bytes actually read once per decode token**, from
`buffer_init_tensor` footprints × k=8 routing. Distinct from:
- dir 7 (f16 dequant amplification on the oneDNN expert path only)
- dir 11 (roofline on expert GEMM stream only)
- dir 17 (tensor metadata topology without a full-family byte sum)
- lm_head ROI probe (single tensor wall-time, not cross-family ledger)
- host↔device logits traffic (D2H, not weight reads)

## Evidence

- `results/ktrace-tip-20260730/decode-ggml/trace.log` — first-seen `.weight` tensors
- `results/LATEST_SCORE.json` / `results/20260731T172351Z/llama-bench.log` — tg=138.02
- `notes/FRONTIER_20260802_roofline_throughput_diagnostic.md` — dense-path peak BW 443.6 GB/s

## Findings

| Family | MB/token | % of active |
|--------|--------:|------------:|
| **attn Q/K/V/O/gate** | **815.74** | **46.2** |
| moe routed k=8 of 256 | 593.17 | 33.6 |
| lm_head (output.weight Q6_K) | 168.59 | 9.5 |
| router ffn_gate_inp f32 | 81.79 | 4.6 |
| shexp gate+up+down | 74.15 | 4.2 |
| dense FFN blk.0 | 32.64 | 1.8 |
| **TOTAL active** | **1766.07** | 100 |

1. **Attention weight traffic exceeds routed-expert traffic by 1.375×** (815.7 vs 593.2 MB/tok).
   The 92.5% expert *time* share is an overhead artifact of tiny-N launches, not a byte share.
2. **Q+O alone = 707.79 MB (40.1%)** — larger than the entire k=8 expert bank read (593 MB).
   Subtypes: Q 353.9 + O 353.9 + V 58.0 + K 47.2 + gate 2.8.
3. **Router f32 (81.8 MB) > shexp (74.1 MB)** — full-precision gate logits weights out-read the
   shared expert FFN every token; prior shexp-tier timing work missed this peer.
4. **Full expert banks = 18.98 GB of the 20.27 GB model** (93.6% of file size); only **3.12%**
   of that bank is touched per token at k=8. Resident cold experts dominate VRAM without
   contributing steady-state decode bytes.
5. At scored **138.02 tok/s**, active-weight BW demand = **243.8 GB/s**. Against measured peak
   **~444 GB/s** (dense M8192 path): weights-only floor = **3.98 ms/tok** vs wall **7.25 ms/tok**
   → floor is **54.9%** of wall. The other ~45% is launch/host/activation/FA-KV/non-weight work
   — consistent with overhead-floor expert kernels, but now bounded as a residual after a
   full-family byte floor rather than an expert-only roofline.
6. Scored decode arm logs **`kernel=VEC`** FA (`Q=[128,1,48] K=[128,256]`); prefill logs
   **`kernel=ONEDNN`**. Finding-#8's oneDNN SDP partition story is **not** the scored decode
   attention substrate.

## Hypotheses

1. **Attention Q/O quant downgrade (Q4→IQ3/TQ)** targets the single largest byte family
   (40% in Q+O alone) without touching MoE correctness surface — quality gate via PPL/golden.
2. **Fuse or quantize router** (`ffn_gate_inp` f32 → Q8/Q4) removes 82 MB/tok of full-precision
   traffic; already partially fused (gemv+sigmoid+add) but still f32 weight reads.
3. **Cold-expert compression / paging** only helps VRAM headroom and first-touch reorder, not
   steady-state tg (3.12% working set) — do not expect tg wins from expert offload.
4. Re-rank fusion ROI: a 10% cut in Q+O bytes ≈ 71 MB/tok ≈ 0.16 ms @444 GB/s theoretical,
   vs another MoE launch fuse that only attacks the residual 45% overhead budget.

## Not claimed

- Wall-time share of attention GEMVs under the *native* path (needs a non-oneDNN time board).
- Whether Q/O are already at the BW roofline (large nrows GEMV — likely yes, like lm_head).
