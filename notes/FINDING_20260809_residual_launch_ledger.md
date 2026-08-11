# FINDING — decode residual is a device-side launch-cost surface (2026-08-09T0418Z)

Artifact: results/lx-residual-ledger-20260809T041820Z/
  ctrl.log / ctrlb.log   tg128 scored shape, champion .so 5e8446c9, llama-bench 55d9290d
  skip15.log             GGML_SYCL_DIAG_SKIP_DECODE=15 (elides fused-QKV launch + MoE
                         mmvq-fused launches; bits 4/8 are unwired no-ops)
  ctrl.chrono.bin / skip15.chrono.bin  GGML_SYCL_LX_CHRONO=1 LX_CHRONO_NAMES=1 dumps
  parsed with scripts/chrono-decode.py + per-name gap ledger (this iteration)

## Measured (fresh, one variable: the skip env)
- ctrl:  142.89 t/s (±1.68)  ctrlb: 144.14 (±0.06)   — the known 143.8–144.1 env band
- skip15: 335.59 t/s (±10.14) = 2.98 ms/token residual wall (v2 measured 342 t/s)
- ctrl host span median 3776.8 us/step (641 graph computes); skip15 2711.8 us/step
  => in skip15 host (2.71 ms) ≈ device (2.98 ms): the pipeline is BALANCED there, so
  per-op host stamp gaps are a first-order estimate of per-op device cost.

## Per-op-class device ledger of the residual (skip15, last-20-step means)
  class                 us/tok   n/tok   avg us   meaning
  Qcur                  483.2     80     6.04    QKV fuse: activation quantize + elided launch + Q-post
  ffn_moe_probs         276.7     78     3.55    router topk chain (2 stamps/layer)
  ffn_moe_up            237.8     39     6.10    MoE dual gate+up GEMV (NOT elided by bit 2)
  ffn_up                231.3     40     5.78    dense shexp dual GEMV (real)
  attn_norm             203.7     40     5.09    rms+mul fused kernel (2048 elems!)
  ffn_norm              200.0     40     5.00    rms+mul fused kernel
  attn_gate_softplus    167.9     80     2.10    gate softplus fuse (2 stamps/layer)
  ffn_moe_weighted      165.9    351     0.47    weighted-reduce consumed view/mul nodes — HOST loop tax
  Kcur_normed           161.5     40     4.04    K norm+mul+rope+set_rows fuse (one launch)
  ffn_out               160.8     40     4.02    MoE down — elided => quantize + idle
  ffn_inp               157.8     40     3.94    dense inp GEMV (real)
  Kcur_rope             145.5     80     1.82    K rope stamps (2/layer)
  result_norm / lm_head ~5.0 + 4.1 + 3.7        tail ops
  TOTAL host span                               2711.8 us/token (vs 2984 us wall)

## What this proves
1. The residual is NOT host dispatch (open lead 1 refuted again, now with names):
   a 2048-element rms_norm kernel shows a 5.09 us device-visible gap; a 4MB
   gate+up GEMV shows 5.78 us. Per-launch device cost floor is ~4–6 us on this
   GPU at decode; the K-fuse's 193 us/120 launches = 1.6 us/launch is the LOW
   bound (chain-internal launches overlap GEMM tails).
2. The activation-quantize launches are the largest single reducible term:
   Qcur-class (elided launch, only quantize still runs) = 483 us/token for 40
   layers (2 stamps/layer), ffn_out-class = 161 us/token. ~5–6 us device-visible
   per quantize launch. ~322 quantize launches/token (qkv, dense dual, moe dual,
   moe down, attn_o, shexp, lm_head) → folding quantize INTO the GEMM kernels is
   worth ~500–1500 us/token ≈ +7–20% decode (open lead 6 was rated "low expected
   value" on the host-dispatch model — that model is wrong; it is device cost).
3. Host loop tax quantified: 351 consumed weighted-reduce view/mul nodes at
   0.47 us (skip15) / 1.11 us (ctrl) host each = 166–390 us/token. Invisible to
   wall at current config (device-bound, 3.1 ms host slack) — a future constraint
   once device time shrinks.
4. The Q-ROPE (open lead 10) is ALREADY fused: qcur_normed / qcur_rope stamps are
   0.04–0.05 us (consumed by the QKV fuse's interleaved post-chain, ggml-sycl.cpp:7704).
   No 40-launch class remains there.

## Next candidate (spec)
Fold the q8_1 activation quantize into the mmvq GEMM kernels (start with the
standalone dense path: attn_o_proj / ffn_inp / ffn_shexp / l_out). Each workgroup
loads its activation row, quantizes q8_1 in registers (deterministic per row →
bit-identical q8_1 → bit-identical dot), then streams weights as today. Kills
~120–160 launches/token of the ~645 live launches. Numerics: q8_1 quantization is
pure per-row deterministic (max→scale→round), so bit-exactness is preserved.
Gate: golden-smoke → KLD → bench-serial (AGENTS.md mandatory sequence).
