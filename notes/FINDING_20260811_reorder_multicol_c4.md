# FINDING 2026-08-11 — C4 reorder-multicol fall-through: +400% real-text prefill, decode held, numerically CLOSER to canonical than ship; pinned KLD gate fails by construction

Receipts: `results/reorder-multicol-20260811T201630Z/` (summary.txt is the index),
KLD receipts `results/kld-20260811T21{0450,0631,1358,1434}Z/`.
Candidate: worktree `/home/frosty40/turbo/worktrees/lx-reorder-multicol`, branch
`lx/reorder-multicol-mkl`, commit `57cffae17` on champion base `c7d3bfe6d`,
`libggml-sycl.so.0.17.0` sha256 `f2b485a5…`. One hunk, +23/−1, default-inert
behind `GGML_SYCL_LX_REORDER_MULTICOL_MKL=1`.

Context: PLAN_20260811_expert_loop_attack step-0 found the entire ~5x real-text
prefill headroom is the blanket reorder guard (`ggml-sycl.cpp:4714-4744`)
force-routing every wide matmul onto 8-column reorder-MMVQ once warmup latches
the reorder flag. C4 narrows the guard to `ne[1] <= MMVQ_MAX_BATCH_SIZE` so
decode keeps reorder-MMVQ and wide batches fall through to fp16/oneMKL.

(Validation was interrupted 2026-08-11 ~15:00 by a hard B70 wedge — a SIGKILLed
`-fa off` leg left the xe blitter queue in a permanent timed-out-job reset loop,
every model load failing as fake OOM; see device-wedge-diagnosis.txt. Cleared by
the 15:59 reboot. Standing hazard: never SIGKILL a llama process on this box.)

## Measurements (all on the candidate binary, ship env, GPU-locked)

| gate | knob OFF | knob ON |
|---|---|---|
| golden-smoke | PASS (bit-parity) | PASS (identical greedy text) |
| KLD vs pinned base | PASS, bit-identical (−0.0 / 100%) | **FAIL 0.053016 / 91.422%** |
| real-text 23K prefill (llama-cli) | 307.3 t/s | **1535.6 t/s (+400%)** |
| real-text decode | 81.0 t/s | **81.4 t/s (held)** |
| tg128 d0 (llama-bench) | — | **153.15 ± 0.22 ≥ 152.5 floor** |
| pp512 (llama-bench) | — | 1160.6 ± 12.3 (no reorder latch in pp-only bench ⇒ unchanged path; −1.1% vs 1174 is noise) |

Unlike the `GGML_SYCL_ENABLE_OPT=0` proxy (which cost −14.7% decode), the
narrowed guard keeps decode on reorder-MMVQ: decode is unharmed.

## The KLD failure is base-path bias, not a quality regression

Three facts, in evidence order:

1. **Every historical KLD pass on record is mean_kld −0.0 / same_top 100%** —
   bit-identical. The pinned base (captured from base-control, post-warmup, in
   the gate env) is itself the reorder-MMVQ path, and every candidate until now
   rode that same path at wide N. The gate has never measured anything but
   bit-identity; C4 is the first candidate that legitimately changes wide-N
   numerics, and distance-to-pinned-base can only punish that.
2. **Canonical triangulation** (driver-canon.sh; reference = `ENABLE_OPT=0`
   linear-weight fp16/oneMKL logits, the upstream ground-truth path for this
   quant; flag verified to gate only reorder machinery):

   | vs canonical | mean KLD | same top-1 | RMS Δp |
   |---|---|---|---|
   | ship path (knob OFF) | 0.056310 | 91.667% | 6.595% |
   | knob ON | **0.033825** | **93.995%** | **4.764%** |

   **The shipped path deviates from canonical MORE than knob-ON does.** C4 moves
   the output distribution toward ground truth on every metric.
3. **Absolute perplexity does not regress — it improves slightly.** At the gate
   geometry (16 chunks), knob-ON PPL 20.468 vs the ship path's ~20.606 once the
   +0.016271 u16-store format constant is accounted for. At 64 chunks
   (driver-ppl64.sh): **OFF 14.7836 ± 0.42131, ON 14.7339 ± 0.41888** (−0.34%,
   within error, direction consistent).

Mechanism of the residual (why knob-ON ≠ canonical exactly): the
`[lx-control-qkv]` fused prefill path intercepts the dense Q/K/V projections at
graph level *before* the `ggml_sycl_mul_mat` guard (marker present in both
ab-on.err and ab-off.err), so the dense projections stay on reordered-MMVQ
(q8_1-quantized activations) under the knob. Same residual explains the prefill
gap to the OPT=0 ceiling (1535.6 vs 1610.8 t/s).

The reorder-aware fp16 dequant kernels were audited read-vs-write for q4_K and
q6_K, dense and MoE layouts (`convert.cpp`/`dequantize.hpp` vs
`reorder_qw_*`): consistent. Note these kernels were dead code in the DNN-off
ship config until this bypass — the guard always diverted before them.

## Status: SHIP-BLOCKED ON GATE POLICY, not on evidence

C4 passes golden, holds every throughput floor, delivers +400% prefill, and is
numerically closer to canonical than what ships today. It cannot pass the KLD
gate *as written* because the gate's reference encodes the old path. Per the
gate's own rule a candidate must never re-pin its reference — so the decision
is the operator's, with three options:

- (a) re-pin the base to the canonical `ENABLE_OPT=0` capture (from the
  control binary, `--capture-base` with `GGML_SYCL_ENABLE_OPT=0`), making
  distance-to-ground-truth the standing metric — ship path would then FAIL the
  gate at today's thresholds (0.056 > 0.01), which is honest but re-baselines
  the whole board;
- (b) keep the pinned base for bit-parity candidates and add the canonical
  triangulation as the arbiter when a candidate legitimately changes numerics
  (amend quality-gate-kld.sh with an `LX_KLD_CANON` second leg);
- (c) accept this receipt as a one-off override and enable the knob in
  serve-laguna.sh.

Until that decision: the knob stays OFF everywhere; serve-laguna.sh untouched;
no promotion recorded. Follow-up candidate **C4b**: width-gate the
`lx-control-qkv` fuse (decode-only) or extend the bypass to it — expected to
close most of the residual KLD-to-canonical and ~5% more prefill.
