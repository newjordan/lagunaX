# Loop charter — 2026-08-11, self-directed campaign (user-approved)

Operator approval (2026-08-11 evening session): set my own loop, run the lx
campaign as a self-directed project, subagents permitted. This file is the
loop's re-entry point: each iteration reads it, does ONE gated unit of work,
updates it, commits findings. AGENTS.md rules bind every iteration unchanged.

## Standing boundaries (not mine to move)

- **KLD gate policy** stays as-is until the operator answers the question in
  FINDING_20260811_reorder_multicol_c4 (re-pin to canonical / add canonical
  arbiter / override). Until then: no promotion of wide-N-numerics candidates,
  `serve-laguna.sh` untouched, the pinned base never re-captured.
- Decode floors: tg128 d0 ≥ 152.5; real-text at-depth decode ≥ its same-session
  control. Never buy prefill with decode.
- GPU: exclusive lock always; never SIGKILL a llama process (blitter wedge —
  reboot required); settle-and-retry after any abnormal exit.
- Receipts under `results/<stamp>/` with binary sha; findings committed same
  session (no AI attribution on commits); one variable per measurement.
- Subagents: read-only recon/analysis in parallel is fine; GPU measurements
  stay serialized through the lock, one at a time.

## Priority queue (owner: the loop; reorder with reasons, in commits)

P1 **Decode-at-depth wall** (user priority: max context). 81 t/s at 23K depth
   vs 152.5 at d0 — the un-attacked frontier. First unit: attribution, not
   mutation — where does the at-depth decode token go? (KV-read BW at f16
   ctk/ctv? TILE-FA kernel time? host?) Instruments: b70-profile skill
   (power/energy saturation), b70-kernel-trace, chrono ledger at depth.
   Candidate axes after attribution: KV cache q8_0 (quality-gated — it changes
   numerics, so it inherits the same canonical-triangulation methodology as
   C4), FA tile shape at depth, KV layout.
P2 **C1: expert-tile quantized GEMM** on the post-C4 oneMKL path (now live:
   258K wide expert GEMMs/pass exist again). Step-0 histogram fixes the tile:
   token-tile 64, grid-stride over N. Arithmetic ceiling ~200 ms/ubatch floor
   vs 34 s measured today → the next multiple lives here. Evidence-first:
   count/size the post-C4 oneMKL expert calls at serving geometry before
   writing the kernel.
P3 **Narrow-slice unification** (quality convergence): route prefill-context
   N≤8 expert slices to oneMKL via a caller flag; expected speed-neutral,
   closes most of the remaining ON-vs-canonical distance. Only worth building
   if the gate-policy answer makes convergence-to-canonical the metric.
P4 **Gate tooling**: add an OPTIONAL canonical-triangulation leg to
   quality-gate-kld.sh (`LX_KLD_CANON=…` second score, informational output,
   pinned verdict unchanged) so future wide-N candidates get both numbers in
   one run. Does not change promotion semantics — that stays with the operator.

## State (update every iteration)

- 2026-08-11: C4 validated end-to-end and certified at -c 131072
  (308→1540 t/s prefill, decode held, ON closer to canonical than ship at all
  geometries). Ship-blocked on gate policy only. Receipts:
  results/reorder-multicol-20260811T201630Z. B70 healthy post-reboot.
- 2026-08-11 iter 1 (P1 attribution) DONE: decode-at-depth is FA-kernel-bound
  (~42% of peak BW on the ~10 full-attention layers; Laguna is interleaved
  SWA-512, GGUF-decoded). q8_0 KV = DEAD (-27% d0, -69% d24.5K). Power 139 W
  of 230 W cap. Prize: +24% decode at 24.5K, ~+65% at 100K if BW-ideal.
  FINDING_20260811_p1_decode_depth_attribution. Receipts:
  results/p1-decode-depth-20260811T214831Z.
- 2026-08-11 iter 2 DONE: decode FA is the VEC kernel (not tile); mechanism
  = ncols2=1 ⇒ 6× KV re-read per token (604 MB vs 100.7 MB unique at 24.5K)
  + parallel_blocks capped at 4. Env probes: FORCE_TILE dead (−26% at depth);
  DECODE_NTHREADS=128 marginal (+2.4% depth, −0.7% d0). Finding addendum in
  FINDING_20260811_p1_decode_depth_attribution.
- Next (iter 3): M1 parallel_blocks env override (build+gate), then M2
  vec-kernel ncols2=2 GQA batching. Both stack on the C4 branch, env-gated
  default-OFF, d0 floor 152.5 mandatory.
