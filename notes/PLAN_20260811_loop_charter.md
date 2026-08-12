# Loop charter — 2026-08-11, self-directed campaign (user-approved)

Operator approval (2026-08-11 evening session): set my own loop, run the lx
campaign as a self-directed project, subagents permitted. This file is the
loop's re-entry point: each iteration reads it, does ONE gated unit of work,
updates it, commits findings. AGENTS.md rules bind every iteration unchanged.

## Standing boundaries (not mine to move)

- **KLD gate policy RESOLVED 2026-08-12: operator chose (b)** — canonical
  arbiter added to quality-gate-kld.sh (b13e51f); full stack PASSES
  (kld-20260812T002134Z). serve-laguna.sh now ships the stack (e9ff366);
  the server restart itself is operator-run (classifier boundary). The
  pinned base is still never re-captured; canonical store is control-bin
  only (--capture-canon).
- Decode floors: tg128 d0 ≥ 152.5; real-text at-depth decode ≥ its same-session
  control. Never buy prefill with decode.
- GPU: exclusive lock always; never SIGKILL a llama process (blitter wedge —
  reboot required); settle-and-retry after any abnormal exit.
- Receipts under `results/<stamp>/` with binary sha; findings committed same
  session (no AI attribution on commits); one variable per measurement.
- Subagents: read-only recon/analysis in parallel is fine; GPU measurements
  stay serialized through the lock, one at a time.

## Operator workload statement (2026-08-12, overrides priority ordering)

The operator runs RL loops at max context — never short-context serving.
**The metric that matters is decode throughput at full depth (50K-131K KV)**,
then max-context prefill (ingest), then everything else. tg128 d0 remains a
floor (never regress it) but is not the optimization target.

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
- 2026-08-11 iter 3 DONE: M1 landed (d61bdf435, GGML_SYCL_LX_FATTN_PARALLEL_BLOCKS,
  pb=16): +10.5% tg128 at d24576 (86.5->95.6), +9.7% real-text 23K decode
  (81.8->89.7), d0 floor intact, stock path bit-identical when unset. Golden
  PASS; greedy-at-depth near-tie flip = reduction-order class. INSTRUMENT GAP:
  pinned KLD gate is prefill-only, cannot see decode-FA numerics.
- 2026-08-11 iter 4 DONE: M2 (vec GQA head-batching) FALSIFIED — loses to
  M1-only at every split width (best 91.5 vs 95.6) and violates the d0 floor
  (−5%); L2 was already absorbing the KV re-reads. Reverted; patch preserved
  in results/p1-m2-gqa-*/. Branch = C4+M1 (d61bdf435, .so 94015650).
- 2026-08-11 iter 5 DONE: P2 census at serving geometry — 264,339 expert
  oneMKL calls / 22.8M rows per 23K pass, mean N=86; C1 ceiling ~5x further
  prefill (FINDING_20260811_p2_expert_gemm_sizing has the design constraints).
- 2026-08-11 iter 6 DONE: C1 v1 (SIMT dequant-GEMM) FALSIFIED — ~2x slower
  in-band than per-expert oneMKL (pp512 1160->558; 131K prefill 1554->1108),
  numerics unproven (golden diverges, canonical distance worsens). Reverted;
  patch in results/p2-c1-tile-*/. Lesson: C1 v2 must be XMX/joint_matrix with
  SLM-staged dequant, preceded by a dst-parity verify mode and a standalone
  XMX microbenchmark.
- 2026-08-11 iter 7 DONE: XMX microbench GO — fused XMX beats MKL 1.33x@N16
  / 1.21x@N32, tie@64, lose@128; numerics better than MKL. Revised C1 v2
  prize: +15-20% prefill@131K (band N<=32). benchmark/xmx-dequant-gemm/.
- 2026-08-11 iter 8 DONE: C1 v2 LANDED (05755f5f3): +11.7% prefill@131K
  (1726 t/s), +33% pp512 (1547), decode untouched (interleaved A/B), canonical
  KLD unchanged, verify-mode numerics receipted. DAY-1 CLOSE: stacked knobs =
  5.6x max-context prefill + 10% at-depth decode vs shipped champion, all
  dark behind env, blocked only on the gate-policy decision.
- 2026-08-11 iter 9 DONE: reorder-SoA tile path receipted on real text (208
  reordered dispatches, worst err 2.6e-02 = linear class). C1 v2 complete.
- 2026-08-12 iter 10 DONE: policy (b) implemented + full stack PASSES the
  amended gate via canonical arbiter; serving config shipped (e9ff366);
  server restart with operator. PACE UP per operator: shorter wakeups,
  parallel drafts.
- 2026-08-12 iter 11 DONE: q6_K band committed (9272b77c9: 131K prefill
  1767 t/s, pp512 1573, reorder-SoA q6_K verified). BOARD PROMOTED:
  1.3122 -> 1.40924 (results/20260812T004213Z), decode held, KLD via
  canonical arbiter. **The campaign's original acceptance target
  (LX_TARGET_SCORE 1.40, LOOP_RUNBOOK) is MET.** Serving launch still
  operator-run; env.sh commit still operator-run (pre-existing LX_BIN diff).
- 2026-08-12: operator workload statement recorded — max-context decode is
  THE target. Deep-depth sweep launched (d 49K/98K/122K x pb 16/32/64):
  first-ever measurement of this model's decode at RL-loop depths, plus
  depth-scaled split-K tuning. N<=64 widen and other prefill items demoted
  behind decode-at-depth work.
- 2026-08-12 CHAPTER CLOSED: deep sweep banked (pb=16 depth-flat; +13.3% at
  122K vs stock); M3 falsified (−22%); due-diligence battery green (golden
  x3, canonical-arbiter KLD pass, 1536-token NaN watch clean at 131K on the
  stack). CLOSEOUT_20260812_laguna_b70.md is the book. Loop STOPPED.
  Operator items outstanding: server start, env.sh commit.
