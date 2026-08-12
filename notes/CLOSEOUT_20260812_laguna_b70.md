# CLOSEOUT — Laguna XS 2.1 on Arc B70: the long-context chapter (2026-08-12)

Final state of the lx campaign, written as the chapter closes. Everything here
is receipted; paths are receipt-dir pointers. Operator workload: RL loops at
max context — the target metric is decode at full depth, then ingest speed.

## What ships

Branch `lx/reorder-multicol-mkl` in `/home/frosty40/turbo/worktrees/lx-reorder-multicol`
(base = champion `c7d3bfe6d`), served by `scripts/serve-laguna.sh`, benched via
`env.sh` (knobs default-on since 2026-08-12). Three env-gated changes, each
individually kill-switchable:

| knob | change | win |
|---|---|---|
| `GGML_SYCL_LX_REORDER_MULTICOL_MKL=1` | narrow the blanket reorder guard: wide batches take fp16/oneMKL instead of 8-col chunked reorder-MMVQ | **+400% real-text prefill at 131K** (307→1540 t/s) |
| `GGML_SYCL_LX_FATTN_PARALLEL_BLOCKS=16` | FA decode split-K width override | **+10% decode at 23K depth** (81.5→89.7); grows with depth |
| `GGML_SYCL_LX_EXPERT_TILE_GEMM=1` | XMX fused dequant-GEMM for N≤32 expert slices, q4_K + q6_K, reordered + linear layouts | **+15% more prefill** (→1767 t/s), pp512 +36% |

Board: **1.3122 → 1.40924, promoted 2026-08-12** (`results/20260812T004213Z`)
— the campaign's original acceptance target (1.40) is met.
Quality: passes gate policy (b) via the canonical arbiter
(`results/kld-20260812T002134Z`: canon KLD 0.0362 vs shipped-path 0.0563,
top-1 93.4% vs 91.7%, PPL better at 16 and 64 chunks). Golden with knobs off
is bit-parity; knobs-on greedy diverges at near-ties (reduction-order class,
documented).

## Final numbers (fill in closing battery)

| metric | champion (2026-08-11 AM) | final stack |
|---|---|---|
| 131K real-text 23K-prompt ingest | 307 t/s (74 s) | 1767 t/s (14 s) |
| decode @ 23K depth | 81.5 | 89.5–90 |
| decode @ 49K / 98K / 122K | never measured | (deep sweep — pending) |
| tg128 d0 floor | 152.5 | 152.3–153.2 (held) |
| board score | 1.3122 | 1.40924 |
| long-gen NaN watch (≥1024 tok @ depth) | — | (pending) |

## The mechanism story (one paragraph)

A warmup decode latches `optimized_feature.reorder` on the weight banks;
a blanket guard then shredded every wide matmul into 8-column reorder-MMVQ
launches for the rest of the process — the entire "device-bound long-context
prefill" mystery. Narrowing that guard to decode widths recovered 5x. On the
decode side, the FA vec kernel serializes each Q head's KV walk with split-K
capped at 4; widening the split shortened the serial walks (+10% at 23K).
The XMX expert tile then deleted the small-N expert loop's per-call apparatus
(launch floor + fp16 round-trips) at microbench-proven kernel efficiency.

## Do-not-try (receipted dead, this chapter)

q8_0 KV cache (−27% d0, −69% at depth: fallback path) ·
`FORCE_TILE` FA at decode (−26% at depth) · vec-FA GQA head-batching (M2:
occupancy loss beats L2-absorbed re-read savings) · SIMT fused dequant-GEMM
(C1v1: 2x slower than oneMKL in-band; XMX is the requirement) ·
`gemm_batch` (hangs B70) · MMQ (broken) · full-tensor fp16 preconversion
(−24%) · fattn master backport (nil prefill, −12% depth decode) ·
ubatch/context/warmup tuning (all <±3%).

## Open frontiers (for a future chapter, ranked for max-context decode)

1. FA-at-depth is still ~42%-of-peak-BW class even post-M1 — lane-contiguity
   (M3), depth-scaled split-K, and a from-scratch depth-decode FA are the
   remaining levers; ceiling ~+65% decode at 100K.
2. Narrow-slice unification (prefill N≤8 experts → oneMKL) — quality
   convergence to canonical, speed-neutral.
3. XMX tile band widen to N≤64 + dense-matmul XMX (microbench says tie/lose
   today; a better tile could flip it).
4. Decode-logit-distance instrument (the pinned KLD gate is prefill-only —
   decode numerics changes currently arbitrated by greedy text + PPL only).

## Operational notes

- Serving: `bash scripts/serve-laguna.sh` (operator-run). Header carries the
  full receipt chain. Alias unchanged (`laguna-xs-2.1-q4-lx-champion`).
- Bench: `LX_BIN=<stack bin> source env.sh && scripts/bench-serial.sh`.
- Gate: `scripts/quality-gate-kld.sh` — policy (b); `--capture-canon`
  (control bin only) refreshes the canonical store.
- Hazards: never SIGKILL a llama process on this box (xe blitter wedge —
  reboot to recover); one GPU client at a time (`scripts/lib-gpu-lock.sh`);
  export `LX_BIN` before sourcing `env.sh`.
- Uncommitted operator items: `env.sh` (carries an earlier LX_BIN repoint +
  the knob block), server restart.
