# Action Plan — from board 1.2287 toward the 1.40 acceptance

Date: 2026-08-07 (20:36Z) · Author: angel (this session)

## Goal & done-condition

- **Goal:** make Laguna XS 2.1 serial decode+prefill on the Arc B70 faster by
  editing kernel source in `benchmark/kernel/`, with **no quality regression of
  any form** (KLD gate + golden smoke + anti-downgrade promotion).
- **Done-condition:** `bash scripts/loop-accept.sh` passes — score ≥ **1.40**
  (decode exponent 0.75; decode is the load-bearer).
- Current board: **1.228740** (`results/20260807T183027Z`, decode 140.60 t/s =
  7.1345 ms/step, prefill 1155.9 t/s). Blob sha `ba06bbd3`, KLD receipt
  `kld-20260807T183231Z` (mean_kld −0.0, same_top 100%).

## Gap math (what 1.40 actually needs)

- 1.40 requires decode_speedup ≈ 1.40^(1/0.75) = 1.573 → **167.46 t/s =
  5.9715 ms/step** → must **save 1.163 ms/step** from 7.1345 ms (prefill term
  is pinned at 1.0147×; it cannot buy decode).
- Repriced wall (bwbench14/15 audit, zero kernel edits — this run's decisive
  result): 7.1345 ms ≈ **6.3 ms (~88%) per-dispatch turnaround / dependency
  stall** + **~0.8 ms warm deck streaming** (978.7 MiB at ~1.4 TB/s warm).
  The old 219–227 GB/s anchors were cold-first-touch artifacts; rate-parity
  lever (lead 6) is dead. Byte-bound theories are dead.
- Funded mechanism classes, with measured warm prices:
  - **Per-dispatch deletion** ~1.2–3.5 µs/launch (memo A/B: 1.19 µs).
  - **Dependency-hop collapse** ~7–9 µs/op — the only 1.40-scale class.
- Budget to 1.40: ~1.16 ms/step. Plan below stacks per-dispatch deletion first
  (cheap, KLD-white), then dependency-hop collapse on the two hottest chains.

## Mandatory sequence — every candidate, no exceptions

```
mutate kernel source -> build -> scripts/golden-smoke.sh
  -> LX_BIN=<bin> bash scripts/quality-gate-kld.sh
  -> bash scripts/bench-serial.sh --note "<what changed>"
```

- GPU lock: `scripts/lib-gpu-lock.sh` (never kill someone else's job).
- **Always pass explicit `LX_BIN`** — env.sh default points at a KLD-failing
  binary.
- One variable per measurement: same flags, same binary, one change.
- Never re-capture `correctness/golden.json`; never work around promote-gate.
- KLD `mean_ln_ppl_ratio 0.016271` is a **u16-store constant**, not candidate
  drift — ignore it, watch `mean_kld` and `same_top_pct` instead.
- `benchmark/kernel/` is git-untracked — kernel edits live on disk only; save
  every change as a patch under `results/<stamp>/` and record the build sha.

## Candidate pipeline (in order)

### A. q8_1 conversion memo for attn_norm-X (per-dispatch deletion ×120) — NEXT
- **Lead:** decode launch ledger (129-step trace): 241 mm ops/step =
  482 kernels (241 quantize_row_q8_1 + 241 reorder_mul_mat_vec) + 80 rope +
  80 set_rows + 2 get_rows + 1 add ≈ 965 visible launches/step. attn_norm-X is
  q8_1-converted **4×/layer** (Q, K, V, attn_gate_proj consecutive consumers;
  516 src1 hits per layer = 4×129) → 160/241 conversions are redundant.
- **Change:** memoize the q8_1 quantize+reorder of the shared src1 row per
  (tensor, graph step); the 3 duplicate consumers reuse the buffer. The
  conversion is deterministic → **bit-identical q8_1 values → KLD-white by
  construction** (no arithmetic change).
- **Expected:** delete ~120 launches/step × 1.2–3.5 µs ≈ **0.14–0.42 ms/step**
  (12–36% of the 1.163 ms budget) → decode ≈ 143–149 t/s, score ≈ 1.24–1.25.
- **Risk/verification:** memo keying + invalidation per graph step; verify
  golden match + KLD same_top 100% + RMS dp ~0.001% (identical-arithmetic
  proof). If numeric parity breaks, revert — not worth a KLD fight.
- Files: `benchmark/kernel/ggml/src/ggml-sycl/mmvq.cpp` (quantize sites),
  `ggml-sycl.cpp` (dispatch).

### B. Epilogue 3-in-1: gate+softplus+O on the MoE down path (hop collapse) — next
- **Lead:** open lead 23 — the +1.64% PPL drift rides the rms_norm+mul / mm+add
  path; the softplus-mul fuse (`ggml_sycl_fuse_softplus_mul`,
  ggml-sycl.cpp:4963) and MoE dual-swiglu (`mul_mat_vec_q_moe_dual_swiglu_reorder`,
  mmvq.cpp:2767) exist but the down path's weighted reduce + gate apply +
  output write are still separate hops.
- **Step 0 (spike):** locate the exact down-path fusion site, confirm current
  per-op warm prices for its hops (reuse bwbench-style A/B, one variable), and
  confirm the graph order actually permits the 3-in-1 before writing it.
- **Change:** fuse the residual-add / gate-multiply / output write into the
  down MMVQ kernel (extend `g_mmvq_row_addend` mechanism).
- **Expected:** ~7–9 µs/op × number of hops collapsed; this is the 1.40-scale
  class — the only mechanism that can clear the full 1.163 ms on its own.
- **Risk:** numerics — keep fused math in the same order as the unfused graph
  (gate*softplus then *down then +residual); KLD must stay −0.0/100%.

### C. Re-audit the launch ledger after A+B land
- Re-run the 129-step trace + warm audit; re-account wall = deck streaming
  (~0.8 ms) + turnaround remainder. Re-rank remaining hops by µs. If the wall
  is still turnaround-bound with headroom inside kernels → D.

### D. Spike only: XMX dpas on the q4_K dot loop
- `vecdotq.hpp:227-249` = scalar dp4a, 4 dp4a per 32 weights; `gpu_has_xmx`
  (common.cpp:58) has zero call sites. dpas is the only 8–16× lever on the
  per-kernel compute — **but** the warm audit says the wall is turnaround, not
  compute. Only pursue if C shows compute exposure inside kernels. Microbench
  dp4a dot vs pure read first; device-lost risk (use nd_range=(ndr,WG)).

### E. Prefill-only (guarded, secondary)
- MMVQ ncols cap 8→16/32 (moves N=9..16 groups off 2 MiB f16 conv + 17–21 µs
  oneDNN floor; KLD risk = q8_1 vs f16 numerics) or shape-keyed f16-view cache
  for hot expert slices (VRAM-bound). Prefill term is 1.0147×; each point must
  clear the decode floor — never buy pp with decode.

## Dead / closed (do not reopen without new evidence)
- Runtime-knob plane: exhausted, all drifts are regressions (21-receipt ledger).
- Rate-parity envelope (219→253 GB/s): dead — warm rate ~1.4 TB/s, 6× funding.
- q6_K→q4_K re-quant: ~0.4% at warm — dead as a rate lever.
- Rope+set_rows fuse: already banked (scored tg128 fires mode=2; slower than
  champion's mmid path).
- MMVQ ns=4: rejected (decode +0.51%, prefill −0.7%). Board stays ns=8.
- lm_head prune / prefetch: probe-only; gate detects lm_head tampering; prefetch
  A/B was champion-vs-champion (mmvq.cpp TU never compiled) — fixed harness
  exists (lmhead-q8-cycle.sh) but the axis measured null.

## Integrity bookkeeping per candidate
1. Save patch → `results/<stamp>/`; record blob sha (`sha256sum` of
   `libggml-sycl.so`, first 8 hex).
2. KLD receipt for the candidate bin (not control-vs-control) — promote-gate
   build-match guard enforces this.
3. `results/LATEST_*` updated by scripts; governance commit `2af8ca5` pattern
   (harden gate capture guard, record rejected state) — keep it clean.
4. Update `notes/` ledger entry for the landing (SHIP_ / FINDING_ per
   convention) before starting the next candidate.

## Non-goals / traps
- No new knob sweeps (~700 run dirs, ~300 audit scripts already exist).
- No multi-slot campaigns — serial track only.
- No `LX_ALLOW_UNGATED` for candidates; no decode-regression acceptance
  without an explicit written reason.
- OneAPI setvars chain clobbers `BIN_DIR` — namespace script-local vars
  (`EMB_BIN` pattern).
