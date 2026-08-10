# FINDING 2026-08-10 — real-text long-ctx prefill is DEVICE-bound; upstream master is 4.8x faster there (MKL/oneDNN FA); champion keeps decode +51%

## Receipts (same 23K wikitext prompt, same geometry ub2048/b4096/c32768, same box)

| build | prefill | decode |
|---|---|---|
| champion (7e1e28cae base, TILE FA at depth, DNN off) | 308.8 t/s | 92.4 t/s |
| master dd1ea5243 (MKL XMX FA + oneDNN SDPA, stock env) | **1493.7 t/s (4.84x)** | 61.3 t/s |

Chain that got here (all receipted in scratchpad + results):
1. Chrono ledger (150K stamps): 69.9% of champion real-text wall in the
   ffn_moe_gate bucket (60.5ms avg vs up 11.1ms — same shape/type).
2. WAIT_TOTAL_NS: the guarded mmid stream->wait() is only 0.19s/76.6s — the
   drain hides inside the UNguarded ids D2H stream->memcpy (UR yield-spin).
3. gdb whole-process sampling (16 rounds x all threads): 504/518 samples idle;
   busy signature = sched_yield spin in UR appendKernelLaunch. Host is NOT
   the wall on real text (unlike synthetic pp512: FINDING_20260810 hostbound,
   98.6% CPU — that regime holds only at shallow depth / tiny device work).
4. Ergo the wall is DEVICE attention-at-depth (TILE FA) + serialization at the
   per-layer ids sync — and master's XMX prompt paths (9d9a6d29f oneMKL FA,
   66fa168a5 oneDNN SDPA non-F16-KV, both Q>=32 K>=1024 gated) attack exactly
   this: measured 4.84x on real text.

## Consequences

- The champion's prefill weakness is invisible on synthetic pp512 (1171 vs
  master ~1147 — looks fine) and catastrophic on real long-context serving.
  Bench-synthetic prefill numbers must never drive serving decisions
  (see FINDING_20260810_serving_prefill_routing_skew for the routing half).
- Serving wants champion-decode + master-prefill. Vehicles, in order:
  (a) BACKPORT 9d9a6d29f + 66fa168a5 (+ fattn-mkl.cpp deps) onto the champion
      tree, DNN runtime-enabled for prefill paths — bounded, attempt started
      in worktree lx-dev-mmid-batch (branch lx/mmid-device-batched-sgemm,
      repurposed);
  (b) the PR-series sycl/all-stack (master + A..H) becomes the serving build
      as the series lands — the strategic fix.
- gate-up-concat revival: DEAD on this model regardless of env — the
  multitoken dual-down path requires down type == gate type in {Q4_K,Q5_K,
  Q6_K}; Laguna down is IQ4_NL (types_ok can never pass). The Aug-10
  "rejection" measured a pure control.
- DNN-off (env.sh line 126) is a DECODE-era verdict; for serving prefill it
  disables the exact paths that win 4.8x. Serve env must diverge from bench
  env once the backport lands.

## Backport v1 attempt (2026-08-10, branch lx/mmid-device-batched-sgemm @ lx-dev-mmid-batch)

Cherry-picked 9d9a6d29f + 66fa168a5 alone onto the champion tag (clean
auto-merge, .so 0ddbff80). Real-text 23K A/B (DNN on, champion env):
**prefill 273.3 / decode 84.6 — WORSE than plain champion (308.8/92.4)**,
champion fuses still firing. Verdict: the two commits without the
intermediate fattn dispatch chain (eef5f3e34 and kin) misroute; a proper
backport must bring the full fattn commit sequence 7e1e28cae..dd1ea5243
(fattn.cpp/fattn-onednn.cpp/fattn-mkl.cpp/fattn-vec.hpp) or the serving
build should simply wait for sycl/all-stack (master + PR series) instead.
Branch kept for the follow-up; do NOT serve from it.
