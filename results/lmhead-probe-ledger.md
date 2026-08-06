# lm_head probe ledger — fused lm_head group wall-time measurement

Date: 2026-08-06 (probed binary = source-repro champion equivalent, src-lmhead-build,
instrumented with env-gated std::chrono probe around the fused mul_mat+add+add2
l_out-39 dispatch; probe is OFF unless GGML_SYCL_LMHEAD_TIMER is set)

## Official geometry (pp512/tg128, timer ON vs SKIP)

| test | timer ON (t/s) | SKIP lm_head (t/s) | delta |
|------|----------------|--------------------|-------|
| pp512 | 1134.76 ± 14.34 | 1129.94 ± 14.14 | −0.42% (noise) |
| tg128 | 135.46 ± 0.23 | 142.39 ± 0.27 | **+5.12%** |

FINAL (timer): 647 lm_head calls, 7,534,309 µs; per-suffix hits: 39:647
(single-suffix — every fused lm_head dispatch is the true l_out-39 group; no collateral)

## tg-only geometry (p=1 n=256, r=1)

| test | timer ON | SKIP | delta |
|------|----------|------|-------|
| tg256 | 136.48 t/s (9117.41 µs/cycle) | 143.00 t/s (8763.79 µs/cycle) | **+4.78%** |

fused lm_head group wall share = 9117.41 − 8763.79 = **353.6 µs per decode token**
(= 4.8% of the ~7.3 ms decode iteration; iteration = gap between consecutive lm_head dispatches)

## Interpretation

- The fused Q6_K lm_head (vocab 100352, 168 MB/token) costs **354 µs/token = 4.8% of decode wall**.
  Ideal bandwidth bound (168 MB @ ~2 TB/s) ≈ 84 µs → the fused kernel runs ~4× off-ideal;
  launch/layout overhead dominates, so a candidate-table prune (exact-greedy + fallback
  recompute, FRONTIER idea #7) can at most recover toward the 354 µs, bounded by +5.12% tg
  (the skip ceiling) — and realistically half that with quality-safe partial prune.
- Hypotheses killed/quantified:
  - Lead 23's "≥2 ms of the 7.2 ms budget" → measured 354 µs, KILLED.
  - Lead 18's "~+3.7% tg via prune" → bounded: ceiling +5.12%, but reaching it requires
    removing the *entire* fused group (skip), which is not quality-safe; realistic
    bitexact partial prune gain ≈ +1.5–2.5% tg → score ≈ 1.24 — insufficient alone for 1.25.
- Prefill (pp512) is lm_head-insensitive (skip delta −0.42%, inside noise) — prefill is
  batch-parallel; lm_head is a single-token-decode-only lever.
- Quality gate: golden-smoke PASSED with the timer instrumentation live (instrumentation
  is invisible; probe is a pure chrono read + stderr print, env-gated).

## Harness

scripts/lmhead-probe-cycle.sh — compile probe object (icpx, probe-build.sh) → swap into
build tree → relink via CMake link.txt (cwd = ggml/src/ggml-sycl, triple-dirname) →
golden-smoke with timer ON → official timed bench → skip-mode ceiling bench.
Env: GGML_SYCL_LMHEAD_TIMER=1 (measure), GGML_SYCL_SKIP_LMHEAD=1 (skip fused group;
output-corrupting, ceiling bound only), GGML_SYCL_LMHEAD_LAYER (default 39).

Raw: results/lmhead-probe-20260806T110533Z/{bench-timer.log,bench-timer.stderr,lmhead-final.txt}
