# ubatch sweep — 20260806T175255Z — champion binary, official geometry, same window

## What
Runtime-level test of the "split the fused 512-token MoE down-GEMM" idea
(open lead 9) WITHOUT a rebuild: llama-bench `-ub` (n_ubatch) chunks the
prompt, and for a 512-token prompt the fused ffn_out GEMM splits exactly
when `-ub < 512`. 6 arms, same window, 3 samples each, r=3, -b 2048,
official geometry otherwise (model, -ngl 99 -t 16 -sm layer -mg 0 -ts 0,
ctk/ctv f16, GGML_SYCL_DISABLE_GRAPH/DNN=1).

## Results (avg_ts over 3 samples)
| arm   | ubatch | pp512 t/s | tg128 t/s |
|-------|--------|-----------|-----------|
| ub2048-0 | 2048 | 1155.34 (sd 4.47) | 138.186 (sd 0.24) |
| ub1024   | 1024 | 1148.61 (sd 4.45) | 138.194 (sd 0.28) |
| ub512    |  512 | 1157.33 (sd 2.53) | 138.148 (sd 0.26) |
| ub256    |  256 |  852.55 (sd 2.71) | 137.525 (sd 0.22) |
| ub128    |  128 |  598.40 (sd 5.33) | 137.661 (sd 0.26) |
| ub2048-1 | 2048 | 1151.92 (sd 9.29) | 137.899 (sd 0.28) |

## Findings
1. **Chunking prefill is catastrophic, not a lever.** ub256 = −26.0% pp,
   ub128 = −48.1% pp vs ctrl mean (1153.63). The fused single-dispatch
   down-GEMM is load-optimal; the "split into 2x256 / 4x128" experiment
   (open lead 9) is falsified with a runtime knob — no source edit needed.
   [evidence: results/ubatch-sweep-20260806T175255Z/ub2048-0.json]
2. **The drop is a step function of split count, monotonic and
   quantitative**: 1 dispatch (ub>=512) = 1155-1157; 2 dispatches (ub256)
   = 852; 4 dispatches (ub128) = 598. Per-split overhead ≈ 138 µs ≈ the
   per-token decode time — the down-GEMM dispatch is GPU-side BW-bound at
   large size and overhead-bound when fragmented.
   [evidence: benchmark:results/ubatch-sweep-20260806T175255Z/ub128.json]
3. **Decode (tg128) is ubatch-robust**: 137.5-138.2 t/s across ALL arms
   (max spread 0.5%), i.e. decode CPU-side launch cost is invisible at
   this occupancy — kernel time dominates; consistent with lm_head axis
   being closed (VDR/prefetch/format all null).
   [evidence: benchmark:results/ubatch-sweep-20260806T175255Z/ub2048-0.json]
4. **ub512 ≡ ub2048 within drift** (pp +0.32%, tg +0.08% vs ctrl mean,
   both inside ±0.68%) — for a 512-token prompt ub512 never splits, so
   this is the internal consistency check, not a win. Champion -ub 2048
   stays; no submission from this axis.
   [evidence: benchmark:results/ubatch-sweep-20260806T175255Z/ub512.json]
5. **Scoring-model reframe**: board score = decode^0.75 × prefill^0.25
   (LATEST_SCORE.json formula field). With decode pinned at 1.2932× and
   pp at 1.0181×, the literal 2.0 target requires BOTH sides to move
   (pp alone would need 7.4×; decode alone 2.52× = 270 t/s). Prefill-side
   work has 4× less leverage than decode-side work under the formula —
   decode remains the primary lever even though lm_head is closed.
   [evidence: file:results/LATEST_SCORE.json:19]

## Harness bugs fixed along the way
- `with-gpu-lock` is NOT on PATH (it is scripts/with-gpu-lock, a script);
  bare invocation → rc=127 command-not-found. Use $ROOT/scripts/with-gpu-lock.
- llama-bench JSON is a multi-line pretty-printed ARRAY; the line-based
  json.loads parser could never parse it. Whole-file parse with
  [lx-*]/[layer-timer] line filter (finding 21 class).
- BENCH resolution needed the champion worktree path first
  (build-mmadd-decode/bin/llama-bench); $BINTREE/$LX_BIN were empty in a
  non-login shell.
- GEOM needed explicit `-m "$LX_MODEL"` (default models/7B...gguf fails).

## State
- No candidate .so built; champion binary untouched (this axis is
  env/runtime-only). Board: results/LATEST_SCORE.json unchanged at
  1.2181469734433867 (no run in this sweep was a scored submission).
