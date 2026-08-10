# PR-A evidence sheet — sycl/fattn-vec-gqa-decode

Branch: `sycl/fattn-vec-gqa-decode` @ `14228a37d` (worktree
`turbo/worktrees/pr-a-fattn-vec`), off upstream master `dd1ea5243`.
Diff: 2 files, +11/−4 (`fattn.cpp` dispatch, `fattn-vec.hpp` nbatch).
Commit message is a placeholder — reword it yourself before pushing
(`git commit --amend`); upstream requires human-authored prose + the
AI-usage disclosure line in the PR template.

## What the change is (two units, one theme: decode fattn)

1. `fattn.cpp`: for single-token decode (`Q->ne[1]==1`, non-quantized KV),
   return the VEC kernel also when the GQA optimization applies (previously
   fell through to TILE). One Q row cannot fill TILE's KQ tiles.
2. `fattn-vec.hpp`: in the 256-thread VEC branch, pass `nbatch_fa =
   nthreads_hw` (256) instead of `D` (128) — nbatch must equal the kernel's
   K stride; at 128 every second K work-group was dead and
   `parallel_blocks > 1` forced the combine pass + two dst_tmp allocations
   per layer.

## Hardware/software block (verbatim for PR body)

- GPU: Intel Arc Pro B70, 256 CUs, 30.3 GiB (`xe` driver)
- Host: AMD Ryzen 9 5950X, Linux 7.0.0-28-generic (Ubuntu 24.04)
- Runtime: compute-runtime 26.18.38308.1, libze1 1.28.2
- Compiler: Intel oneAPI DPC++/C++ 2026.0.0 (icpx), Release,
  `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_TARGET=INTEL`

## Correctness

| check | result | receipt |
|---|---|---|
| `test-backend-ops test -b SYCL0 -o FLASH_ATTN_EXT` (PR-A build) | **OK, 0 FAIL** | task log bn00pfdh7, 2026-08-10 |
| full `test-backend-ops test -b SYCL0` (PR-A build) | PENDING | `results/lx-pra-public-ab-*/pra-full-ops.log` |
| KLD gate vs pinned base (internal, Laguna) | PENDING (run after public chain) | |
| context: `-o MUL_MAT_ID` on clean master | OK (issue #25455 does not reproduce here) | `notes/RECEIPT_20260810_mmid_precheck_master.md` |

## Performance — internal (Laguna XS 2.1 Q4_K_M, serial pp512+tg128, 5 reps)

NOT for the PR body (private model) — internal confirmation only:

| build | decode tg128 | prefill pp512 | receipt |
|---|---|---|---|
| master `dd1ea5243` | 111.31 t/s | 1147.3 (1.0071×) | `results/20260810T151824Z` |
| PR-A | **113.78 t/s (+2.22%)** | 1151.8 (1.0111×) | `results/20260810T151906Z` |

## Performance — public models

llama-bench `-ngl 99 -fa 1 -p 512 -n 128 -r 5` (depth runs `-r 3`), same box.
Receipts: `results/lx-pra-public-ab-20260810T152747Z/`.

Depth 0 (shallow KV — deltas within noise):

| model | metric | master | PR-A | delta |
|---|---|---|---|---|
| Qwen3.5-35B-A3B Q4_K_M (MoE) | tg128 | 86.20 | 86.00 | −0.2% |
| Qwen3.5-35B-A3B Q4_K_M | pp512 | 1286.47 | 1299.62 | +1.0% |
| Qwen3.5-4B Q4_K_M (dense) | tg128 | 108.90 | 109.33 | +0.4% |
| Qwen3.5-4B Q4_K_M | pp512 | 4372.31 | 4405.51 | +0.8% |

**Depth sweep (MoE) — BLOCKING FINDING, 2026-08-10:**

| depth | metric | master | PR-A | delta |
|---|---|---|---|---|
| 4096 | tg128 | 81.59 ±0.15 | 79.73 ±0.25 | **−2.3%** |
| 16384 | tg128 | 74.47 ±0.06 | 66.89 ±0.28 | **−10.2%** |
| 4096 | pp512 | 1062.80 | 1212.96 | +14.1% (unexplained, investigate) |
| 16384 | pp512 | 968.74 | 884.85 | −8.7% (unexplained, investigate) |

**RESOLVED 2026-08-10 — attribution complete, PR-A re-scoped.**

Isolation depth sweep, Qwen MoE tg128 @ d4096 / d16384:

| variant | @4096 | @16384 | verdict |
|---|---|---|---|
| master | 81.59 ±0.15 | 74.47 ±0.06 | reference |
| nbatch-only | **82.41 ±0.03** | **75.27 ±0.02** | +1.0% / +1.1% — unconditional win |
| dispatch-only | 79.91 ±0.51 | 67.50 ±0.06 | reproduces the full regression |

Laguna champion (via `GGML_SYCL_FATTN_FORCE_TILE`), tg128:
VEC 102.71 vs TILE 79.55 at d16384 → **VEC +29.1% on Laguna** — the same
dispatch that loses 10% on Qwen. Kernel choice is model-shape-dependent
(GQA ratio / SWA mix / head geometry).

Decisions:
- **PR-A re-scoped to the nbatch fix alone**: branch amended to `a9751d6c0`
  "sycl: fix flash-attention vec nbatch_fa to match the kernel K stride",
  1 file, +7/−1. Wins or is neutral on every model/depth measured.
- **VEC-for-GQA-decode dispatch DEFERRED** (removed from the series): needs a
  cross-model heuristic (crossover data above), not a constant policy. Stays
  in the champion tree (correct for Laguna; +29% at 16K depth strengthens the
  serving package's long-context claims). Revisit with a gqa_ratio/K-length
  model after the main series lands.

Full-ops context: full `test-backend-ops -b SYCL0` aborts identically on
clean master and PR-A (16 CONV_2D + CPY pre-existing B70/driver failures,
same crash signature; receipts `master-full-ops.log` / `pra-full-ops.log`).
PR-A introduces zero new failures; FLASH_ATTN_EXT green.

## Pre-push checklist

- [ ] full test-backend-ops green
- [ ] KLD pass on PR-A build
- [ ] `ci/run.sh` local pass
- [ ] attribution grep empty: `git log --format='%an %ae%n%(trailers)' origin/master..HEAD | grep -iE 'co-authored|generated|claude|anthropic'`
- [ ] commit message reworded by you; PR body written by you; AI-disclosure line answered
- [ ] `git push fork sycl/fattn-vec-gqa-decode` then `gh pr create --repo ggml-org/llama.cpp --head newjordan:sycl/fattn-vec-gqa-decode`
- Serial-filing rule: keep only this PR open; B–H stay local until it merges.
