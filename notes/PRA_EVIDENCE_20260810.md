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

## Performance — public models (FOR the PR body)

llama-bench `-ngl 99 -fa 1 -p 512 -n 128 -r 5`, same box:

| model | metric | master | PR-A | delta |
|---|---|---|---|---|
| Qwen3.5-35B-A3B Q4_K_M (MoE, GQA) | tg128 | PENDING | PENDING | |
| Qwen3.5-35B-A3B Q4_K_M | pp512 | PENDING | PENDING | |
| Qwen3.5-4B Q4_K_M (dense) | tg128 | PENDING | PENDING | |
| Qwen3.5-4B Q4_K_M | pp512 | PENDING | PENDING | |

Receipts: `results/lx-pra-public-ab-<stamp>/{moe,dense}-{master,pra}.json`.

## Pre-push checklist

- [ ] full test-backend-ops green
- [ ] KLD pass on PR-A build
- [ ] `ci/run.sh` local pass
- [ ] attribution grep empty: `git log --format='%an %ae%n%(trailers)' origin/master..HEAD | grep -iE 'co-authored|generated|claude|anthropic'`
- [ ] commit message reworded by you; PR body written by you; AI-disclosure line answered
- [ ] `git push fork sycl/fattn-vec-gqa-decode` then `gh pr create --repo ggml-org/llama.cpp --head newjordan:sycl/fattn-vec-gqa-decode`
- Serial-filing rule: keep only this PR open; B–H stay local until it merges.
