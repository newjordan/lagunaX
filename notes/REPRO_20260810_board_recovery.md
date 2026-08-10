# REPRO 2026-08-10 — board 152.5 recovered; Aug-10 regression explained

## Verdict

The 1.3105 board (decode 152.545) is REAL and reproduces from the git-preserved
champion source. The Aug-10 "regression" to ~129.7 had TWO stacked causes:

1. **Env-order contamination** (already documented in
   FINDING_20260810_lxbin_env_order_contamination.md): the Aug-10 morning runs
   (llama-bench sha `45e07d09`, 13:20–13:34 series) loaded the treebeard
   `.so` (`abb033c4`) via captured LD_LIBRARY_PATH → they measured the Tier-1
   library (~130 t/s decode = its true speed), not the kernel tree.
2. **A bad rebuild of `benchmark/kernel/build`**: `CMakeCache.txt` was modified
   2026-08-10 08:30 *without* a configure-log update (hand-edited cache or
   `cmake -D`), triggering a full recompile at 08:33 (`04d6d625` — even
   never-touched files like conv.cpp/upscale.cpp rebuilt). That binary runs all
   champion fuse paths (fuse-hit lines identical to the board log) yet decodes
   at 130.17 (leg A, `results/20260810T144440Z`) — a build-config, not code,
   regression. The board-era config was overwritten and is unrecoverable;
   don't use this build dir.

## Receipts

| leg | build | env | decode | score | receipt |
|---|---|---|---|---|---|
| board (2026-08-09) | kernel tree, board-era config | correct | 152.545 | 1.310474 | `results/20260809T191208Z` |
| A | `benchmark/kernel/build` Aug-10 rebuild (`04d6d625`) | correct order, FUSE_NORM_ROPE=1, proof-of-load verified | 130.17 | 1.1539 | `results/20260810T144440Z` |
| D | **fresh canonical build of tag `lx-champion-1.3105-20260810`** (`.so ae6407a4`, flags: Release, icpx, GGML_SYCL=ON, F16=ON, TARGET=INTEL, LLAMA_BUILD_TOOLS=ON) | correct | **152.5418** | **1.305739** | `results/20260810T145715Z` |

Leg D decode matches the board to 0.003 t/s. Prefill 1154.2 vs 1171.0 (−1.4%,
within its historical variance, stddev ~9.7). Score gap −0.36% is prefill-side.

Leg B (FUSE_NORM_ROPE=0 quantification) not re-run: already receipted on sha
55d9290d (143.93 off / 147.47 on, env.sh:79-85 comment + commit bf85534).
Leg C (deliberate contamination) not re-run: FINDING_20260810 already carries
the LD_DEBUG proof; the 45e07d09 sha trail above confirms it.

## Canonical champion binary (new)

`/home/frosty40/turbo/worktrees/lx-champion-tier12/build/bin`
built from hub commit `c7d3bfe6d` (tag `lx-champion-1.3105-20260810`).
- llama-bench `5eba67f6427c83dd1080e9a9fe3364e05d45fb875b0c5e579a78e31c0d6af3a2`
- libggml-sycl.so.0.17.0 `ae6407a41512147447696e4ad6b11069d1444f9e5230a04bbae5a0be67fa2a6e`

`env.sh` LX_BIN default now points here. KLD receipt for this binary: run
recorded in `results/LATEST_KLD.json` (gate rerun 2026-08-10).

## Harness hardening (this commit)

- `bench-serial.sh`: records `so_sha256` (the kernel library, which
  `binary_sha256` never covered — llama-bench's sha was constant Aug 7→9
  while every kernel change shipped in the `.so`) and hard-fails if the
  loader resolves libggml-sycl from outside `$LX_BIN`.
- `env.sh`: stale `LX_BIN` default (treebeard mmadd-decode) → canonical
  champion build; comment states the export-before-source rule.

## Rules going forward

- Never bench `benchmark/kernel/build` again; the canonical build is the
  worktree one, from tagged source, with the flags above.
- A board claim requires: correct-order env + `so_sha256` in the receipt +
  KLD receipt bound to the same binary.
