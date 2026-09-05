# September 4 PR handoff

Internal working note for tomorrow; not an upstream PR description.

## llama.cpp: two independent commits ready for human review

Both branches start at upstream `4d9176092d00586775af140581bb0b558ddc4389`.
No PR has been opened. The contributor should review the code, write the PRs,
and disclose AI assistance; both commits carry `Assisted-by: Codex`.

| Branch | Commit | Scope |
|---|---|---|
| `sycl/q8-moe-review-20260904` | `088103051` | Q8_0 expert reorder with original MoE accumulation order |
| `sycl/dmmv-tail-20260904` | `2ae2b703e` | Bounds check for DMMV row tails |

Local worktrees under `/home/frosty40/turbo/worktrees/` are respectively
`pr-sycl-q8-review-20260904` and `pr-sycl-dmmv-tail-20260904`.

Q8_0: 16/16 shared-weight reuse, 75/75 Q8_0 MoE, and 39/39 selected dense
tests pass. On B70, Qwen3.6-35B-A3B UD-Q5_K_XL decode improves from 83.161 to
85.196 tokens/s (+2.45%); prefill is neutral. Two measured expert kernel
geometries improve 6.17x and 6.67x. Four WikiText chunks give matching PPL
8.732186 and 100% top-token agreement with upstream; this is a short sample,
not a broad quality guarantee.

DMMV: upstream reproduces a NaN failure on a 96-column input in three runs.
The fix passes 16/16 reuse and 230/230 quantized MoE cases with reorder disabled.
Aligned-row performance is neutral. Both branches add the same regression
fixture; retain it only once when integrating both changes. Release builds and
diff checks pass; other GPUs and full upstream CI were not tested.

The relevant open item is [llama.cpp #27517](https://github.com/ggml-org/llama.cpp/issues/27517).
The account has no open authored upstream issues or PRs; all four authored PRs,
including RMS fusion #27610, are merged as of this audit.

## Publishing boundary

The llama.cpp commits are **local only**. The fork has Actions enabled and its
master is substantially behind upstream. A new-branch push can include upstream
changes matching branch-unrestricted path filters, including Python type and
requirements checks, pre-tokenizer hashes, operator docs, and Copilot setup.
Standing machine policy requires explicit authorization for hosted workflow
runs; the request to push alone does not grant it. No settings or workflows
were changed. Resolve that authorization before executing:

```sh
git -C /home/frosty40/turbo/worktrees/pr-sycl-q8-review-20260904 push -u fork sycl/q8-moe-review-20260904
git -C /home/frosty40/turbo/worktrees/pr-sycl-dmmv-tail-20260904 push -u fork sycl/dmmv-tail-20260904
```

Recheck upstream before submission. Do not push to `ggml-org/llama.cpp` directly.

## Turbo fixes

The GPU lock now recognizes Linux's truncated backend-test process name,
does not exempt GPU embedding servers, and distinguishes explicit zero GPU
layers from values such as `01`. The audit exercises five actual process names;
all pass, as do shell syntax and focused argument-classification checks.
These are harness fixes, not a new model champion or board promotion.

The `lagunaX` main push also includes the existing local commit `2265a38`, which
preserves stderr when releasing the GPU lock. Unrelated untracked work was not
staged. This repository has no GitHub Actions workflows.

## Local evidence

Full audit, commands, model census, patches, and final measurements:
`/home/frosty40/turbo/results/pr-readiness-20260904/README.md`.
Use the `final/` v2 results; the earlier Q8 candidate was superseded after
detecting arithmetic drift. The local gpt-oss file named Q8_0 has MXFP4 experts
and is not evidence for Q8_0 expert speedups. Benchmarks are finished and the
GPU lock is released. Large measurement artifacts remain local.
