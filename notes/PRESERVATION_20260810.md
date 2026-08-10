# PRESERVATION 2026-08-10 — champion + Tier-1 trees under git

Step zero of the upstream PR-packaging campaign: both modification tiers are
now committed and tagged in the hub repo (`/home/frosty40/llama.cpp`).

## Commits

| tree | branch | commit | tag | contents |
|---|---|---|---|---|
| Champion Tier 1+2 (`lx/benchmark/kernel`) | `private/lx-champion-tier12-20260810` (worktree `turbo/worktrees/lx-champion-tier12`) | `c7d3bfe6d` | `lx-champion-1.3105-20260810` | 16 files, +7446/−441 on upstream `7e1e28cae` |
| Tier-1 base-control | `private/treebeard-base-control-20260728` (worktree `turbo/worktrees/treebeard-base-control-latest`) | `97cb79588` | `lx-tier1-snapshot-20260810` | 8 files, +4114/−189 on upstream `7e1e28cae` |

**Correction to prior surveys:** the champion diff is 16 files, not 13. The
extra three, caught by full-tree diff during preservation:

- `src/llama-graph.cpp` (+9/−4) — **ISWA k-last expand reorder**: expands
  `k_cur` after `v_cur` in `build_attn` so the rope fuse can write directly
  into the KV cache. **Load-bearing for the rope-fuse family (PR-G).**
  Not a ggml-sycl-local change; needs its own upstream justification.
- `ggml/src/ggml-sycl/CMakeLists.txt` (+5/−1) — hardcoded oneAPI link
  fallback for this box. Local-only, never upstream.
- `tools/CMakeLists.txt` (+6/−1) — makes mtmd tool optional (submodule absent
  in vendored tree). Local-only, never upstream.

## Binary hash ledger

| binary | sha256 | role |
|---|---|---|
| `lx/benchmark/kernel/build/bin/llama-bench` | `0f450b3418646d6a8cba06cd298b004b64419de31cfb46c07de5c8607a95e8f7` | current champion-tree build (2026-08-10 rebuild) |
| `lx/benchmark/kernel/build/bin/libggml-sycl.so.0.17.0` | `04d6d625a4f56b2c86e4b5ff7edbb2bde3b647240d2f8e02b60d867f1e11b5e5` | same rebuild — benches tg128 ~129.7, NOT the board binary; board-era `.so` was not preserved. Repro campaign: `REPRO_20260810_board_recovery.md` |
| `treebeard-base-control-latest/build-mmadd-decode/bin/libggml-sycl.so.0` | `abb033c4c37a5733578642b80464c3acdaf5390fb55fc0b59186a42e855db82e` | the env-order contaminant (`FINDING_20260810_lxbin_env_order_contamination.md`); matches the finding's recorded sha |
| `treebeard-base-control-latest/build-base-control/bin/llama-perplexity` | `4d7342b034789ffb51de6bd8526ae1f67ff0135fa39c53bd49c007058bebd059` | pinned KLD-gate baseline — never rebuild/clean |

## Rules carried forward

- `private/lx-champion-*` stays local or on a private remote only; the raw
  research tree (diagnostics, `[lx-*]` logging) is not pushed to the public
  fork.
- PR branches are built by hand-selecting hunks from
  `lx-champion-1.3105-20260810`, not by replaying `patches/`.
- Do not build on `pr/sycl-moe-multislot-package` or `lx/serial-kernel-*`
  (they delete upstream sources).
