# PR Package — Laguna SYCL upgrades — unit map (2026-08-04)

Source of truth for the ggml-org/llama.cpp submission campaign.
Champion = `treebeard-base-control-latest` working tree diff (57 hunks, +3787/-176,
8 files under `ggml/src/ggml-sycl/`), formal 1.227 (pp512 1183.3 / tg128 139.3 vs pin
1139.2/107.35), golden OK, wikitext-2 PPL ~12.6. Champion formal artifact:
`results/20260731T141436Z/score.json`.

Only default-ON, quality-green units below. Known-broken and research code does not ship.

## Global porting rules (ice-cold-clean)

- Strip all research scaffolding: hybrid modes 0-7/9 (ship only the mode-8 path),
  one-shot diag prints (`[lx-control-moe-dual]`), `GGML_SYCL_GRAPH_CHECKSUM`,
  `GGML_SYCL_DEBUG_MOE_DOWN_DIFF`, `GGML_SYCL_MOE_DOWN_DUMP`, dead `topk_swap`,
  unused `ggml_sycl_op_rms_norm_fused_add`.
- Do NOT port opt-in switches for broken paths: `ENABLE_MUL_MAT_ADD_ANY_BATCH`,
  `ENABLE_MOE_DUAL_DOWN`, dual-multitoken, QKV shared quant, MMID fused batch.
  (Broken code must not exist in the PR even disabled.)
- Every shipped feature keeps a `GGML_SYCL_DISABLE_*` kill switch matching the
  existing backend style.
- Comments: technical, no campaign references, no notes/ links, no "Laguna".
- AI usage disclosed in each PR; all posted text authored/approved by operator.

## PR units (submission order)

### PR-1 — fattn: VEC kernel for GQA decode
- Hunks: fattn.cpp H01 (~13 lines, policy change only — VEC kernel pre-exists).
- Change: select VEC for `Q->ne[1]==1` decode even when GQA opt applies; kill switch
  `GGML_SYCL_FATTN_FORCE_TILE=1` (rename to DISABLE-style if upstream prefers).
- Benefit: any GQA model on SYCL decode. Evidence: +3.4 tg128 (Laguna XS 2.1 Q4_K_M).
- Not bitexact vs TILE — PPL parity gate required. Bench: Qwen3.6-35B-A3B + a dense GQA model.

### PR-2 — topk-moe: fused sigmoid+bias router chain (true top-k + gather/sum/norm + router GEMV)
- Hunks: topk-moe.cpp H47-H56 (~1150 lines), exports H02 (argsort) + H09 (non-static
  compute_forward), ggml-sycl.cpp fuse-dispatch part of H55.
- Sub-units (one PR, separable commits): ROUTER_SIGMOID_ADD, ROUTER_GEMV_FUSE
  (decode-only by construction, n_rows==1), ROUTER_TRUE_TOPK (+gather/sum),
  ROUTER_TRUE_TOPK_NORM, TOPK_MOE_BIAS hybrid — **mode-8 path only**.
- Benefit: MoE models with sigmoid + e_score_correction_bias routers
  (Qwen3.6-35B-A3B class). Evidence: +3.4 tg GEMV, +0.9 tg true-topk, +1.0 tg norm.
- Kill switches: DISABLE_TOPK_MOE, DISABLE_ROUTER_GEMV_FUSE, DISABLE_ROUTER_TRUE_TOPK,
  DISABLE_ROUTER_TRUE_TOPK_NORM, DISABLE_ROUTER_SIGMOID_ADD.

### PR-3 — mmvq/moe: dual-SwiGLU fused gate+up (MOE_DUAL_SWIGLU)
- Hunks: ggml-sycl.cpp H04 sub 1 + fuse fn sub 8 (~370 lines), mmvq.cpp H36 sub 2
  (dual_swiglu_reorder kernel), mmvq.hpp decls.
- Benefit: any SwiGLU MoE decode (Q4/Q5/Q6_K). Evidence: +1.6% formal, first kernel win.

### PR-4 — moe-down: weighted reduce + device expert sort (MOE_DOWN_WEIGHTED + MMID_DEVICE_SORT)
- Hunks: H08 sub 1,2,3,7,8 (~375 lines: weighted-reduce kernel incl. k=8 unroll + sgs=8,
  device counting-sort/prefix-scan/event-wait, fuse fn), H05-H07 decode-path plumbing only.
- Excludes: expert loop, packed reduce, compose (dual_down family — broken), INTEGRATED.
- Evidence: +6 tg weighted, +143 pp prefill two-step, infra sort ≈ tip.

### PR-5 — dense: dual-SwiGLU shared-expert fuse (DENSE_DUAL_SWIGLU)
- Hunks: H04 sub 9,10 (~255 lines), mmvq.cpp H36 sub 3 (dense_dual_swiglu kernel).
- Benefit: MoE models with shared/dense experts. Evidence: +3 tg decode, +16 pp multicol.

### PR-6 — norm: RMS_NORM+MUL fuse (mirrors existing CUDA rms_norm_fused)
- Hunks: norm.cpp H39-H45, norm.hpp H46, ggml-sycl.cpp H04 sub 3 (~243 lines).
- Strong precedent: CUDA backend already has `ggml_cuda_op_rms_norm_fused`.

### PR-7 — fuse: ADD+ADD residual
- Hunks: H04 sub 6 (~73 lines, inline kernel).

### PR-8 — fuse: softplus×mul attention gate
- Hunks: H04 sub 5 (~203 lines).
- **GENERICITY GATE**: verify a public model hits this pattern before submitting;
  if only Laguna-class custom archs benefit, hold back. Verify in Phase 3.

### PR-9 — fuse: ROPE+VIEW+SET_ROWS (dispatch only)
- Hunks: H04 sub 4 (~57 lines; kernel `ggml_sycl_rope_fused` pre-exists at rope.cpp:637).
- Open item: 0024 patch notes an "ISWA k-last expand (q→v→k then cpy_k)" fix in
  llama graph code — confirm whether already upstream or needed; verify in Phase 3.

### PR-10 — mmvq: MUL_MAT+ADD(+ADD) residual epilogue, decode-only
- Hunks: H04 sub 2, mmvq.cpp H13-H28 row-addend API (~265 lines).
- Source gate `ne11==1` (replaces the champion's 10-byte binary patch at 0x22952f).
- Any-batch form is a known PPL break (1e5+) — gate is load-bearing; PR text must say so.
- Evidence: +1.9 tg / +1.4 pp decode-only; PPL 12.60 == quality-safe.

## Excluded (documented, not shipped)

| Unit | Why |
|---|---|
| MOE_DUAL_DOWN whole family (expert loop, packed reduce, compose, graph tensors) | PPL -nan on reordered SoA quant (MMQ/oneDNN fallthrough). mul_mat chunk fix exists in source (H03) but unit measured neutral-to-invalid |
| MOE_DUAL_MULTITOKEN | Same bug class; on fixed source: quality OK but zero speedup (+63% was fast garbage) |
| MOE_DOWN_INTEGRATED | Golden FAIL 2026-07-30; source opt-in only. Revisit as separate follow-up after PR-4 |
| MMID_FUSED_BATCH / SINGLE | Opt-in, golden-unproven at multitoken |
| QKV shared quant | Golden FAIL + UR device-lost history |
| MUL_MAT_ADD any-batch | PPL 1e5+ |
| Hybrid modes 0-7/9, mode6 warp-norm | Research matrix; mode 8 is the ship path |
| Debug/diag infra, 6 OFF-default MoE knobs (FRONTIER_20260806) | Unmeasured/dev-only |
| SYCL graph capture | Measured loss; stays disabled |

## Dependencies / ordering rationale

- PR-1 independent, tiny → first (establishes contributor track record; one open PR rule).
- PR-2/3/4/5 independent of each other; PR-3 before PR-4 (shares mmvq reorder infra).
- PR-6/7/8/9 independent small fuses.
- PR-10 last (touches mmvq broadly; needs its own PPL-parity spotlight).

## Validation matrix per PR (all on fresh master, public weights)

Bench target: Qwen3.6-35B-A3B (public GGUF on disk) + secondary model exercising the path.
1. SYCL build clean (oneAPI, -Werror as per CI).
2. `test-backend-ops` SYCL vs CPU (add cases if new op patterns).
3. `llama-perplexity` wikitext-2 parity vs master (same chunks, report both).
4. `llama-bench` pp512/pp2048/pp8192 + tg128, 5 reps, f16 KV, master vs PR.
5. Long-context: needle/dossier smoke at 32k+ (server).
6. Real-world: chat smoke, tool-call smoke, greedy determinism check.
7. Local CI (`ci/README.md`) before submission.
8. Final combined-stack branch: full matrix + compare vs champion 1.227 baseline.

## Open items for operator review

1. PR-8 softplus gate — hold if no public model exercises it.
2. PR-9 ISWA k-last expand — confirm upstream status.
3. MOE_DOWN_INTEGRATED decode-only: champion binary era note says default-ON decode-only,
   current source is opt-in OFF. Proposal: exclude from PR-4, separate follow-up later.
4. Router PR-2 size (~1150 lines) — may need maintainer pre-pble via tracking issue;
   alternative split: (a) true top-k+norm, (b) GEMV+sigmoid-add. Decide at issue stage.
