# PR PACKAGE v2 — Tier 1+2 unit map (supersedes PR_PACKAGE_20260804_unit_map.md)

Target: incremental PR series to ggml-org/llama.cpp covering the full champion
stack (`lx-champion-1.3105-20260810`, hub commit `c7d3bfe6d`: 16 files,
+7446/−441 on upstream `7e1e28cae`). Board receipt: `results/20260809T191208Z`
— score 1.310474, tg128 152.545, pp512 1170.98, KLD mean −0.0 / same_top 100%.

Authority for hunk extraction: `git diff 7e1e28cae..lx-champion-1.3105-20260810`
in the hub repo. Env-var spellings below were extracted from that diff
(2026-08-10); at port time re-grep the diff, not this doc.

Upstream constraints (verified in local CONTRIBUTING.md / pull_request_template.md):
AI-usage disclosure line per PR; PR prose/commit messages human-written by the
contributor (newjordan); serial filing (one open PR); squash-merge; title style
`sycl: <lowercase imperative>`; `@ggml-org/ggml-sycl` owns the path.

---

## PR series (filing order)

### PR-A `sycl: prefer VEC flash-attention for GQA decode and widen vec nbatch`
- Units: fattn VEC policy for `Q->ne[1]==1` (fattn.cpp, ~28-line diff);
  `nbatch_fa` 128→256 at nthreads=256 (fattn-vec.hpp).
- Evidence: +3.4–3.6 tg128 (policy, disable −1.69%); +0.33% score
  (`results/20260807T113536Z`, nbatch leg). Precedent: upstream `c1063ac9d`.
- Env: none new needed if policy is expressed as dispatch heuristic; else
  `GGML_SYCL_DISABLE_FATTN_VEC_DECODE`.
- Genericity: fully generic (any GQA model, decode).
- Tests: FLASH_ATTN_EXT decode-shaped GQA cases in test-backend-ops.
- Strip during port: `GGML_SYCL_FATTN_FORCE_TILE`, `GGML_SYCL_FATTN_DECODE_NTHREADS`
  debug knobs, `[lx-control-fattn]` logging.

### PR-B `sycl: fuse RMS_NORM+MUL and ADD+ADD`
- Units: RMS_NORM+MUL (norm.cpp/hpp; disable −1.81% tg); ADD+ADD residual
  (ggml-sycl.cpp; disable −0.08%, keeps graph fusion contiguity for later PRs).
- Kill switches (already spelled): `GGML_SYCL_DISABLE_RMS_NORM_FUSE`,
  `GGML_SYCL_DISABLE_ADD_ADD_FUSE`.
- Genericity: generic; CUDA parity (`rms_norm_fused`). Tests exist
  (`test_rms_norm_mul_add`); add ADD+ADD chain case.

### PR-C `sycl: memoize q8_1 quantization of shared src1`
- Unit: q8_1 src1 quant memo arena (common.hpp `quant_memo_*` + quantize.hpp,
  +35/+108). Q/K/V/gate share one quantized `attn_norm` row; ~120 launches/step
  removed. Bit-identical by construction.
- Evidence: board 1.2211 (`results/20260807T051009Z` era, SHIP note).
- Kill switch (new): `GGML_SYCL_DISABLE_Q8_MEMO` (name at port time).
- Genericity: generic decode-path infra; REQUIRED BY PR-F (QKV fuse).
- Risk: invalidation correctness — PR body must state the memo key
  (src1 pointer/generation) and include repeated-src1 graph test.

### PR-D `sycl: extend fused top-k MoE to sigmoid+bias routers`
- Units: router chain mode-8 (fused sigmoid+bias, router GEMV fuse, true
  top-k + gather/sum/norm) + `LX_IDS_ONCE` MoE ids memo (common.hpp:461 area).
- Evidence: GEMV fuse disable −2.87% tg; true-topk disable −1.68%; bias off
  −7.5%; ids-once board 1.248→1.252 (`results/20260808T191654Z` era).
- Kill switches: `GGML_SYCL_DISABLE_TOPK_MOE` (upstream-existing),
  `GGML_SYCL_DISABLE_ROUTER_TRUE_TOPK`, `_ROUTER_TRUE_TOPK_NORM`,
  `_ROUTER_GEMV_FUSE`, `_ROUTER_SIGMOID_ADD`. Fold `GGML_SYCL_ENABLE_TOPK_MOE_BIAS`
  (opt-in) into default-ON. Rename `LX_IDS_ONCE` → `GGML_SYCL_DISABLE_MOE_IDS_MEMO`.
- STRIP: `GGML_SYCL_TOPK_MOE_HYBRID_MODE` knob + modes 0–7/9 code (keep the
  mode-8 path only), `GGML_SYCL_ROUTER_MULTIROW` (unvalidated), router timers.
- Genericity: any sigmoid-routed MoE with score-correction bias (afmoe-class,
  Laguna-class). Extend `test_topk_moe` with sigmoid+bias ± norm cases.
- Size ~1200 lines → open tracking issue first; split option: (a) true-topk
  + norm, (b) GEMV+sigmoid-add.

### PR-E `sycl: MMVQ MoE decode fusions (dual-SwiGLU, weighted down-reduce, device expert sort)`
- Units: MoE dual-SwiGLU gate+up (`MUL_MAT_ID×2+GLU`, decode `ne12==1`;
  +1.63% score, SHIP_20260729_dual_swiglu); moe-down weighted reduce + device
  counting-sort/prefix-sum (disable −3.2% tg, knob-ab-ledger ×4);
  single-token weighted-reduce specialization (board 1.2318); reorder-MMVQ
  `num_subgroups` 16→8 (board 1.2287); buffer-overlap reject narrowed to
  `weights`/`mmid` only (SHIP_20260808_moe-down-mul-alias — rationale must be
  restated in PR discussion).
- Kill switches: `GGML_SYCL_DISABLE_MOE_DOWN_WEIGHTED`,
  `GGML_SYCL_DISABLE_MOE_PACKED_REDUCE`, `GGML_SYCL_DISABLE_MMID_DEVICE_SORT`
  (+ dual-SwiGLU switch — grep diff for exact spelling).
- Depends: PR-C. Hazard: upstream #25455 (MUL_MAT_ID wrong results on B70) —
  run master `test-backend-ops -o MUL_MAT_ID` first; if our sort/weighted path
  fixes it, file that separately BEFORE this.
- Tests: `test_mul_mat_id_fusion`, `test_mul_mat_vec_fusion` + weighted-reduce
  and expert-sort cases, Q4_K/Q5_K/Q6_K/IQ4_NL coverage.

### PR-F `sycl: dense dual-SwiGLU and fused QKV decode GEMV`
- Units: dense/shared-expert dual SwiGLU (disable −2.77% tg); fused QKV decode
  GEMV — Q(q4_K)+K(q4_K)+V(q6_K) in one reorder-MMVQ launch, V published
  directly to F16 KV cache (board 1.2497, biggest single Tier-2 jump 142→144).
- Kill switches: `GGML_SYCL_DISABLE_DENSE_DUAL_SWIGLU`, `GGML_SYCL_DISABLE_QKV_FUSE`.
- Depends: PR-C (shared quant memo feeds the single launch).
- GENERICITY GATE: fuse must key on runtime pattern (3 same-src1 GEMVs,
  ne11==1, compatible quants) — verify it triggers on Qwen3.5-35B-A3B before
  filing; if Laguna-only in practice, demote to follow-up.

### PR-G `sycl: rope-side decode fusions (norm+rope, rope+set_rows, merged QK, V-cache direct write)`
- Units: ROPE+VIEW+SET_ROWS (`GGML_SYCL_DISABLE_ROPE_SET_ROWS_FUSE`, −0.36%
  when off); K-path RMS_NORM+MUL+ROPE+VIEW+SET_ROWS single launch
  (`rope_neox_normed_sycl`; board 148.03/1.278); Q-path twin (board
  152.36/1.308); merged QK rope `rope_neox_normed_qk` via
  `ggml_can_fuse_subgraph_ext`, decode-only `ne[2]==1` (board 152.55/1.3105);
  NEOX 64-thread WG for 128-wide heads (rope.cpp:343 area); q6_K V-MMVQ →
  direct indexed F16 V-cache publication (`GGML_SYCL_DISABLE_V_MMVQ_SET_ROWS_FUSE`,
  board 1.2354).
- **REQUIRED CO-CHANGE: `src/llama-graph.cpp` k-last expand reorder**
  (discovered during preservation; expands `k_cur` after `v_cur` in
  `build_attn` so the rope fuse can write directly into the KV cache).
  Outside ggml-sycl — needs its own justification paragraph and a check that
  no other backend's fusion assumptions break. Check during drift survey
  whether upstream already reordered this.
- REQUIRED FLIP: `GGML_SYCL_FUSE_NORM_ROPE` (opt-in, source-default OFF) →
  default-ON with `GGML_SYCL_DISABLE_NORM_ROPE_FUSE`. This flip carries the
  148→144 delta documented in commit bf85534 and env.sh:79-85.
- Also: `GGML_SYCL_DISABLE_QK_ROPE_MERGE` stays as the merged-QK switch.
- Tests: `test_rms_norm_mul_rope`, `test_rope_set_rows` + normed+set_rows,
  merged-QK, 128-head WG, V-cache publication cases.
- Split-on-request: merged-QK and V-cache publication are separable.

### PR-H `sycl: decode-only MUL_MAT+ADD residual epilogue`
- Unit: MUL_MAT+ADD(+ADD) epilogue, decode-only `ne11==1`
  (`GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE`; +2.1–2.5 tg128,
  SHIP_20260731_mmadd_decode_only).
- The `ne11==1` gate is load-bearing: any-batch form breaks wikitext PPL
  (1e5–1e6). PR text states this measured failure and the gate explicitly;
  test-backend-ops must include a batch>1 case asserting NO fuse.
- STRIP: `GGML_SYCL_ENABLE_MUL_MAT_ADD_ANY_BATCH` and its code path entirely.

---

## Hold / drop

| item | verdict |
|---|---|
| softplus×mul attention gate fuse (`GGML_SYCL_DISABLE_SOFTPLUS_MUL_FUSE`, −1.02% off) + gate-softplus-MUL MMVQ fuse (`GGML_SYCL_DISABLE_GATE_SOFTPLUS_FUSE`, board 1.2556) | HOLD behind genericity check: grep current-master `src/models/` for softplus attention-gate pattern; drop from series if Laguna-only |
| MOE_DUAL_DOWN family (`GGML_SYCL_DISABLE_MOE_DUAL_DOWN` guard) | EXCLUDE — NaN/neg-logprob; delete code, not just guard |
| `GGML_SYCL_ENABLE_MOE_DOWN_INTEGRATED` | EXCLUDE — graph-wiring golden FAIL |
| any-batch MUL_MAT_ADD (`GGML_SYCL_ENABLE_MUL_MAT_ADD_ANY_BATCH`) | EXCLUDE — PPL explosion |
| QKV shared quant / QKVG gate segment | EXCLUDE — golden FAIL / device-lost / tg stall history |
| `GGML_SYCL_ENABLE_MMID_FUSED_BATCH` / `GGML_SYCL_DISABLE_MMID_FUSED_BATCH` / `GGML_SYCL_MMID_FUSED_SINGLE` | EXCLUDE — null axis (structurally decode-only) |
| `GGML_SYCL_LX_MMVQ_PREFILL` (−17%), `GGML_SYCL_LX_GEMM_BATCH` (hangs B70) | EXCLUDE — dead ends |
| hybrid router modes 0–7/9, `GGML_SYCL_TOPK_MOE_HYBRID_MODE`, `GGML_SYCL_ROUTER_MULTIROW` | EXCLUDE — keep mode-8 path only |

## Strip list (every PR, exact spellings from champion diff)

Diagnostics/timers: `GGML_SYCL_DIAG_SKIP{,_DECODE,_TINY_N,_QUANT,_LARGE_N,_F32}`,
`GGML_SYCL_DIAG_NAME_{QUANT,MMQ}`, `GGML_SYCL_LX_DIAG_EXPERT_CAP`,
`GGML_SYCL_LX_CHRONO` + `LX_CHRONO_{OUT,NAMES}`, `GGML_SYCL_LX_ROUTER_TIMER` +
`LX_ROUTER_TIMER`, `GGML_SYCL_DEBUG_QKV_VSET`, `GGML_SYCL_NR_DEBUG*`,
`GGML_SYCL_NR_NO_NORM`, `LX_DIAG_*`, `LX_PROBE_*`, `LX_QUANT_NAME{,S}`,
`LX_WAIT_{COUNT,TOTAL_NS}`, `GGML_SYCL_DIAG_SKIP_MMVQ_TYPES` and kin.
All `[lx-*]` fprintf logging: 123 added sites (69 ggml-sycl.cpp,
16 topk-moe.cpp, 13 rope.cpp, 10 mmvq.cpp, 15 elsewhere).
Local build workarounds: `ggml/src/ggml-sycl/CMakeLists.txt` MKL fallback,
`tools/CMakeLists.txt` mtmd option — never upstream.
`GGML_SYCL_MMV_Y` hits are uses of the upstream-existing constant — not a knob,
do not strip blindly. `GGML_SYCL_ENABLE_MMQ` hits touch upstream-existing knob —
review per hunk.

## Porting rules (unchanged from v1 + additions)

1. Guards = runtime shape/type conditions with clean fallback; no model names,
   no "Laguna"/campaign vocabulary in code or comments.
2. Every feature default-ON with `GGML_SYCL_DISABLE_*` (getenv-once static
   bool, per upstream `GGML_SYCL_DISABLE_OPT/GRAPH/DNN` pattern).
3. Hand-select hunks from the champion tag; never replay `patches/`.
4. Update `docs/backend/SYCL.md` env table per PR.
5. Reviewer-facing numbers: public models only (Qwen3.5-35B-A3B Q4_K_M MoE +
   dense Qwen), master-vs-PR-vs-PR+kill-switch, mean±sd, 5 reps. Laguna board
   numbers stay internal.
6. All commit messages / PR prose / review replies authored by newjordan; AI
   disclosure line per template; no AI attribution trailers anywhere.

## Validation matrix (per PR)

full `test-backend-ops test -b SYCL0` green ×2 (default + kill switch) →
golden-smoke → quality-gate-kld (≤0.010 / ≥99.0) → bench-serial vs stack →
public-model llama-bench + llama-perplexity A/B → local `ci/run.sh` →
attribution grep → file.
