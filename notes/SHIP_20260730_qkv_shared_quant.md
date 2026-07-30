# Research — Q/K/V shared Q8 quantize (2026-07-30)

## Status: **NO WIN / BLOCKED** — tip binary preserved; GPU device-lost after probe

| arm | result |
|-----|--------|
| **tip** (prior formal) | pp3730 / tg138.4 / **+62.75%** |
| QKV shared quant (this fire) | **device-lost / hang** — not scored |
| tip binary + `GGML_SYCL_DISABLE_QKV_SHARED_QUANT=1` | worked briefly (tg16~138) pre-wedge |

## Lever

Laguna has separate `attn_q/k/v` (+ gate) MUL_MATs on the same `attn_norm` activation.
Decode pays 3–4 Q8 quantize launches of the same 2048-vector per layer.
Goal: one soa Q8 quantize + N reorder MMVQs.

## Findings

1. Graph expand order is **Q → RESHAPE → … → V → K** (not consecutive MUL_MATs).
   `can_fuse_subgraph` on a contiguous span fails; need non-consecutive precompute + skip.
2. Matching by `src1->data` pointer is **unsafe** (allocator reuses buffers across layers)
   → incorrectly precomputed future layers; first probe had `n_mm=8` and hung.
3. Exact `src1` tensor pointer match yields **Q/V/K** (`n_mm=3`) correctly.
4. Stock `ggml_sycl_mul_mat` precompute/skip path **progresses** (validated with logs).
5. Direct `ggml_sycl_op_mul_mat_vec_q` + shared soa q8 → **UR device-lost** on B70
   (`UR_RESULT_ERROR_DEVICE_LOST` on stream wait). Cause not fully isolated
   (async q8 lifetime vs kernel params vs reorder state).
6. Accidental `git checkout` during revert **wiped uncommitted tip sources** for
   `ggml-sycl.cpp` / `topk-moe.*`. Sequential `patches/*.patch` do **not** cleanly
   re-stack on current HEAD (many empty/stale). Tip **binary** backed up:
   `baseline/tip-binary-backup-20260730T141542Z/` (QKV kill required if that .so
   still contains the experimental fuse).

## Tip

**Unchanged** at formal **+62.75%** (`20260730T141542Z`). No ship.

## Recovery notes (next agent)

1. **GPU:** sycl-ls empty after device-lost — **reboot** then `scripts/resume-after-reboot.sh`.
2. **Source tip stack:** restore from prior agent state if available; else re-port from
   notes + fullsnippets (`0047` = tip `topk-moe.cpp` without QKV). Do **not** rebuild
   over tip .so until source is restored and verified.
3. **QKV lever:** only revisit with (a) exact src1 pointer, (b) stock mul_mat path or
   proven shared-q8 lifetime, (c) kill-switch default OFF until golden+bench, (d) never
   match by data* alone.

## Next lever (after GPU + source recovery)

- Prefer non-QKV theory under +62.8% (attn gate GEMV+softplus low ROI; MoE expert
  geometry; DNNL gemm→sig epilogue for prefill).
- Or re-do QKV shared quant **opt-in only** with stock mul_mat first (dispatch-only)
  then careful shared q8.
