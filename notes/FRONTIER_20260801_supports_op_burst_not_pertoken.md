# Frontier — device_supports_op is graph-split bursty, not per-token (2026-08-01)

## Direction (PIVOT — inverts dir 23 / findings #35–#38)

**Scheduler op-support query frequency model inverted.** Prior work treated 677,040
`device_supports_op` calls as ~5,290/decode-step continuous revalidation. The
trace shows the opposite structure: **exactly 24 contiguous bursts of 28,210
queries each**, with **zero** support queries in 521/545 inter-logits windows.
Distinct from:
- dir 23 (volume-only characterization of supports_op)
- dir 19 (graph-capture gates)
- host↔device leaf traffic (set/get/sync per step)
- dir 24 (in-order queue)

## Evidence

- Trace: `results/ktrace-tip-20260730/decode-ggml/trace.log`
- Code: `/home/frosty40/llama.cpp/ggml/src/ggml-backend.cpp` (`ggml_backend_sched_split_graph` multi-pass `ggml_backend_supports_op`)
- Code: `/home/frosty40/llama.cpp/src/llama-context.cpp` (`pipeline_parallel` → `n_copies`)

## Findings

1. **24 × 28,210 = 677,040** — total matches dir-23 volume exactly; distribution is
   not uniform per token.
2. **521 of 545** get_tensor→get_tensor intervals contain **0** `device_supports_op`
   lines. Mid-decode steady state (e.g. between get #99 and #100) has SUP=0,
   SET=9, SYNC=11, MM=280.
3. Each burst is a **contiguous 28,212-line** block of pure supports_op logging,
   then `buffer_reset` + mass `buffer_init_tensor` for the next graph shape
   (prefill embd `[2048,256]` after burst 0; decode embd `[2048,1]` after burst 1).
4. Burst #0 op mix (exact): NONE 7570, ADD 4310, MUL_MAT 3600, MUL 2400,
   RMS_NORM 1610, MUL_MAT_ID 1170, SET_ROWS/VIEW/PERMUTE/ROPE 800 each,
   FLASH_ATTN_EXT **400**, … — FA and MUL_MAT_ID are *queried* but **never**
   appear under `[SYCL][OP]` in any steady-state decode window.
5. Steady-state decode OP log (one step): 280 mul_mat + 241 quantize_q8 + 40 rope
   + 40 rope_fused + 40 set_rows + 2 get_rows + 1 add = **644** launches;
   **zero** `ffn_moe_swiglu`/`ffn_moe_down`/`exps`/`mul_mat_id`/`dual` strings —
   dual-MoE expert work is invisible to SYCL[OP] instrumentation.
6. Decode-shaped (ne[1]==1) full-trace mul_mat counts: ffn_moe_logits **20,608**,
   ffn_moe_swiglu **224**, ffn_moe_down **112** — expert gate/up/down almost never
   take the logged mul_mat path in decode (dual control kernels own them).
7. Single-GPU forces `pipeline_parallel = false` → `sched->n_copies = 1`
   (`GGML_SCHED_MAX_COPIES` unused). No multi-copy input overlap on this track.

## Killed claims

- Finding #37 (“~5,290 supports_op per decode step = full graph re-validated
  every token”) — **false** for this trace; rate is ~0 for 95.6% of steps and
  28,210 only at split/reserve boundaries.
- Open leads #16–#18 framed as “per-step memoization of 5,290 queries” — ROI
  must be re-scoped to **burst sites only** (24 events / whole bench), not the
  per-token critical path.

## Hypotheses

1. Bursts fire when `ggml_backend_sched_split_graph` runs on a **new graph shape**
   (prefill ubatch vs decode n_tokens=1, and llama-bench rep/warmup boundaries
   with gets_before pattern 0,2,35,68,70,103… ≈ cycles of +33,+33,+2). Caching
   supports_op across identical shapes would only help those 24 sites.
2. Because dual-MoE and FLASH_ATTN_EXT leave no `[SYCL][OP]` breadcrumb, all
   launch-count budgets derived from SYCL debug undercount the true decode
   kernel mix — need UR/oneDNN/xe counters for expert+attention, not OP logs.
3. Wall-time of one 28,210-query burst is still unmeasured; if each query is
   ~100 ns, a burst is ~3 ms — material for cold graph shape, invisible on
   steady tg128 after warmup.

## Not claimed

- Wall-time share of supports_op bursts vs 7.2 ms/token decode.
- Exact llama-bench rep mapping of the +33/+2 get cadence.
