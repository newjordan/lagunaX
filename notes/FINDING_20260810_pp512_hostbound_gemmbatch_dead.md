# FINDING 20260810 — pp512 prefill is HOST-CPU-BOUND; expert-GEMM loop is the wall; gemm_batch is dead on B70

Date: 2026-08-10. Build: benchmark/kernel build dir, .so sha (post-diag) — all edits env-gated default OFF.
Verifier: default-path golden-smoke GOLDEN OK; tg128 147.80 t/s (champion-era window); pp512 ctrl 1174-1177 tok/s.

## 1. pp512 pacing model (host vs device) — measured, closes open lead 9's "compute-bound" claim
- pp512 pass: wall 434.5ms, host submission span 428.4ms, device tail 6.15ms (1.4%). [lx-pp512-chrono-20260809T112925Z + 113943Z chrono bins; scripts/lx-chrono-analyze.py]
- CPU-time proof (r1 vs r20 differencing, /usr/bin/time): Δwall/19 passes = 435.8ms/pass; ΔCPU/19 = 429.5ms/pass → CPU fraction 0.986 of wall. The host is genuinely busy 430ms/pass.
- stream->wait() instrumentation (WAIT_TOTAL_NS/WAIT_COUNT, same GGML_SYCL_LX_CHRONO=1): 56.7ms total over 228 waits (6 passes) → 249µs avg per layer's ids D2H wait. All mid-pass device drains ≈ 9.5ms/pass. The 6.3ms tail is the final drain.
- ⇒ The wall is HOST per-op dispatch work, NOT device execution. Device total ≈ 15ms/pass (9.5 drain + 6.3 tail).

## 2. Where the host time goes — the 3 per-layer MoE dispatches
Steady-pass chrono (names on) per layer (layer-1 example): ffn_moe_gate-1 dispatch 7.6ms, ffn_moe_up-1 4.2ms, ffn_moe_down-1 3.6ms (weighted consumed by the down fuse). Averages across 39 layers: gate ≈ 5.1ms, up ≈ 2.8ms, down/swiglu ≈ 2.85ms → ~10.75ms/layer × 39 ≈ 419ms of the 434ms pass. Every other op class (attn op29 4ms/pass, elementwise, fused-launch chains) is noise. The MoE per-expert GEMM loop in ggml_sycl_mul_mat_id (ne12>1) IS the pp512 wall.

## 3. Expert-cap probe — sizes the loop
GGML_SYCL_LX_DIAG_EXPERT_CAP (measurement-only, dst stale): cap=0 → 1168.4 tok/s (434.6ms); cap=64 → 3946.6 (129.7ms); cap=128 → 2929.1 (174.8ms); cap=256 → 1170.0 (433.9ms). Per-expert loop ≈ 300-345ms/pass; non-loop floor ≈ 85-130ms/pass. At T=512 all 256 experts are non-empty (cap=256 == control proves the probe fired). Per-call cost ≈ 10-27µs (slope-dependent), dominated by the single-gemm submission + dispatch setup, NOT by the per-expert conversions.

## 4. oneMKL gemm_batch is dead on this B70 — two forms, both fail (kills open leads 8/11)
- GROUPED form (oneapi::mkl::blas::row_major::gemm_batch, group_count per distinct n, flat pointer arrays): abort, UR_RESULT_ERROR_OUT_OF_RESOURCES (level_zero error 40) on the first dispatch. [bench /tmp/gb-1.err]
- STRIDED form (dpct::gemm_batch padded to n_max=256 experts, one call per dispatch): hard device hang (busy-spin, no output in 3+ min; SIGTERM). [bench /tmp/st-1.err]
- Same disease class as MMQ (finding 18). The ONLY working GEMM surface on this device is the per-call single dpct::gemm.

## 5. Conversion-hoisting is NOT the lever (kills open leads 16/17's submission-cost premise)
Full-tensor fp16 pre-conversion (src0 whole tensor + src1 whole buffer, one launch each) + per-expert single gemm + one dst convert: 896.2 tok/s vs 1174.4 ctrl = −24%. The per-expert conversions were nearly free (device-overlapped, pipelined); hoisting them serializes the device on the big conversion kernels (537MB fp16/family/layer). The per-call cost is the gemm submission itself.

## 6. Implications (what is NOT left for pp512)
- gemm_batch coalescing: dead (measured, both forms).
- conversion memo (fp16 weight cache): dead as a wall lever (conversions aren't the wall); the full-shadow is also memory-impossible (finding 17 stands).
- QKV/attn fusion (open lead 20): attn MUL_MAT host cost is 4ms/pass — nothing to win there.
- Remaining live levers: (a) mmvq/q8_1 path for per-expert ncols>1 via the proven reorder-MMVQ chunked path (2 lighter submissions/call, no oneMKL host cost) — UNTESTED; (b) per-layer fused gate+up+down dispatch (collapse 3 dispatches → 1, reusing one mmid sort) — the dual-swiglu fuse exists but is env-killed for decode-quality reasons; a prefill-only variant is untested; (c) accept prefill ~1.02× and keep decode wins.

## Artifacts
- results/lx-pp512-chrono-20260809T112925Z/{chrono.bin, pp.json, pp.stderr}
- results/lx-pp512-chrono-20260809T113943Z/ (with WAIT_TOTAL_NS=56718910 WAIT_COUNT=228)
- /tmp/expcap2-{0,64,128,256}.json, /tmp/gb-{0,1}.json, /tmp/gb3-{0,1}.json, /tmp/st-{0,1}.{json,err}
- scripts/lx-pp512-chrono.sh, scripts/lx-chrono-analyze.py, scripts/lx-chrono-pername.py, scripts/lx-chrono-gaps.py
- Source edits (all env-gated default-off): ggml-sycl.cpp lx_chrono wait-timer; GGML_SYCL_LX_DIAG_EXPERT_CAP; GGML_SYCL_LX_GEMM_BATCH batched path (grouped → strided, both failing at runtime, default off).
