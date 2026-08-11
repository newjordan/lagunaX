# FINDING — pp512 wall decomposition: the prefill path has NO native quantized
# batched GEMM (2026-08-09T030915Z) — answers the AGENTS.md open lead.

## Evidence artifact
`results/diag-pp512-20260809T030915Z/run.log` (all runs: current live
`benchmark/kernel/build/bin/llama-bench`, `-ngl 99 -t 16 -ub 2048 -b 2048
-ctk f16 -ctv f16 -r 2`, GGML_SYCL_DISABLE_GRAPH=1, GGML_SYCL_DISABLE_DNN=1,
scored flags). In-tree diag probes: `GGML_SYCL_DIAG_SKIP[_LARGE_N/_TINY_N]`
(ggml-sycl.cpp:2588-2668, env-gated no-ops otherwise) zero the dst of
`ggml_sycl_op_mul_mat_sycl` (the F32/oneMKL row_gemm path, reached only from
the else branch of ggml_sycl_mul_mat, ggml-sycl.cpp:4680) instead of running
the GEMM. `gguf-tensor-types.txt` = tensor-type census of the scored GGUF.

## Methodology guard (important)
Skipping GEMM exec of the LARGE (ncols>32) F32-path calls — which include the
pp512 MoE router GEMV (`ffn_gate_inp` src0 is F32, ncols=512) — zeroes the
router logits, collapsing downstream expert routing: per-pass F32-path call
count drops 29526 → 3801 (run B vs C). So runs B/D/G (skip-large / skip-all /
skip-all-with-conversions) measure a DEGRADED workload and their wall times
(5233/5230/8235 t/s) are NOT a faithful decomposition. Only run C (skip tiny
≤32 ncols; router GEMM is large → intact) and run E (tg128) are clean.

## Clean measured facts (one pp512 pass ≈ 438.5 ms @ 1167.6 t/s)
| fact | value | evidence |
|---|---|---|
| tiny-slice (ncols≤32) F32-path GEMM exec | **66.6 ms/pass (15.2% of wall)** | A 438.5 ms → C 371.9 ms; avg ~6.78 µs/call |
| tiny F32-path calls per pass | **~9820** | run C counts: 19641 over r=2 |
| large (ncols>32) F32-path calls per pass | **~4943** | run C counts: 9885 over r=2 |
| tg128 F32-path calls | **zero** (no LX_DIAG_COUNTS line) | run E, 144.20 t/s unchanged |
| large-slice exec upper bound | < 371.9 ms/pass | 438.5 − 66.6 (unmeasured cleanly; see guard) |

## Structural fact: prefill has no native quantized batched GEMM
- `can_use_mul_mat_vec_q` caps `src1->ne[1] <= MMVQ_MAX_BATCH_SIZE`
  (ggml-sycl.cpp:4561) and `MMVQ_MAX_BATCH_SIZE` = 8 (common.hpp:177).
- `ggml_sycl_supports_reorder_mmvq` excludes GGML_TYPE_IQ4_NL
  (ggml-sycl.cpp:3885-3896) and `reorder_qw` has no iq4_nl reorder
  (ggml-sycl.cpp:4446-4478).
- `ggml_sycl_supports_mmq` excludes IQ4_NL and MMQ is env-gated OFF by default
  (ggml-sycl.cpp:3823-3843, GGML_SYCL_ENABLE_MMQ default 0).
- Consequence: every batched quantized-src0 GEMM at prefill (MoE expert slices
  with >8 routed tokens, attn projections at T=512, q6_K dense ffn_shexp)
  dequantizes src0 to F32 and runs oneMKL sgemm.
- Native iq4_nl/q6_k GEMV kernels DO exist for ncols≤8
  (`mul_mat_vec_iq4_nl_q8_1_sycl` mmvq.cpp:2229, dispatch mmvq.cpp:2622) —
  decode already uses them; the batched (>8) variant does not exist.

## Model layout (scored GGUF, 678 tensors)
- `ffn_gate_exps`/`ffn_up_exps` = Q6_K (39+39), `ffn_down_exps` = IQ4_NL (39),
  `ffn_*_shexp` dense = Q6_K, router `ffn_gate_inp` = F32, norms F32.
- iq4_nl is only 60/678 tensors, all routed MoE down-projections.

## MMQ is broken, not just off
- `GGML_SYCL_ENABLE_MMQ=1` hangs: pp512 >120s no sample (killed); pp64 timed
  out at 75s (run F rc=124). GPU healthy after (1160.66 t/s recheck). The
  compiled MMQ kernels are unusable on the scored path as shipped.

## What this funds (next candidates)
1. Native batched (multi-col, ncols>8) iq4_nl/q6_k mmvq-style GEMM for pp512
   slices (q8_1 activations, extend the existing kernels). Cleanly measured
   tiny-slice term is 66.6 ms; the large-slice term is the remaining unmeasured
   ~300+ ms class. KLD arbiter (decode already KLD-clean on q8_1 activations).
2. MMQ repair (hang first) for the q6_k batched path.
- Rejected-with-new-evidence: cap 8→32 alone only touches the 66.6 ms tiny term
  (~15%); the large-slice wall needs a real batched kernel (the 2026-07-30
  "chunked MMVQ slower" result never touched ncols>32 slices).
- Probe upgrade: a clean large-slice decomposition needs a skip that exempts
  the F32 router GEMM (e.g. gate by src0 type != F32), so routing stays intact.
