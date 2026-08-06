# MMID fused-batch cap extension A/B — 20260806T171017Z

Candidate: source edit ggml-sycl.cpp — env-extensible GGML_SYCL_MMID_FUSED_MAX_TOKENS
(ne12_cap 64 → 512, default 64 = champion), cand arm ran MAX_TOKENS=512 +
ENABLE_MMID_FUSED_BATCH=1. Official geometry, same-window ctrl/cand/ctrl.

| arm | tg | pp |
|---|---|---|
| ctrl-a | 135.372 | 1122.805 |
| cand   | 135.912 | 1119.222 |
| ctrl-b | 135.991 | 1124.736 |

cand pp -0.40% vs ctrl mean (1123.77) — inside ±0.68% drift bound → null.

Root cause of null: fused mul_mat_id path is structurally decode-only in the
standard layout. Gate at ggml-sycl.cpp:5694 requires ne11==1 (decode) or
ne11==n_ids_per_group (transposed layout); pp512 builds ne11=512 → return false
BEFORE the cap matters. [lx-control-mmid] marker fired 0 times in all 3 arms.

Bonus root-cause: [lx-control-moe-down] logs "skip: buffer overlap with dst"
at n_rows=512 (prefill) in every arm, vs "fuse hit (weighted reduce)" at
tokens=1 (decode) — prefill ffn_out runs the UNFUSED counting-sort
grouped-GEMM path; that is the 123.5ms/call budget bucket, and the fused
weighted-reduce path refuses batch due to dst aliasing.

Post-cycle: source reverted (MMID_FUSED_MAX_TOKENS count 0), champion .so
md5 2361042a… pristine, golden passed on the probe lib before revert.
