# VDR depth ledger — ncols==1 q6_K mmvq (lm_head) work-distribution A/B

Same-window ctrl / VDR-N / ctrl, official pp512/tg128 geometry (champion flags,
reps=5, env: ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0
GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_DISABLE_DNN=1, model Laguna-XS-2.1-Q4_K_M,
binary src-lmhead-build probe relink, build_commit 7e1e28cae).

Bit-exact: VDR only changes per-thread work split in mul_mat_vec_q; arithmetic
identical. Default (env unset) = champion VDR=1 path.

| stamp | VDR | ctrl-a tg | VDR tg | ctrl-b tg | Δ vs sandwich mean | vs bound (±0.68%) | verdict |
|---|---|---|---|---|---|---|---|
| 20260806T150619Z | 8 | 136.085 (0.219) | 136.163 (0.229) | 135.971 (0.231) | +0.099% | inside | null |
| 20260806T150846Z | 4 | 135.478 (0.208) | 135.921 (0.244) | 135.922 (0.191) | +0.163% | inside | null |

Rules:
- sandwich mean = (ctrl-a + ctrl-b)/2; Δ% = (vdr − mean)/mean.
- Verdict null if |Δ| < 0.68% between-run drift bound (findings 8/10).
- VDR=2 was measured null in a prior window (open-lead 22, same harness family).

Prior data point (vdr2-cycle.sh, earlier window): VDR=2 null (+0.158% vs sandwich).

## Interpretation
VDR=8 null (+0.099%, inside ±0.68% bound): both extreme depths (2, 8) are null
→ the work-distribution axis of the lm_head q6_K MMVQ is closed with data; the
kernel is not thread-partition-limited. Consistent with the 475 GB/s effective
bandwidth (compute/latency-limited) reading: adding ILP per thread does not
touch the per-block dequant cost.
VDR=4 detail: 135.921 vs ctrl-b 135.922 (0.001% apart) — the +0.163% vs sandwich
mean is entirely ctrl-a window drift (135.478, 0.33% below ctrl-b).
