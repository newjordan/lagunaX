# Research — multi-row router F32 GEMV under tip (2026-07-30)

## Status: **CLOSED** — no tip change (decode-only GEMV remains)

| arm | pp512 | tg64/128 | notes |
|-----|------:|---------:|-------|
| **tip decode-only GEMV** formal | **3730** | tg128 **138.4** / **+62.75%** | MKL prefill |
| expert×row custom GEMV all-N | ~3231 | tg~138 | thrash W re-read |
| expert-outer (1 WG/expert, loop rows) | ~3330 | tg64 ~139 | better but still −11% pp |
| expert-outer + SLM W row | ~2814 | tg64 ~140 | worse pp (SLM tax) |
| float4 decode GEMV | ~3736 | tg128 ~138.3 / +62.75% | flat score; **golden FAIL** vs tip oracle |

## What was tried

Unlock prefill `n_rows>1` for fused F32 router GEMV+sigmoid+bias:

1. **expert×row** launch (prior ship, crushed pp) — re-reads W n_rows× under thrash.
2. **expert-outer** — one WG per expert, sequential rows (W reuse in L1/L2).
3. **SLM stage W[e]** then loop rows.
4. **float4** decode path (micro) — score flat, tokens diverge → reverted to scalar.

## Conclusion

Stock **MKL GEMM + fused sigmoid+add** remains best for multi-row prefill.
Decode-only custom GEMV tip stands. Do not default multi-row custom GEMV without a
path that matches MKL bandwidth (true GEMM / DNNL), not a tall thin GEMV loop.

## Tip unchanged

`+62.75%` router gemv decode (`20260730T141542Z`). Kill: `GGML_SYCL_DISABLE_ROUTER_GEMV_FUSE=1`.

## Next

1. New theory under +62.8% tip (not multi-row thrash).
2. Optional: DNNL/MKL gemm epilogue that writes sig+add without custom tall GEMV.
3. Attn / QKV / residual theories remain open.
