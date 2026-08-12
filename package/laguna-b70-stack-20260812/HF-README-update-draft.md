# Draft section for Frosty40/Laguna-XS-2.1-ArcB70-GGUF (publish at your discretion)

Suggested: add the promo image to `assets/`, then insert this section after
the existing "Results (tip vs base)" section of the README.

---

## 2026-08-12 update — the long-context stack (5.75x ingest, +13% deep decode)

The follow-up campaign targeted **max-context** workloads at the full
131,072-token window. Three env-gated kernel changes (tag
`lx-stack-1.4092-20260812`) on top of the previous champion:

| | champion | long-context stack |
|---|---|---|
| 23K-token real-text ingest @ 131K ctx | 307 t/s | **1,764 t/s (5.75x)** |
| decode @ 23K depth | 81.5 t/s | **90 t/s** |
| decode @ 122K depth | 36.0 t/s | **40.8 t/s** |
| tg128 (short context) | 152.5 t/s | 152.5 t/s |

```bash
# the three knobs (on top of the existing ship env)
export GGML_SYCL_LX_REORDER_MULTICOL_MKL=1   # wide batches -> fp16/oneMKL (the 5x)
export GGML_SYCL_LX_FATTN_PARALLEL_BLOCKS=16 # FA decode split-K width
export GGML_SYCL_LX_EXPERT_TILE_GEMM=1       # XMX fused dequant-GEMM, small-N experts
```

Mechanism in one line: a warmup decode permanently latches weight reordering,
after which a blanket dispatch guard shredded every wide matmul into 8-column
MMVQ chunks — narrowing that guard to decode widths recovers 5x prefill at
identical decode; split-K widening and an XMX expert-tile kernel add the rest.

Quality: bit-parity with the champion when the knobs are off; knobs-on output
measures closer to canonical fp16 math than the champion did (KLD-to-canonical
0.045 vs 0.056, top-1 agreement 92.8% vs 91.7%) with slightly better wikitext
PPL, verified per-dispatch against an fp32 reference and NaN-watched over
1536-token generations at full context.
