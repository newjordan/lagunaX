# Frontier — Non-matmul SYCL op launch budget + supports_op composition (2026-08-01)

Direction 25 candidate. Distinct from dir 16 (native mul_mat), 17 (weight tensor meta), 23 (supports_op *volume only*).

## Evidence base
`results/ktrace-tip-20260730/decode-ggml/trace.log` (ggml SYCL debug, tip binary)

## Key numbers
### device_supports_op by op type (677,040 total; sums exact)
| op | count | % |
|----|------:|--:|
| NONE | 181,680 | 26.8 |
| ADD | 103,440 | 15.3 |
| MUL_MAT | 86,400 | 12.8 |
| MUL | 57,600 | 8.5 |
| RMS_NORM | 38,640 | 5.7 |
| MUL_MAT_ID | 28,080 | 4.1 |
| VIEW/SET_ROWS/ROPE/PERMUTE | 19,200 each | |
| FLASH_ATTN_EXT | 9,600 | 1.4 |
| Matmul family (MUL_MAT+ID) | 114,480 | **16.9** |

### SYCL[OP] entry launches (exclude `done`)
| op | launches |
|----|--------:|
| mul_mat | 159,064 |
| quantize_row_q8_1 | 127,793 |
| rope (Q) | 21,760 |
| rope_fused (K) | 21,760 |
| set_rows | 21,760 |
| to_fp16 | 13,242 |
| to_fp32 | 3,098 |
| get_rows | 1,088 |
| add | 544 |

## Structural claims
1. K-cache path is rope_fused (mode=2 rope+view+set_rows); V-cache is a separate set_rows f32→f16 — 100% of set_rows dsts are `cache_v_*`.
2. Q rope never fused to cache (always plain `ggml_sycl_rope` → Qcur_rope).
3. ADD supports:launches ≈ 190:1 — residual fuses kill launches but not support queries.
4. FLASH_ATTN_EXT is supported and dispatched (`ggml-sycl.cpp:6986-6988`) but **never** appears under `[SYCL][OP]` — instrumentation blind spot for all prior launch-count work.
5. Q-head heterogeneity: 10 layers × 48 heads (Q ne1=6144), 30 layers × 64 heads (Q ne1=8192), KV always 8 heads — root of SDP 6:1 vs 8:1 partitions.

## Hypotheses
- Fuse V set_rows (f32→f16 scatter) into V projection or a dual-KV write kernel (mirror K rope_fused).
- Memoize supports_op by (op, type) — NONE+ADD alone are 42% of 677k queries.
- Instrument fattn path with SYCL[OP] timing to put attention on the same budget board as matmul.
