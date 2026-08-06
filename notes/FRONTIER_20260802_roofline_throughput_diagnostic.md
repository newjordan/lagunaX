# FRONTIER: Compute-Throughput / Roofline Position of the Expert GEMM Stream

## New angle (direction #13)

No prior direction computed **achieved compute throughput (TFLOPS)**, **arithmetic intensity (AI = FLOP/byte)**, or the **BW-roofline-derived overhead fraction**. All 12 prior directions analyzed call-counts, per-call latency, or bandwidth in isolation — never cross-referenced FLOPs/byte/time into a roofline position.

## Measured data (ktrace-post-brownout-20260731, computed from all 190,211 exec lines)

### Per-shape-class aggregate (flops=2*M*K*N, bytes=2*K*N+2*M*K+4*M*N)

| Class | Count | Time(ms) | TFLOPS | AI | BW(GB/s) |
|---|---|---|---|---|---|
| M512_K2048 (gate/up) | 121,666 | 2333.3 | 4.12 | 34.0* | 121.4 |
| M2048_K512 (down) | 60,833 | 1063.9 | 4.51 | 32.3* | 139.7 |
| M8192_K2048 (dense) | 1,024 | 99.2 | 88.64 | 199.8 | 443.6 |
| M2048_K8192 (dense) | 992 | 114.7 | 74.29 | 215.6 | 344.6 |
| M1024_K2048 (dense) | 2,560 | 91.1 | 30.17 | 170.7 | 176.8 |

(*aggregate AI = total_flops/total_bytes, not the simple mean of per-call AI)

Total: 39.00 TFLOP / 3.828s = 10.19 TFLOPS aggregate. Expert: 14.41 TFLOP = 37% of FLOPs but 89% of time.

### Per-call AI by N (gate/up family, M=512 K=2048)

| N | FLOP/call | Bytes/call | AI | Measured time | Achieved TFLOPS |
|---|---|---|---|---|---|
| 9 | 18.9M | 2.15 MB | 8.8 | ~0.018 ms | 1.04 |
| 16 | 33.6M | 2.20 MB | 15.3 | ~0.019 ms | 1.77 |
| 128 | 268M | 2.88 MB | 93.1 | ~0.025 ms | 10.7 |
| 256 | 537M | 3.67 MB | 146.3 | ~0.030 ms | 17.9 |
| 256 (dense M8192) | 8.6G | 43.0 MB | 199.8 | 0.097 ms | 88.6 |

### BW-roofline overhead decomposition (the new diagnostic)

Using the dense M8192_K2048 path as the GPU's measured peak BW (443.6 GB/s):

For gate/up at N=16 (dominant expert tier):
- Bytes/call = 2.20 MB
- Minimum GPU time at peak BW: 2.20 MB / 444 GB/s = **4.95 µs**
- Measured per-call time: **~19 µs**
- Implied overhead: **14.05 µs = 74% of measured wall-time**
- Implied GPU-only BW if all time were GPU: 2.20 MB / 19 µs = 116 GB/s (matches the 121 GB/s aggregate)

For dense M8192_K2048:
- Bytes/call = 43.0 MB
- Minimum GPU time at peak BW: 43.0 MB / 444 GB/s = **96.8 µs**
- Measured per-call time: **97 µs**
- Implied overhead: **~0.2 µs = 0.2% of measured wall-time** (negligible — confirms the dense path is at the BW roofline)

This derives the 74% overhead fraction from a BW-roofline model — a methodology distinct from the N-tile launch-floor (direction 3), the cache-hit tax (direction 2), or the API-overhead tier decomposition (direction 7).

## Per-call traffic composition (gate/up, N=16)

| Component | Bytes | Share |
|---|---|---|
| src (weight matrix, 512×2048 f16) | 2,097,152 | **95.4%** |
| wei (activation, 2048×16 f16) | 65,536 | 3.0% |
| dst (output, 512×16 f32) | 32,768 | **1.6%** |

The weight matrix (A/src in oneDNN, d_expert×d_model) dominates per-call traffic at 95.4%; the f32 output write that direction 4 targets is only 1.6%.

## Total expert weight re-read traffic

- Gate/up: 121,666 × 2.1 MB = **255 GB** of weight reads
- Down: 60,833 × 2.1 MB = **125 GB** of weight reads
- Total expert weight traffic: **380 GB** (each expert's 2 MB weight matrix re-read for every GEMM call that hits it)

## Distinct from all 12 prior directions

- Not launch-count (dir 1), not cache-hit tax (dir 2), not N-tile floor (dir 3)
- Not output precision (dir 4 — here shown to be 1.6% of per-call traffic)
- Not epilogue fusion (dir 5), not kernel selection (dir 6), not API overhead tiers (dir 7)
- Not descriptor rank (dir 8), not weight precision BW (dir 9 — complementary: that quantified the f16-vs-Q4_K amplification, this quantifies the absolute weight traffic and AI)
- Not gate/up concat (dir 10), not host-exec serialization (dir 11), not logging contamination (dir 12)
