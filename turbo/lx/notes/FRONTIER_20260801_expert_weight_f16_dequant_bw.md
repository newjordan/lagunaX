# FRONTIER 20260801: Expert GEMM weight precision / dequant bandwidth

## Direction
The model ships as **Q4_K** (18.88 GiB for 33.44 B params, verified in-trace).
The dense-FFN, shared-expert, and attention-projection paths correctly read
**quantized weights directly** via the ggml `mmvq` custom kernels
(`[lx-control-mm-add] wtype=q4_K`, `wtype=q6_K`, `[lx-control-dense-dual] mmvq`,
`[lx-control-moe-down] weighted-mmvq`).

**But the expert GEMM stream does NOT.** All 188,995 `primitive,exec,gpu,matmul`
expert calls carry `wei:f16::blocked:cab` — full fp16 weights — meaning the Q4_K
expert weights are dequantized to f16 before each oneDNN GEMM, paying a 3.6×
weight-bandwidth penalty that the quantized custom-kernel path avoids entirely.

This is **distinct from direction 4 (dst output precision)**: that axis was about
the *output* (`dst:f32` vs `dst:f16`); this is about the *weight input*
(`wei:f16` dequantized-from-Q4_K vs `wei:q4_K` direct).

## Evidence (from ktrace-post-brownout-20260731/trace.log)

### Weight precision split
| path | weight format | kernel | bandwidth |
|------|-------------|--------|-----------|
| expert GEMM (188,995 calls) | `wei:f16` (dequantized from Q4_K) | oneDNN `jit:gemm:any` | 31.708 GB read |
| dense/shared-expert (mmvq) | Q4_K / Q6_K direct | ggml custom `mmvq` | ~8.918 GB equivalent |
| f32-src dense (1,216 calls) | `wei:f32` | oneDNN `jit:gemm:any` | — |

### Quantification
- f16 expert weight reads: **31.708 GB** across 188,995 calls
- If weights were Q4_K (0.5625 B/elem): **8.918 GB** → **22.790 GB waste** (3.6× amplification)
- Expert achieved BW: 122–132 GB/s (gate/up + down families) vs dense 340–444 GB/s
- The expert path is **25–35% of the GPU's own dense BW** — consistent with weight-bandwidth-bound,
  not compute-bound

### Why this matters for speed
The expert GEMM stream is 86.3% of the 3828.33 ms matmul exec total (finding #13).
For the BW-bound tiers (N=129–256 prefill = 212 ms, N=17–128 = 1093 ms),
weight traffic dominates. A 3.6× reduction in weight reads would proportionally
relieve the bandwidth bottleneck that caps expert throughput at 25–35% of dense.

### Why this is distinct from all 8 tried directions
1. NOT launch-coalescing (keeps GEMM count, changes weight format)
2. NOT cache-hit host tax (GPU-side BW, not host-side lookup)
3. NOT N-tile ladder (weight precision, not N-padding)
4. NOT dst output precision (this is the **weight input** `wei:f16`, not `dst:f32`)
5. NOT post-op epilogue fusion (different problem entirely)
6. NOT kernel selection (same `jit:gemm:any` kernel; different weight dtype)
7. NOT 3-tier cache nesting decomposition (GPU compute-side BW)
8. NOT unit-batch descriptor tax (weight memory, not descriptor format)

## Hypotheses (untested)
- The expert GEMM is routed to oneDNN (which only accepts f16/f32/f8 inputs)
  because the ggml SYCL backend lacks a quantized MoE expert GEMM path; the dense
  path uses mmvq (custom quantized kernel) but the expert dispatch forces
  pre-dequantization to f16.
- If the expert weights could be kept in Q4_K and a quantized GEMV/GEMM used
  (like mmvq), the 22.79 GB weight-bandwidth waste is eliminated → proportional
  speedup on the BW-bound expert tiers.
- The f16 dequant may also cost extra VRAM: 256 experts × 2 matrices (gate+up)
  × 512×2048 × 2 bytes = ~537 MB per layer in f16 vs ~151 MB in Q4_K, but this
  is dominated by the per-call bandwidth cost, not VRAM residency.
- Alternatively: if a fused Q4_K→f16 dequant + GEMM existed (dequant-on-load
  inside the kernel), the weight-read traffic would drop to Q4_K size while
  keeping the oneDNN GEMM compute path — the ggml mmvq kernels already do this
  for dense/shexp.
