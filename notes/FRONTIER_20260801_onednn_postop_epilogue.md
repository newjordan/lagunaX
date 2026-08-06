# Frontier: oneDNN primitive post-op epilogue fusion (zero-post-op GEMMs)

## Core finding
ALL 190,211 `primitive,exec,gpu,matmul` calls are **bare GEMMs with zero oneDNN post-ops**.
Every exec line carries only `attr-scratchpad:user attr-fpmath:f16` — no `post_ops` chain
(bias/eltwise/sum/reduction). The MoE epilogue work (swiglu on gate/up, weighted-reduce on down)
is handled by *separate* ggml control kernels ([lx-control-moe-dual], [lx-control-moe-down]),
never fused inside the oneDNN primitive.

## Evidence (from results/ktrace-post-brownout-20260731/trace.log)
- `grep -c 'post_op\|post-op\|attr-post'` on all exec lines = **0**
- Attr field is uniformly `attr-scratchpad:user attr-fpmath:f16` (190,211 / 190,211)
- Separate epilogue kernels confirmed: `[lx-control-moe-dual] fuse hit (gate+up+swiglu)`,
  `[lx-control-moe-down] fuse hit (weighted reduce)`

## Cost-tier distribution (TIME-weighted, not just count)
| tier | calls | time (ms) | avg (ms) |
|------|-------|-----------|----------|
| <20µs | 138,474 | 2409.85 | 0.0174 |
| 20-30µs | 44,303 | 976.71 | 0.0220 |
| 30-50µs | 4,712 | 170.43 | 0.0362 |
| 50-100µs | 1,639 | 146.00 | 0.0891 |
| 100-500µs | 1,082 | 124.73 | 0.1153 |
| >500µs | 1 | 0.60 | 0.6050 |

The <20µs floor-bound tier = 2409.85 ms = **63% of the 3828 ms matmul total**.

## Why this is distinct from all 4 tried directions
1. NOT launch-coalescing (still 1 GEMM per expert, just with fused epilogue)
2. NOT cache-hit-tax (host-side; this is GPU-kernel-elimination)
3. NOT N-tile ladder
4. NOT output-precision (f32 vs f16 dst)

oneDNN natively supports matmul `post_ops`: `eltwise` (for swiglu/silu), `binary` (add bias),
`sum` (accumulate into residual). Fusing the epilogue INTO the GEMM kernel would eliminate the
separate consumer kernel launch entirely.

## Hypothesis (untested)
- The ggml control fusions (moe-dual, moe-down) already combine mul_mat+activation/reduce into
  a custom kernel. Moving the epilogue INTO the oneDNN post_op would not add compute but WOULD
  eliminate the separate kernel launch + its memory round-trip. Net win depends on whether the
  custom ggml kernel is already as fast as a oneDNN fused GEMM+post_op.
- For the down-family (weighted-reduce), the reduction dimension maps poorly to a oneDNN matmul
  post_op (it's a weighted-sum across experts, not a simple bias-add), so the gate/up swiglu
  fusion is the more tractable first target.
