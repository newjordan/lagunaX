# Ship note — RMS_NORM + MUL(weight) fuse (2026-07-30)

## Status: **SCORED TIP** (default ON)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior tip integrated decode-only | 1147.6 | 122.4 | +10.55% | OK |
| **+ RMS_NORM+MUL fuse** | **1146.4** | **123.2** | **+11.04%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T072939Z/`

Hit:
```
[lx-control-rms] fuse hit (rms_norm+mul) ne0=2048 ne1=1   # decode
[lx-control-rms] fuse hit (rms_norm+mul) ne0=2048 ne1=512 # prefill
```

## What

Port package/CUDA-style **RMS_NORM → MUL(weight)** fuse onto control:

- One kernel: `out = (rsqrt(mean(x²)+ε) * x) * w` with sequential `(scale*x)*w` (stock order)
- Graph: `ggml_can_fuse_subgraph(RMS_NORM, MUL)` → skip following MUL
- Laguna hits: attn_norm, ffn_norm, output_norm, Q/K norms every layer

Kill:
```bash
export GGML_SYCL_DISABLE_RMS_NORM_FUSE=1
```

## Why it wins

Decode-weighted score: ~**+0.8 tg128** formal. Elides a full embd-sized write+read of the unscaled RMS intermediate on every pre-norm (×~3+ per layer ×40).

## Tip stack (default ON)

1. MoE dual SwiGLU  
2. Hybrid mode7 + fused sigmoid+add + noop reshape + skip DIV store  
3. Dense dual shexp  
4. MoE down k8 unroll + **integrated decode-only**  
5. Device mmid sort/prefix/event  
6. **RMS_NORM+MUL fuse**

## Code

- `norm.cpp` / `norm.hpp`: optional mul epilogue + `ggml_sycl_op_rms_norm_fused`
- `ggml-sycl.cpp`: `ggml_sycl_fuse_rms_norm_mul`
- `topk-moe.cpp`: wire into `ggml_sycl_fuse`

## Next

1. Multi-token dual/MMVQ (mirror host-sort multi-token mmid).  
2. Optional RMS+MUL+ADD residual if graph ever stacks that way.  
3. lm_head.
