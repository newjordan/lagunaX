# Ship note — softplus×mul attn gate fuse (2026-07-30)

## Status: **SCORED TIP** (default ON)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior tip ADD+ADD | 1145.0 | 125.2 | +12.35% | OK |
| **+ softplus×mul** | **1158.8** | **127.2** | **+14.07%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T074337Z/`

Hit:
```
[lx-control-softplus-mul] fuse hit (xs2 per-head) hd=128 nh=48 T=1
[lx-control-softplus-mul] fuse hit (xs2 per-head) hd=128 nh=48 T=512
```

## What

Fuse Laguna XS.2 attention output gate:

```
gate = softplus(g_proj)
attn = reshape [head_dim, n_head, T]
gate = reshape [1, n_head, T]
out  = attn * gate   # broadcast over head_dim
```

into one kernel:

```
out[d,h,t] = attn[d,h,t] * softplus(gate_logits[h,t])
```

with stock softplus: `max(x,0) + log1p(exp(-|x|))`.

- Skips SOFTPLUS + intermediate RESHAPEs + MUL (+ optional view RESHAPE)
- Default ON; kill: `GGML_SYCL_DISABLE_SOFTPLUS_MUL_FUSE=1`

## Why it wins

~**+2 tg128** and prefill lift formal. Softplus+broadcast-mul every attention layer was multiple launches; now one.

## Tip stack (default ON)

1. MoE dual SwiGLU  
2. Hybrid mode7 + sigmoid+add + noop reshape + skip DIV  
3. Dense dual shexp  
4. MoE down k8 + integrated decode-only  
5. Device mmid sort/prefix/event  
6. RMS_NORM+MUL  
7. ADD+ADD residual  
8. **softplus×mul attn gate**

## Next

1. Multi-token dual/MMVQ (host-sort parity).  
2. o_proj MUL_MAT+ADD residual epilogue.  
3. lm_head.
