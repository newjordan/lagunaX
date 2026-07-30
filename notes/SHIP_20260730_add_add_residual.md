# Ship note — ADD+ADD residual fuse (2026-07-30)

## Status: **SCORED TIP** (default ON)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior tip RMS+MUL | 1146.4 | 123.2 | +11.04% | OK |
| **+ ADD+ADD residual** | **1145.0** | **125.2** | **+12.35%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T073706Z/`

Hit:
```
[lx-control-add] fuse hit (add+add residual) nelt=2048 ne0=2048 ne1=1    # decode
[lx-control-add] fuse hit (add+add residual) nelt=1048576 ne0=2048 ne1=512 # prefill
```

## What

Fuse Laguna FFN residual chain:

```
ffn_out = moe_out + ffn_shexp     # ADD
l_out   = ffn_out + ffn_inp       # ADD
```

into one kernel with **left-to-right** `(a+b)+c` (matches two stock ADDs):

- Contiguous same-shape F32 only (no broadcast)
- Graph: `can_fuse_subgraph(ADD, ADD)` → write final buffer only
- Default ON; kill: `GGML_SYCL_DISABLE_ADD_ADD_FUSE=1`

## Why it wins

~**+2 tg128** formal. One fewer full-embd read/write per MoE layer (×~39) on the hot residual path.

## Tip stack (default ON)

1. MoE dual SwiGLU  
2. Hybrid mode7 + sigmoid+add + noop reshape + skip DIV  
3. Dense dual shexp  
4. MoE down k8 + integrated decode-only  
5. Device mmid sort/prefix/event  
6. RMS_NORM+MUL  
7. **ADD+ADD residual**

## Next

1. Multi-token dual/MMVQ (host-sort parity).  
2. Softplus+mul attn gate (reshape surface).  
3. lm_head.
