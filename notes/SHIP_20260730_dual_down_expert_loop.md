# Ship note — MoE dual+down multi-token **expert-loop** (2026-07-30)

## Status: **SCORED TIP** (default ON)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior dual+down graph compose | 3161.0 | 129.0 | +48.09% | OK |
| **+ expert-loop (no glu materialize)** | **3266.0** | **128.9** | **+49.29%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T095249Z/`

Hit:
```
[lx-control-moe-dual] multi-token dual+down EXPERT-LOOP n_tokens=512 k=8 n_experts=256
[lx-control-moe-dual] fuse hit (dual+down multi-token) tokens=512 k=8
```

## What

Replace compose path (dual → write glu → re-sort → down mmid → reduce) with **one** device-sort + pack:

1. Pack activations by expert (shared ids).
2. Per expert with rows: gate GEMM → up GEMM → swiglu → down GEMM (all expert-batch layout).
3. Scatter down outputs → `[embd,k,T]`, weighted reduce → residual.

No intermediate glu materialize; no second ids counting-sort/pack for down.

Mul_mat shells templated from **live** weight/act tensors (stack-only shells segfaulted).

Fallback: compose (dual fuse + stock mmid + reduce) if expert-loop fails.  
Kill expert-loop only: `GGML_SYCL_DISABLE_MOE_DUAL_DOWN_EXPERT_LOOP=1`  
Kill dual+down: `GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1`

## Why it wins

Prefill: ~**+105 pp** formal by dropping glu HBM write/read + duplicate sort/pack for down.  
Decode unchanged (ne12==1 still dual MMVQ + integrated down).

## Tip stack

Prior dual+down tip + **expert-loop** path default ON.

## Next

1. lm_head (decode-weighted).  
2. First-layer can_fuse still sometimes dual-only before expert-loop.  
3. Dense shexp multi-col GEMM dual (not MMVQ).
