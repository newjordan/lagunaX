# Ship note — MoE down **prefill two-step weighted reduce** (2026-07-30)

## Status: **SCORED TIP** (default ON; extends prior moe-down weighted)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior tip rope+set_rows | 3005.5 | 128.9 | +46.18% | OK |
| **+ prefill two-step weighted** | **3148.4** | **128.9** | **+47.88%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal tip: `results/20260730T085918Z/`

Hits:
```
[lx-control-moe-down] two-step scratch experts=0 weights=1 tokens=512
[lx-control-moe-down] fuse hit (weighted reduce) embd=2048 k=8 tokens=512
[lx-control-moe-down] multi-token two-step weighted reduce embd=2048 k=8 tokens=512
# decode unchanged:
[lx-control-moe-down] fuse hit (integrated weighted-mmvq) embd=2048 k=8 tokens=1
```

## What

1. Raise two-step weighted-reduce fuse `n_tokens` cap **32 → 2048** (serial ubatch).
2. Prefill product allocator often **aliases** `weights`/`mmid` with fuse `dst`. Prior hard-skip left prefill on stock VIEW×8+ADD×7.
3. On alias: copy tiny weights `[1,k,T]` and/or experts buffer to **pool scratch**, then `mul_mat_id` + `moe_weighted_reduce` (same k=8 unroll contract).
4. Integrated one-kernel path stays **decode-only** (`n_tokens==1`, no alias).

Kill (unchanged): `GGML_SYCL_DISABLE_MOE_DOWN_WEIGHTED=1`

## Why it wins

Prefill: collapse MUL + 8 VIEWs + 7 ADDs into one weighted reduce after expert down GEMM.  
~**+143 pp** formal; decode flat (integrated path unchanged). Score weight favors decode but prefill lift is large.

## Tip stack (default ON)

Prior rope tip + **prefill moe-down two-step weighted reduce** (scratch on buffer alias).

## Next

1. Multi-token dual/MMVQ (still GEMM-vs-MMVQ golden risk).  
2. lm_head.  
3. Further epilogues / true TOP_K.
