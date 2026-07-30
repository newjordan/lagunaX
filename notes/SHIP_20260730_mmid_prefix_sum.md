# Ship note — mmid device exclusive prefix-sum (2026-07-30)

## Status: **DEFAULT ON** (golden OK; score ≈ tip, not a formal beat)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| tip fused sigmoid+add | 1148.9 | 120.2 | **+9.09%** | OK |
| **+ device exclusive prefix-sum** | **1143.0** | **120.3** | **+8.99%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal: `results/20260730T054207Z/` · hit  
`[lx-control-mmid] device counting-sort+prefix n_tokens=512 k=8 n_experts=256`

Kill (same as prior device sort): `GGML_SYCL_DISABLE_MMID_DEVICE_SORT=1`

## What

On multi-token `mul_mat_id` device counting-sort path, replace:

1. hist → D2H counts + wait → **host exclusive scan** → **H2D next[]** → fill → pack → GEMMs

with:

1. hist → **device exclusive scan** (seeds `next[]`) → fill → D2H counts + wait → host offsets from counts → pack → GEMMs

Removes H2D of the per-expert offset vector. Expert GEMM regroup numerics unchanged (same class as device counting-sort).

## What did **not** work (measured)

**Wait after pack / deferred counts D2H** to “overlap” fill+pack with D2H:

| variant | pp512 | tg128 | score |
|---------|------:|------:|------:|
| wait after pack (v1/v2) | ~1127–1129 | ~119–120 | **~+8.1–8.7%** |
| early wait before pack (this) | 1143 | 120.3 | +8.99% |

Root cause: host bubble between pack completion and expert-GEMM enqueue regresses prefill ~20 t/s on B70. Keep pack→GEMM continuous on the in-order queue.

## Why not a score tip bump

Prefill −6 t/s vs tip formal (noise / small tax of extra 1-thread scan kernel); decode flat.  
Ships as default: golden-safe, drops H2D next, documents the wait-ordering constraint for next prefill work.

## Tip stack (scored claim unchanged)

Scored tip stamp remains **fused sigmoid+add** (`20260730T053204Z`, +9.09%).  
This change is default infrastructure under the same mmid device-sort kill-switch.

1. MoE dual SwiGLU  
2. Hybrid mode2 + fused sigmoid+add  
3. Dense dual shexp  
4. MoE down weighted  
5. Device mmid counting-sort + **prefix-sum** (this)

## Next

1. ~~True zero-wait / better wait~~ → **event-wait** shipped (`SHIP_20260730_mmid_counts_event.md`, ~+9.04%).  
2. Pinned/USM counts or compact non-empty list.  
3. Bitexact multi-token dual/MMVQ (still golden-fail).  
4. Fused sum/div router norm (still golden-fail).
