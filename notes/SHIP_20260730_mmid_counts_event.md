# Ship note — mmid counts **event-wait** (pack overlap) 2026-07-30

## Status: **DEFAULT ON** (golden OK; score ≈ tip, not a formal beat)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| tip fused sigmoid+add | 1148.9 | 120.2 | **+9.09%** | OK |
| prefix-sum + **stream->wait** | 1143.0 | 120.3 | +8.99% | OK |
| **prefix-sum + counts event-wait** | **1144.2** | **120.3** | **+9.04%** | **OK** |
| wait-after-pack (stream) | ~1127–1129 | ~120 | ~+8.1–8.7% | OK (regressed) |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal: `results/20260730T055755Z/` · hit  
`[lx-control-mmid] device counting-sort+prefix+ev n_tokens=512 k=8 n_experts=256`

Kill: `GGML_SYCL_DISABLE_MMID_DEVICE_SORT=1` (same as device sort/prefix)

## What

Multi-token `mul_mat_id` device counting-sort path:

```
hist → exclusive_scan → fill mapping → D2H counts (sycl::event)
     → pack src1
     → event.wait()          // hist..D2H only — NOT full stream wait
     → host offsets from counts
     → expert GEMMs → scatter
```

vs prior `stream->memcpy` + `stream->wait()` **before** pack (serialized pack behind host), or full `stream->wait()` **after** pack (host bubble before GEMMs, ~−20 pp).

Event wait covers only the D2H command’s dependency chain (hist/scan/fill/D2H). Pack enqueued after D2H can still be in flight while host builds expert views; GEMMs stay after pack on the in-order queue.

## Why not tip stamp

Composite +9.04% vs tip +9.09% (pp slightly under tip formal noise). Ships as default infra under the same kill-switch.

## Tip stack

Scored tip stamp remains **fused sigmoid+add** (`20260730T053204Z`).  
Default mmid path: device counting-sort + prefix-sum + **event counts wait**.

## Next

1. Pinned/USM host counts if L0 pageable D2H still blocks enqueue.  
2. Compact non-empty expert list (shorter host GEMM loop) without mid-op full wait.  
3. Bitexact multi-token dual/MMVQ.  
4. Fused router sum/div (golden-fail).
