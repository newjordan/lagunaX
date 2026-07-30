# Research — lm_head ROI + prune ceiling (2026-07-30)

## Status: **no tip change** (probe opt-in only; tip path default OFF)

`output.weight` is **Q6_K** `[2048 × 100352]` (~160.8 MB / decode step). Prior packing probes
were flat; this fire measures **wall-time share** and **upper bound** if vocab rows could be
skipped (ideal prune).

## Probe (control `mmvq.cpp`, default OFF)

| env | effect |
|-----|--------|
| `GGML_SYCL_PROFILE_LM_HEAD=1` | host-time large-nrows Q6_K reorder GEMV (stderr) |
| `GGML_SYCL_LM_HEAD_ROW_LIMIT=N` | compute first N rows only; tail logits = −inf (**not golden**) |

Both require `nrows ≥ 65536` and `ncols == 2048` (lm_head-scale).

## Measured (B70, tip stack, tg64 r=2)

| arm | tg64 | notes |
|-----|-----:|-------|
| tip (no env) | **130.81** | packed-reduce tip binary |
| profile ON (full vocab) | 132.07 | kernel **~0.296 ms** avg (128+ calls) |
| `ROW_LIMIT=65536` (~65%) | 132.19 | |
| `ROW_LIMIT=32768` (~33%) | 133.87 | |
| `ROW_LIMIT=16384` (~16%) | 134.69 | |
| `ROW_LIMIT=8192` (~8%) | **135.64** | near full-elimination of lm_head GEMV |

### Share of decode

- Token budget @130 t/s ≈ **7.7 ms**
- Full Q6_K lm_head GEMV ≈ **0.30 ms** → **~3.8–4%** of serial decode wall
- Eliminating nearly all of it (8k-row ceiling) → **~+4.8 tg** (~+3.7% decode)
- Formal score class if tg 130.2→135 and pp flat: roughly **+3–4 pp** composite max
  (decode^0.75 weighted) — real golden-safe prune would capture only a fraction of that

## Implications

1. **lm_head is real but not dominant** under the current dual+down+topk+packed tip.
   MoE expert traffic still dominates (kernel-trace / power profile).
2. **Prune/mask design is optional**, not the highest-leverage open lever for +10% score.
   Worth doing only if a **cheap golden-safe** coarse path appears (M5-style table), not a
   multi-day custom quant path for ≤4 tg theoretical.
3. Launch packing (prior fire) correctly flat: GEMV is already BW-efficient at ~0.3 ms for
   161 MB → effective ~540 GB/s class on the weight stream.

## Tip unchanged

Packed reduce formal **+53.41%** (`20260730T115251Z`). Probe code stays opt-in for future
coarse-mask experiments; do **not** default `ROW_LIMIT`.

## Next

1. Prefer levers with **>5% decode** headroom (remaining MoE / attn graph), not full lm_head
   redesign.
2. If revisiting lm_head: offline coarse candidate table + sparse Q6 refine, golden oracle
   only; target K such that second pass ≪ 0.3 ms (e.g. K≤4k).
3. Re-trace under packed tip (UR/oneDNN) for new hottest non-lm_head ops.
