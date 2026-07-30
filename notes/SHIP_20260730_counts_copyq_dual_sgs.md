# Research — counts copy-q + dual sgs on packed-reduce tip (2026-07-30)

## Status: **no tip change**

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| **tip packed reduce** | **3540.3** | **130.2** | **+53.41%** | OK |
| dual sgs=8 env | 3538.9 | 129.9 | +53.20% | OK under |
| **counts D2H copy-q default ON** | **3367.4** | **129.8** | **+51.22%** | OK **regress** |
| tip recheck (copy-q OFF) | 3508.1 | 130.4 | +53.23% | OK ~tip noise |

Formals: `20260730T120056Z` (dual8), `20260730T120357Z` (copy-q), `20260730T120539Z` (recheck).

## What

1. **Dual sgs=8** remeasure on packed tip — still under (tg/pp flat-under).
2. **Secondary SYCL queue** for expert-loop counts D2H after hist only, overlapping
   scan+fill+pack on main in-order stream. Theory: main queue was serializing
   D2H before pack. Practice: **~−170 pp** — L0 dual-queue contention on B70.
3. Code retained **opt-in** `GGML_SYCL_ENABLE_MOE_COUNTS_COPY_Q=1` (default OFF).
   Device hist buffers lifetime fixed (hoisted until after wait).

## Tip unchanged

`+53.41%` packed reduce (`20260730T115251Z`).

## Next

1. lm_head prune/mask with golden oracle.
2. Avoid dual-queue D2H without profiled L0 copy engine win.
3. Expert-loop host tax: try other approaches (not second queue).
