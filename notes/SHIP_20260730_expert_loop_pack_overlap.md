# Research — expert-loop pack before counts-wait (2026-07-30)

## Status: **no tip change** (golden OK; score flat within noise)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| **tip packed reduce** | **3540.3** | **130.2** | **+53.41%** | OK |
| pack-before-counts-wait | 3541.2 | 130.1 | **+53.39%** | **OK** |

Formal: `results/20260730T123248Z/`

## What

On dual+down **expert-loop** prefill device-sort path, pack only needs `dev_row_mapping`.
Previously:

1. hist → scan → fill → **D2H counts + host wait + exclusive-scan offsets**
2. pack activations
3. expert GEMMs

Now:

1. hist → scan → fill → async D2H counts
2. **pack activations** (queued; mapping only)
3. host wait D2H + build offsets (can overlap pack)
4. expert GEMMs

Keep `dev_counts`/`dev_next` alive until after D2H wait (pool LIFO-safe). Bitexact;
no second queue (copy-q / shared-USM still closed).

## Why flat

Host exclusive-scan of 256 ints is negligible vs pack + expert GEMMs. Overlap does not
move pp/tg under formal noise. Confirms remaining prefill tax is **GEMM / weight traffic**,
not the small counts sync.

## Tip unchanged

Keep packed-reduce tip (`20260730T115251Z`). Reorder stays in control tree (harmless default).

## Next

1. **Re-trace under tip** (ggml/UR) for non-MoE remaining wall — attn gate, o_proj, FA.
2. Avoid further counts-D2H thrash (copy-q, shared-USM, this overlap — all closed).
3. Candidate: fuse **attn gate proj → softplus** or o_proj path if trace shows share.
