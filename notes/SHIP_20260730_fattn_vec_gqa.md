# Research — FA VEC for Laguna GQA decode (2026-07-30)

## Status: **opt-in only** — golden FAIL if defaulted; tip unchanged

| arm | tg64 (r=2–3) | golden | notes |
|-----|-------------:|:------:|-------|
| **tip TILE** (GQA default) | **~131.5–131.6** | **OK** | stock selection |
| **FORCE_VEC** decode | **~135.2** | **FAIL** | ~+3.6 tg64 |

## What

Stock FA selection for F16 K/V:

```text
if (Q.ne[1]==1 && !gqa_opt_applies) → VEC
else if can_use_vector → may fall through
→ TILE
```

Laguna GQA (e.g. Q heads 48 / KV 8, `gqa_ratio≥2`) sets `gqa_opt_applies` → **decode uses TILE**, not VEC.

Measured on tip stack:

```
[lx-control-fattn] kernel=VEC  Q=[128,1,48] K=[128,256]  → tg64 ~135.2
[lx-control-fattn] kernel=TILE Q=[128,1,48] K=[128,256]  → tg64 ~131.6
```

## Opt-in

```bash
export GGML_SYCL_FATTN_FORCE_VEC=1   # decode GQA → VEC (~+3.6 tg, golden FAIL)
```

Default remains TILE for GQA decode (golden-safe tip).

## Why not tip

Greedy golden **mismatch** with FORCE_VEC — FA numerics differ enough to change tokens.
Do not default ON without a bitexact VEC path or a separate non-golden speed track.

## Tip unchanged

`+56.53%` mm-add+add decode (`20260730T125600Z`).

## TILE geometry probes (same fire follow-up) — all keep TILE, no golden win

| experiment | tg64 | vs tip TILE |
|------------|-----:|------------:|
| TILE GQA pack ncols2=2 (stock tip) | **~131.5** | — |
| TILE force ncols2=1 for decode | ~123.8 | **regress** |
| TILE force parallel_blocks=1 decode | ~117.2 | **regress** |
| FORCE_VEC | ~135.3 | +3.7, golden FAIL |

GQA head-packing (ncols2=2 for ratio 6) and multi-block K-split **help** TILE speed;
stripping them moves away from VEC, not toward it. Speed gap is algorithmic (VEC kernel),
not just bad TILE launch geometry.

## Next

1. **Bitexact VEC**: match TILE softmax/K-loop order and GQA pairing (ncols2=2 semantics
   in VEC, or fp32 path) — still the highest decode leverage (~+3–4 tg).
2. Diff first-token FA outputs TILE vs VEC on a fixed prompt to locate divergence.
3. Non-FA levers if FA bitexact stalls.
