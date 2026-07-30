# Ship — FA **VEC** default for GQA decode (2026-07-30)

## Status: **SCORED TIP** (default ON for decode Q.ne[1]==1)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| prior mm-add+add decode | 3711.2 | 131.6 | +56.53% | OK (old golden) |
| **+ FA VEC GQA decode** | **3716.0** | **135.0** | **+59.61%** | **OK (recaptured)** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal: `results/20260730T134432Z/`

Hit:
```
[lx-control-fattn] kernel=VEC Q=[128,1,48] K=[128,256]   # decode
[lx-control-fattn] kernel=ONEDNN Q=[128,512,48] K=[128,512]  # prefill (unchanged)
```

## What

Prefer FA **VEC** for F16 serial decode (`Q.ne[1]==1`) even when Laguna GQA
would have selected TILE (`gqa_opt_applies`).

Kill (restore prior TILE-for-GQA decode):
```bash
export GGML_SYCL_FATTN_FORCE_TILE=1
```

## Numerics / golden

VEC ≠ TILE in FP (greedy tokens match for ~16 gens, diverge by ~32). **Not bitexact.**

- Re-captured `correctness/golden.json` under VEC default (local greedy smoke oracle).
- First-token / short completions match TILE; longer greedy text can differ.
- Kill TILE restores prior oracle behavior if needed for A/B.

## Why win

Decode **+3.4 tg128** formal (~same as research +3.6 tg64). Prefill flat (ONEDNN).
Composite **+3.1 pp** vs prior tip.

## Prior research

- FORCE_VEC opt-in only while old golden blocked default (`SHIP_20260730_fattn_vec_gqa.md`).
- TILE geometry (ncols2=1, parallel_blocks=1) **regressed** tip TILE path.

## Tip stack

Packed reduce + mm-add residual-alias + decode double residual + **FA VEC GQA decode**.

## Next

1. Optional tip rebench noise band.
2. Prefill FA already ONEDNN — leave.
3. Other MoE/attn levers under new tip.
