# FINDING 2026-08-11 — P1: decode-at-depth is FA-kernel-bound (~42% of peak BW on the full-attention layers); q8_0 KV is a dead lever; Laguna is interleaved-SWA (only ~10 deep layers)

Receipts: `results/p1-decode-depth-20260811T214831Z/` (depth-sweep.md,
depth-sweep-q8kv.md, energy.tsv, lx-chrono.bin, drivers). Binary: the C4
candidate (`lx-reorder-multicol`, knob ON — decode path identical to champion;
knob only accelerates the -d prefill priming). Loop charter iteration 1.

## Architecture (GGUF metadata, hand-parsed — settles the KV arithmetic)

`sliding_window = 512`, per-layer `head_count = [48,64,64,64, 48,64,…]`
(period 4): **~10 full-attention layers** (48 Q heads, rope 500K/yarn×32) and
**~30 SWA-512 layers** (64 Q heads, rope 10K). KV uniform: 8 KV heads × 128 =
4 KiB/token/layer (f16 K+V). Structural proof SWA-KV allocation is live: full
40-layer KV at 131072 would need ~21 GiB (doesn't fit beside the 18.9 GiB
model); 10 full + 30×512 ≈ 5 GiB — and serve-laguna runs 131072 on this card.
Max-context decode therefore only pays deep-KV reads on ~10 layers.

## Depth sweep (tg128, f16 KV, ub2048, fa on, r=2)

| depth | t/s | ms/token | Δms per 4096 depth |
|---|---|---|---|
| 0 | 152.52 ± 0.23 | 6.556 | — |
| 4096 | 122.54 ± 11.30 | 8.161 | +1.605 (first step: window fill + fixed) |
| 8192 | 114.24 ± 9.26 | 8.753 | +0.592 |
| 16384 | 98.38 ± 6.48 | 10.165 | +0.706/4K |
| 24576 | 86.54 ± 5.31 | 11.555 | +0.695/4K |

Asymptotic slope **0.171 µs per token-of-depth**. Peak-BW cost for the ~10
full layers: 40 KiB/token-of-depth ÷ 567 GB/s = **0.072 µs** ⇒ the at-depth
FA read runs at **~42% of peak bandwidth**. Device power during the sweep:
139 W of the 230 W cap (60%) — not power/compute-saturated. ⇒ The wall is
the FA kernel's memory-access efficiency at long KV, not bytes, not power.

## q8_0 KV probe — DEAD, do not retry

| depth | f16 KV t/s | q8_0 KV t/s |
|---|---|---|
| 0 | 152.52 | 111.58 (−27%) |
| 8192 | 114.24 | 42.45 (−63%) |
| 24576 | 86.54 | 26.48 (−69%) |

Halving KV bytes should *win* if BW-bound; instead q8_0 KV collapses decode at
every depth — the quantized-KV FA path on this stack is catastrophically worse
(kernel fallback class). Dead lever on this build; add to the do-not-try table.

## Prize arithmetic (why this matters for max context)

If the FA at-depth linear term reached peak BW: d24576 token 11.56 → ~9.3 ms
(**+24% decode at 24.5K**). At d≈100K the linear term dominates: measured-slope
token ≈ 24.7 ms (40 t/s) vs BW-ideal ≈ 14.8 ms (67 t/s) — **~+65% at 100K**.
The lever grows with context, which is the stated priority.

Note the constraint from FINDING_20260811_fattn_fullchain_backport_no_win:
master's FA chain was **−12%** at-depth decode on this box — importing kernels
is receipted-worse; the work is making OUR TILE-FA path BW-efficient at long
KV (GQA 48Q/8KV, head_dim 128), or overlapping the 10 deep layers' reads.

Chrono ledger captured (lx-chrono.bin) but per-name attribution is sync-point
polluted (known artifact) — not load-bearing here; the sweep+power+probe carry
the conclusion.

## Next (iteration 2)

Read the TILE-FA decode path for the at-depth shape (which kernel, workgroup
geometry, KV access pattern per 8-KV-head/128-dim GQA at ne1=1); find why
effective BW is 42%; candidate mutations behind env knobs, gated per AGENTS.md
+ canonical triangulation.

## Iteration 2 addendum — mechanism located; env levers measured

Recon (fattn.cpp / fattn-vec.hpp / fattn-common.hpp, C4 tree): decode (ne1=1)
runs `flash_attn_ext_vec` D=128 WG=256 warp16 — NOT tile. Mechanism of the 42%
efficiency: `ncols2` is hardcoded 1 in the vec launch (fattn-vec.hpp:623,630)
⇒ one workgroup per Q head ⇒ the shared KV head is re-read gqa_ratio (6)×:
at 24.5K depth, 604 MB fetched per token where 100.7 MB is unique (L2 absorbs
part — that is why the slope is 2.4× ideal rather than 6×). Split-K exists but
`parallel_blocks` is capped at max_wg_per_cu=4 (fattn-common.hpp:1048-1092;
a commented-out uncap sits at :1070) ⇒ 192 WGs at 48 heads, each serially
walking 6144 KV rows at 24.5K. No mask early-exit at decode (KV_max null,
gate at fattn-common.hpp:1013). K-pass SIMD16 issues two disjoint 128 B
segments per load (lanes split across rows 8 apart) — secondary suspect.

Env-only probes at d {0, 8192, 24576} (fa-tile.md, fa-nt128.md):
  baseline           152.52 / 114.24 / 86.54
  FORCE_TILE=1       147.82 /  97.49 / 63.61   DEAD (tile kernel loses more
                                               than GQA batching saves)
  DECODE_NTHREADS=128 151.45 / 115.92 / 88.63  +2.4% at depth, −0.7% d0 —
                                               marginal, floor-grazing
Ranked mutations (iteration 3+, all env-gated default-OFF on the C4 branch):
  M1 parallel_blocks env override (3-line; more split-K, shorter serial walks)
  M2 vec-kernel ncols2=2 for gqa%2==0 (halves KV re-read 6→3×; the principled
     fix; kernel surgery in fattn-vec.hpp + launch_fattn ncols2 plumbing)
  M3 K-load lane mapping → one contiguous 256 B segment per SIMD16
Prize bound unchanged: +24% decode at 24.5K, more at 100K+.
